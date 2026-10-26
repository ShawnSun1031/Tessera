"""Convert TIF images (16-bit, float, multi-page) to 8-bit PNGs.

Output is suitable for Label Studio and YOLO training.
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageSequence

TIF_SUFFIXES = (".tif", ".tiff")


def to_uint8(
    arr: np.ndarray, low_pct: float = 0.5, high_pct: float = 99.5
) -> np.ndarray:
    """Percentile-stretch an array of any dtype to uint8."""
    if arr.dtype == np.uint8:
        return arr
    arr = arr.astype(np.float32)
    lo, hi = np.percentile(arr, (low_pct, high_pct))
    if hi <= lo:
        hi = lo + 1
    return (np.clip((arr - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


def frame_to_image(frame: Image.Image) -> Image.Image:
    if frame.mode in ("RGB", "L"):
        return frame.copy()
    if frame.mode in ("RGBA", "LA", "P", "1", "CMYK", "YCbCr"):
        return frame.convert("RGB")
    # I;16, I, F and friends: normalize to 8-bit grayscale
    return Image.fromarray(to_uint8(np.array(frame)), mode="L")


def convert_tif(path: str | Path, out_dir: str | Path) -> list[Path]:
    """Write one PNG per page of `path` into `out_dir`; return the written paths."""
    path, out_dir = Path(path), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(path) as im:
        frames = [frame_to_image(f) for f in ImageSequence.Iterator(im)]
    if len(frames) == 1:
        names = [f"{path.stem}.png"]
    else:
        names = [f"{path.stem}_p{i:03d}.png" for i in range(len(frames))]
    outs = []
    for frame, name in zip(frames, names):
        frame.save(out_dir / name)
        outs.append(out_dir / name)
    return outs


def convert_dir(src: str | Path, out_dir: str | Path) -> list[Path]:
    src = Path(src)
    if src.is_file():
        files = [src]
    else:
        files = sorted(p for p in src.iterdir() if p.suffix.lower() in TIF_SUFFIXES)
    return [o for f in files for o in convert_tif(f, out_dir)]


def main() -> None:
    p = argparse.ArgumentParser(
        description="Convert a TIF file or a directory of TIFs to 8-bit PNGs."
    )
    p.add_argument("src", help="TIF file or directory")
    p.add_argument("--out", default="data")
    args = p.parse_args()
    for path in convert_dir(args.src, args.out):
        print(path)


if __name__ == "__main__":
    main()
