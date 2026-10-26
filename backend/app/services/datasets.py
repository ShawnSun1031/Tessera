"""YOLO dataset helpers: stats, data.yaml and importing a labeled zip."""

import random
import shutil
import tempfile
import zipfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from app.services.ingest import IMAGE_SUFFIXES
from app.services.tif import TIF_SUFFIXES, convert_tif

SPLITS = ("train", "val")


def write_data_yaml(root: Path, names: Sequence[str], relative: bool = False) -> Path:
    """data.yaml for Ultralytics.

    `relative` omits `path:` so Ultralytics resolves the splits against the
    folder holding data.yaml, wherever the dataset is unpacked.
    """
    yaml = [] if relative else [f"path: {root.resolve()}"]
    yaml += ["train: images/train", "val: images/val"]
    yaml += ["names:"] + [f"  {i}: {n}" for i, n in enumerate(names)]
    (root / "data.yaml").write_text("\n".join(yaml) + "\n")
    return root / "data.yaml"


def dataset_stats(root: Path, names: Sequence[str]) -> dict[str, Any]:
    """Images per split, instances per class, and how many images have no labels."""
    stats: dict[str, Any] = {"splits": {}, "instances": dict.fromkeys(names, 0)}
    for split in SPLITS:
        imgs = [p for p in (root / "images" / split).glob("*") if p.is_file()]
        empty = 0
        for img in imgs:
            label = root / "labels" / split / f"{img.stem}.txt"
            lines = label.read_text().split("\n") if label.exists() else []
            lines = [ln for ln in lines if ln.strip()]
            empty += not lines
            for ln in lines:
                cls = int(ln.split()[0])
                if 0 <= cls < len(names):
                    stats["instances"][names[cls]] += 1
        stats["splits"][split] = {"images": len(imgs), "unlabeled": empty}
    return stats


def guess_split(path: Path) -> str | None:
    parts = {p.lower() for p in path.parts[:-1]}
    if parts & {"val", "valid", "validation", "test"}:
        return "val"
    if "train" in parts:
        return "train"
    return None


def import_yolo_zip(
    src: Path,
    out: Path,
    names: Sequence[str],
    task_type: str,
    val_ratio: float = 0.2,
    seed: int = 0,
) -> Path:
    """Build a dataset from a zip of images + YOLO .txt labels matched by file stem.

    Train/val split is taken from folder names (train/, val/) when present,
    otherwise images are split randomly.
    """
    with zipfile.ZipFile(src) as zf, tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)
        labels: dict[str, str] = {}
        images: list[tuple[Path, str | None]] = []
        for info in zf.infolist():
            member = Path(info.filename)
            if info.is_dir() or member.name.startswith("."):
                continue
            suffix = member.suffix.lower()
            if suffix == ".txt" and member.name != "classes.txt":
                labels[member.stem] = zf.read(info).decode()
            elif suffix in (*IMAGE_SUFFIXES, *TIF_SUFFIXES):
                # flatten member paths: never trust directory parts from the archive
                extracted = tmp_root / f"{len(images)}_{member.name}"
                with zf.open(info) as fin, extracted.open("wb") as fout:
                    shutil.copyfileobj(fin, fout)
                images.append((extracted, guess_split(member)))
        if not images:
            raise ValueError("The zip contains no images")
        if not labels:
            raise ValueError("The zip contains no YOLO .txt label files")

        unsplit = [i for i, (_, split) in enumerate(images) if split is None]
        random.Random(seed).shuffle(unsplit)
        n_val = max(1, int(len(unsplit) * val_ratio)) if len(unsplit) > 1 else 0
        val_ids = set(unsplit[:n_val])
        assigned = [
            split or ("val" if i in val_ids else "train")
            for i, (_, split) in enumerate(images)
        ]

        shutil.rmtree(out, ignore_errors=True)
        for split in SPLITS:
            (out / "images" / split).mkdir(parents=True)
            (out / "labels" / split).mkdir(parents=True)
        for (path, _), split in zip(images, assigned):
            stem = path.name.split("_", 1)[1].rsplit(".", 1)[0]
            if path.suffix.lower() in TIF_SUFFIXES:
                image = convert_tif(path, tmp_root / "png")[0]
            else:
                image = path
            dest = out / "images" / split / f"{stem}{image.suffix}"
            shutil.copyfile(image, dest)
            text = labels.get(stem, "")
            validate_labels(text, len(names), task_type, stem)
            (out / "labels" / split / f"{stem}.txt").write_text(text)

    if not any((out / "images" / "val").iterdir()):
        # all images were in train/: validate on train like the LS exporter does
        for kind in ("images", "labels"):
            train, val = out / kind / "train", out / kind / "val"
            shutil.copytree(train, val, dirs_exist_ok=True)
    return write_data_yaml(out, names)


def validate_labels(text: str, n_classes: int, task_type: str, stem: str) -> None:
    for n, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        where = f"{stem}.txt line {n}"
        try:
            cls, coords = int(parts[0]), [float(v) for v in parts[1:]]
        except ValueError:
            raise ValueError(f"{where}: not a YOLO label line") from None
        if not 0 <= cls < n_classes:
            raise ValueError(f"{where}: class id {cls} but project has {n_classes}")
        if task_type == "detect" and len(coords) != 4:
            raise ValueError(f"{where}: detection labels need 4 values")
        if task_type == "segment" and (len(coords) < 6 or len(coords) % 2):
            raise ValueError(f"{where}: segmentation labels need x y pairs (>=3)")


def label_objects(label_file: Path) -> list[dict[str, Any]]:
    """Objects in a YOLO label file as normalized outline points.

    Boxes (cx cy w h) are returned as their 4 corners so callers can draw
    detection and segmentation labels the same way.
    """
    if not label_file.exists():
        return []
    objects = []
    for line in label_file.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls, coords = int(parts[0]), [float(v) for v in parts[1:]]
        if len(coords) == 4:
            cx, cy, w, h = coords
            x0, y0, x1, y1 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
            points = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
            kind = "box"
        else:
            points = [[coords[i], coords[i + 1]] for i in range(0, len(coords) - 1, 2)]
            kind = "polygon"
        objects.append({"class_id": cls, "kind": kind, "points": points})
    return objects


def list_items(
    root: Path,
    split: str,
    class_id: int | None = None,
    unlabeled: bool = False,
) -> list[dict[str, Any]]:
    """Images of one split with their objects, optionally filtered."""
    items = []
    images = sorted(p for p in (root / "images" / split).glob("*") if p.is_file())
    for img in images:
        objects = label_objects(root / "labels" / split / f"{img.stem}.txt")
        if unlabeled and objects:
            continue
        if class_id is not None and not any(o["class_id"] == class_id for o in objects):
            continue
        items.append({"split": split, "name": img.name, "objects": objects})
    return items


def preview_image(source: Path, cache_dir: Path, max_size: int) -> Path:
    """Browser-viewable PNG/JPEG of a dataset image, resized and cached.

    TIFs (including 16-bit) are converted since browsers can't show them.
    """
    from PIL import Image

    from app.services.tif import frame_to_image

    suffix = ".png" if source.suffix.lower() in (".png", *TIF_SUFFIXES) else ".jpg"
    cached = cache_dir / f"{source.parent.name}_{max_size}_{source.stem}{suffix}"
    if cached.exists() and cached.stat().st_mtime >= source.stat().st_mtime:
        return cached
    cache_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as im:
        frame = frame_to_image(im)
    frame.thumbnail((max_size, max_size))
    if suffix == ".jpg" and frame.mode not in ("RGB", "L"):
        frame = frame.convert("RGB")
    tmp = cached.with_suffix(f".tmp{suffix}")
    frame.save(tmp, quality=88) if suffix == ".jpg" else frame.save(tmp)
    tmp.replace(cached)
    return cached
