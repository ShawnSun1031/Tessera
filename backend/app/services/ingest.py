"""Turn uploaded files (TIFs, plain images, zips of either) into PNG/JPG images."""

import re
import shutil
import tempfile
import uuid
import zipfile
from pathlib import Path

from app.services.tif import TIF_SUFFIXES, convert_tif

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
ACCEPTED_SUFFIXES = (*TIF_SUFFIXES, *IMAGE_SUFFIXES, ".zip")


def natural_key(name: str) -> list[int | str]:
    """Sort key that orders "2.tif" before "10.tif", like a file browser."""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", name.lower())]


def sortable_name(name: str, width: int = 6) -> str:
    """Zero-pad numbers so text sorting matches number order.

    "2.png" -> "000002.png", which sorts before "000010.png".
    """
    return re.sub(r"\d+", lambda m: m.group().zfill(width), name)


def unique_path(directory: Path, name: str) -> Path:
    path = directory / name
    if path.exists():
        stem, suffix = Path(name).stem, Path(name).suffix
        path = directory / f"{stem}_{uuid.uuid4().hex[:8]}{suffix}"
    return path


def ingest_file(src: Path, images_dir: Path) -> list[tuple[Path, str]]:
    """Convert/copy `src` into `images_dir`; return [(image_path, source_name)]."""
    images_dir.mkdir(parents=True, exist_ok=True)
    suffix = src.suffix.lower()
    if suffix in TIF_SUFFIXES:
        with tempfile.TemporaryDirectory() as tmp:
            outs = []
            for png in convert_tif(src, tmp):
                dest = unique_path(images_dir, png.name)
                shutil.move(png, dest)
                outs.append((dest, src.name))
            return outs
    if suffix in IMAGE_SUFFIXES:
        dest = unique_path(images_dir, src.name)
        shutil.copyfile(src, dest)
        return [(dest, src.name)]
    if suffix == ".zip":
        return ingest_zip(src, images_dir)
    raise ValueError(f"Unsupported file type: {src.name}")


def ingest_zip(src: Path, images_dir: Path) -> list[tuple[Path, str]]:
    outs = []
    with zipfile.ZipFile(src) as zf, tempfile.TemporaryDirectory() as tmp:
        members = sorted(zf.infolist(), key=lambda i: natural_key(i.filename))
        for info in members:
            # flatten member paths: never trust directory parts from the archive
            name = Path(info.filename).name
            if info.is_dir() or name.startswith(".") or not name:
                continue
            if Path(name).suffix.lower() not in (*TIF_SUFFIXES, *IMAGE_SUFFIXES):
                continue
            extracted = Path(tmp) / name
            with zf.open(info) as fin, extracted.open("wb") as fout:
                shutil.copyfileobj(fin, fout)
            outs += ingest_file(extracted, images_dir)
            extracted.unlink()
    return outs
