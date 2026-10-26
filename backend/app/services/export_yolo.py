"""Convert Label Studio exports into Ultralytics YOLO datasets (detect or segment).

Supported LS result types: brushlabels (RLE), polygonlabels, rectanglelabels.
In `segment` mode rectangles become 4-point polygons; in `detect` mode brushes and
polygons become their bounding boxes.
"""

import argparse
import io
import math
import os
import random
import re
import shutil
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np
from label_studio_sdk.converter.brush import decode_rle
from PIL import Image

from app.services.datasets import write_data_yaml
from app.services.ingest import IMAGE_SUFFIXES
from app.services.tif import TIF_SUFFIXES
from app.services.labelstudio import LabelStudio, load_env

TASK_TYPES = ("detect", "segment")

Task = dict[str, Any]
Result = dict[str, Any]
Shapes = tuple[str, list[np.ndarray], int, int]


def class_names(label_config: str) -> list[str]:
    # the same labels may appear under several control tags (e.g. brush + polygon)
    return list(dict.fromkeys(re.findall(r'<Label\s+value="([^"]+)"', label_config)))


def image_key(label_config: str) -> str:
    """Task data field holding the image, e.g. `image` for value="$image"."""
    m = re.search(r'<Image\b[^>]*\bvalue="\$([\w.-]+)"', label_config)
    return m.group(1) if m else "image"


def infer_task_type(label_config: str) -> str | None:
    """`segment` for brush/polygon configs, `detect` for rectangles, else None."""
    if re.search(r"<(BrushLabels|PolygonLabels)\b", label_config):
        return "segment"
    if re.search(r"<RectangleLabels\b", label_config):
        return "detect"
    return None


def image_name(url: str, data: bytes) -> str:
    """File name for a task image URL, with an image suffix Ultralytics accepts."""
    parsed = urlparse(url)
    # local-files storage: /data/local-files/?d=folder/img.png
    name = Path(parse_qs(parsed.query).get("d", [parsed.path])[0]).name or "image"
    if Path(name).suffix.lower() not in (*IMAGE_SUFFIXES, *TIF_SUFFIXES):
        with Image.open(io.BytesIO(data)) as im:
            name = f"{name}.{(im.format or 'png').lower()}"
    return name


def mask_to_polygons(mask: np.ndarray, min_area: float) -> list[np.ndarray]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in contours:
        if cv2.contourArea(c) < min_area or len(c) < 3:
            continue
        polys.append(c.reshape(-1, 2))
    return polys


def rect_corners(v: dict[str, Any], w: int, h: int) -> np.ndarray:
    """Pixel corners of an LS rectangle (percent coords, rotated about top-left)."""
    x, y = v["x"] / 100 * w, v["y"] / 100 * h
    rw, rh = v["width"] / 100 * w, v["height"] / 100 * h
    a = math.radians(v.get("rotation", 0) or 0)
    cos, sin = math.cos(a), math.sin(a)
    offsets = ((0, 0), (rw, 0), (rw, rh), (0, rh))
    return np.array(
        [(x + dx * cos - dy * sin, y + dx * sin + dy * cos) for dx, dy in offsets]
    )


def result_to_shapes(r: Result, min_area: float) -> Shapes | None:
    """Return (label, [polygon_px, ...], w, h) for one LS result, or None."""
    kind, v = r.get("type"), r.get("value", {})
    w, h = r.get("original_width"), r.get("original_height")
    if not w or not h:
        return None
    if kind == "brushlabels" and v.get("format") == "rle":
        # decoded RLE is RGBA; alpha channel holds the mask
        mask = decode_rle(v["rle"]).reshape(h, w, 4)[:, :, 3]
        polys = mask_to_polygons((mask > 127).astype(np.uint8), min_area)
        return v["brushlabels"][0], polys, w, h
    if kind == "polygonlabels":
        poly = np.array(v["points"], dtype=float) * [w / 100, h / 100]
        return v["polygonlabels"][0], [poly] if len(poly) >= 3 else [], w, h
    if kind == "rectanglelabels":
        return v["rectanglelabels"][0], [rect_corners(v, w, h)], w, h
    return None


def annotation_to_lines(
    results: list[Result],
    class_ids: dict[str, int],
    task_type: str,
    min_area: float = 4.0,
) -> list[str]:
    lines = []
    for r in results:
        shapes = result_to_shapes(r, min_area)
        if shapes is None:
            continue
        label, polys, w, h = shapes
        if label not in class_ids:
            continue
        cls = class_ids[label]
        for poly in polys:
            poly = np.clip(np.asarray(poly, dtype=float) / [w, h], 0, 1)
            if task_type == "segment":
                coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in poly)
                lines.append(f"{cls} {coords}")
            else:
                (x0, y0), (x1, y1) = poly.min(0), poly.max(0)
                if x1 > x0 and y1 > y0:
                    cx, cy, bw, bh = (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0
                    lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return lines


def latest_annotation(task: Task) -> dict[str, Any] | None:
    anns = [
        a
        for a in task.get("annotations", [])
        if not a.get("was_cancelled") and a.get("result")
    ]
    return anns[-1] if anns else None


def split_tasks(
    tasks: Sequence[Task], val_ratio: float, seed: int
) -> dict[str, list[Task]]:
    tasks = list(tasks)
    random.Random(seed).shuffle(tasks)
    n_val = int(len(tasks) * val_ratio) if len(tasks) > 1 else 0
    if n_val == 0:
        # too few images to split; validate on train
        return {"train": tasks, "val": tasks}
    return {"val": tasks[:n_val], "train": tasks[n_val:]}


def write_dataset(
    tasks: Sequence[Task],
    names: list[str],
    out: str | Path,
    fetch_image: Callable[[str], bytes],
    task_type: str = "segment",
    val_ratio: float = 0.2,
    min_area: float = 4.0,
    seed: int = 0,
    log: Callable[[str], None] = print,
    data_key: str = "image",
) -> Path:
    """Write a YOLO dataset from LS tasks. Returns the path to data.yaml."""
    if task_type not in TASK_TYPES:
        raise ValueError(f"task_type must be one of {TASK_TYPES}")
    class_ids = {n: i for i, n in enumerate(names)}
    tasks = [t for t in tasks if latest_annotation(t)]
    log(f"{len(tasks)} annotated tasks, classes: {class_ids}")

    out = Path(out)
    shutil.rmtree(out, ignore_errors=True)
    for split, items in split_tasks(tasks, val_ratio, seed).items():
        img_dir, lbl_dir = out / "images" / split, out / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for t in items:
            ann = latest_annotation(t)
            if ann is None:
                continue
            url = t["data"].get(data_key)
            if not isinstance(url, str):
                log(f"[{split}] task {t['id']}: no '{data_key}' image field, skipped")
                continue
            data = fetch_image(url)
            name = image_name(url, data)
            if (img_dir / name).exists():
                name = f"{t['id']}_{name}"
            (img_dir / name).write_bytes(data)
            lines = annotation_to_lines(ann["result"], class_ids, task_type, min_area)
            text = "\n".join(lines) + ("\n" if lines else "")
            (lbl_dir / f"{Path(name).stem}.txt").write_text(text)
            log(f"[{split}] {name}: {len(lines)} objects")

    data_yaml = write_data_yaml(out, names)
    log(f"Dataset written to {data_yaml}")
    return data_yaml


def export_project(
    ls: LabelStudio,
    project_id: int,
    out: str | Path,
    task_type: str = "segment",
    val_ratio: float = 0.2,
    min_area: float = 4.0,
    seed: int = 0,
    log: Callable[[str], None] = print,
) -> Path:
    """Export a Label Studio project to a YOLO dataset."""
    config = ls.project(project_id)["label_config"]
    tasks = ls.export_tasks(project_id)
    return write_dataset(
        tasks, class_names(config), out, ls.download, task_type, val_ratio,
        min_area, seed, log, data_key=image_key(config),
    )  # fmt: skip


def main() -> None:
    p = argparse.ArgumentParser(
        description="Export a Label Studio project to an Ultralytics YOLO dataset."
    )
    p.add_argument("--host", default="http://localhost:8080")
    p.add_argument("--project", type=int, required=True)
    p.add_argument("--out", default="datasets/labelstudio_seg")
    p.add_argument("--task", choices=TASK_TYPES, default="segment")
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument(
        "--min-area", type=float, default=4.0, help="drop contours below this px^2"
    )
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    load_env()
    ls = LabelStudio.from_key(args.host, os.environ["LABEL_STUDIO_API_KEY"])
    export_project(
        ls, args.project, args.out, args.task, args.val_ratio, args.min_area, args.seed
    )


if __name__ == "__main__":
    main()
