"""Thin wrappers around Ultralytics train / predict / export, free of DB code."""

import json
import math
import shutil
import tempfile
import zipfile
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from ultralytics.engine.results import Results

ARCHITECTURES: list[dict[str, str]] = [
    {"family": "YOLO26", "size": "n", "note": "Fastest, smallest. Good first try."},
    {"family": "YOLO26", "size": "s", "note": "Small: better accuracy, still fast."},
    {"family": "YOLO26", "size": "m", "note": "Medium: balanced speed and accuracy."},
    {"family": "YOLO26", "size": "l", "note": "Large: high accuracy, slower."},
    {"family": "YOLO26", "size": "x", "note": "Extra large: best accuracy, slowest."},
    {"family": "YOLO11", "size": "n", "note": "Previous generation, nano."},
    {"family": "YOLO11", "size": "s", "note": "Previous generation, small."},
    {"family": "YOLO11", "size": "m", "note": "Previous generation, medium."},
]

# Ultralytics augmentation overrides; "default" keeps Ultralytics' defaults
AUGMENT_PRESETS: dict[str, dict[str, float]] = {
    "none": {
        "mosaic": 0.0, "fliplr": 0.0, "hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.0,
        "translate": 0.0, "scale": 0.0, "erasing": 0.0, "auto_augment": 0.0,
    },
    "light": {"mosaic": 0.5, "scale": 0.25, "translate": 0.05, "hsv_s": 0.3},
    "default": {},
    "strong": {"degrees": 15.0, "flipud": 0.5, "scale": 0.7, "mixup": 0.1},
}

# metric keys shown as the model's headline numbers
PREDICT_BATCH = 16  # images per GPU call; halved automatically on out-of-memory
SAVE_THREADS = 8  # threads drawing and writing overlay PNGs

SUMMARY_KEYS = (
    "metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)",
    "metrics/mAP50-95(B)", "metrics/mAP50(M)", "metrics/mAP50-95(M)",
)  # fmt: skip


def weights_name(family: str, size: str, task_type: str) -> str:
    suffix = "-seg" if task_type == "segment" else ""
    return f"{family.lower()}{size}{suffix}.pt"


def architectures(task_type: str) -> list[dict[str, str]]:
    return [
        {**a, "weights": weights_name(a["family"], a["size"], task_type)}
        for a in ARCHITECTURES
    ]


def clean_metrics(values: dict[str, Any]) -> dict[str, float]:
    out = {}
    for k, v in values.items():
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f):
            out[k] = round(f, 6)
    return out


class Cancelled(Exception):
    pass


def train(
    data_yaml: Path,
    base_weights: str,
    task_type: str,
    run_dir: Path,
    params: dict[str, Any],
    device: str,
    on_epoch: Callable[[int, dict[str, float]], None],
    on_batch: Callable[[dict[str, int]], bool],
) -> dict[str, float]:
    """Train and return the final (best.pt) metrics.

    `on_batch` returns True to request a graceful stop: Ultralytics then
    validates, saves and finishes, so best.pt still exists.
    """
    from ultralytics import YOLO

    model = YOLO(base_weights)
    if model.task != task_type:  # possible with weights uploaded by the user
        raise ValueError(
            f"{Path(base_weights).name} is a {model.task} model, "
            f"but this project needs a {task_type} model"
        )

    def fit_epoch_end(trainer: Any) -> None:  # noqa: ANN401
        # final_eval re-validates best.pt and reports it as epoch N+1 without an
        # epoch start; those are the returned metrics, not a training epoch
        if not getattr(trainer, "_platform_in_epoch", False):
            return
        trainer._platform_in_epoch = False
        losses = trainer.label_loss_items(trainer.tloss) if trainer.tloss else {}
        values = clean_metrics({**losses, **trainer.metrics, **trainer.lr})
        on_epoch(trainer.epoch + 1, values)

    def batch_end(trainer: Any) -> None:  # noqa: ANN401
        nb = len(trainer.train_loader)
        progress = {
            "epoch": trainer.epoch + 1,
            "epochs": trainer.epochs,
            "batch": getattr(trainer, "_platform_batch", 0) % nb + 1,
            "batches": nb,
        }
        trainer._platform_batch = getattr(trainer, "_platform_batch", 0) + 1
        if on_batch(progress):
            trainer.stop = True

    def epoch_start(trainer: Any) -> None:  # noqa: ANN401
        trainer._platform_batch = 0
        trainer._platform_in_epoch = True

    model.add_callback("on_fit_epoch_end", fit_epoch_end)
    model.add_callback("on_train_batch_end", batch_end)
    model.add_callback("on_train_epoch_start", epoch_start)

    augment = AUGMENT_PRESETS[params.get("augment", "default")]
    args = {k: params[k] for k in ("epochs", "imgsz", "batch", "lr0", "patience")}
    model.train(
        data=str(data_yaml),
        project=str(run_dir.parent),
        name=run_dir.name,
        exist_ok=True,
        device=device or None,
        workers=params.get("workers", 4),
        **args,
        **augment,
    )
    metrics = getattr(model.trainer, "metrics", None) or {}
    return clean_metrics(metrics)


def result_objects(result: "Results") -> list[dict[str, Any]]:
    boxes = result.boxes
    if boxes is None:
        return []
    polys = result.masks.xyn if result.masks is not None else None
    objects = []
    for j in range(len(boxes)):
        cls = int(boxes.cls[j])
        obj: dict[str, Any] = {
            "class_id": cls,
            "class": result.names[cls],
            "confidence": round(float(boxes.conf[j]), 4),
            "box_xyxy": [round(float(v), 2) for v in boxes.xyxy[j]],
        }
        if polys is not None:
            obj["polygon_xyn"] = [
                [round(float(x), 5), round(float(y), 5)] for x, y in polys[j]
            ]
        objects.append(obj)
    return objects


def predict(
    weights: Path,
    inputs: list[Path],
    out_dir: Path,
    conf: float,
    iou: float,
    imgsz: int | None,
    device: str,
    on_progress: Callable[[int, int], bool],
) -> dict[str, Any]:
    """Run inference; writes overlays to out_dir/images and out_dir/predictions.json.

    `imgsz` None uses the size stored in the checkpoint (its training size).

    Images go through the GPU in batches while overlays are drawn and saved on
    other threads (PNG encoding takes about as long as the inference itself).
    """
    import torch
    from ultralytics import YOLO

    model = YOLO(str(weights))
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    records = []
    counts: dict[str, int] = {}
    batch = PREDICT_BATCH
    size: dict[str, Any] = {"imgsz": imgsz} if imgsz else {}
    saving: list[Future[Any]] = []
    with ThreadPoolExecutor(SAVE_THREADS) as pool:
        i = 0
        while i < len(inputs):
            if on_progress(i, len(inputs)):
                raise Cancelled
            chunk = inputs[i : i + batch]
            try:
                results = model.predict(
                    [str(p) for p in chunk], conf=conf, iou=iou,
                    device=device or None, batch=batch, verbose=False,
                    **size,
                )  # fmt: skip
            except torch.cuda.OutOfMemoryError:
                if batch == 1:
                    raise
                batch //= 2  # big model or image size: retry with smaller batches
                torch.cuda.empty_cache()
                continue
            # keep at most one batch of overlays waiting to be written
            for f in saving:
                f.result()
            saving = []
            for path, result in zip(chunk, cast("list[Results]", results)):
                overlay = str(out_dir / "images" / f"{path.stem}.png")
                saving.append(pool.submit(result.save, filename=overlay))
                objects = result_objects(result)
                for obj in objects:
                    counts[obj["class"]] = counts.get(obj["class"], 0) + 1
                height, width = result.orig_shape[:2]
                records.append(
                    {"image": path.name, "width": width, "height": height,
                     "objects": objects}
                )  # fmt: skip
            i += len(chunk)
        for f in saving:
            f.result()
    on_progress(len(inputs), len(inputs))
    (out_dir / "predictions.json").write_text(json.dumps(records, indent=1))
    classes = [model.names[i] for i in sorted(model.names)]
    return {
        "images": len(inputs), "objects": sum(counts.values()), "per_class": counts,
        "classes": classes,
    }  # fmt: skip


def zip_output(out_dir: Path, archive: Path) -> None:
    """Zip inference output, storing PNGs as-is: recompressing them gains nothing."""
    partial = archive.with_suffix(".part")
    with zipfile.ZipFile(partial, "w", zipfile.ZIP_STORED) as zf:
        for path in sorted(p for p in out_dir.rglob("*") if p.is_file()):
            png = path.suffix == ".png"
            method = zipfile.ZIP_STORED if png else zipfile.ZIP_DEFLATED
            zf.write(path, path.relative_to(out_dir), compress_type=method)
    partial.replace(archive)


def yolo_label_lines(record: dict[str, Any], segment: bool) -> list[str]:
    """YOLO label lines for one predictions.json record: polygons when `segment`,
    else boxes (class cx cy w h), all normalized to 0..1."""
    w, h = record["width"], record["height"]
    lines = []
    for obj in record["objects"]:
        cls = obj["class_id"]
        if segment:
            poly = obj.get("polygon_xyn") or []
            if len(poly) >= 3:
                lines.append(f"{cls} " + " ".join(f"{x:.6f} {y:.6f}" for x, y in poly))
            continue
        x0, y0, x1, y1 = obj["box_xyxy"]
        cx, cy = (x0 + x1) / 2 / w, (y0 + y1) / 2 / h
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}")
    return lines


def prediction_classes(records: list[dict[str, Any]]) -> list[str]:
    """Class names seen in predictions, for runs from before `classes` was saved;
    ids without a detection get a placeholder name."""
    seen = {o["class_id"]: o["class"] for r in records for o in r["objects"]}
    return [seen.get(i, f"class{i}") for i in range(max(seen, default=-1) + 1)]


def zip_yolo(
    records: list[dict[str, Any]], inputs: Path, classes: list[str], archive: Path
) -> None:
    """Zip the input images with the predictions as YOLO labels, plus data.yaml.

    Segmentation models' output becomes polygons, detection models' boxes.
    """
    segment = any("polygon_xyn" in o for r in records for o in r["objects"])
    yaml = ["train: images", "val: images", "names:"]
    yaml += [f"  {i}: {n}" for i, n in enumerate(classes)]
    with tempfile.NamedTemporaryFile(dir=archive.parent, delete=False) as tmp:
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("data.yaml", "\n".join(yaml) + "\n")
            for record in records:
                image = inputs / record["image"]
                # PNGs gain nothing from recompressing
                stored = zipfile.ZIP_STORED
                zf.write(image, f"images/{image.name}", compress_type=stored)
                lines = yolo_label_lines(record, segment)
                text = "\n".join(lines) + ("\n" if lines else "")
                zf.writestr(f"labels/{image.stem}.txt", text)
    # atomic: concurrent downloads never see a half-written zip
    Path(tmp.name).replace(archive)


def export_onnx(weights: Path, imgsz: int) -> Path:
    from ultralytics import YOLO

    exported = Path(YOLO(str(weights)).export(format="onnx", imgsz=imgsz))
    target = weights.with_suffix(".onnx")
    if exported != target:
        shutil.move(exported, target)
    return target
