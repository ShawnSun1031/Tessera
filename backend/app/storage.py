"""Filesystem layout under STORAGE_DIR.

projects/{project_id}/images/        uploaded images (converted to PNG)
projects/{project_id}/datasets/{id}/ YOLO datasets (data.yaml, images/, labels/)
projects/{project_id}/base_weights/  user-uploaded .pt files to start training from
jobs/{job_id}/                       job log, inputs and outputs
jobs/{job_id}/uploads/               files sent for inference, before conversion
jobs/{job_id}/train/                 Ultralytics run dir (weights/best.pt, plots)
weights/                             downloaded pretrained weights
label-studio.log                     output of a Label Studio started from the UI
"""

from pathlib import Path

from fastapi import HTTPException, status

from app.config import get_settings


def root() -> Path:
    return get_settings().storage_dir


def project_dir(project_id: int) -> Path:
    return root() / "projects" / str(project_id)


def images_dir(project_id: int) -> Path:
    return project_dir(project_id) / "images"


def dataset_dir(project_id: int, dataset_id: int) -> Path:
    return project_dir(project_id) / "datasets" / str(dataset_id)


def base_weights_dir(project_id: int) -> Path:
    return project_dir(project_id) / "base_weights"


def job_dir(job_id: int) -> Path:
    return root() / "jobs" / str(job_id)


def uploads_dir(job_id: int) -> Path:
    return job_dir(job_id) / "uploads"


def output_zip(job_id: int) -> Path:
    return job_dir(job_id) / "output.zip"


def log_path(job_id: int) -> Path:
    return job_dir(job_id) / "log.txt"


def train_dir(job_id: int) -> Path:
    return job_dir(job_id) / "train"


def weights_path(train_job_id: int) -> Path:
    return train_dir(train_job_id) / "weights" / "best.pt"


def onnx_path(train_job_id: int) -> Path:
    return weights_path(train_job_id).with_suffix(".onnx")


def label_studio_log() -> Path:
    return root() / "label-studio.log"


def weights_cache() -> Path:
    return root() / "weights"


def safe_child(base: Path, relative: str) -> Path:
    """Resolve `relative` under `base`, refusing anything that escapes it."""
    path = (base / relative).resolve()
    if not path.is_relative_to(base.resolve()) or not path.is_file():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "File not found")
    return path
