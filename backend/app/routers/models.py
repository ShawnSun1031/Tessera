import shutil
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlmodel import Session, col, select

from app import storage
from app.auth import CurrentUser, SessionDep
from app.config import get_settings
from app.deps import JobDep, ModelDep, ProjectDep, enqueue
from app.db import get_engine
from app.models import Dataset, Job, Project, TrainedModel
from app.routers.jobs import JobRead, to_read
from app.services import training
from app.services.ingest import ACCEPTED_SUFFIXES

router = APIRouter(tags=["models"])


class TrainRequest(BaseModel):
    dataset_id: int
    base_model: str = Field(
        description="Catalog weights name, model:<id> or upload:<file name>"
    )
    name: str | None = None
    epochs: int = Field(100, ge=1, le=2000)
    imgsz: int = Field(640, ge=64, le=2048, multiple_of=32)
    batch: int = Field(16, ge=-1, le=512)
    lr0: float = Field(0.01, gt=0, le=1)
    patience: int = Field(50, ge=0, le=2000)
    augment: Literal["none", "light", "default", "strong"] = "default"


class ModelRead(BaseModel):
    id: int
    project_id: int
    job_id: int
    dataset_id: int
    name: str
    task_type: str
    architecture: str
    classes: list[str]
    metrics: dict[str, float]
    created_at: datetime
    onnx_available: bool
    plots: list[str]
    imgsz: int


def model_read(model: TrainedModel) -> ModelRead:
    run = storage.train_dir(model.job_id)
    plots = sorted(
        str(p.relative_to(run)) for p in run.glob("*") if p.suffix in (".png", ".jpg")
    )
    return ModelRead(
        **model.model_dump(),
        onnx_available=storage.onnx_path(model.job_id).exists(),
        plots=plots,
        imgsz=trained_imgsz(model),
    )


def trained_imgsz(model: TrainedModel) -> int:
    """Image size the model was trained at; inference should use the same."""
    with Session(get_engine()) as session:
        job = session.get(Job, model.job_id)
        return int(job.params.get("imgsz", 640)) if job else 640


@router.get("/projects/{project_id}/architectures")
def list_architectures(project: ProjectDep) -> dict[str, Any]:
    return {
        "architectures": training.architectures(project.task_type),
        "augment_presets": list(training.AUGMENT_PRESETS),
    }


@router.post("/projects/{project_id}/train", status_code=201)
def start_training(
    body: TrainRequest, project: ProjectDep, session: SessionDep, user: CurrentUser
) -> JobRead:
    dataset = session.get(Dataset, body.dataset_id)
    if dataset is None or dataset.project_id != project.id or not dataset.stats:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Dataset not found")
    check_base_model(session, project, body.base_model)
    job = enqueue(session, project, user, "train", body.model_dump())
    return to_read(session, job)


def check_base_model(session: SessionDep, project: Project, base: str) -> None:
    if base.startswith("model:"):
        try:
            model = session.get(TrainedModel, int(base.split(":", 1)[1]))
        except ValueError:
            model = None
        if model is None or model.project_id != project.id:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Base model not found")
        return
    if base.startswith("upload:"):
        assert project.id is not None
        storage.safe_child(storage.base_weights_dir(project.id), base[len("upload:") :])
        return
    names = {a["weights"] for a in training.architectures(project.task_type)}
    if base not in names:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT, f"Unknown architecture {base}"
        )


class BaseWeightsRead(BaseModel):
    name: str
    size: int
    uploaded_at: datetime


def base_weights_read(path: Path) -> BaseWeightsRead:
    stat = path.stat()
    return BaseWeightsRead(
        name=path.name,
        size=stat.st_size,
        uploaded_at=datetime.fromtimestamp(stat.st_mtime, UTC),
    )


@router.get("/projects/{project_id}/base-weights")
def list_base_weights(project: ProjectDep) -> list[BaseWeightsRead]:
    """Weights files uploaded to this project, newest first."""
    assert project.id is not None
    folder = storage.base_weights_dir(project.id)
    if not folder.is_dir():
        return []
    files = [base_weights_read(p) for p in folder.glob("*.pt")]
    return sorted(files, key=lambda f: f.uploaded_at, reverse=True)


@router.post("/projects/{project_id}/base-weights", status_code=201)
def upload_base_weights(file: UploadFile, project: ProjectDep) -> BaseWeightsRead:
    """Store a .pt trained elsewhere so training can start from it.

    Only the file format is checked here; the worker checks that it is an
    Ultralytics model of the project's task when a training run loads it.
    """
    assert project.id is not None
    original = Path(file.filename or "")
    if original.suffix.lower() != ".pt":
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "Upload a PyTorch .pt file"
        )
    stem = "".join(c if c.isalnum() or c in "-_." else "_" for c in original.stem)
    stem = stem.strip(".") or "weights"
    folder = storage.base_weights_dir(project.id)
    folder.mkdir(parents=True, exist_ok=True)
    dest, n = folder / f"{stem}.pt", 1
    while dest.exists():
        n += 1
        dest = folder / f"{stem}-{n}.pt"
    partial = dest.with_name(f".{dest.name}.part")
    with partial.open("wb") as out:
        shutil.copyfileobj(file.file, out)
    if not zipfile.is_zipfile(partial):  # torch.save writes a zip archive
        partial.unlink()
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            f"{original.name} is not a PyTorch weights file",
        )
    partial.replace(dest)
    return base_weights_read(dest)


@router.delete("/projects/{project_id}/base-weights/{name}", status_code=204)
def delete_base_weights(name: str, project: ProjectDep) -> None:
    assert project.id is not None
    storage.safe_child(storage.base_weights_dir(project.id), name).unlink()


@router.get("/projects/{project_id}/models")
def list_models(project: ProjectDep, session: SessionDep) -> list[ModelRead]:
    rows = session.exec(
        select(TrainedModel)
        .where(TrainedModel.project_id == project.id)
        .order_by(col(TrainedModel.id).desc())
    ).all()
    return [model_read(m) for m in rows]


@router.get("/models/{model_id}")
def get_model(model: ModelDep) -> ModelRead:
    return model_read(model)


@router.get("/models/{model_id}/files/{path:path}")
def model_file(model: ModelDep, path: str) -> FileResponse:
    return FileResponse(storage.safe_child(storage.train_dir(model.job_id), path))


@router.get("/models/{model_id}/download")
def download_model(
    model: ModelDep, format: Literal["pt", "onnx"] = "pt"
) -> FileResponse:
    if format == "pt":
        path = storage.weights_path(model.job_id)
    else:
        path = storage.onnx_path(model.job_id)
    if not path.exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No {format} file yet")
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in model.name)
    return FileResponse(path, filename=f"{safe_name}.{format}")


class OnnxRequest(BaseModel):
    # defaults to the training image size
    imgsz: int | None = Field(None, ge=64, le=2048, multiple_of=32)


@router.post("/models/{model_id}/export-onnx", status_code=201)
def export_onnx(
    body: OnnxRequest, model: ModelDep, session: SessionDep, user: CurrentUser
) -> JobRead:
    project = session.get(Project, model.project_id)
    assert project is not None
    params = {"model_id": model.id, "imgsz": body.imgsz or trained_imgsz(model)}
    return to_read(session, enqueue(session, project, user, "onnx", params))


@router.post("/models/{model_id}/predict", status_code=201)
def predict(
    model: ModelDep,
    session: SessionDep,
    user: CurrentUser,
    conf: Annotated[float, Form(gt=0, lt=1)] = 0.25,
    iou: Annotated[float, Form(gt=0, lt=1)] = 0.7,
    imgsz: Annotated[int | None, Form(ge=64, le=2048, multiple_of=32)] = None,
) -> JobRead:
    """Create an inference job. Send images in parts with POST /jobs/{id}/inputs,
    then queue it with POST /jobs/{id}/start."""
    project = session.get(Project, model.project_id)
    assert project is not None
    params = {
        "model_id": model.id, "conf": conf, "iou": iou,
        "imgsz": imgsz or trained_imgsz(model),
    }  # fmt: skip
    job = enqueue(session, project, user, "predict", params, status_="uploading")
    return to_read(session, job)


@router.post("/projects/{project_id}/base-weights/{name}/predict", status_code=201)
def predict_with_uploaded(
    name: str,
    project: ProjectDep,
    session: SessionDep,
    user: CurrentUser,
    conf: Annotated[float, Form(gt=0, lt=1)] = 0.25,
    iou: Annotated[float, Form(gt=0, lt=1)] = 0.7,
    imgsz: Annotated[int | None, Form(ge=64, le=2048, multiple_of=32)] = None,
) -> JobRead:
    """Like POST /models/{id}/predict, with a .pt uploaded to the project.
    Without `imgsz` the image size stored in the checkpoint is used."""
    assert project.id is not None
    path = storage.safe_child(storage.base_weights_dir(project.id), name)
    params = {
        "weights": f"upload:{path.name}", "conf": conf, "iou": iou, "imgsz": imgsz,
    }  # fmt: skip
    job = enqueue(session, project, user, "predict", params, status_="uploading")
    return to_read(session, job)


class InputsRead(BaseModel):
    files: int


def uploading_job(job: Job) -> int:
    if job.kind != "predict" or job.status != "uploading":
        raise HTTPException(status.HTTP_409_CONFLICT, "This job is not taking uploads")
    assert job.id is not None
    return job.id


@router.post("/jobs/{job_id}/inputs")
def add_inputs(
    files: list[UploadFile],
    job: JobDep,
    offset: Annotated[int, Form(ge=0)] = 0,
) -> InputsRead:
    """Store one part of an upload. `offset` is the index of the first file in the
    whole upload: it keeps the original order and makes retrying a part safe."""
    job_id = uploading_job(job)
    if offset + len(files) > get_settings().max_predict_images:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            f"Up to {get_settings().max_predict_images} files per run",
        )
    for f in files:
        if Path(f.filename or "").suffix.lower() not in ACCEPTED_SUFFIXES:
            raise HTTPException(
                status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, f"{f.filename}: unsupported"
            )
    uploads = storage.uploads_dir(job_id)
    uploads.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(files):
        dest = uploads / f"{offset + i:06d}-{Path(f.filename or 'upload').name}"
        partial = dest.with_name(f".{dest.name}.part")
        with partial.open("wb") as out:
            shutil.copyfileobj(f.file, out)
        partial.replace(dest)
    return InputsRead(files=sum(1 for p in uploads.iterdir() if p.name[0] != "."))


@router.post("/jobs/{job_id}/start")
def start_job(job: JobDep, session: SessionDep) -> JobRead:
    """Queue an inference job once all its files are uploaded."""
    job_id = uploading_job(job)
    uploads = storage.uploads_dir(job_id)
    if not uploads.is_dir() or not any(p.name[0] != "." for p in uploads.iterdir()):
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT, "Upload at least one image first"
        )
    job.status = "queued"
    session.add(job)
    session.commit()
    session.refresh(job)
    return to_read(session, job)
