import shutil
from datetime import datetime
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
    base_model: str = Field(description="Catalog weights name or model:<id>")
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
    names = {a["weights"] for a in training.architectures(project.task_type)}
    if base not in names:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT, f"Unknown architecture {base}"
        )


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
