import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Form, HTTPException, Query, UploadFile, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlmodel import col, select

from app import storage
from app.auth import CurrentUser, SessionDep
from app.deps import DatasetDep, ProjectDep, enqueue
from app.models import Dataset
from app.routers.jobs import JobRead, to_read
from app.services.datasets import (
    dataset_stats,
    import_yolo_zip,
    list_items,
    preview_image,
    write_data_yaml,
)

router = APIRouter(tags=["datasets"])


class DatasetRead(BaseModel):
    id: int
    project_id: int
    name: str
    source: str
    classes: list[str]
    stats: dict[str, Any]
    created_at: datetime


class ExportRequest(BaseModel):
    name: str | None = None
    val_ratio: float = Field(0.2, ge=0, le=0.5)
    seed: int = 0
    min_area: float = Field(4.0, ge=0)


def dataset_read(dataset: Dataset) -> DatasetRead:
    return DatasetRead.model_validate(dataset, from_attributes=True)


@router.get("/projects/{project_id}/datasets")
def list_datasets(project: ProjectDep, session: SessionDep) -> list[DatasetRead]:
    rows = session.exec(
        select(Dataset)
        .where(Dataset.project_id == project.id)
        .order_by(col(Dataset.id).desc())
    ).all()
    # datasets still being exported have no stats yet
    return [dataset_read(d) for d in rows if d.stats]


@router.post("/projects/{project_id}/datasets/export", status_code=201)
def export_from_label_studio(
    body: ExportRequest, project: ProjectDep, session: SessionDep, user: CurrentUser
) -> JobRead:
    if project.ls_project_id is None:
        raise HTTPException(
            status.HTTP_409_CONFLICT, "Send images to Label Studio first"
        )
    job = enqueue(session, project, user, "export", body.model_dump())
    return to_read(session, job)


@router.post("/projects/{project_id}/datasets/upload", status_code=201)
def upload_dataset(
    file: UploadFile,
    project: ProjectDep,
    session: SessionDep,
    name: Annotated[str, Form()] = "",
    val_ratio: Annotated[float, Form(ge=0, le=0.5)] = 0.2,
) -> DatasetRead:
    """Import an already-labeled YOLO zip (images + .txt labels)."""
    if not (file.filename or "").lower().endswith(".zip"):
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "Upload a .zip")
    assert project.id is not None
    dataset = Dataset(
        project_id=project.id,
        name=name or Path(file.filename or "upload").stem,
        source="upload",
        classes=project.classes,
    )
    session.add(dataset)
    session.commit()
    session.refresh(dataset)
    assert dataset.id is not None
    out = storage.dataset_dir(project.id, dataset.id)
    with tempfile.NamedTemporaryFile(suffix=".zip") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp.flush()
        try:
            import_yolo_zip(
                Path(tmp.name), out, project.classes, project.task_type, val_ratio
            )
        except Exception as e:
            session.delete(dataset)
            session.commit()
            shutil.rmtree(out, ignore_errors=True)
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(e)) from e
    dataset.stats = dataset_stats(out, project.classes)
    session.add(dataset)
    session.commit()
    session.refresh(dataset)
    return dataset_read(dataset)


@router.get("/datasets/{dataset_id}")
def get_dataset(dataset: DatasetDep) -> DatasetRead:
    return dataset_read(dataset)


@router.get("/datasets/{dataset_id}/download")
def download_dataset(dataset: DatasetDep) -> FileResponse:
    """The dataset as a YOLO zip (data.yaml, images/, labels/), usable anywhere."""
    assert dataset.id is not None
    root = storage.dataset_dir(dataset.project_id, dataset.id)
    if not dataset.stats or not (root / "data.yaml").exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Dataset is not ready")
    archive = root.with_name(f"{dataset.id}.zip")
    if not archive.exists():
        # data.yaml holds this server's absolute path; ship a portable one instead
        with tempfile.TemporaryDirectory() as staging:
            copy = Path(staging) / "dataset"
            shutil.copytree(root, copy)
            write_data_yaml(copy, dataset.classes, relative=True)
            built = shutil.make_archive(str(Path(staging) / "dataset"), "zip", copy)
            # atomic: concurrent downloads never see a half-written zip
            shutil.move(built, archive.with_suffix(".tmp"))
            archive.with_suffix(".tmp").replace(archive)
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in dataset.name)
    return FileResponse(archive, filename=f"{safe or 'dataset'}_yolo.zip")


class LabeledObject(BaseModel):
    class_id: int
    kind: Literal["box", "polygon"]
    points: list[list[float]]


class DatasetItem(BaseModel):
    split: str
    name: str
    objects: list[LabeledObject]


class DatasetItems(BaseModel):
    total: int
    items: list[DatasetItem]


@router.get("/datasets/{dataset_id}/items")
def dataset_items(
    dataset: DatasetDep,
    split: Literal["train", "val"] = "train",
    class_id: int | None = None,
    unlabeled: bool = False,
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=200)] = 24,
) -> DatasetItems:
    """Images of a split with their labels, for previewing before training."""
    assert dataset.id is not None
    root = storage.dataset_dir(dataset.project_id, dataset.id)
    items = list_items(root, split, class_id, unlabeled)
    page = items[offset : offset + limit]
    return DatasetItems(
        total=len(items), items=[DatasetItem.model_validate(i) for i in page]
    )


@router.get("/datasets/{dataset_id}/preview/{split}/{name}")
def dataset_preview_image(
    dataset: DatasetDep,
    split: Literal["train", "val"],
    name: str,
    size: Annotated[int, Query(ge=64, le=2048)] = 512,
) -> FileResponse:
    assert dataset.id is not None
    root = storage.dataset_dir(dataset.project_id, dataset.id)
    source = storage.safe_child(root / "images" / split, name)
    cache = root.with_name(f"{dataset.id}.preview")
    # round so arbitrary sizes can't fill the cache
    size = min((512, 1024, 2048), key=lambda s: abs(s - size))
    return FileResponse(
        preview_image(source, cache, size),
        headers={"Cache-Control": "private, max-age=3600"},
    )
