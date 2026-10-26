import shutil
import tempfile
from pathlib import Path
from typing import Any, Literal

import httpx

from fastapi import APIRouter, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlmodel import col, select

from app.auth import CurrentUser, SessionDep
from app.config import get_settings
from app.deps import LabelStudioDep, ProjectDep, can_access
from app.models import Image, Project, ProjectMember, User
from app.routers.labelstudio import describe, ensure_ls_project
from app.services.ingest import ACCEPTED_SUFFIXES, ingest_file
from app.storage import images_dir, safe_child

router = APIRouter(prefix="/projects", tags=["projects"])

LS_IMPORT_BATCH = 50


class ProjectCreate(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    task_type: Literal["detect", "segment"]
    classes: list[str] = Field(min_length=1)


class ProjectRead(BaseModel):
    id: int
    name: str
    task_type: str
    classes: list[str]
    owner_id: int
    ls_project_id: int | None
    ls_url: str | None


class MemberAdd(BaseModel):
    username: str


class ImageRead(BaseModel):
    id: int
    filename: str
    source_name: str
    in_label_studio: bool


class LabelStudioStatus(BaseModel):
    ls_project_id: int | None
    ls_url: str | None
    images: int
    pending_upload: int
    tasks: int = 0
    annotated: int = 0


def to_read(project: Project) -> ProjectRead:
    ls_url = None
    if project.ls_project_id:
        base = get_settings().label_studio_public_url.rstrip("/")
        ls_url = f"{base}/projects/{project.ls_project_id}"
    return ProjectRead(
        **project.model_dump(exclude={"created_at"}), ls_url=ls_url
    )


@router.post("", status_code=status.HTTP_201_CREATED)
def create_project(
    body: ProjectCreate, session: SessionDep, user: CurrentUser
) -> ProjectRead:
    classes = [c.strip() for c in body.classes if c.strip()]
    if len(set(classes)) != len(classes) or not classes:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT, "Class names must be unique"
        )
    assert user.id is not None
    project = Project(
        name=body.name, task_type=body.task_type, classes=classes, owner_id=user.id
    )
    session.add(project)
    session.commit()
    session.refresh(project)
    return to_read(project)


class FromLabelStudio(BaseModel):
    ls_project_id: int
    name: str | None = Field(None, max_length=128)


@router.post("/from-labelstudio", status_code=status.HTTP_201_CREATED)
def create_from_label_studio(
    body: FromLabelStudio, session: SessionDep, user: CurrentUser, ls: LabelStudioDep
) -> ProjectRead:
    """New project that uses an existing, already-labeled Label Studio project.

    Task type and object types are taken from its labeling config. Brush labels
    become YOLO segmentation polygons when a dataset is built.
    """
    info = describe(ls_project_or_404(ls, body.ls_project_id))
    if info.task_type is None or not info.classes:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, info.problem)
    assert user.id is not None
    project = Project(
        name=body.name or info.title,
        task_type=info.task_type,
        classes=info.classes,
        owner_id=user.id,
        ls_project_id=info.id,
    )
    session.add(project)
    session.commit()
    session.refresh(project)
    return to_read(project)


class LinkRequest(BaseModel):
    ls_project_id: int


@router.post("/{project_id}/labelstudio/link")
def link_label_studio(
    body: LinkRequest, project: ProjectDep, session: SessionDep, ls: LabelStudioDep
) -> ProjectRead:
    """Use an existing Label Studio project for this project's labels."""
    if project.ls_project_id is not None:
        raise HTTPException(
            status.HTTP_409_CONFLICT, "This project is already linked to Label Studio"
        )
    info = describe(ls_project_or_404(ls, body.ls_project_id))
    if info.task_type is None:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, info.problem)
    if project.task_type == "segment" and info.task_type == "detect":
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            "That project only has rectangles; segmentation needs brush or polygon"
            " labels. Create a detection project from it instead.",
        )
    unknown = [c for c in info.classes if c not in project.classes]
    if unknown:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            f"Label Studio uses labels this project doesn't have: {', '.join(unknown)}."
            " Create a new project from it instead.",
        )
    project.ls_project_id = info.id
    session.add(project)
    session.commit()
    session.refresh(project)
    return to_read(project)


def ls_project_or_404(ls: LabelStudioDep, ls_project_id: int) -> dict[str, Any]:
    try:
        return ls.project(ls_project_id)
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            raise HTTPException(
                status.HTTP_404_NOT_FOUND, "Label Studio project not found"
            ) from e
        raise


@router.get("")
def list_projects(session: SessionDep, user: CurrentUser) -> list[ProjectRead]:
    projects = session.exec(select(Project).order_by(col(Project.id))).all()
    return [to_read(p) for p in projects if can_access(session, user, p)]


@router.get("/{project_id}")
def get_project(project: ProjectDep) -> ProjectRead:
    return to_read(project)


@router.post("/{project_id}/members", status_code=status.HTTP_204_NO_CONTENT)
def add_member(body: MemberAdd, project: ProjectDep, session: SessionDep) -> None:
    member = session.exec(select(User).where(User.username == body.username)).first()
    if member is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "User not found")
    assert project.id is not None and member.id is not None
    if not session.get(ProjectMember, (project.id, member.id)):
        session.add(ProjectMember(project_id=project.id, user_id=member.id))
        session.commit()


@router.post("/{project_id}/uploads", status_code=status.HTTP_201_CREATED)
def upload_files(
    files: list[UploadFile], project: ProjectDep, session: SessionDep
) -> list[ImageRead]:
    assert project.id is not None
    for f in files:
        if Path(f.filename or "").suffix.lower() not in ACCEPTED_SUFFIXES:
            raise HTTPException(
                status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                f"{f.filename}: accepted types are {', '.join(ACCEPTED_SUFFIXES)}",
            )
    created = []
    with tempfile.TemporaryDirectory() as tmp:
        for f in files:
            src = Path(tmp) / Path(f.filename or "upload").name
            with src.open("wb") as out:
                shutil.copyfileobj(f.file, out)
            try:
                results = ingest_file(src, images_dir(project.id))
            except Exception as e:
                raise HTTPException(
                    status.HTTP_422_UNPROCESSABLE_CONTENT, f"{f.filename}: {e}"
                ) from e
            for path, source_name in results:
                image = Image(
                    project_id=project.id, filename=path.name, source_name=source_name
                )
                session.add(image)
                created.append(image)
    session.commit()
    return [ImageRead.model_validate(i, from_attributes=True) for i in created]


@router.get("/{project_id}/images")
def list_images(project: ProjectDep, session: SessionDep) -> list[ImageRead]:
    images = session.exec(
        select(Image).where(Image.project_id == project.id).order_by(col(Image.id))
    ).all()
    return [ImageRead.model_validate(i, from_attributes=True) for i in images]


@router.get("/{project_id}/images/{image_id}/file")
def image_file(image_id: int, project: ProjectDep, session: SessionDep) -> FileResponse:
    image = session.get(Image, image_id)
    if image is None or image.project_id != project.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Image not found")
    assert project.id is not None
    return FileResponse(safe_child(images_dir(project.id), image.filename))


@router.post("/{project_id}/labelstudio/sync")
def sync_label_studio(
    project: ProjectDep, session: SessionDep, ls: LabelStudioDep
) -> LabelStudioStatus:
    """Create the LS project if needed and upload images not yet sent to it."""
    assert project.id is not None
    ls_project_id = ensure_ls_project(project, session, ls)

    pending = session.exec(
        select(Image).where(
            Image.project_id == project.id, col(Image.in_label_studio).is_(False)
        )
    ).all()
    for start in range(0, len(pending), LS_IMPORT_BATCH):
        batch = pending[start : start + LS_IMPORT_BATCH]
        paths = [images_dir(project.id) / i.filename for i in batch]
        ls.import_files(ls_project_id, paths)
        # commit per batch so a failure part-way doesn't re-upload what succeeded
        for image in batch:
            image.in_label_studio = True
            session.add(image)
        session.commit()
    return label_studio_status(project, session, ls)


@router.get("/{project_id}/labelstudio/status")
def label_studio_status(
    project: ProjectDep, session: SessionDep, ls: LabelStudioDep
) -> LabelStudioStatus:
    images = session.exec(select(Image).where(Image.project_id == project.id)).all()
    status_ = LabelStudioStatus(
        ls_project_id=project.ls_project_id,
        ls_url=to_read(project).ls_url,
        images=len(images),
        pending_upload=sum(not i.in_label_studio for i in images),
    )
    if project.ls_project_id is not None:
        info = ls.project(project.ls_project_id)
        status_.tasks = info.get("task_number", 0)
        status_.annotated = info.get("num_tasks_with_annotations", 0)
    return status_
