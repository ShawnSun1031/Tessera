"""Shared request dependencies."""

from functools import lru_cache
from typing import Annotated, Any

from fastapi import Depends, HTTPException, status
from sqlmodel import select

from app.auth import CurrentUser, SessionDep
from app.config import get_settings
from app.models import Dataset, Job, Project, ProjectMember, TrainedModel, User
from app.services.labelstudio import LabelStudio


@lru_cache
def get_labelstudio() -> LabelStudio:
    settings = get_settings()
    if not settings.label_studio_api_key:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "LABEL_STUDIO_API_KEY is not set"
        )
    return LabelStudio.from_key(
        settings.label_studio_url, settings.label_studio_api_key
    )


LabelStudioDep = Annotated[LabelStudio, Depends(get_labelstudio)]


def can_access(session: SessionDep, user: User, project: Project) -> bool:
    if user.is_admin or project.owner_id == user.id:
        return True
    member = session.exec(
        select(ProjectMember).where(
            ProjectMember.project_id == project.id, ProjectMember.user_id == user.id
        )
    ).first()
    return member is not None


def project_for_user(
    project_id: int, session: SessionDep, user: CurrentUser
) -> Project:
    project = session.get(Project, project_id)
    # 404 rather than 403 so project ids of other teams aren't revealed
    if project is None or not can_access(session, user, project):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Project not found")
    return project


ProjectDep = Annotated[Project, Depends(project_for_user)]


def owned[T: (Dataset, Job, TrainedModel)](
    model: type[T], obj_id: int, session: SessionDep, user: User
) -> T:
    """Load a project-scoped row, 404 unless the user can access its project."""
    obj = session.get(model, obj_id)
    project = session.get(Project, obj.project_id) if obj else None
    if obj is None or project is None or not can_access(session, user, project):
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"{model.__name__} not found")
    return obj


def job_for_user(job_id: int, session: SessionDep, user: CurrentUser) -> Job:
    return owned(Job, job_id, session, user)


def dataset_for_user(
    dataset_id: int, session: SessionDep, user: CurrentUser
) -> Dataset:
    return owned(Dataset, dataset_id, session, user)


def model_for_user(
    model_id: int, session: SessionDep, user: CurrentUser
) -> TrainedModel:
    return owned(TrainedModel, model_id, session, user)


JobDep = Annotated[Job, Depends(job_for_user)]
DatasetDep = Annotated[Dataset, Depends(dataset_for_user)]
ModelDep = Annotated[TrainedModel, Depends(model_for_user)]


def enqueue(
    session: SessionDep,
    project: Project,
    user: User,
    kind: str,
    params: dict[str, Any],
    status_: str = "queued",
) -> Job:
    assert project.id is not None and user.id is not None
    job = Job(
        project_id=project.id, user_id=user.id, kind=kind, params=params,
        status=status_,
    )  # fmt: skip
    session.add(job)
    session.commit()
    session.refresh(job)
    return job
