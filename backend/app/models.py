"""Database tables."""

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, Column
from sqlmodel import Field, SQLModel


def now() -> datetime:
    return datetime.now(UTC)


class User(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    username: str = Field(unique=True, index=True)
    password_hash: str
    is_admin: bool = False
    created_at: datetime = Field(default_factory=now)


class ProjectMember(SQLModel, table=True):
    project_id: int = Field(foreign_key="project.id", primary_key=True)
    user_id: int = Field(foreign_key="user.id", primary_key=True)


class Project(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str
    task_type: str  # "detect" | "segment"
    classes: list[str] = Field(sa_column=Column(JSON, nullable=False))
    owner_id: int = Field(foreign_key="user.id")
    ls_project_id: int | None = None
    created_at: datetime = Field(default_factory=now)


class Image(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    project_id: int = Field(foreign_key="project.id", index=True)
    filename: str  # file under the project's images/ dir
    source_name: str  # original upload name (e.g. the TIF it came from)
    in_label_studio: bool = False
    created_at: datetime = Field(default_factory=now)


class Dataset(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    project_id: int = Field(foreign_key="project.id", index=True)
    name: str
    source: str  # "labelstudio" | "upload"
    classes: list[str] = Field(sa_column=Column(JSON, nullable=False))
    stats: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=now)


class Job(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    project_id: int = Field(foreign_key="project.id", index=True)
    user_id: int = Field(foreign_key="user.id")
    kind: str  # "export" | "train" | "predict" | "onnx"
    status: str = Field(default="queued", index=True)
    params: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    result: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    progress: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    error: str | None = None
    cancel_requested: bool = False
    worker: str | None = None
    created_at: datetime = Field(default_factory=now)
    started_at: datetime | None = None
    finished_at: datetime | None = None


class Metric(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    job_id: int = Field(foreign_key="job.id", index=True)
    epoch: int
    values: dict[str, float] = Field(sa_column=Column(JSON, nullable=False))


class TrainedModel(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    project_id: int = Field(foreign_key="project.id", index=True)
    job_id: int = Field(foreign_key="job.id")
    dataset_id: int = Field(foreign_key="dataset.id")
    name: str
    task_type: str
    architecture: str
    classes: list[str] = Field(sa_column=Column(JSON, nullable=False))
    metrics: dict[str, float] = Field(default_factory=dict, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=now)


class UserPreference(SQLModel, table=True):
    """Per-user UI settings (last-used training options etc.), as free-form JSON."""

    user_id: int = Field(foreign_key="user.id", primary_key=True)
    data: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    updated_at: datetime = Field(default_factory=now)
