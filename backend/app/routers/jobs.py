import asyncio
import json
import shutil
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from sqlmodel import Session, col, select

from app import storage
from app.auth import CurrentUser, SessionDep
from app.db import get_engine
from app.deps import JobDep, ProjectDep
from app.models import Job, Metric, User
from app.services import training

router = APIRouter(tags=["jobs"])

TERMINAL = ("succeeded", "failed", "cancelled")
LOG_CHUNK = 64 * 1024


class JobRead(BaseModel):
    id: int
    project_id: int
    username: str | None
    kind: str
    status: str
    params: dict[str, Any]
    result: dict[str, Any]
    progress: dict[str, Any]
    error: str | None
    cancel_requested: bool
    queue_position: int
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None


class MetricRead(BaseModel):
    epoch: int
    values: dict[str, float]


def queue_position(session: Session, job: Job) -> int:
    """Number of jobs ahead of a queued job (0 = runs next)."""
    if job.status != "queued":
        return 0
    ahead = session.exec(
        select(Job.id).where(
            col(Job.status).in_(["queued", "running"]), col(Job.id) < job.id
        )
    ).all()
    return len(ahead)


def to_read(session: Session, job: Job) -> JobRead:
    user = session.get(User, job.user_id)
    return JobRead(
        **job.model_dump(exclude={"user_id", "worker"}),
        username=user.username if user else None,
        queue_position=queue_position(session, job),
    )


@router.get("/projects/{project_id}/jobs")
def list_jobs(
    project: ProjectDep, session: SessionDep, kind: str | None = None
) -> list[JobRead]:
    query = select(Job).where(Job.project_id == project.id)
    if kind:
        query = query.where(Job.kind == kind)
    jobs = session.exec(query.order_by(col(Job.id).desc())).all()
    return [to_read(session, j) for j in jobs]


class QueueEntry(BaseModel):
    id: int
    kind: str
    status: str
    username: str | None
    mine: bool
    queue_position: int


@router.get("/queue")
def active_queue(session: SessionDep, user: CurrentUser) -> list[QueueEntry]:
    """Running and queued jobs of all users, so people see what they wait behind.

    Only kind/owner are shown for jobs in projects the user cannot access.
    """
    jobs = session.exec(
        select(Job)
        .where(col(Job.status).in_(["queued", "running"]))
        .order_by(col(Job.id))
    ).all()
    entries = []
    for job in jobs:
        owner = session.get(User, job.user_id)
        assert job.id is not None
        entries.append(
            QueueEntry(
                id=job.id, kind=job.kind, status=job.status,
                username=owner.username if owner else None,
                mine=job.user_id == user.id,
                queue_position=queue_position(session, job),
            )  # fmt: skip
        )
    return entries


@router.get("/jobs/{job_id}")
def get_job(job: JobDep, session: SessionDep) -> JobRead:
    return to_read(session, job)


@router.post("/jobs/{job_id}/cancel")
def cancel_job(job: JobDep, session: SessionDep) -> JobRead:
    if job.status == "uploading":
        assert job.id is not None
        job.status = "cancelled"
        shutil.rmtree(storage.uploads_dir(job.id), ignore_errors=True)
    elif job.status == "queued":
        job.status = "cancelled"
    elif job.status == "running":
        job.cancel_requested = True
    else:
        raise HTTPException(status.HTTP_409_CONFLICT, f"Job is already {job.status}")
    session.add(job)
    session.commit()
    session.refresh(job)
    return to_read(session, job)


@router.get("/jobs/{job_id}/metrics")
def job_metrics(job: JobDep, session: SessionDep) -> list[MetricRead]:
    rows = session.exec(
        select(Metric).where(Metric.job_id == job.id).order_by(col(Metric.epoch))
    ).all()
    return [MetricRead(epoch=m.epoch, values=m.values) for m in rows]


def read_log(job_id: int, offset: int) -> tuple[str, int]:
    path = storage.log_path(job_id)
    if not path.exists():
        return "", offset
    with path.open("rb") as f:
        f.seek(offset)
        data = f.read(LOG_CHUNK)
    return data.decode(errors="replace"), offset + len(data)


@router.get("/jobs/{job_id}/log")
def job_log(job: JobDep, offset: int = 0) -> dict[str, Any]:
    assert job.id is not None
    text, end = read_log(job.id, offset)
    return {"text": text, "offset": end}


def sse(event: str, data: Any) -> str:  # noqa: ANN401
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


@router.get("/jobs/{job_id}/events")
async def job_events(job: JobDep, request: Request) -> StreamingResponse:
    """Server-Sent Events: `job` on any change, `metric` per epoch, `log` text."""
    job_id = job.id
    assert job_id is not None

    def snapshot(seen_epochs: dict[int, dict[str, float]]) -> tuple[JobRead, list]:
        with Session(get_engine()) as session:
            current = session.get(Job, job_id)
            assert current is not None
            rows = session.exec(
                select(Metric)
                .where(Metric.job_id == job_id)
                .order_by(col(Metric.epoch))
            ).all()
            changed = [
                MetricRead(epoch=m.epoch, values=m.values).model_dump()
                for m in rows
                if seen_epochs.get(m.epoch) != m.values
            ]
            for m in rows:
                seen_epochs[m.epoch] = m.values
            return to_read(session, current), changed

    async def stream() -> AsyncIterator[str]:
        seen: dict[int, dict[str, float]] = {}
        last_job = None
        offset = 0
        while not await request.is_disconnected():
            current, metrics = await asyncio.to_thread(snapshot, seen)
            data = current.model_dump(mode="json")
            if data != last_job:
                last_job = data
                yield sse("job", data)
            for m in metrics:
                yield sse("metric", m)
            while True:
                text, offset = await asyncio.to_thread(read_log, job_id, offset)
                if not text:
                    break
                yield sse("log", text)
            if current.status in TERMINAL:
                yield sse("end", {})
                return
            await asyncio.sleep(1)

    headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    return StreamingResponse(stream(), media_type="text/event-stream", headers=headers)


@router.get("/jobs/{job_id}/files/{path:path}")
def job_file(job: JobDep, path: str) -> FileResponse:
    assert job.id is not None
    return FileResponse(storage.safe_child(storage.job_dir(job.id), path))


@router.get("/jobs/{job_id}/download")
def download_output(job: JobDep) -> FileResponse:
    """Zip of a finished inference job's output (overlays + predictions.json)."""
    assert job.id is not None
    out = storage.job_dir(job.id) / "output"
    if job.status != "succeeded" or not out.is_dir():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No output to download")
    archive = storage.output_zip(job.id)
    if not archive.exists():  # jobs from before the worker wrote the zip itself
        training.zip_output(out, archive)
    return FileResponse(archive, filename=f"predictions_job{job.id}.zip")
