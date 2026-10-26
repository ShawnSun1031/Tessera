import json
import shutil
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from sqlmodel import Session

from app import storage
from app.auth import CurrentUser, SessionDep
from app.config import get_settings
from app.deps import JobDep, LabelStudioDep
from app.models import Image, Project, TrainedModel
from app.services import ls_server
from app.services.export_yolo import class_names, infer_task_type
from app.services.ingest import sortable_name, unique_path
from app.services.labelstudio import LabelStudio, label_config
from app.services.ls_predictions import labeling_tools, prediction_results

router = APIRouter(prefix="/labelstudio", tags=["labelstudio"])


class LabelStudioProject(BaseModel):
    id: int
    title: str
    tasks: int
    annotated: int
    task_type: str | None
    classes: list[str]
    problem: str | None  # why it can't be used for training, if it can't


def describe(p: dict[str, Any]) -> LabelStudioProject:
    config = p.get("label_config") or ""
    task_type, classes = infer_task_type(config), class_names(config)
    problem = None
    if task_type is None:
        problem = "Uses no brush, polygon or rectangle labels"
    elif not classes:
        problem = "Has no label names defined"
    elif not p.get("num_tasks_with_annotations"):
        problem = "Has no labeled images yet"
    return LabelStudioProject(
        id=p["id"],
        title=p.get("title") or f"Project #{p['id']}",
        tasks=p.get("task_number") or 0,
        annotated=p.get("num_tasks_with_annotations") or 0,
        task_type=task_type,
        classes=classes,
        problem=problem,
    )


class ServerStatus(BaseModel):
    url: str
    reachable: bool
    can_start: bool
    starting: bool
    reason: str | None


def server_status() -> ServerStatus:
    s = get_settings()
    found = ls_server.state(
        s.label_studio_url, s.label_studio_command, storage.label_studio_log()
    )
    return ServerStatus(url=s.label_studio_public_url, **vars(found))


@router.get("/server")
def get_server(_: CurrentUser) -> ServerStatus:
    """Whether Label Studio is up, and whether this server can start it."""
    return server_status()


@router.post("/server/start")
def start_server(_: CurrentUser) -> ServerStatus:
    """Launch a local Label Studio; poll GET /server until it is reachable."""
    current = server_status()
    if current.reachable:
        raise HTTPException(status.HTTP_409_CONFLICT, "Label Studio is already running")
    if not current.can_start:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, current.reason)
    s = get_settings()
    ls_server.start(
        s.label_studio_url, s.label_studio_command, storage.label_studio_log()
    )
    return server_status()


@router.get("/projects")
def list_label_studio_projects(
    ls: LabelStudioDep, _: CurrentUser
) -> list[LabelStudioProject]:
    """Projects in Label Studio, with what the platform can infer from each."""
    return [describe(p) for p in ls.list_projects()]


def ensure_ls_project(project: Project, session: Session, ls: LabelStudio) -> int:
    """The project's Label Studio project id, creating one if it has none."""
    if project.ls_project_id is None:
        config = label_config(project.task_type, project.classes)
        created = ls.create_project(f"{project.name} (#{project.id})", config)
        project.ls_project_id = created["id"]
        session.add(project)
        session.commit()
    assert project.ls_project_id is not None
    return project.ls_project_id


class SentToLabelStudio(BaseModel):
    ls_project_id: int
    ls_url: str
    images: int
    objects: int
    skipped_objects: int
    sent_at: str


@router.post("/send-predictions/{job_id}")
def send_predictions(
    job: JobDep, session: SessionDep, ls: LabelStudioDep, force: bool = False
) -> SentToLabelStudio:
    """Upload a prediction run's images to Label Studio with the predictions as
    pre-labels, so people correct them instead of labeling from scratch.

    Only submitted (corrected) labels are used when a dataset is built, so
    unchecked predictions never end up in training data.
    """
    assert job.id is not None
    if job.kind != "predict" or job.status != "succeeded":
        raise HTTPException(
            status.HTTP_409_CONFLICT, "Only finished prediction runs can be sent"
        )
    if job.result.get("label_studio") and not force:
        raise HTTPException(
            status.HTTP_409_CONFLICT, "These predictions were already sent"
        )
    project = session.get(Project, job.project_id)
    model = session.get(TrainedModel, job.params.get("model_id"))
    assert project is not None and project.id is not None
    predictions = storage.job_dir(job.id) / "output" / "predictions.json"
    if not predictions.exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Prediction output is missing")
    records = json.loads(predictions.read_text())

    ls_project_id = ensure_ls_project(project, session, ls)
    try:
        tools = labeling_tools(ls.project(ls_project_id)["label_config"])
    except (ValueError, ET.ParseError) as e:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(e)) from e
    version = f"{model.name} (model #{model.id})" if model else f"job {job.id}"
    inputs = storage.job_dir(job.id) / "inputs"
    objects = skipped = 0
    for record in records:
        # keep a copy with the project's images so it shows on the Upload step
        source = inputs / record["image"]
        image_path = unique_path(storage.images_dir(project.id), source.name)
        image_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, image_path)
        task_id = ls.import_file(ls_project_id, image_path)
        ls.update_task_data(task_id, {"filename": sortable_name(image_path.name)})
        results, n_skipped = prediction_results(record, tools)
        if results:
            scores = [r["score"] for r in results if r.get("score") is not None]
            score = sum(scores) / len(scores) if scores else None
            ls.create_prediction(task_id, results, score, version)
        objects += len(results)
        skipped += n_skipped
        session.add(
            Image(
                project_id=project.id,
                filename=image_path.name,
                source_name=f"{record['image']} (predictions #{job.id})",
                in_label_studio=True,
            )
        )
        session.commit()

    if objects:
        ls.use_predictions_for_prelabeling(ls_project_id, version)
    base = get_settings().label_studio_public_url.rstrip("/")
    sent = SentToLabelStudio(
        ls_project_id=ls_project_id,
        ls_url=f"{base}/projects/{ls_project_id}",
        images=len(records),
        objects=objects,
        skipped_objects=skipped,
        sent_at=datetime.now(UTC).isoformat(),
    )
    job.result = {**job.result, "label_studio": sent.model_dump()}
    session.add(job)
    session.commit()
    return sent
