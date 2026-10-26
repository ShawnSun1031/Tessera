"""Job worker: `python -m app.worker`. Run one per GPU (set DEVICE=0, 1, ...)."""

import logging
import os
import shutil
import socket
import tempfile
import time
import traceback
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from sqlalchemy import update
from sqlmodel import Session, col, select

from app import storage
from app.config import get_settings
from app.db import get_engine, init_db
from app.models import Dataset, Job, Metric, Project, TrainedModel
from app.services import training
from app.services.datasets import dataset_stats
from app.services.export_yolo import image_key, write_dataset
from app.services.ingest import ingest_file, natural_key
from app.services.labelstudio import LabelStudio

log = logging.getLogger("platform.worker")
log.setLevel(logging.INFO)
TERMINAL = ("succeeded", "failed", "cancelled")
PROGRESS_INTERVAL = 1.0  # seconds between progress writes / cancel checks
UPLOAD_TTL = timedelta(hours=24)  # unfinished inference uploads are dropped after this


def worker_name() -> str:
    """Stable across restarts so a restarted worker can recover its own jobs."""
    device = get_settings().device or "auto"
    return os.environ.get("WORKER_NAME") or f"{socket.gethostname()}:{device}"


def claim_next(engine: Any, name: str) -> int | None:  # noqa: ANN401
    """Atomically move the oldest queued job to running; return its id."""
    with Session(engine) as session:
        queued = session.exec(
            select(Job.id).where(Job.status == "queued").order_by(col(Job.id)).limit(5)
        ).all()
        for job_id in queued:
            claimed = session.exec(  # type: ignore[call-overload]
                update(Job)
                .where(col(Job.id) == job_id, col(Job.status) == "queued")
                .values(status="running", worker=name, started_at=datetime.now(UTC))
            )
            session.commit()
            if claimed.rowcount == 1:
                return job_id
    return None


class JobContext:
    """DB helpers for a running job, each using a short-lived session."""

    def __init__(self, engine: Any, job_id: int) -> None:  # noqa: ANN401
        self.engine = engine
        self.job_id = job_id
        self.last_progress = 0.0
        self.cancelled = False

    def job(self) -> Job:
        with Session(self.engine) as session:
            job = session.get(Job, self.job_id)
            assert job is not None
            return job

    def update(self, **fields: Any) -> None:  # noqa: ANN401
        with Session(self.engine) as session:
            session.exec(  # type: ignore[call-overload]
                update(Job).where(col(Job.id) == self.job_id).values(**fields)
            )
            session.commit()

    def progress(self, progress: dict[str, Any], force: bool = False) -> bool:
        """Throttled progress write; returns True if cancellation was requested."""
        now = time.monotonic()
        if force or now - self.last_progress >= PROGRESS_INTERVAL:
            self.last_progress = now
            self.update(progress=progress)
            self.cancelled = self.cancelled or self.job().cancel_requested
        return self.cancelled

    def metric(self, epoch: int, values: dict[str, float]) -> None:
        with Session(self.engine) as session:
            existing = session.exec(
                select(Metric).where(
                    Metric.job_id == self.job_id, Metric.epoch == epoch
                )
            ).first()
            if existing:
                # Ultralytics re-reports the final epoch with best.pt metrics
                existing.values = values
                session.add(existing)
            else:
                session.add(Metric(job_id=self.job_id, epoch=epoch, values=values))
            session.commit()


def run_export(ctx: JobContext, job: Job, project: Project) -> dict[str, Any]:
    settings = get_settings()
    if project.ls_project_id is None:
        raise RuntimeError("Project has not been sent to Label Studio yet")
    ls = LabelStudio.from_key(settings.label_studio_url, settings.label_studio_api_key)
    p = job.params
    with Session(ctx.engine) as session:
        assert project.id is not None
        dataset = Dataset(
            project_id=project.id,
            name=p.get("name") or f"Label Studio export (job {job.id})",
            source="labelstudio",
            classes=project.classes,
        )
        session.add(dataset)
        session.commit()
        session.refresh(dataset)
        assert dataset.id is not None
        dataset_id = dataset.id
    out = storage.dataset_dir(project.id, dataset_id)
    try:
        config = ls.project(project.ls_project_id)["label_config"]
        write_dataset(
            ls.export_tasks(project.ls_project_id), project.classes, out, ls.download,
            project.task_type, p.get("val_ratio", 0.2), p.get("min_area", 4.0),
            p.get("seed", 0), log=log.info, data_key=image_key(config),
        )  # fmt: skip
        stats = dataset_stats(out, project.classes)
    except BaseException:
        with Session(ctx.engine) as session:
            session.delete(session.get(Dataset, dataset_id))
            session.commit()
        raise
    with Session(ctx.engine) as session:
        dataset = session.get(Dataset, dataset_id)
        assert dataset is not None
        dataset.stats = stats
        session.add(dataset)
        session.commit()
    if not stats["splits"]["train"]["images"]:
        log.warning("No annotated images found in Label Studio")
    return {"dataset_id": dataset_id, "stats": stats}


def resolve_base_weights(session: Session, project: Project, base: str) -> str:
    """Catalog weights live in the weights cache; `model:<id>` fine-tunes a model;
    `upload:<name>` starts from a .pt uploaded to the project."""
    if base.startswith("model:"):
        model = session.get(TrainedModel, int(base.split(":", 1)[1]))
        if model is None or model.project_id != project.id:
            raise RuntimeError(f"Base model {base} not found in this project")
        return str(storage.weights_path(model.job_id))
    if base.startswith("upload:"):
        assert project.id is not None
        folder = storage.base_weights_dir(project.id)
        path = folder / Path(base[len("upload:") :]).name
        if not path.is_file():
            raise RuntimeError(f"Uploaded weights {path.name} no longer exist")
        return str(path)
    names = {a["weights"] for a in training.architectures(project.task_type)}
    if base not in names:
        raise RuntimeError(f"Unknown architecture {base}")
    storage.weights_cache().mkdir(parents=True, exist_ok=True)
    return str(storage.weights_cache() / base)


def run_train(ctx: JobContext, job: Job, project: Project) -> dict[str, Any]:
    assert job.id is not None and project.id is not None
    p = job.params
    with Session(ctx.engine) as session:
        dataset = session.get(Dataset, p["dataset_id"])
        if dataset is None or dataset.project_id != project.id:
            raise RuntimeError("Dataset not found")
        base = resolve_base_weights(session, project, p["base_model"])
    data_yaml = storage.dataset_dir(project.id, p["dataset_id"]) / "data.yaml"

    def on_batch(progress: dict[str, int]) -> bool:
        return ctx.progress(progress)

    metrics = training.train(
        data_yaml, base, project.task_type, storage.train_dir(job.id), p,
        get_settings().device, ctx.metric, on_batch,
    )  # fmt: skip
    if not storage.weights_path(job.id).exists():
        raise RuntimeError("Training finished without producing best.pt")
    with Session(ctx.engine) as session:
        model = TrainedModel(
            project_id=project.id,
            job_id=job.id,
            dataset_id=p["dataset_id"],
            name=p.get("name") or f"Model from job {job.id}",
            task_type=project.task_type,
            architecture=p["base_model"],
            classes=dataset.classes,
            metrics={k: v for k, v in metrics.items() if k in training.SUMMARY_KEYS},
        )
        session.add(model)
        session.commit()
        session.refresh(model)
        return {"model_id": model.id, "metrics": model.metrics}


def load_model(ctx: JobContext, project: Project, model_id: int) -> TrainedModel:
    with Session(ctx.engine) as session:
        model = session.get(TrainedModel, model_id)
        if model is None or model.project_id != project.id:
            raise RuntimeError("Model not found")
        return model


def prepare_inputs(ctx: JobContext, job_id: int) -> tuple[list[Path], int]:
    """Convert uploaded files (TIFs, images, zips) to model inputs, in upload order.

    Returns the inputs and how many uploaded files could not be read.
    """
    uploads, inputs = storage.uploads_dir(job_id), storage.job_dir(job_id) / "inputs"
    if not uploads.is_dir():  # jobs from before uploads were converted here
        return sorted(inputs.glob("*"), key=lambda p: natural_key(p.name)), 0
    files = sorted(p for p in uploads.iterdir() if p.name[0] != ".")
    limit = get_settings().max_predict_images
    outs: list[Path] = []
    skipped = 0
    for i, src in enumerate(files):
        if ctx.progress({"prepared": i, "files": len(files)}):
            raise training.Cancelled
        with tempfile.TemporaryDirectory() as tmp:
            # drop the "000123-" order prefix so outputs keep the original name
            named = Path(tmp) / src.name.split("-", 1)[1]
            shutil.move(src, named)
            try:
                outs += [path for path, _ in ingest_file(named, inputs)]
            except Exception as e:
                log.warning(f"Skipped {named.name}: {e}")
                skipped += 1
        if len(outs) > limit:
            raise RuntimeError(f"More than {limit} images; split them into runs")
    shutil.rmtree(uploads, ignore_errors=True)
    return outs, skipped


def run_predict(ctx: JobContext, job: Job, project: Project) -> dict[str, Any]:
    assert job.id is not None
    p = job.params
    model = load_model(ctx, project, p["model_id"])
    inputs, skipped = prepare_inputs(ctx, job.id)
    if not inputs:
        raise RuntimeError("None of the uploaded files could be read as images")

    def on_progress(done: int, total: int) -> bool:
        return ctx.progress({"done": done, "total": total}, force=done == total)

    out = storage.job_dir(job.id) / "output"
    result = training.predict(
        storage.weights_path(model.job_id), inputs, out,
        p.get("conf", 0.25), p.get("iou", 0.7), p.get("imgsz", 640),
        get_settings().device, on_progress,
    )  # fmt: skip
    ctx.progress({"packaging": 1}, force=True)
    training.zip_output(out, storage.output_zip(job.id))
    return {**result, "skipped_files": skipped}


def run_onnx(ctx: JobContext, job: Job, project: Project) -> dict[str, Any]:
    model = load_model(ctx, project, job.params["model_id"])
    path = training.export_onnx(storage.weights_path(model.job_id), job.params["imgsz"])
    return {"model_id": model.id, "size_bytes": path.stat().st_size}


RUNNERS: dict[str, Callable[[JobContext, Job, Project], dict[str, Any]]] = {
    "export": run_export,
    "train": run_train,
    "predict": run_predict,
    "onnx": run_onnx,
}


def run_job(engine: Any, job_id: int) -> None:  # noqa: ANN401
    ctx = JobContext(engine, job_id)
    job = ctx.job()
    with Session(engine) as session:
        project = session.get(Project, job.project_id)
    assert project is not None

    path = storage.log_path(job_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path)
    handler.setFormatter(logging.Formatter("%(message)s"))
    loggers = [logging.getLogger("ultralytics"), log]
    for lg in loggers:
        lg.addHandler(handler)
    log.info(f"Job {job_id} ({job.kind}) started on {worker_name()}")
    try:
        result = RUNNERS[job.kind](ctx, job, project)
        status = "cancelled" if ctx.cancelled else "succeeded"
        ctx.update(status=status, result=result, finished_at=datetime.now(UTC))
        log.info(f"Job {job_id} {status}")
    except training.Cancelled:
        ctx.update(status="cancelled", finished_at=datetime.now(UTC))
        log.info(f"Job {job_id} cancelled")
    except Exception as e:
        log.error(traceback.format_exc())
        ctx.update(status="failed", error=str(e) or type(e).__name__,
                   finished_at=datetime.now(UTC))  # fmt: skip
    finally:
        for lg in loggers:
            lg.removeHandler(handler)
        handler.close()


def expire_uploads(engine: Any, max_age: timedelta = UPLOAD_TTL) -> None:  # noqa: ANN401
    """Cancel inference jobs whose upload was abandoned (e.g. the tab was closed)."""
    cutoff = datetime.now(UTC) - max_age
    with Session(engine) as session:
        jobs = session.exec(select(Job).where(Job.status == "uploading")).all()
        for job in jobs:
            created = job.created_at
            if created.tzinfo is None:  # SQLite drops the timezone
                created = created.replace(tzinfo=UTC)
            if created > cutoff:
                continue
            assert job.id is not None
            shutil.rmtree(storage.uploads_dir(job.id), ignore_errors=True)
            job.status = "cancelled"
            job.error = "The upload was not finished"
            job.finished_at = datetime.now(UTC)
            session.add(job)
        session.commit()


def recover_orphans(engine: Any, name: str) -> None:  # noqa: ANN401
    """Fail jobs this worker left running (e.g. after a crash or restart)."""
    with Session(engine) as session:
        jobs = session.exec(
            select(Job).where(Job.status == "running", Job.worker == name)
        ).all()
        for job in jobs:
            job.status = "failed"
            job.error = "Worker restarted while the job was running"
            job.finished_at = datetime.now(UTC)
            session.add(job)
        session.commit()


def isolate_ultralytics() -> None:
    """Keep Ultralytics settings, downloads and stray files inside STORAGE_DIR.

    Must run before ultralytics is imported; the user's global settings file
    would otherwise decide where AMP-check weights and runs land.
    """
    cache = storage.weights_cache()
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(storage.root() / "ultralytics"))
    os.chdir(cache)
    import ultralytics.utils
    import ultralytics.utils.checks

    ultralytics.utils.SETTINGS.update(
        weights_dir=str(cache),
        runs_dir=str(storage.root() / "runs"),
        datasets_dir=str(storage.root() / "datasets"),
        sync=False,
    )
    # WEIGHTS_DIR is computed at import (defaulting to <git repo>/weights when
    # the venv lives inside a repo) and the AMP check downloads there
    ultralytics.utils.WEIGHTS_DIR = cache
    setattr(ultralytics.utils.checks, "WEIGHTS_DIR", cache)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    settings = get_settings()
    engine = get_engine()
    init_db(engine)
    isolate_ultralytics()
    name = worker_name()
    recover_orphans(engine, name)
    log.info(f"Worker {name} polling every {settings.worker_poll_seconds}s")
    last_expiry = 0.0
    while True:
        job_id = claim_next(engine, name)
        if job_id is None:
            if time.monotonic() - last_expiry > 600:
                expire_uploads(engine)
                last_expiry = time.monotonic()
            time.sleep(settings.worker_poll_seconds)
            continue
        run_job(engine, job_id)


if __name__ == "__main__":
    main()
