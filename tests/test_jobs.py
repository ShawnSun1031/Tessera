import io
import zipfile
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session

from app import worker
from app.db import get_engine
from app.models import Job, TrainedModel
from app.storage import train_dir
from conftest import FakeLabelStudio, make_user, new_project, png_bytes, rect_task


def detect_project(client: TestClient, headers: dict[str, str]) -> dict[str, Any]:
    body = {"name": "Holes", "task_type": "detect", "classes": ["Crack", "Hole"]}
    return client.post("/api/projects", json=body, headers=headers).json()


def yolo_zip(n: int = 5, with_split: bool = False, label: str = "1 0.5 0.5 0.2 0.2") -> bytes:  # noqa: E501
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for i in range(n):
            folder = ("val/" if i == 0 else "train/") if with_split else ""
            zf.writestr(f"images/{folder}img{i}.png", png_bytes())
            zf.writestr(f"labels/{folder}img{i}.txt", label + "\n")
    return buf.getvalue()


def upload_dataset(client: TestClient, headers: dict[str, str], pid: int, data: bytes) -> Any:  # noqa: ANN401, E501
    files = {"file": ("ds.zip", data, "application/zip")}
    return client.post(
        f"/api/projects/{pid}/datasets/upload", files=files, headers=headers
    )


def test_upload_yolo_zip(client: TestClient, admin: dict[str, str]) -> None:
    pid = detect_project(client, admin)["id"]
    resp = upload_dataset(client, admin, pid, yolo_zip(with_split=True))
    assert resp.status_code == 201, resp.text
    stats = resp.json()["stats"]
    assert stats["splits"]["train"]["images"] == 4
    assert stats["splits"]["val"]["images"] == 1
    assert stats["instances"] == {"Crack": 0, "Hole": 5}

    bad = upload_dataset(client, admin, pid, yolo_zip(label="7 0.5 0.5 0.2 0.2"))
    assert bad.status_code == 422
    assert "class id 7" in bad.json()["detail"]
    seg = upload_dataset(client, admin, pid, yolo_zip(label="1 0.1 0.1 0.2"))
    assert seg.status_code == 422
    assert len(client.get(f"/api/projects/{pid}/datasets", headers=admin).json()) == 1


def test_export_job_runs_in_worker(
    client: TestClient,
    admin: dict[str, str],
    fake_ls: FakeLabelStudio,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pid = detect_project(client, admin)["id"]
    url = f"/api/projects/{pid}"
    assert client.post(f"{url}/datasets/export", json={}, headers=admin).status_code == 409

    client.post(f"{url}/labelstudio/sync", headers=admin)
    fake_ls.tasks = [rect_task(i) for i in range(10)]
    monkeypatch.setattr(worker.LabelStudio, "from_key", lambda *a: fake_ls)
    job = client.post(f"{url}/datasets/export", json={"val_ratio": 0.2}, headers=admin)
    assert job.status_code == 201
    job_id = job.json()["id"]

    engine = get_engine()
    assert worker.claim_next(engine, "test") == job_id
    assert worker.claim_next(engine, "test") is None  # already claimed
    worker.run_job(engine, job_id)

    done = client.get(f"/api/jobs/{job_id}", headers=admin).json()
    assert done["status"] == "succeeded", done["error"]
    datasets = client.get(f"{url}/datasets", headers=admin).json()
    assert datasets[0]["id"] == done["result"]["dataset_id"]
    assert datasets[0]["stats"]["splits"]["val"]["images"] == 2
    assert datasets[0]["stats"]["instances"]["Hole"] == 10

    log = client.get(f"/api/jobs/{job_id}/log", headers=admin).json()
    assert "10 annotated tasks" in log["text"]

    # SSE stream ends by itself for a finished job
    with client.stream("GET", f"/api/jobs/{job_id}/events", headers=admin) as resp:
        body = "".join(resp.iter_text())
    assert "event: job" in body and "event: end" in body


def test_train_request_validation_and_cancel(
    client: TestClient, admin: dict[str, str]
) -> None:
    pid = detect_project(client, admin)["id"]
    ds = upload_dataset(client, admin, pid, yolo_zip()).json()["id"]
    url = f"/api/projects/{pid}/train"
    archs = client.get(f"/api/projects/{pid}/architectures", headers=admin).json()
    assert "yolo26n.pt" in [a["weights"] for a in archs["architectures"]]

    def train(**overrides: Any) -> Any:  # noqa: ANN401
        body = {"dataset_id": ds, "base_model": "yolo26n.pt", "epochs": 1, **overrides}
        return client.post(url, json=body, headers=admin)

    assert train(base_model="yolo26n-seg.pt").status_code == 422  # wrong task
    assert train(base_model="model:999").status_code == 404
    assert train(imgsz=100).status_code == 422  # not a multiple of 32
    assert train(dataset_id=999).status_code == 404

    first, second = train().json(), train().json()
    assert (first["queue_position"], second["queue_position"]) == (0, 1)
    queue = client.get("/api/queue", headers=admin).json()
    assert [q["id"] for q in queue] == [first["id"], second["id"]]

    resp = client.post(f"/api/jobs/{first['id']}/cancel", headers=admin)
    assert resp.json()["status"] == "cancelled"
    assert client.post(f"/api/jobs/{first['id']}/cancel", headers=admin).status_code == 409
    second = client.get(f"/api/jobs/{second['id']}", headers=admin).json()
    assert second["queue_position"] == 0


def test_jobs_and_files_are_project_scoped(
    client: TestClient, admin: dict[str, str]
) -> None:
    alice = make_user(client, admin, "alice")
    pid = new_project(client, alice)["id"]
    zip_bytes = yolo_zip(label="0 0.1 0.1 0.5 0.1 0.5 0.5")
    ds = upload_dataset(client, alice, pid, zip_bytes).json()["id"]
    body = {"dataset_id": ds, "base_model": "yolo26n-seg.pt", "epochs": 1}
    job = client.post(f"/api/projects/{pid}/train", json=body, headers=alice).json()

    # pretend the job trained a model with a plot
    with Session(get_engine()) as session:
        db_job = session.get(Job, job["id"])
        assert db_job is not None
        db_job.status = "succeeded"
        model = TrainedModel(
            project_id=pid, job_id=job["id"], dataset_id=ds, name="m",
            task_type="segment", architecture="yolo26n-seg.pt", classes=["a"],
        )  # fmt: skip
        session.add_all([db_job, model])
        session.commit()
        model_id = model.id
    run = train_dir(job["id"])
    (run / "weights").mkdir(parents=True)
    (run / "results.png").write_bytes(png_bytes())
    (run / "weights" / "best.pt").write_bytes(b"weights")

    bob = make_user(client, admin, "bob")
    assert client.get(f"/api/jobs/{job['id']}", headers=bob).status_code == 404
    assert client.get(f"/api/models/{model_id}", headers=bob).status_code == 404

    info = client.get(f"/api/models/{model_id}", headers=alice).json()
    assert info["plots"] == ["results.png"]
    token = alice["Authorization"].split()[1]
    img = client.get(f"/api/models/{model_id}/files/results.png?access_token={token}")
    assert img.status_code == 200
    escape = client.get(f"/api/models/{model_id}/files/../../../platform.db", headers=alice)
    assert escape.status_code == 404
    pt = client.get(f"/api/models/{model_id}/download", headers=alice)
    assert pt.content == b"weights"
    onnx = client.get(f"/api/models/{model_id}/download?format=onnx", headers=alice)
    assert onnx.status_code == 404


def test_inference_defaults_to_training_image_size(
    client: TestClient, admin: dict[str, str]
) -> None:
    pid = detect_project(client, admin)["id"]
    ds = upload_dataset(client, admin, pid, yolo_zip()).json()["id"]
    body = {"dataset_id": ds, "base_model": "yolo26n.pt", "epochs": 1, "imgsz": 1056}
    job = client.post(f"/api/projects/{pid}/train", json=body, headers=admin).json()
    assert job["params"]["imgsz"] == 1056
    with Session(get_engine()) as session:
        model = TrainedModel(
            project_id=pid, job_id=job["id"], dataset_id=ds, name="m",
            task_type="detect", architecture="yolo26n.pt", classes=["Crack", "Hole"],
        )  # fmt: skip
        session.add(model)
        session.commit()
        model_id = model.id

    assert client.get(f"/api/models/{model_id}", headers=admin).json()["imgsz"] == 1056
    onnx = client.post(f"/api/models/{model_id}/export-onnx", json={}, headers=admin)
    assert onnx.json()["params"]["imgsz"] == 1056
    files = [("files", ("a.png", png_bytes(), "image/png"))]
    pred = client.post(f"/api/models/{model_id}/predict", files=files, headers=admin)
    assert pred.json()["params"]["imgsz"] == 1056
    override = client.post(
        f"/api/models/{model_id}/predict", files=files, data={"imgsz": "640"}, headers=admin
    )
    assert override.json()["params"]["imgsz"] == 640
