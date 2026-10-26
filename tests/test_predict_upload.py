import io
import json
import zipfile
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session

from app import storage, worker
from app.db import get_engine
from app.models import TrainedModel
from conftest import png_bytes
from test_jobs import detect_project, upload_dataset, yolo_zip


def make_model(client: TestClient, headers: dict[str, str]) -> int:
    pid = detect_project(client, headers)["id"]
    ds = upload_dataset(client, headers, pid, yolo_zip()).json()["id"]
    body = {"dataset_id": ds, "base_model": "yolo26n.pt", "epochs": 1}
    job = client.post(f"/api/projects/{pid}/train", json=body, headers=headers).json()
    client.post(f"/api/jobs/{job['id']}/cancel", headers=headers)
    with Session(get_engine()) as session:
        model = TrainedModel(
            project_id=pid, job_id=job["id"], dataset_id=ds, name="m",
            task_type="detect", architecture="yolo26n.pt", classes=["Crack", "Hole"],
        )  # fmt: skip
        session.add(model)
        session.commit()
        assert model.id is not None
        return model.id


def image(name: str) -> tuple[str, tuple[str, bytes, str]]:
    return ("files", (name, png_bytes(), "image/png"))


def fake_predict(seen: list[str]) -> Any:  # noqa: ANN401
    def predict(weights: Path, inputs: list[Path], out: Path, *args: Any) -> dict:  # noqa: ANN401
        seen.extend(p.name for p in inputs)
        (out / "images").mkdir(parents=True)
        for p in inputs:
            (out / "images" / p.name).write_bytes(p.read_bytes())
        (out / "predictions.json").write_text(json.dumps([{"image": n} for n in seen]))
        return {"images": len(inputs), "objects": 0, "per_class": {}}

    return predict


def test_upload_in_parts_then_predict(
    client: TestClient, admin: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    job = client.post(
        f"/api/models/{make_model(client, admin)}/predict", data={"conf": "0.4"},
        headers=admin,
    ).json()  # fmt: skip
    assert (job["status"], job["params"]["conf"]) == ("uploading", 0.4)
    url = f"/api/jobs/{job['id']}"
    assert client.post(f"{url}/start", headers=admin).status_code == 422

    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, "w") as zf:
        zf.writestr("20.png", png_bytes())
        zf.writestr("3.png", png_bytes())
    # parts may arrive in any order; a retried part replaces itself
    part2 = [
        ("files", ("batch.zip", zbuf.getvalue(), "application/zip")),
        ("files", ("broken.png", b"not an image", "image/png")),
    ]
    resp = client.post(f"{url}/inputs", files=part2, data={"offset": "2"}, headers=admin)
    assert resp.json() == {"files": 2}
    for _ in range(2):
        part1 = [image("1.png"), image("2.png")]
        resp = client.post(f"{url}/inputs", files=part1, data={"offset": "0"}, headers=admin)
        assert resp.json() == {"files": 4}
    bad = [("files", ("x.exe", b"MZ", "application/octet-stream"))]
    assert client.post(f"{url}/inputs", files=bad, headers=admin).status_code == 415

    assert client.post(f"{url}/start", headers=admin).json()["status"] == "queued"
    late = client.post(f"{url}/inputs", files=[image("9.png")], headers=admin)
    assert late.status_code == 409

    seen: list[str] = []
    monkeypatch.setattr(worker.training, "predict", fake_predict(seen))
    engine = get_engine()
    assert worker.claim_next(engine, "test") == job["id"]
    worker.run_job(engine, job["id"])

    done = client.get(url, headers=admin).json()
    assert done["status"] == "succeeded", done["error"]
    assert done["result"]["skipped_files"] == 1
    assert seen == ["1.png", "2.png", "3.png", "20.png"]
    assert not storage.uploads_dir(job["id"]).exists()

    resp = client.get(f"{url}/download", headers=admin)
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        assert "images/20.png" in zf.namelist() and "predictions.json" in zf.namelist()
        assert zf.getinfo("images/20.png").compress_type == zipfile.ZIP_STORED


def test_upload_limit(
    client: TestClient, admin: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "app.routers.models.get_settings",
        lambda: type("S", (), {"max_predict_images": 3})(),
    )
    job = client.post(f"/api/models/{make_model(client, admin)}/predict", headers=admin)
    url = f"/api/jobs/{job.json()['id']}/inputs"
    ok = client.post(url, files=[image("1.png")], data={"offset": "2"}, headers=admin)
    assert ok.status_code == 200
    over = client.post(url, files=[image("1.png")], data={"offset": "3"}, headers=admin)
    assert over.status_code == 422


def test_abandoned_uploads_are_cleaned_up(
    client: TestClient, admin: dict[str, str]
) -> None:
    model_id = make_model(client, admin)
    ids = []
    for _ in range(2):
        job = client.post(f"/api/models/{model_id}/predict", headers=admin).json()
        client.post(f"/api/jobs/{job['id']}/inputs", files=[image("1.png")], headers=admin)
        assert storage.uploads_dir(job["id"]).is_dir()
        ids.append(job["id"])

    cancelled = client.post(f"/api/jobs/{ids[0]}/cancel", headers=admin).json()
    assert cancelled["status"] == "cancelled"
    assert not storage.uploads_dir(ids[0]).exists()

    worker.expire_uploads(get_engine())  # too recent to expire
    assert client.get(f"/api/jobs/{ids[1]}", headers=admin).json()["status"] == "uploading"
    worker.expire_uploads(get_engine(), max_age=timedelta(0))
    expired = client.get(f"/api/jobs/{ids[1]}", headers=admin).json()
    assert (expired["status"], expired["error"]) == ("cancelled", "The upload was not finished")
    assert not storage.uploads_dir(ids[1]).exists()
