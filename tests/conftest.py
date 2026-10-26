

import io
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from PIL import Image as PILImage

from app.config import get_settings
from app.db import get_engine
from app.deps import get_labelstudio


class FakeLabelStudio:
    def __init__(self) -> None:
        self.projects: dict[int, dict[str, Any]] = {}
        self.uploaded: list[str] = []
        self.predictions: dict[int, list[dict[str, Any]]] = {}
        self.task_data: dict[int, dict[str, Any]] = {}

    def import_file(self, project_id: int, path: Path) -> int:
        assert path.exists()
        self.uploaded.append(path.name)
        return 1000 + len(self.uploaded)  # task id

    def use_predictions_for_prelabeling(self, project_id: int, version: str) -> None:
        self.projects[project_id]["model_version"] = version

    def create_prediction(
        self, task_id: int, result: list[dict[str, Any]], score: float | None,
        model_version: str,
    ) -> None:  # fmt: skip
        self.predictions.setdefault(task_id, []).append(
            {"result": result, "score": score, "model_version": model_version}
        )

    def create_project(self, title: str, config: str) -> dict[str, Any]:
        pid = len(self.projects) + 1
        self.projects[pid] = {"id": pid, "title": title, "label_config": config}
        return self.projects[pid]

    def import_files(self, project_id: int, paths: list[Path]) -> list[int]:
        assert all(p.exists() for p in paths)
        start = len(self.uploaded)
        self.uploaded += [p.name for p in paths]
        return [1001 + start + i for i in range(len(paths))]

    def update_task_data(self, task_id: int, fields: dict[str, Any]) -> None:
        self.task_data.setdefault(task_id, {}).update(fields)

    def project(self, project_id: int) -> dict[str, Any]:
        if project_id not in self.projects:
            request = httpx.Request("GET", f"http://ls/api/projects/{project_id}")
            raise httpx.HTTPStatusError(
                "not found", request=request, response=httpx.Response(404, request=request)
            )
        counts = {"task_number": len(self.uploaded), "num_tasks_with_annotations": 1}
        return {**counts, **self.projects[project_id]}

    def list_projects(self) -> list[dict[str, Any]]:
        return [self.project(pid) for pid in self.projects]

    def add_existing(self, config: str, title: str = "Labeled elsewhere") -> int:
        """A project someone labeled directly in Label Studio."""
        pid = len(self.projects) + 1
        self.projects[pid] = {"id": pid, "title": title, "label_config": config}
        return pid

    # export side: tasks with one rectangle annotation each
    tasks: list[dict[str, Any]] = []

    def export_tasks(self, project_id: int) -> list[dict[str, Any]]:
        return self.tasks

    def download(self, url: str) -> bytes:
        return png_bytes()


def png_bytes(w: int = 64, h: int = 64) -> bytes:
    buf = io.BytesIO()
    PILImage.new("RGB", (w, h), "gray").save(buf, "PNG")
    return buf.getvalue()


def rect_task(i: int, label: str = "Hole") -> dict[str, Any]:
    value = {"x": 10, "y": 10, "width": 20, "height": 30, "rectanglelabels": [label]}
    result = {"type": "rectanglelabels", "value": value,
              "original_width": 64, "original_height": 64}  # fmt: skip
    return {"id": i, "data": {"image": f"/data/upload/1/img{i}.png"},
            "annotations": [{"result": [result]}]}  # fmt: skip


@pytest.fixture
def fake_ls() -> FakeLabelStudio:
    return FakeLabelStudio()


@pytest.fixture
def client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_ls: FakeLabelStudio
) -> Iterator[TestClient]:
    monkeypatch.setenv("STORAGE_DIR", str(tmp_path / "storage"))
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/test.db")
    monkeypatch.setenv("ADMIN_USERNAME", "admin")
    monkeypatch.setenv("ADMIN_PASSWORD", "admin-password")
    monkeypatch.setenv("LABEL_STUDIO_PORT", "")  # ignore one set in a local .env
    get_settings.cache_clear()
    get_engine.cache_clear()
    from app.main import app

    app.dependency_overrides[get_labelstudio] = lambda: fake_ls
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()
    get_settings.cache_clear()
    get_engine.cache_clear()


def login(client: TestClient, username: str, password: str) -> dict[str, str]:
    resp = client.post(
        "/api/auth/login", data={"username": username, "password": password}
    )
    assert resp.status_code == 200, resp.text
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


@pytest.fixture
def admin(client: TestClient) -> dict[str, str]:
    return login(client, "admin", "admin-password")


def make_user(client: TestClient, admin: dict[str, str], name: str) -> dict[str, str]:
    body = {"username": name, "password": "password123"}
    assert client.post("/api/auth/users", json=body, headers=admin).status_code == 201
    return login(client, name, "password123")


def new_project(client: TestClient, headers: dict[str, str]) -> dict[str, Any]:
    body = {"name": "Cracks", "task_type": "segment", "classes": ["Crack", "Hole"]}
    resp = client.post("/api/projects", json=body, headers=headers)
    assert resp.status_code == 201, resp.text
    return resp.json()

