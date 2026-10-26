import io
import zipfile
from typing import Any

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from conftest import FakeLabelStudio, login, make_user, new_project


def tif_bytes() -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.arange(64 * 64, dtype=np.uint16).reshape(64, 64)).save(
        buf, "TIFF"
    )
    return buf.getvalue()


def test_login_and_auth_required(client: TestClient, admin: dict[str, str]) -> None:
    assert client.get("/api/projects").status_code == 401
    assert client.get("/api/auth/me", headers=admin).json()["is_admin"] is True
    bad = {"username": "admin", "password": "nope"}
    assert client.post("/api/auth/login", data=bad).status_code == 401


def test_only_admin_creates_users(client: TestClient, admin: dict[str, str]) -> None:
    alice = make_user(client, admin, "alice")
    body = {"username": "bob", "password": "password123"}
    assert client.post("/api/auth/users", json=body, headers=alice).status_code == 403
    assert client.post("/api/auth/users", json=body, headers=admin).status_code == 201
    assert client.post("/api/auth/users", json=body, headers=admin).status_code == 409


def test_project_visibility_and_members(
    client: TestClient, admin: dict[str, str]
) -> None:
    alice = make_user(client, admin, "alice")
    bob = make_user(client, admin, "bob")
    project = new_project(client, alice)
    url = f"/api/projects/{project['id']}"

    assert client.get(url, headers=bob).status_code == 404
    assert client.get("/api/projects", headers=bob).json() == []
    assert client.get(url, headers=admin).status_code == 200

    resp = client.post(f"{url}/members", json={"username": "bob"}, headers=alice)
    assert resp.status_code == 204
    assert client.get(url, headers=bob).status_code == 200


def test_duplicate_classes_rejected(client: TestClient, admin: dict[str, str]) -> None:
    body = {"name": "x", "task_type": "detect", "classes": ["a", "a"]}
    assert client.post("/api/projects", json=body, headers=admin).status_code == 422


def test_upload_and_sync(
    client: TestClient, admin: dict[str, str], fake_ls: FakeLabelStudio
) -> None:
    project = new_project(client, admin)
    url = f"/api/projects/{project['id']}"

    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, "w") as zf:
        zf.writestr("nested/../b.tif", tif_bytes())
        zf.writestr("notes.txt", "ignored")
    files = [
        ("files", ("a.tif", tif_bytes(), "image/tiff")),
        ("files", ("batch.zip", zbuf.getvalue(), "application/zip")),
    ]
    resp = client.post(f"{url}/uploads", files=files, headers=admin)
    assert resp.status_code == 201, resp.text
    assert [i["filename"] for i in resp.json()] == ["a.png", "b.png"]

    bad = [("files", ("x.exe", b"MZ", "application/octet-stream"))]
    assert client.post(f"{url}/uploads", files=bad, headers=admin).status_code == 415

    status = client.get(f"{url}/labelstudio/status", headers=admin).json()
    assert (status["ls_project_id"], status["pending_upload"]) == (None, 2)

    status = client.post(f"{url}/labelstudio/sync", headers=admin).json()
    assert status["ls_project_id"] == 1
    assert status["pending_upload"] == 0
    assert status["ls_url"].endswith("/projects/1")
    assert fake_ls.uploaded == ["a.png", "b.png"]
    assert "<BrushLabels" in fake_ls.projects[1]["label_config"]

    # a second sync uploads only new images
    client.post(f"{url}/labelstudio/sync", headers=admin)
    assert fake_ls.uploaded == ["a.png", "b.png"]


def test_upload_keeps_numeric_order(
    client: TestClient, admin: dict[str, str], fake_ls: FakeLabelStudio
) -> None:
    url = f"/api/projects/{new_project(client, admin)['id']}"
    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, "w") as zf:
        for n in (100, 20, 9):
            zf.writestr(f"{n}.tif", tif_bytes())
    files = [("files", (f"{n}.tif", tif_bytes(), "image/tiff")) for n in (10, 2, 1)]
    files.append(("files", ("batch.zip", zbuf.getvalue(), "application/zip")))
    assert client.post(f"{url}/uploads", files=files, headers=admin).status_code == 201

    client.post(f"{url}/labelstudio/sync", headers=admin)
    expected = ["1.png", "2.png", "10.png", "9.png", "20.png", "100.png"]
    assert fake_ls.uploaded == expected
    images = client.get(f"{url}/images", headers=admin).json()
    assert [i["filename"] for i in images] == expected
    names = [fake_ls.task_data[1001 + i]["filename"] for i in range(len(expected))]
    assert names == ["000001.png", "000002.png", "000010.png", "000009.png", "000020.png", "000100.png"]


def test_preferences_are_per_user_and_merge(client: TestClient, admin: dict[str, str]) -> None:
    alice = make_user(client, admin, "alice")
    url = "/api/auth/me/preferences"
    assert client.get(url, headers=alice).json() == {}
    assert client.patch(url, json={"predict_conf": 0.4}, headers=alice).status_code == 200
    train = {"epochs": 50, "imgsz": 1024, "base_model": {"segment": "yolo26s-seg.pt"}}
    merged = client.patch(url, json={"train": train}, headers=alice).json()
    assert merged == {"predict_conf": 0.4, "train": train}
    assert client.get(url, headers=alice).json() == merged
    assert client.patch(url, json={"predict_conf": None}, headers=alice).json() == {"train": train}

    assert client.get(url, headers=admin).json() == {}  # other users unaffected
    huge = {"x": "a" * 40_000}
    assert client.patch(url, json=huge, headers=alice).status_code == 413
    assert client.get(url).status_code == 401
