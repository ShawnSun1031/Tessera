import json
import socket
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.services import ls_server

# Stands in for `label-studio start`: answers /health and records its environment
FAKE_LABEL_STUDIO = """
import http.server, json, os, sys
port, env_file = int(sys.argv[1]), sys.argv[2]
json.dump(dict(os.environ), open(env_file, "w"))

class Health(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200 if self.path == "/health" else 404)
        self.end_headers()

http.server.HTTPServer(("127.0.0.1", port), Health).serve_forever()
"""


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(autouse=True)
def no_leftover_process() -> Iterator[None]:
    yield
    if ls_server._process is not None:
        ls_server._process.kill()
        ls_server._process.wait()
    ls_server._process = None


def configure(monkeypatch: pytest.MonkeyPatch, url: str, command: str = "") -> None:
    monkeypatch.setenv("LABEL_STUDIO_URL", url)
    monkeypatch.setenv("LABEL_STUDIO_PUBLIC_URL", url)
    monkeypatch.setenv("LABEL_STUDIO_COMMAND", command)
    get_settings.cache_clear()


def wait_for(client: TestClient, headers: dict[str, str], key: str) -> dict:
    status: dict = {}
    for _ in range(100):
        status = client.get("/api/labelstudio/server", headers=headers).json()
        if status[key]:
            return status
        time.sleep(0.1)
    raise AssertionError(f"Label Studio never became {key}: {status}")


def test_start_local_label_studio(
    client: TestClient, admin: dict[str, str], monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:  # fmt: skip
    port = free_port()
    script, env_file = tmp_path / "fake_ls.py", tmp_path / "env.json"
    script.write_text(FAKE_LABEL_STUDIO)
    configure(
        monkeypatch, f"http://localhost:{port}", f"{sys.executable} {script} {port} {env_file}"
    )

    assert client.get("/api/labelstudio/server").status_code == 401
    status = client.get("/api/labelstudio/server", headers=admin).json()
    assert status == {
        "url": f"http://localhost:{port}", "reachable": False, "can_start": True,
        "starting": False, "reason": None,
    }  # fmt: skip

    started = client.post("/api/labelstudio/server/start", headers=admin).json()
    assert started["starting"] or started["reachable"]
    first = ls_server._process
    client.post("/api/labelstudio/server/start", headers=admin)  # double click
    assert ls_server._process is first

    assert wait_for(client, admin, "reachable")["can_start"] is False
    again = client.post("/api/labelstudio/server/start", headers=admin)
    assert again.status_code == 409
    # the platform's own settings are not passed on to Label Studio
    env = json.loads(env_file.read_text())
    assert "STORAGE_DIR" not in env and "DATABASE_URL" not in env


def test_failed_start_shows_the_log(
    client: TestClient, admin: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    fail = f"{sys.executable} -c \"print('port 8080 is already in use'); exit(1)\""
    configure(monkeypatch, f"http://127.0.0.1:{free_port()}", fail)
    client.post("/api/labelstudio/server/start", headers=admin)
    assert ls_server._process is not None
    ls_server._process.wait()
    status = client.get("/api/labelstudio/server", headers=admin).json()
    assert (status["starting"], status["can_start"]) == (False, True)
    assert "port 8080 is already in use" in status["reason"]


def test_remote_label_studio_cannot_be_started(
    client: TestClient, admin: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    configure(monkeypatch, "http://labelstudio.invalid:8080")
    status = client.get("/api/labelstudio/server", headers=admin).json()
    assert status["can_start"] is False
    assert "another machine" in status["reason"]
    resp = client.post("/api/labelstudio/server/start", headers=admin)
    assert resp.status_code == 422


def test_default_command_uses_the_url_port(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ls_server.shutil, "which", lambda name: "/opt/bin/label-studio")
    assert ls_server.start_command("http://localhost:8081", "") == [
        "/opt/bin/label-studio", "start", "--port", "8081", "--no-browser",
    ]  # fmt: skip
    assert ls_server.start_command("http://localhost:8081", "docker start ls") == [
        "docker", "start", "ls",
    ]  # fmt: skip
