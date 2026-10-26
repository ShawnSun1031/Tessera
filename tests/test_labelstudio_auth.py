import threading
import time
from collections.abc import Iterator
from typing import Any

import httpx
import jwt
import pytest
from fastapi.testclient import TestClient

from app.deps import get_labelstudio
from app.services import labelstudio
from app.services.labelstudio import LabelStudio, LabelStudioAuthError

HOST = "http://ls.test"
SIGNING_KEY = "test-signing-key-at-least-32-bytes-long"
REFRESH_KEY = jwt.encode({"token_type": "refresh"}, SIGNING_KEY, algorithm="HS256")
LIFETIME = 300


class FakeServer:
    """Label Studio stand-in issuing short-lived access tokens."""

    def __init__(self, accept_refresh: bool = True) -> None:
        self.accept_refresh = accept_refresh
        self.exchanges = 0
        self.valid: set[str] = set()
        self.lock = threading.Lock()

    def handler(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/token/refresh":
            if not self.accept_refresh:
                return httpx.Response(401, json={"detail": "Token is invalid"})
            with self.lock:
                self.exchanges += 1
                n = self.exchanges
            now = time.time()
            token = jwt.encode({"n": n, "iat": now, "exp": now + LIFETIME}, SIGNING_KEY)
            self.valid.add(token)
            return httpx.Response(200, json={"access": token})
        auth = request.headers.get("Authorization", "")
        if auth.removeprefix("Bearer ") in self.valid or auth == "Token legacy-key":
            return httpx.Response(200, json={"id": 6, "auth": auth})
        return httpx.Response(401, json={"detail": "Authentication credentials were not provided."})

    def client(self, key: str = REFRESH_KEY) -> LabelStudio:
        return LabelStudio(HOST, key, transport=httpx.MockTransport(self.handler))


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[float]]:
    """Controllable time for the token expiry check."""
    now = [time.time()]
    monkeypatch.setattr(labelstudio.time, "time", lambda: now[0])
    yield now


def test_token_renewed_before_it_expires(clock: list[float]) -> None:
    server = FakeServer()
    ls = server.client()
    assert ls.project(6)["id"] == 6
    clock[0] += 120
    ls.project(6)
    assert server.exchanges == 1  # still fresh: reused
    clock[0] += LIFETIME - 120 - 10  # within the renewal margin
    ls.project(6)
    assert server.exchanges == 2


def test_401_triggers_one_renewal_and_retry() -> None:
    server = FakeServer()
    ls = server.client()
    ls.project(6)
    server.valid.clear()  # e.g. Label Studio restarted / token revoked
    assert ls.project(6)["id"] == 6
    assert server.exchanges == 2


def test_concurrent_requests_renew_once(clock: list[float]) -> None:
    server = FakeServer()
    ls = server.client()
    ls.project(6)
    clock[0] += LIFETIME  # expired for everyone at once
    threads = [threading.Thread(target=ls.project, args=(6,)) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert server.exchanges == 2


def test_legacy_token_is_sent_as_is() -> None:
    server = FakeServer()
    ls = server.client("legacy-key")
    assert ls.project(6)["auth"] == "Token legacy-key"
    assert server.exchanges == 0


def test_rejected_key_raises_clear_error() -> None:
    ls = FakeServer(accept_refresh=False).client()
    with pytest.raises(LabelStudioAuthError, match="rejected LABEL_STUDIO_API_KEY"):
        ls.project(6)


def linked_project(client: TestClient, admin: dict[str, str]) -> int:
    from sqlmodel import Session

    from app.db import get_engine
    from app.models import Project

    body = {"name": "p", "task_type": "segment", "classes": ["Crack"]}
    pid = client.post("/api/projects", json=body, headers=admin).json()["id"]
    with Session(get_engine()) as session:
        project = session.get(Project, pid)
        assert project is not None
        project.ls_project_id = 6
        session.add(project)
        session.commit()
    return pid


@pytest.mark.parametrize(
    ("ls_factory", "message"),
    [
        (lambda: FakeServer(accept_refresh=False).client(), "rejected LABEL_STUDIO_API_KEY"),
        (
            lambda: LabelStudio(HOST, "legacy-key", transport=httpx.MockTransport(
                lambda r: (_ for _ in ()).throw(httpx.ConnectError("refused", request=r)))),
            "Can't reach Label Studio",
        ),
        (
            lambda: LabelStudio(HOST, "legacy-key", transport=httpx.MockTransport(
                lambda r: httpx.Response(500))),
            "Label Studio answered 500 for /api/projects/6",
        ),
    ],
)
def test_label_studio_failures_are_502_not_500(
    client: TestClient, admin: dict[str, str], ls_factory: Any, message: str  # noqa: ANN401
) -> None:
    pid = linked_project(client, admin)
    client.app.dependency_overrides[get_labelstudio] = ls_factory  # type: ignore[attr-defined]
    resp = client.get(f"/api/projects/{pid}/labelstudio/status", headers=admin)
    assert resp.status_code == 502
    assert message in resp.json()["detail"]


def test_status_keeps_working_after_token_expiry(
    client: TestClient, admin: dict[str, str], clock: list[float]
) -> None:
    """The reported bug: a long-lived cached client started failing after 5 minutes."""
    pid = linked_project(client, admin)
    server = FakeServer()
    ls = server.client()
    client.app.dependency_overrides[get_labelstudio] = lambda: ls  # type: ignore[attr-defined]
    url = f"/api/projects/{pid}/labelstudio/status"
    for _ in range(4):
        assert client.get(url, headers=admin).status_code == 200
        clock[0] += 200  # several token lifetimes in total
    # t=0 and t=400 need a new token; t=200 and t=600 reuse the current one
    assert server.exchanges == 2
