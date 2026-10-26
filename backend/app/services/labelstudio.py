"""Minimal Label Studio HTTP client."""

import os
import threading
import time
from collections.abc import Generator, Sequence
from pathlib import Path
from typing import Any
from xml.sax.saxutils import quoteattr

import httpx
import jwt


def load_env(path: str = ".env") -> None:
    if Path(path).exists():
        for line in Path(path).read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("\"'"))


class LabelStudioAuthError(Exception):
    """The configured API key was rejected by Label Studio."""


class TokenAuth(httpx.Auth):
    """Label Studio auth that keeps itself valid.

    Personal Access Tokens are JWT refresh tokens: they are exchanged for an
    access token that expires after ~5 minutes, so it is renewed shortly before
    expiry and once more if Label Studio still answers 401. Legacy (non-JWT)
    tokens are sent as-is and never expire.
    """

    REFRESH_MARGIN = 30  # seconds before expiry to renew

    def __init__(
        self, host: str, key: str, transport: httpx.BaseTransport | None = None
    ) -> None:
        self.host = host.rstrip("/")
        self.key = key
        self.transport = transport
        self.refreshable = key.count(".") == 2  # JWT: header.payload.signature
        self.lock = threading.Lock()
        self.header: str | None = None if self.refreshable else f"Token {key}"
        self.expires_at: float | None = None

    def current(self, force: bool = False) -> str:
        with self.lock:
            stale = self.expires_at is not None and (
                time.time() > self.expires_at - self.REFRESH_MARGIN
            )
            if self.header is None or (self.refreshable and (force or stale)):
                self.header, self.expires_at = self.exchange()
            return self.header

    def exchange(self) -> tuple[str, float | None]:
        with httpx.Client(transport=self.transport, timeout=30) as client:
            resp = client.post(
                f"{self.host}/api/token/refresh", json={"refresh": self.key}
            )
        if resp.status_code in (400, 401):
            raise LabelStudioAuthError(
                "Label Studio rejected LABEL_STUDIO_API_KEY. Create a new access"
                " token in Label Studio (Account & Settings) and update .env."
            )
        resp.raise_for_status()
        access = resp.json()["access"]
        try:
            claims = jwt.decode(access, options={"verify_signature": False})
            expires_at = float(claims["exp"])
        except (jwt.PyJWTError, KeyError, TypeError, ValueError):
            expires_at = None  # unknown lifetime: rely on the 401 retry
        return f"Bearer {access}", expires_at

    def auth_flow(
        self, request: httpx.Request
    ) -> Generator[httpx.Request, httpx.Response, None]:
        request.headers["Authorization"] = self.current()
        response = yield request
        if response.status_code == 401 and self.refreshable:
            request.headers["Authorization"] = self.current(force=True)
            yield request


def label_config(task_type: str, classes: Sequence[str]) -> str:
    """LS labeling config: rectangles for detect; brush + polygon for segment."""
    labels = "".join(f"<Label value={quoteattr(c)}/>" for c in classes)
    if task_type == "detect":
        tags = ["RectangleLabels"]
    else:
        tags = ["BrushLabels", "PolygonLabels"]
    controls = "".join(
        f'<{t} name="{t.lower()}" toName="image">{labels}</{t}>' for t in tags
    )
    return f'<View><Image name="image" value="$image" zoom="true"/>{controls}</View>'


class LabelStudio:
    def __init__(
        self, host: str, key: str, transport: httpx.BaseTransport | None = None
    ) -> None:
        self.host = host.rstrip("/")
        self.auth = TokenAuth(self.host, key, transport)
        self.client = httpx.Client(
            base_url=self.host, auth=self.auth, timeout=60, transport=transport
        )

    @classmethod
    def from_key(cls, host: str, key: str) -> "LabelStudio":
        return cls(host, key)

    def get_json(self, path: str) -> Any:  # noqa: ANN401
        resp = self.client.get(path)
        resp.raise_for_status()
        return resp.json()

    def project(self, project_id: int) -> dict[str, Any]:
        return self.get_json(f"/api/projects/{project_id}")

    def list_projects(self) -> list[dict[str, Any]]:
        """All projects visible to the API token (follows pagination)."""
        projects: list[dict[str, Any]] = []
        page = 1
        while True:
            data = self.get_json(f"/api/projects?page={page}&page_size=100")
            if isinstance(data, list):  # older Label Studio: unpaginated
                return data
            projects += data["results"]
            if not data.get("next"):
                return projects
            page += 1

    def export_tasks(self, project_id: int) -> list[dict[str, Any]]:
        return self.get_json(f"/api/projects/{project_id}/export?exportType=JSON")

    def create_project(self, title: str, config: str) -> dict[str, Any]:
        resp = self.client.post(
            "/api/projects", json={"title": title, "label_config": config}
        )
        resp.raise_for_status()
        return resp.json()

    def delete_project(self, project_id: int) -> None:
        self.client.delete(f"/api/projects/{project_id}").raise_for_status()

    def import_files(self, project_id: int, paths: Sequence[Path]) -> int:
        """Upload image files as new tasks; returns the number of tasks created."""
        handles = [p.open("rb") for p in paths]
        try:
            files = [("files", (p.name, f)) for p, f in zip(paths, handles)]
            resp = self.client.post(f"/api/projects/{project_id}/import", files=files)
        finally:
            for f in handles:
                f.close()
        resp.raise_for_status()
        return resp.json().get("task_count", 0)

    def import_file(self, project_id: int, path: Path) -> int:
        """Upload one image as a new task; returns the task id."""
        with path.open("rb") as f:
            resp = self.client.post(
                f"/api/projects/{project_id}/import?return_task_ids=true",
                files=[("files", (path.name, f))],
            )
        resp.raise_for_status()
        task_ids = resp.json().get("task_ids") or []
        if len(task_ids) != 1:
            raise RuntimeError(f"Label Studio created {len(task_ids)} tasks for 1 file")
        return int(task_ids[0])

    def create_prediction(
        self, task_id: int, result: list[dict[str, Any]], score: float | None,
        model_version: str,
    ) -> None:  # fmt: skip
        body = {"task": task_id, "result": result, "model_version": model_version}
        if score is not None:
            body["score"] = score
        self.client.post("/api/predictions", json=body).raise_for_status()

    def use_predictions_for_prelabeling(self, project_id: int, version: str) -> None:
        """Make annotators start from these predictions.

        Label Studio only pre-fills tasks with predictions matching the project's
        selected model version (or all of them when none is selected).
        """
        body = {"model_version": version, "show_collab_predictions": True}
        self.client.patch(f"/api/projects/{project_id}", json=body).raise_for_status()

    def download(self, url: str) -> bytes:
        # only LS-hosted files need our auth header; don't leak it to external hosts
        if url.startswith("http") and not url.startswith(self.host):
            resp = httpx.get(url, timeout=60)
        else:
            resp = self.client.get(url.removeprefix(self.host))
        resp.raise_for_status()
        return resp.content
