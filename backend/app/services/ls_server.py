"""Start Label Studio on this machine when it isn't running.

Only for a Label Studio at a local address (LABEL_STUDIO_URL on localhost).
It runs detached from the API, so restarting or stopping the API leaves it up.
"""

import os
import shlex
import shutil
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

import httpx

LOCAL_HOSTS = ("localhost", "127.0.0.1", "::1")
# our own settings; Label Studio reads some of the same names (e.g. SECRET_KEY)
PLATFORM_ENV = (
    "DATABASE_URL", "STORAGE_DIR", "SECRET_KEY", "TOKEN_MINUTES", "DEVICE",
    "LABEL_STUDIO_URL", "LABEL_STUDIO_PORT", "LABEL_STUDIO_PUBLIC_URL",
    "LABEL_STUDIO_API_KEY", "LABEL_STUDIO_COMMAND", "ADMIN_USERNAME", "ADMIN_PASSWORD",
    "WORKER_NAME", "WORKER_POLL_SECONDS", "MAX_PREDICT_IMAGES",
)  # fmt: skip

_lock = threading.Lock()
_process: subprocess.Popen[bytes] | None = None


@dataclass
class ServerState:
    reachable: bool
    can_start: bool
    starting: bool
    reason: str | None = None  # why it can't be started, or why the last start failed


def reachable(url: str) -> bool:
    try:
        return httpx.get(f"{url.rstrip('/')}/health", timeout=2).status_code == 200
    except httpx.HTTPError:
        return False


def start_command(url: str, configured: str) -> list[str] | None:
    """LABEL_STUDIO_COMMAND as given, else `label-studio start` on the URL's port."""
    if configured:
        return shlex.split(configured)
    exe = shutil.which("label-studio")
    # `uv tool install label-studio` puts it here, which a service PATH may lack
    fallback = Path.home() / ".local" / "bin" / "label-studio"
    if exe is None and fallback.exists():
        exe = str(fallback)
    if exe is None:
        return None
    port = urlsplit(url).port or 8080
    return [exe, "start", "--port", str(port), "--no-browser"]


def log_tail(log: Path, lines: int = 5) -> str:
    if not log.exists():
        return ""
    return "\n".join(log.read_text(errors="replace").splitlines()[-lines:])


def state(url: str, configured: str, log: Path) -> ServerState:
    if reachable(url):
        return ServerState(reachable=True, can_start=False, starting=False)
    if urlsplit(url).hostname not in LOCAL_HOSTS:
        return unavailable(
            f"Label Studio runs on another machine ({url}); ask its admin to start it."
        )
    if start_command(url, configured) is None:
        return unavailable(
            "The label-studio command was not found on the server. Install it"
            " (uv tool install label-studio) or set LABEL_STUDIO_COMMAND."
        )
    with _lock:
        running = _process is not None and _process.poll() is None
        failed = _process is not None and _process.returncode not in (None, 0)
    reason = f"The last start failed:\n{log_tail(log)}" if failed else None
    return ServerState(reachable=False, can_start=True, starting=running, reason=reason)


def unavailable(reason: str) -> ServerState:
    return ServerState(reachable=False, can_start=False, starting=False, reason=reason)


def start(url: str, configured: str, log: Path) -> None:
    """Launch Label Studio in the background; no-op if a launch is under way."""
    global _process
    command = start_command(url, configured)
    assert command is not None
    with _lock:
        if _process is not None and _process.poll() is None:
            return
        log.parent.mkdir(parents=True, exist_ok=True)
        env = {k: v for k, v in os.environ.items() if k not in PLATFORM_ENV}
        with log.open("ab") as out:
            _process = subprocess.Popen(
                command, stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT,
                env=env, start_new_session=True,
            )  # fmt: skip
