"""Settings read from the environment (and `.env`)."""

import os
import secrets
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlsplit

from app.services.labelstudio import load_env
from app.services.ls_server import LOCAL_HOSTS


@dataclass(frozen=True)
class Settings:
    database_url: str
    storage_dir: Path
    secret_key: str
    token_minutes: int
    label_studio_url: str
    label_studio_public_url: str
    label_studio_api_key: str
    label_studio_command: str
    admin_username: str
    admin_password: str
    device: str
    worker_poll_seconds: float
    max_predict_images: int


def persistent_secret(storage_dir: Path) -> str:
    """Random signing key created on first run and kept in the storage dir."""
    path = storage_dir / ".secret_key"
    if not path.exists():
        storage_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(secrets.token_urlsafe(48))
        path.chmod(0o600)
    return path.read_text().strip()


def label_studio_url(url: str, port: str) -> str:
    """LABEL_STUDIO_URL with LABEL_STUDIO_PORT, if set, as the port of a local one."""
    if not port:
        return url
    if not port.isdigit() or not 0 < int(port) < 65536:
        raise ValueError(f"LABEL_STUDIO_PORT must be a port number, not {port!r}")
    parts = urlsplit(url)
    if parts.hostname not in LOCAL_HOSTS:
        raise ValueError(
            "LABEL_STUDIO_PORT only applies to a Label Studio on localhost;"
            f" put the port in LABEL_STUDIO_URL ({url}) instead"
        )
    host = f"[{parts.hostname}]" if ":" in parts.hostname else parts.hostname
    return parts._replace(netloc=f"{host}:{port}").geturl()


@lru_cache
def get_settings() -> Settings:
    load_env()
    env = os.environ.get
    ls_url = label_studio_url(
        env("LABEL_STUDIO_URL", "http://localhost:8080"),
        env("LABEL_STUDIO_PORT", "").strip(),
    )
    storage_dir = Path(env("STORAGE_DIR", "storage")).resolve()
    return Settings(
        database_url=env("DATABASE_URL", f"sqlite:///{storage_dir}/platform.db"),
        storage_dir=storage_dir,
        secret_key=env("SECRET_KEY") or persistent_secret(storage_dir),
        token_minutes=int(env("TOKEN_MINUTES", "720")),
        label_studio_url=ls_url,
        label_studio_public_url=env("LABEL_STUDIO_PUBLIC_URL", ls_url),
        label_studio_api_key=env("LABEL_STUDIO_API_KEY", ""),
        label_studio_command=env("LABEL_STUDIO_COMMAND", ""),
        admin_username=env("ADMIN_USERNAME", "admin"),
        admin_password=env("ADMIN_PASSWORD", ""),
        device=env("DEVICE", ""),
        worker_poll_seconds=float(env("WORKER_POLL_SECONDS", "2")),
        max_predict_images=int(env("MAX_PREDICT_IMAGES", "20000")),
    )
