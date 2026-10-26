"""Settings read from the environment (and `.env`)."""

import os
import secrets
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from app.services.labelstudio import load_env


@dataclass(frozen=True)
class Settings:
    database_url: str
    storage_dir: Path
    secret_key: str
    token_minutes: int
    label_studio_url: str
    label_studio_public_url: str
    label_studio_api_key: str
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


@lru_cache
def get_settings() -> Settings:
    load_env()
    env = os.environ.get
    ls_url = env("LABEL_STUDIO_URL", "http://localhost:8080")
    storage_dir = Path(env("STORAGE_DIR", "storage")).resolve()
    return Settings(
        database_url=env("DATABASE_URL", f"sqlite:///{storage_dir}/platform.db"),
        storage_dir=storage_dir,
        secret_key=env("SECRET_KEY") or persistent_secret(storage_dir),
        token_minutes=int(env("TOKEN_MINUTES", "720")),
        label_studio_url=ls_url,
        label_studio_public_url=env("LABEL_STUDIO_PUBLIC_URL", ls_url),
        label_studio_api_key=env("LABEL_STUDIO_API_KEY", ""),
        admin_username=env("ADMIN_USERNAME", "admin"),
        admin_password=env("ADMIN_PASSWORD", ""),
        device=env("DEVICE", ""),
        worker_poll_seconds=float(env("WORKER_POLL_SECONDS", "2")),
        max_predict_images=int(env("MAX_PREDICT_IMAGES", "20000")),
    )
