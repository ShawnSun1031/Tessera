"""FastAPI application entry point: `uvicorn app.main:app`."""

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import get_settings
from app.db import get_engine, init_db
from app.routers import auth, datasets, jobs, labelstudio, models, projects
from app.services.labelstudio import LabelStudioAuthError


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    init_db(get_engine())
    yield


app = FastAPI(title="Tessera", lifespan=lifespan)


@app.exception_handler(httpx.HTTPError)
@app.exception_handler(LabelStudioAuthError)
async def label_studio_error(_: Request, exc: Exception) -> JSONResponse:
    """Label Studio problems are upstream failures (502), with a usable message."""
    if isinstance(exc, LabelStudioAuthError):
        detail = str(exc)
    elif isinstance(exc, httpx.HTTPStatusError):
        detail = (
            f"Label Studio answered {exc.response.status_code} for"
            f" {exc.request.url.path}"
        )
    elif isinstance(exc, httpx.TransportError):
        detail = f"Can't reach Label Studio at {get_settings().label_studio_url}."
    else:
        detail = f"Label Studio request failed: {exc}"
    return JSONResponse(status_code=502, content={"detail": detail})


app.include_router(auth.router, prefix="/api")
app.include_router(projects.router, prefix="/api")
app.include_router(datasets.router, prefix="/api")
app.include_router(jobs.router, prefix="/api")
app.include_router(models.router, prefix="/api")
app.include_router(labelstudio.router, prefix="/api")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


# Serve the built React app (frontend/dist) when present; the Vite dev server
# proxies /api instead during development.
DEFAULT_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"
DIST = Path(os.environ.get("FRONTEND_DIST", DEFAULT_DIST))
if (DIST / "index.html").exists():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str) -> FileResponse:
        if path.startswith("api/"):
            raise HTTPException(404)
        file = (DIST / path).resolve()
        if path and file.is_file() and file.is_relative_to(DIST.resolve()):
            return FileResponse(file)
        return FileResponse(DIST / "index.html")
