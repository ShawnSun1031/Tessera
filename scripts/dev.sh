#!/bin/bash
# Run API (port 8000), worker and Vite dev server (port 5173) together; Ctrl+C stops all.
set -e
cd "$(dirname "$0")/.."
trap 'kill 0' EXIT
uv run uvicorn app.main:app --reload --reload-dir backend --port 8000 &
uv run python -m app.worker &
(cd frontend && npm run dev) &
wait
