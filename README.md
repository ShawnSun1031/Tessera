# Tessera

Web platform for annotating images in Label Studio and training / running Ultralytics YOLO models.
A step-by-step UI walks users through: upload images → annotate → build dataset → train → review results → predict.

```
React (frontend/)  ──REST + SSE──►  FastAPI (backend/app)  ──►  SQLite/Postgres
                                        │   │
                                        │   └──►  Label Studio (HTTP API)
                                        ▼
                                  worker (python -m app.worker), one per GPU:
                                  runs dataset export, training, prediction, ONNX export
```

Already labeled in Label Studio? Choose **New project → Use labels already in Label Studio**
(or link one from the Annotate step). Object types and task come from its labeling config, and
**Build dataset** converts the labels to YOLO format: brush masks and polygons become segmentation
polygons (one per separately painted region), rectangles become boxes. Any dataset can be
downloaded as a ready-to-train YOLO zip.

## Quick start (development)

```bash
uv sync
(cd frontend && npm install)
echo "ADMIN_PASSWORD=change-me" >> .env       # first admin user is created from this
echo "LABEL_STUDIO_API_KEY=..." >> .env       # Label Studio > Account > Access Token
bash scripts/dev.sh                           # API :8000, worker, UI :5173
```

Open http://localhost:5173 and sign in as `admin`.

## Production-style run

```bash
(cd frontend && npm run build)                # FastAPI then serves frontend/dist
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
DEVICE=0 uv run python -m app.worker          # one worker per GPU: DEVICE=0, DEVICE=1, ...
```

Jobs run one at a time per worker, in the order they were queued. Restarting a worker marks
the job it was running as failed.

## Settings (environment or `.env`)

| Variable | Default |
| --- | --- |
| `DATABASE_URL` | `sqlite:///<STORAGE_DIR>/platform.db` |
| `STORAGE_DIR` | `storage` (images, datasets, runs, weights) |
| `SECRET_KEY` | random, persisted in `storage/.secret_key` |
| `ADMIN_USERNAME` / `ADMIN_PASSWORD` | `admin` / unset (no admin bootstrap) |
| `LABEL_STUDIO_URL` | `http://localhost:8080` |
| `LABEL_STUDIO_PORT` | unset; when set, the port of a localhost `LABEL_STUDIO_URL` (e.g. `8081` gives `http://localhost:8081`). Also the port **Start Label Studio** launches on |
| `LABEL_STUDIO_PUBLIC_URL` | same as `LABEL_STUDIO_URL` (used for links shown to users) |
| `LABEL_STUDIO_API_KEY` | required for Label Studio sync and export |
| `LABEL_STUDIO_COMMAND` | how the **Start Label Studio** button launches it; default `label-studio start --port <port of LABEL_STUDIO_URL> --no-browser`. Only offered when `LABEL_STUDIO_URL` is on localhost; output goes to `STORAGE_DIR/label-studio.log` |
| `MAX_PREDICT_IMAGES` | `20000` images per prediction run |
| `DEVICE` | worker GPU, e.g. `0`; empty lets Ultralytics choose |
| `WORKER_NAME` | `<hostname>:<DEVICE>` |
| `WORKER_POLL_SECONDS` | `2` |

The worker keeps its own Ultralytics settings and downloaded weights under `STORAGE_DIR`, so your
global `~/.config/Ultralytics` settings are not changed.

## Command-line tools

```bash
uv run python -m app.services.tif path/to/tifs --out data/png
uv run python -m app.services.export_yolo --project 4 --task segment --out datasets/labelstudio_seg
```

## Development

```bash
uv run pytest
bash scripts/lint.sh                  # ruff + pyright on backend/
(cd frontend && npm run typecheck)
```

API docs: http://localhost:8000/docs
