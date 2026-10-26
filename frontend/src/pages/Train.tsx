import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useMemo, useRef, useState } from "react";
import { Link, useNavigate, useParams, useSearchParams } from "react-router-dom";
import { api, fileUrl, TERMINAL, type Architecture, type Job, type SentToLabelStudio, type TaskType, type TrainRequest } from "../api";
import { TRAIN_DEFAULTS, usePreferences, type Preferences, type PreferencesPatch, type TrainSettings } from "../prefs";
import { useProjectId } from "../components/Layout";
import { accuracySeries, lossSeries, METRIC_LABELS, MetricChart } from "../components/MetricChart";
import { UploadedWeights } from "../components/UploadedWeights";
import { duration, ErrorNotice, JOB_KIND_TEXT, PageHead, ProgressBar, StatusBadge, useJobStream } from "../components/ui";

const AUGMENT_TEXT: Record<TrainRequest["augment"], string> = {
  none: "Off. Only for checking that training works at all.",
  light: "Small shifts and color changes. Good when images all look alike.",
  default: "Ultralytics' standard mix. Right for most projects.",
  strong: "Adds rotation, vertical flips and blending. Helps with few images or objects in any orientation.",
};

export function TrainPage() {
  const pid = useProjectId();
  const { prefs, ready, save } = usePreferences();
  const project = useQuery({ queryKey: ["project", pid], queryFn: () => api.project(pid) });
  const archs = useQuery({ queryKey: ["archs", pid], queryFn: () => api.architectures(pid) });
  if (!ready || !project.data || !archs.data) return <p className="muted">Loading…</p>;
  return <TrainForm prefs={prefs} save={save} taskType={project.data.task_type} architectures={archs.data.architectures} />;
}

function TrainForm({
  prefs,
  save,
  taskType,
  architectures,
}: {
  prefs: Preferences;
  save: (patch: PreferencesPatch) => void;
  taskType: TaskType;
  architectures: Architecture[];
}) {
  const pid = useProjectId();
  const navigate = useNavigate();
  const qc = useQueryClient();
  const [params] = useSearchParams();
  const datasets = useQuery({ queryKey: ["datasets", pid], queryFn: () => api.datasets(pid) });
  const models = useQuery({ queryKey: ["models", pid], queryFn: () => api.models(pid) });

  const catalog = architectures.map((a) => a.weights);
  const savedBase = prefs.train?.base_model?.[taskType];
  const defaultBase = savedBase && catalog.includes(savedBase) ? savedBase : catalog[0];
  const saved = settingsFrom(prefs.train);
  const usingSaved = !!prefs.train;

  const [form, setForm] = useState<TrainRequest>({
    dataset_id: Number(params.get("dataset")) || 0,
    base_model: defaultBase,
    name: "",
    ...saved,
  });
  const set = <K extends keyof TrainRequest>(key: K, value: TrainRequest[K]) => setForm((f) => ({ ...f, [key]: value }));
  const isDefault =
    form.base_model === catalog[0] && (Object.keys(TRAIN_DEFAULTS) as (keyof TrainSettings)[]).every((k) => form[k] === TRAIN_DEFAULTS[k]);

  useEffect(() => {
    if (!form.dataset_id && datasets.data?.length) set("dataset_id", datasets.data[0].id);
  }, [datasets.data, form.dataset_id]);

  const start = useMutation({
    mutationFn: () => api.train(pid, { ...form, name: form.name || undefined }),
    onSuccess: (job) => {
      // remember these settings for next time; own models and uploads are project-specific, so not saved
      const base_model = !catalog.includes(form.base_model)
        ? prefs.train?.base_model
        : { ...prefs.train?.base_model, [taskType]: form.base_model };
      const { epochs, imgsz, batch, lr0, patience, augment } = form;
      save({ train: { epochs, imgsz, batch, lr0, patience, augment, base_model } });
      qc.invalidateQueries({ queryKey: ["queue"] });
      navigate(`../jobs/${job.id}`);
    },
  });

  if (datasets.isSuccess && !datasets.data.length) {
    return (
      <div className="stack">
        <PageHead title="Train" />
        <div className="panel">
          <p>You need a dataset before you can train.</p>
          <Link className="btn primary" to="../dataset">
            Build a dataset
          </Link>
        </div>
      </div>
    );
  }

  return (
    <form
      className="stack"
      onSubmit={(e) => {
        e.preventDefault();
        start.mutate();
      }}
      // open a collapsed section holding an invalid field so the browser can show why
      onInvalidCapture={(e) => (e.target as HTMLElement).closest("details")?.setAttribute("open", "")}
    >
      <PageHead title="Train a model">
        <p>
          {usingSaved
            ? "Settings start from your last training run."
            : "The defaults work well for a first run."}{" "}
          Training happens on the shared GPU; if someone else is training, yours waits its turn.
        </p>
      </PageHead>

      <section className="panel stack">
        <label className="field">
          Dataset
          <span className="help">
            Not sure the labels are right? <Link to={`../datasets/${form.dataset_id}`}>Preview this dataset</Link>.
          </span>
          <select value={form.dataset_id} onChange={(e) => set("dataset_id", Number(e.target.value))}>
            {datasets.data?.map((d) => (
              <option key={d.id} value={d.id}>
                {d.name} ({d.stats.splits.train.images} train / {d.stats.splits.val.images} val images)
              </option>
            ))}
          </select>
        </label>
      </section>

      <section className="panel stack">
        <div>
          <h2>Starting model</h2>
          <p className="muted small">
            Every option starts from a model already trained on everyday photos, so it learns your objects much faster.
            Bigger models are more accurate but slower to train and run.
          </p>
        </div>
        <div className="choice-grid" role="radiogroup">
          {architectures.map((a) => (
            <label className="choice" key={a.weights}>
              <input type="radio" name="base" checked={form.base_model === a.weights} onChange={() => set("base_model", a.weights)} />
              <span>
                <strong>
                  {a.family} {a.size.toUpperCase()}
                </strong>
                <span>{a.note}</span>
              </span>
            </label>
          ))}
        </div>
        {!!models.data?.length && (
          <>
            <h3>Or keep training one of your models</h3>
            <div className="choice-grid">
              {models.data.map((m) => (
                <label className="choice" key={m.id}>
                  <input type="radio" name="base" checked={form.base_model === `model:${m.id}`} onChange={() => set("base_model", `model:${m.id}`)} />
                  <span>
                    <strong>{m.name}</strong>
                    <span>Useful after labeling more images.</span>
                  </span>
                </label>
              ))}
            </div>
          </>
        )}
        <UploadedWeights
          title="Or start from your own weights"
          hint={`An Ultralytics YOLO ${taskType === "segment" ? "segmentation" : "detection"} model trained elsewhere. Its classes may differ from this project's.`}
          selected={form.base_model}
          onSelect={(v) => set("base_model", v)}
          fallback={defaultBase}
        />
      </section>

      <section className="panel stack">
        <div className="row spread">
          <h2 style={{ margin: 0 }}>Settings</h2>
          <button
            type="button"
            className="btn small"
            disabled={isDefault && !usingSaved}
            title="Use the standard settings and forget your saved ones"
            onClick={() => {
              setForm((f) => ({ ...f, ...TRAIN_DEFAULTS, base_model: catalog[0] }));
              save({ train: null });
            }}
          >
            Reset to defaults
          </button>
        </div>
        <div className="grid-3">
          <label className="field">
            Epochs
            <span className="help">Passes over your images. Stops early if it stops improving.</span>
            <input type="number" min={1} max={2000} value={form.epochs} onChange={(e) => set("epochs", Number(e.target.value))} />
          </label>
          <ImageSizeField value={form.imgsz} onChange={(v) => set("imgsz", v)} />
          <label className="field">
            Batch size
            <span className="help">Images per step. Auto picks the most that fits on the GPU.</span>
            <select value={form.batch} onChange={(e) => set("batch", Number(e.target.value))}>
              <option value={-1}>Auto</option>
              {[2, 4, 8, 16, 32, 64].map((b) => (
                <option key={b} value={b}>
                  {b}
                </option>
              ))}
            </select>
          </label>
        </div>
        <label className="field">
          Name
          <span className="help">Optional. Helps you tell runs apart later.</span>
          <input type="text" value={form.name} placeholder="e.g. More hole examples" onChange={(e) => set("name", e.target.value)} />
        </label>
        <details>
          <summary>Advanced</summary>
          <div className="stack" style={{ marginTop: "0.75rem" }}>
            <div className="grid-2">
              <label className="field">
                Learning rate
                <span className="help">How big each correction step is. Keep 0.01 unless training is unstable.</span>
                <input type="number" step="any" min={0.0001} max={1} value={form.lr0} onChange={(e) => set("lr0", Number(e.target.value))} />
              </label>
              <label className="field">
                Patience
                <span className="help">Stop after this many epochs without improvement. 0 never stops early.</span>
                <input type="number" min={0} max={2000} value={form.patience} onChange={(e) => set("patience", Number(e.target.value))} />
              </label>
            </div>
            <fieldset style={{ border: 0, padding: 0, margin: 0 }}>
              <legend style={{ fontWeight: 700, marginBottom: "0.3rem" }}>Augmentation</legend>
              <p className="small muted">Random changes to images during training so the model doesn't memorize them.</p>
              <div className="choice-list">
                {(Object.keys(AUGMENT_TEXT) as TrainRequest["augment"][]).map((k) => (
                  <label className="choice" key={k}>
                    <input type="radio" name="augment" checked={form.augment === k} onChange={() => set("augment", k)} />
                    <span>
                      <strong>{k[0].toUpperCase() + k.slice(1)}</strong>
                      <span>{AUGMENT_TEXT[k]}</span>
                    </span>
                  </label>
                ))}
              </div>
            </fieldset>
          </div>
        </details>
      </section>
      <ErrorNotice error={start.error} />
      <div>
        <button className="btn primary" disabled={!form.dataset_id || !form.base_model || start.isPending}>
          Start training
        </button>
      </div>
    </form>
  );
}

/** Saved settings, ignoring anything a newer version no longer accepts. */
function settingsFrom(saved: Preferences["train"]): TrainSettings {
  const out = { ...TRAIN_DEFAULTS };
  if (!saved) return out;
  const num = (v: unknown, lo: number, hi: number) => typeof v === "number" && v >= lo && v <= hi;
  if (num(saved.epochs, 1, 2000)) out.epochs = saved.epochs!;
  if (num(saved.imgsz, 64, 2048) && saved.imgsz! % 32 === 0) out.imgsz = saved.imgsz!;
  if (num(saved.batch, -1, 512) && saved.batch !== 0) out.batch = saved.batch!;
  if (num(saved.lr0, 1e-6, 1)) out.lr0 = saved.lr0!;
  if (num(saved.patience, 0, 2000)) out.patience = saved.patience!;
  if (saved.augment && saved.augment in AUGMENT_TEXT) out.augment = saved.augment;
  return out;
}

const IMGSZ_MIN = 64;
const IMGSZ_MAX = 2048;
const IMGSZ_SUGGESTIONS = [320, 480, 640, 800, 960, 1024, 1280, 1536];

/** YOLO needs a multiple of 32; round and clamp whatever the user types. */
function normalizeImgsz(raw: string): number {
  const n = Number(raw);
  if (!Number.isFinite(n) || n <= 0) return 640;
  return Math.min(IMGSZ_MAX, Math.max(IMGSZ_MIN, Math.round(n / 32) * 32));
}

function ImageSizeField({ value, onChange }: { value: number; onChange: (v: number) => void }) {
  const [draft, setDraft] = useState(String(value));
  const input = useRef<HTMLInputElement>(null);
  useEffect(() => {
    if (document.activeElement !== input.current) setDraft(String(value));
  }, [value]);
  const adjusted = draft.trim() !== "" && Number(draft) !== value;
  return (
    <label className="field">
      Image size (px)
      <span className="help">Type any size or pick one. Larger finds smaller objects but needs more memory.</span>
      <input
        ref={input}
        type="number"
        inputMode="numeric"
        list="imgsz-suggestions"
        min={IMGSZ_MIN}
        max={IMGSZ_MAX}
        step="any"
        value={draft}
        onChange={(e) => {
          setDraft(e.target.value);
          onChange(normalizeImgsz(e.target.value));
        }}
        onBlur={() => setDraft(String(value))}
      />
      <datalist id="imgsz-suggestions">
        {IMGSZ_SUGGESTIONS.map((s) => (
          <option key={s} value={s} />
        ))}
      </datalist>
      {adjusted && (
        <span className="help" role="status">
          Will train at {value} px: sizes are rounded to a multiple of 32 between {IMGSZ_MIN} and {IMGSZ_MAX}.
        </span>
      )}
    </label>
  );
}

function trainFraction(job: Job): number {
  const { epoch = 0, epochs = 0, batch = 0, batches = 0 } = job.progress;
  if (job.status === "succeeded") return 1;
  if (!epochs) return 0;
  return Math.min(1, (epoch - 1 + (batches ? batch / batches : 0)) / epochs);
}

function eta(job: Job, fraction: number): string | null {
  if (!job.started_at || fraction < 0.02 || job.status !== "running") return null;
  const start = new Date(job.started_at.endsWith("Z") || job.started_at.includes("+") ? job.started_at : `${job.started_at}Z`).getTime();
  const elapsed = (Date.now() - start) / 1000;
  const remaining = (elapsed / fraction) * (1 - fraction);
  if (remaining < 60) return "under a minute left";
  if (remaining < 3600) return `about ${Math.round(remaining / 60)} min left`;
  return `about ${(remaining / 3600).toFixed(1)} h left`;
}

/** Inference runs in stages: convert the uploaded files, predict, zip the results. */
function predictStage(job: Job): { text: string; fraction: number; eta: boolean } {
  const p = job.progress;
  if (p.packaging) return { text: "Packing results for download…", fraction: 1, eta: false };
  if (p.files) return { text: `Preparing images: ${p.prepared ?? 0} of ${p.files} files`, fraction: (p.prepared ?? 0) / p.files, eta: false };
  if (p.total) return { text: `${p.done ?? 0} of ${p.total} images`, fraction: (p.done ?? 0) / p.total, eta: true };
  return { text: "Starting…", fraction: 0, eta: false };
}

export function JobPage() {
  const { jobId } = useParams();
  const id = Number(jobId);
  const qc = useQueryClient();
  const { job, metrics, log, connected } = useJobStream(id);
  const cancel = useMutation({ mutationFn: () => api.cancelJob(id) });
  const logRef = useRef<HTMLPreElement>(null);

  useEffect(() => {
    const el = logRef.current;
    if (el && el.scrollHeight - el.scrollTop - el.clientHeight < 80) el.scrollTop = el.scrollHeight;
  }, [log]);
  useEffect(() => {
    if (job && TERMINAL.includes(job.status)) {
      qc.invalidateQueries({ queryKey: ["models", job.project_id] });
      qc.invalidateQueries({ queryKey: ["datasets", job.project_id] });
      qc.invalidateQueries({ queryKey: ["jobs", job.project_id] });
      qc.invalidateQueries({ queryKey: ["queue"] });
    }
  }, [job?.status]);

  const latest = metrics[metrics.length - 1];
  const acc = useMemo(() => accuracySeries(metrics), [metrics]);
  const loss = useMemo(() => lossSeries(metrics), [metrics]);

  if (!job) return <p className="muted">{connected ? "Loading…" : "Connecting…"}</p>;
  const active = !TERMINAL.includes(job.status);
  const stage = job.kind === "predict" ? predictStage(job) : null;
  const fraction = job.kind === "train" ? trainFraction(job) : stage ? stage.fraction : 0;

  return (
    <div className="stack">
      <header className="page-head row spread">
        <div>
          <h1>
            {JOB_KIND_TEXT[job.kind]} #{job.id}
          </h1>
          <div className="row small muted">
            <StatusBadge status={job.status} />
            <span>by {job.username}</span>
            {job.started_at && <span className="num">{duration(job.started_at, job.finished_at)}</span>}
          </div>
        </div>
        {active && (
          <button className="btn danger" onClick={() => cancel.mutate()} disabled={cancel.isPending || job.cancel_requested}>
            {job.cancel_requested ? "Stopping…" : job.status === "running" ? "Stop" : "Cancel"}
          </button>
        )}
      </header>
      <ErrorNotice error={cancel.error} />

      {job.status === "queued" && (
        <div className="notice run">
          {job.queue_position === 0 ? "Starts as soon as the worker picks it up." : `Waiting for ${job.queue_position} job(s) ahead of it on the GPU.`}
        </div>
      )}
      {job.cancel_requested && job.status === "running" && job.kind === "train" && (
        <div className="notice run">Stopping after the current step. The best model so far is kept.</div>
      )}
      {job.status === "failed" && (
        <div className="notice error">
          <strong>This run failed.</strong> {job.error}
          <div className="small">The log below has the details.</div>
        </div>
      )}

      {job.status === "running" && (job.kind === "train" || job.kind === "predict") && (
        <section className="panel stack" style={{ ["--step" as string]: "0.6rem" }}>
          <div className="row spread small">
            <span className="num">
              {job.kind === "train"
                ? job.progress.epochs
                  ? `Epoch ${job.progress.epoch} of ${job.progress.epochs}`
                  : "Preparing data and model…"
                : stage?.text}
            </span>
            <span className="muted">{!stage || stage.eta ? eta(job, fraction) : null}</span>
          </div>
          <ProgressBar value={fraction} running />
        </section>
      )}

      {job.kind === "train" && metrics.length > 0 && (
        <>
          {latest && (
            <section className="panel">
              <div className="stat-row">
                {acc.slice(0, 2).map((s) => (
                  <div className="stat" key={s.key}>
                    <b>{(latest.values[s.key] ?? 0).toFixed(3)}</b>
                    <span>{s.label}</span>
                  </div>
                ))}
                {["metrics/precision(B)", "metrics/recall(B)"].map((k) => (
                  <div className="stat" key={k}>
                    <b>{(latest.values[k] ?? 0).toFixed(3)}</b>
                    <span>{METRIC_LABELS[k]}</span>
                  </div>
                ))}
              </div>
              <p className="small muted" style={{ marginTop: "0.75rem", marginBottom: 0 }}>
                mAP is the main accuracy score, from 0 to 1. Higher is better; it usually climbs fast at first, then levels off.
              </p>
            </section>
          )}
          <div className="grid-2">
            <section className="panel">
              <h3>Accuracy</h3>
              <MetricChart rows={metrics} series={acc} domain={[0, 1]} />
            </section>
            <section className="panel">
              <h3>Loss</h3>
              <MetricChart rows={metrics} series={loss} />
            </section>
          </div>
        </>
      )}

      {job.status === "succeeded" && <JobResult job={job} />}

      {job.kind === "train" && (
        <details className="panel">
          <summary>Settings used</summary>
          <table className="data" style={{ marginTop: "0.5rem" }}>
            <tbody>
              {Object.entries(job.params).map(([k, v]) => (
                <tr key={k}>
                  <td className="muted">{k}</td>
                  <td>{String(v)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </details>
      )}

      <details className="panel" open={job.status === "failed" || (active && job.kind !== "train")}>
        <summary>Log</summary>
        <pre className="log" ref={logRef} style={{ marginTop: "0.6rem" }}>
          {log || "No output yet."}
        </pre>
      </details>
    </div>
  );
}

function JobResult({ job }: { job: Job }) {
  if (job.kind === "train") {
    return (
      <div className="notice ok row spread">
        <span>Training finished. The model is saved with its best epoch.</span>
        <Link className="btn primary small" to={`../models/${job.result.model_id}`}>
          See results
        </Link>
      </div>
    );
  }
  if (job.kind === "export") {
    return (
      <div className="notice ok row spread">
        <span>
          Dataset built: {job.result.stats?.splits?.train?.images} training and {job.result.stats?.splits?.val?.images} validation images.
        </span>
        <div className="row">
          <Link className="btn small" to={`../datasets/${job.result.dataset_id}`}>
            Preview labels
          </Link>
          <Link className="btn primary small" to={`../train?dataset=${job.result.dataset_id}`}>
            Train on it
          </Link>
        </div>
      </div>
    );
  }
  if (job.kind === "onnx") {
    return (
      <div className="notice ok row spread">
        <span>ONNX file ready.</span>
        <a className="btn primary small" href={fileUrl(`/models/${job.result.model_id}/download?format=onnx`)}>
          Download .onnx
        </a>
      </div>
    );
  }
  return <PredictionResult job={job} />;
}

const GALLERY_PAGE = 60;

export function PredictionResult({ job }: { job: Job }) {
  const preds = useQuery({ queryKey: ["predictions", job.id], queryFn: () => api.predictions(job.id) });
  const perClass: Record<string, number> = job.result.per_class ?? {};
  const [page, setPage] = useState(0);
  const all = preds.data ?? [];
  const pages = Math.max(1, Math.ceil(all.length / GALLERY_PAGE));
  const shown = all.slice(page * GALLERY_PAGE, (page + 1) * GALLERY_PAGE);
  const pager = pages > 1 && (
    <div className="row small">
      <button className="btn small" onClick={() => setPage(page - 1)} disabled={page === 0}>
        Previous
      </button>
      <span className="num">
        Page {page + 1} of {pages} ({all.length.toLocaleString()} images)
      </span>
      <button className="btn small" onClick={() => setPage(page + 1)} disabled={page >= pages - 1}>
        Next
      </button>
    </div>
  );
  return (
    <section className="stack">
      <div className="panel row spread">
        <div className="stat-row">
          <div className="stat">
            <b>{job.result.images}</b>
            <span>images</span>
          </div>
          {Object.entries(perClass).map(([name, n]) => (
            <div className="stat" key={name}>
              <b>{n}</b>
              <span>{name}</span>
            </div>
          ))}
        </div>
        <div className="row">
          <a className="btn" href={fileUrl(`/jobs/${job.id}/files/output/predictions.json`)} download={`predictions_job${job.id}.json`}>
            Coordinates only (.json)
          </a>
          <a className="btn" href={fileUrl(`/jobs/${job.id}/download?format=yolo`)}>
            YOLO dataset (.zip)
          </a>
          <a className="btn primary" href={fileUrl(`/jobs/${job.id}/download`)}>
            Download all (.zip)
          </a>
        </div>
      </div>
      <p className="small muted">
        <strong>Download all</strong> has every image with its detections drawn on, plus predictions.json with exact
        coordinates and confidence. <strong>YOLO dataset</strong> has the original images in images/ and the predictions
        as YOLO labels in labels/, with a data.yaml, ready for other YOLO tools. For thousands of images the zips are
        large; download the .json alone if you only need the coordinates.
      </p>
      {job.result.skipped_files > 0 && (
        <div className="notice error">
          {job.result.skipped_files} uploaded file(s) could not be read as images and were left out. The log lists them.
        </div>
      )}
      <SendToLabelStudio job={job} />
      {pager}
      <div className="gallery">
        {shown.map((p) => {
          const stem = p.image.replace(/\.[^.]+$/, "");
          return (
            <figure key={p.image}>
              <img src={fileUrl(`/jobs/${job.id}/files/output/images/${stem}.png`)} alt={`Predictions on ${p.image}`} loading="lazy" />
              <figcaption>
                {p.image}: {p.objects.length} found
              </figcaption>
            </figure>
          );
        })}
      </div>
      {pager}
    </section>
  );
}

function SendToLabelStudio({ job }: { job: Job }) {
  const qc = useQueryClient();
  const [sent, setSent] = useState<SentToLabelStudio | null>(job.result.label_studio ?? null);
  const send = useMutation({
    mutationFn: () => api.sendPredictions(job.id),
    onSuccess: (data) => {
      setSent(data);
      for (const key of ["ls", "images", "project"]) qc.invalidateQueries({ queryKey: [key, job.project_id] });
    },
  });

  if (sent) {
    return (
      <section className="panel stack" style={{ ["--step" as string]: "0.6rem" }}>
        <div className="row spread">
          <h3 style={{ margin: 0 }}>In Label Studio for correction</h3>
          <a className="btn primary" href={sent.ls_url} target="_blank" rel="noreferrer">
            Open Label Studio
          </a>
        </div>
        <p className="small" style={{ margin: 0 }}>
          Sent {sent.images} image{sent.images === 1 ? "" : "s"} with {sent.objects} predicted object
          {sent.objects === 1 ? "" : "s"} as pre-labels.
          {sent.skipped_objects > 0 &&
            ` ${sent.skipped_objects} predicted object(s) were left out because Label Studio has no matching label.`}
        </p>
        <p className="small muted" style={{ margin: 0 }}>
          Fix the outlines in Label Studio and press Submit on each image. Only submitted images go into a dataset, then{" "}
          <Link to="../dataset">build a new dataset</Link> and train again.
        </p>
      </section>
    );
  }
  return (
    <section className="panel stack" style={{ ["--step" as string]: "0.6rem" }}>
      <div className="row spread">
        <div>
          <h3 style={{ margin: 0 }}>Correct these predictions</h3>
          <p className="small muted" style={{ margin: 0 }}>
            Send the images to Label Studio with the model's findings already drawn in. Fixing them is much faster than
            labeling from scratch, and the corrected images make the next model better.
          </p>
        </div>
        <button className="btn primary" onClick={() => send.mutate()} disabled={send.isPending}>
          {send.isPending ? "Sending…" : "Send to Label Studio"}
        </button>
      </div>
      <ErrorNotice error={send.error} />
    </section>
  );
}
