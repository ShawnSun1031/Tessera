import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { Link, useNavigate, useParams, useSearchParams } from "react-router-dom";
import { api, fileUrl, upload, type Job, type Model } from "../api";
import { useProjectId } from "../components/Layout";
import { PREDICT_CONF_DEFAULT, usePreferences } from "../prefs";
import { accuracySeries, lossSeries, METRIC_LABELS, MetricChart } from "../components/MetricChart";
import { Dropzone, ErrorNotice, JOB_KIND_TEXT, PageHead, ProgressBar, StatusBadge, timeAgo } from "../components/ui";

function headline(m: Model): { key: string; value: number } | null {
  for (const key of ["metrics/mAP50-95(M)", "metrics/mAP50-95(B)"]) {
    if (key in m.metrics) return { key, value: m.metrics[key] };
  }
  return null;
}

function archLabel(arch: string): string {
  if (arch.startsWith("model:")) return `Continued from model #${arch.slice(6)}`;
  return arch.replace(".pt", "").replace(/^yolo(\d+)/, "YOLO$1 ").toUpperCase().replace("-SEG", "");
}

export function ModelsPage() {
  const pid = useProjectId();
  const models = useQuery({ queryKey: ["models", pid], queryFn: () => api.models(pid) });
  const trainJobs = useQuery({ queryKey: ["jobs", pid, "train"], queryFn: () => api.jobs(pid, "train"), refetchInterval: 5000 });
  const active = (trainJobs.data ?? []).filter((j) => j.status === "queued" || j.status === "running");
  const list = models.data ?? [];
  const best = list.reduce<Model | null>((b, m) => ((headline(m)?.value ?? -1) > (b ? (headline(b)?.value ?? -1) : -1) ? m : b), null);

  return (
    <div className="stack">
      <PageHead title="Results">
        <p>Each finished training run gives you a model. Compare them here, then open one to see details or download it.</p>
      </PageHead>
      {active.map((j) => (
        <div className="notice run row spread" key={j.id}>
          <span>
            Training #{j.id} is {j.status === "running" ? "running" : "waiting for the GPU"}.
          </span>
          <Link className="btn small" to={`../jobs/${j.id}`}>
            Watch
          </Link>
        </div>
      ))}
      {models.isSuccess && !list.length && !active.length && (
        <div className="panel">
          <p>No trained models yet.</p>
          <Link className="btn primary" to="../train">
            Train a model
          </Link>
        </div>
      )}
      {list.length > 0 && (
        <div className="panel table-wrap">
          <table className="data">
            <thead>
              <tr>
                <th>Model</th>
                <th>Starting model</th>
                <th className="num">mAP@50–95</th>
                <th className="num">mAP@50</th>
                <th>Trained</th>
              </tr>
            </thead>
            <tbody>
              {list.map((m) => {
                const h = headline(m);
                const m50 = m.metrics["metrics/mAP50(M)"] ?? m.metrics["metrics/mAP50(B)"];
                return (
                  <tr key={m.id}>
                    <td>
                      <Link to={`${m.id}`}>{m.name}</Link>
                      {best?.id === m.id && list.length > 1 && <span className="status succeeded" style={{ marginLeft: 8 }}>Best</span>}
                    </td>
                    <td className="muted">{archLabel(m.architecture)}</td>
                    <td className="num">{h ? h.value.toFixed(3) : "–"}</td>
                    <td className="num">{m50 !== undefined ? m50.toFixed(3) : "–"}</td>
                    <td className="muted">{timeAgo(m.created_at)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

const PLOT_TEXT: Record<string, string> = {
  "results.png": "All training curves",
  "confusion_matrix_normalized.png": "What gets confused with what",
  "MaskPR_curve.png": "Precision vs. recall (masks)",
  "BoxPR_curve.png": "Precision vs. recall (boxes)",
  "val_batch0_pred.png": "Predictions on validation images",
  "val_batch0_labels.png": "Your labels on the same images",
};

export function ModelPage() {
  const pid = useProjectId();
  const { modelId } = useParams();
  const id = Number(modelId);
  const navigate = useNavigate();
  const model = useQuery({ queryKey: ["model", id], queryFn: () => api.model(id) });
  const m = model.data;
  const metrics = useQuery({ queryKey: ["metrics", m?.job_id], queryFn: () => api.metrics(m!.job_id), enabled: !!m });
  const onnx = useMutation({ mutationFn: () => api.exportOnnx(id), onSuccess: (job) => navigate(`../jobs/${job.id}`) });

  if (model.error) return <ErrorNotice error={model.error} />;
  if (!m) return <p className="muted">Loading…</p>;
  const rows = metrics.data ?? [];
  const keys = ["metrics/mAP50-95(M)", "metrics/mAP50(M)", "metrics/mAP50-95(B)", "metrics/mAP50(B)", "metrics/precision(B)", "metrics/recall(B)"].filter(
    (k) => k in m.metrics,
  );
  const featured = Object.keys(PLOT_TEXT).filter((p) => m.plots.includes(p));
  const others = m.plots.filter((p) => !featured.includes(p));

  return (
    <div className="stack">
      <PageHead title={m.name}>
        <p>
          {archLabel(m.architecture)}, trained on dataset #{m.dataset_id} at {m.imgsz} px. <Link to={`../jobs/${m.job_id}`}>Training run #{m.job_id}</Link>
        </p>
      </PageHead>
      <section className="panel stack">
        <div className="stat-row">
          {keys.map((k) => (
            <div className="stat" key={k}>
              <b>{m.metrics[k].toFixed(3)}</b>
              <span>{METRIC_LABELS[k] ?? k}</span>
            </div>
          ))}
        </div>
        <p className="small muted" style={{ marginBottom: 0 }}>
          Measured on the validation images with the best epoch. mAP@50 counts a hit when the outline overlaps the label
          by half or more; mAP@50–95 also rewards tight outlines. Precision: how many detections are right. Recall: how
          many objects were found.
        </p>
        <div className="row">
          <Link className="btn primary" to={`../predict?model=${m.id}`}>
            Run on new images
          </Link>
          <a className="btn" href={fileUrl(`/models/${m.id}/download?format=pt`)}>
            Download .pt
          </a>
          {m.onnx_available ? (
            <a className="btn" href={fileUrl(`/models/${m.id}/download?format=onnx`)}>
              Download .onnx
            </a>
          ) : (
            <button className="btn" onClick={() => onnx.mutate()} disabled={onnx.isPending}>
              Convert to ONNX
            </button>
          )}
        </div>
        <ErrorNotice error={onnx.error} />
        <p className="small muted" style={{ marginBottom: 0 }}>
          .pt works with the Ultralytics Python package. ONNX runs in most other tools (OpenCV, ONNX Runtime, C++, C#).
        </p>
      </section>
      {rows.length > 1 && (
        <div className="grid-2">
          <section className="panel">
            <h3>Accuracy by epoch</h3>
            <MetricChart rows={rows} series={accuracySeries(rows)} domain={[0, 1]} />
          </section>
          <section className="panel">
            <h3>Loss by epoch</h3>
            <MetricChart rows={rows} series={lossSeries(rows)} />
          </section>
        </div>
      )}
      {featured.length > 0 && (
        <section className="stack">
          <h2>Plots</h2>
          <div className="gallery">
            {featured.map((p) => (
              <figure key={p}>
                <a href={fileUrl(`/models/${m.id}/files/${p}`)} target="_blank" rel="noreferrer">
                  <img src={fileUrl(`/models/${m.id}/files/${p}`)} alt={PLOT_TEXT[p]} loading="lazy" />
                </a>
                <figcaption>{PLOT_TEXT[p]}</figcaption>
              </figure>
            ))}
          </div>
        </section>
      )}
      {others.length > 0 && (
        <details className="panel">
          <summary>All {m.plots.length} plots</summary>
          <div className="gallery" style={{ marginTop: "0.75rem" }}>
            {others.map((p) => (
              <figure key={p}>
                <a href={fileUrl(`/models/${m.id}/files/${p}`)} target="_blank" rel="noreferrer">
                  <img src={fileUrl(`/models/${m.id}/files/${p}`)} alt={p} loading="lazy" />
                </a>
                <figcaption>{p}</figcaption>
              </figure>
            ))}
          </div>
        </details>
      )}
      <p className="small">
        <Link to={`/p/${pid}/models`}>Back to all results</Link>
      </p>
    </div>
  );
}

export function PredictPage() {
  const pid = useProjectId();
  const navigate = useNavigate();
  const qc = useQueryClient();
  const [params] = useSearchParams();
  const models = useQuery({ queryKey: ["models", pid], queryFn: () => api.models(pid) });
  const jobs = useQuery({ queryKey: ["jobs", pid, "predict"], queryFn: () => api.jobs(pid, "predict"), refetchInterval: 5000 });
  const [modelId, setModelId] = useState<number>(Number(params.get("model")) || 0);
  const { prefs, save } = usePreferences();
  const [chosenConf, setConf] = useState<number | null>(null);
  const conf = chosenConf ?? prefs.predict_conf ?? PREDICT_CONF_DEFAULT;
  const [files, setFiles] = useState<File[]>([]);
  const [progress, setProgress] = useState<number | null>(null);
  const [error, setError] = useState<unknown>(null);
  const chosen = modelId || models.data?.[0]?.id || 0;

  const run = async () => {
    const form = new FormData();
    files.forEach((f) => form.append("files", f));
    form.append("conf", String(conf));
    save({ predict_conf: conf });
    setError(null);
    setProgress(0);
    try {
      const job = await upload<Job>(`/models/${chosen}/predict`, form, setProgress);
      qc.invalidateQueries({ queryKey: ["queue"] });
      navigate(`../jobs/${job.id}`);
    } catch (e) {
      setError(e);
      setProgress(null);
    }
  };

  if (models.isSuccess && !models.data.length) {
    return (
      <div className="stack">
        <PageHead title="Predict" />
        <div className="panel">
          <p>Train a model first, then come back to run it on new images.</p>
          <Link className="btn primary" to="../train">
            Train a model
          </Link>
        </div>
      </div>
    );
  }

  return (
    <div className="stack">
      <PageHead title="Predict">
        <p>Run a trained model on images it hasn't seen. You get each image with the findings drawn on, plus a file with exact positions.</p>
      </PageHead>
      <section className="panel stack">
        <label className="field">
          Model
          <select value={chosen} onChange={(e) => setModelId(Number(e.target.value))}>
            {models.data?.map((m) => (
              <option key={m.id} value={m.id}>
                {m.name}
                {headline(m) ? ` (mAP ${headline(m)!.value.toFixed(3)})` : ""}
              </option>
            ))}
          </select>
        </label>
        <label className="field">
          Minimum confidence: {Math.round(conf * 100)}%
          <span className="help">Lower finds more objects but adds false alarms. Raise it if you see detections that aren't there.</span>
          <input type="range" min={0.05} max={0.95} step={0.05} value={conf} onChange={(e) => setConf(Number(e.target.value))} />
        </label>
        {conf !== PREDICT_CONF_DEFAULT && (
          <div>
            <button
              type="button"
              className="linkish small"
              onClick={() => {
                setConf(PREDICT_CONF_DEFAULT);
                save({ predict_conf: null });
              }}
            >
              Reset to default ({Math.round(PREDICT_CONF_DEFAULT * 100)}%)
            </button>
          </div>
        )}
        {models.data?.find((m) => m.id === chosen) && (
          <p className="small muted" style={{ margin: 0 }}>
            Images are resized to {models.data.find((m) => m.id === chosen)!.imgsz} px, the size this model was trained at.
          </p>
        )}
        <Dropzone
          accept=".tif,.tiff,.png,.jpg,.jpeg,.bmp,.webp,.zip"
          onFiles={(f) => setFiles((prev) => [...prev, ...f])}
          disabled={progress !== null}
          title={files.length ? `${files.length} file${files.length === 1 ? "" : "s"} chosen. Drop more or click to add` : "Drop images here, or click to choose"}
          hint="TIF, PNG, JPG or a .zip; up to 500 images per run"
        />
        {progress !== null && <ProgressBar value={progress} />}
        <ErrorNotice error={error} />
        <div className="row">
          <button className="btn primary" onClick={run} disabled={!chosen || !files.length || progress !== null}>
            Run predictions
          </button>
          {files.length > 0 && progress === null && (
            <button className="linkish" onClick={() => setFiles([])}>
              Clear files
            </button>
          )}
        </div>
      </section>
      {!!jobs.data?.length && (
        <section className="panel table-wrap">
          <h2>Earlier runs</h2>
          <table className="data">
            <tbody>
              {jobs.data.map((j) => (
                <tr key={j.id}>
                  <td>
                    <Link to={`../jobs/${j.id}`}>
                      {JOB_KIND_TEXT[j.kind]} #{j.id}
                    </Link>
                  </td>
                  <td>
                    <StatusBadge status={j.status} />
                  </td>
                  <td className="num muted">{j.result.images ? `${j.result.images} images, ${j.result.objects} found` : ""}</td>
                  <td className="muted">{timeAgo(j.created_at)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}
    </div>
  );
}
