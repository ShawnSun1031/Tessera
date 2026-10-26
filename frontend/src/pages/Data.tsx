import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { api, fileUrl, TERMINAL, upload, type Dataset, type ImageRow, type LabelStudioProject, type Project } from "../api";
import { LabelStudioPicker } from "../components/LabelStudioPicker";
import { usePreferences, VAL_PCT_DEFAULT } from "../prefs";
import { useProjectId } from "../components/Layout";
import { Dropzone, ErrorNotice, PageHead, ProgressBar, StatusBadge } from "../components/ui";
import { LabelStudioServerNotice, useLabelStudioServer } from "../components/LabelStudioServer";

const IMAGE_ACCEPT = ".tif,.tiff,.png,.jpg,.jpeg,.bmp,.webp,.zip";
const THUMB_LIMIT = 48;

export function UploadPage() {
  const pid = useProjectId();
  const qc = useQueryClient();
  const navigate = useNavigate();
  const images = useQuery({ queryKey: ["images", pid], queryFn: () => api.images(pid) });
  const [progress, setProgress] = useState<number | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [added, setAdded] = useState<number | null>(null);

  const send = useMutation({
    mutationFn: () => api.lsSync(pid),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["ls", pid] });
      qc.invalidateQueries({ queryKey: ["images", pid] });
      qc.invalidateQueries({ queryKey: ["project", pid] });
      navigate("../annotate");
    },
  });

  const onFiles = async (files: File[]) => {
    const form = new FormData();
    files.forEach((f) => form.append("files", f));
    setError(null);
    setAdded(null);
    setProgress(0);
    try {
      const created = await upload<ImageRow[]>(`/projects/${pid}/uploads`, form, setProgress);
      setAdded(created.length);
      qc.invalidateQueries({ queryKey: ["images", pid] });
    } catch (e) {
      setError(e);
    } finally {
      setProgress(null);
    }
  };

  const list = images.data ?? [];
  const pending = list.filter((i) => !i.in_label_studio).length;

  return (
    <div className="stack">
      <PageHead title="Upload images">
        <p>
          Add the images you want to label. TIFs are converted for you: 16-bit and multi-page files become ordinary
          8-bit PNGs, one per page. You can also drop a .zip of images.
        </p>
      </PageHead>
      <Dropzone
        accept={IMAGE_ACCEPT}
        disabled={progress !== null}
        onFiles={onFiles}
        title={progress === null ? "Drop images here, or click to choose" : "Uploading…"}
        hint="TIF, PNG, JPG, BMP, WebP or a .zip of them"
      />
      {progress !== null && (
        <div className="stack" style={{ ["--step" as string]: "0.4rem" }}>
          <ProgressBar value={progress} />
          <span className="small muted">{progress < 1 ? `${Math.round(progress * 100)}% uploaded` : "Converting on the server…"}</span>
        </div>
      )}
      <ErrorNotice error={error} />
      {added !== null && <div className="notice ok">Added {added} image{added === 1 ? "" : "s"}.</div>}

      {list.length > 0 && (
        <section className="panel stack">
          <div className="row spread">
            <h2>
              {list.length} image{list.length === 1 ? "" : "s"}
            </h2>
            {pending > 0 ? (
              <button className="btn primary" onClick={() => send.mutate()} disabled={send.isPending}>
                {send.isPending ? "Sending…" : `Send ${pending} to Label Studio`}
              </button>
            ) : (
              <Link className="btn" to="../annotate">
                Continue to annotation
              </Link>
            )}
          </div>
          <ErrorNotice error={send.error} />
          <div className="thumbs">
            {list.slice(0, THUMB_LIMIT).map((img) => (
              <figure className="thumb" key={img.id}>
                <img src={fileUrl(`/projects/${pid}/images/${img.id}/file`)} alt="" loading="lazy" />
                <figcaption title={img.source_name}>{img.filename}</figcaption>
              </figure>
            ))}
          </div>
          {list.length > THUMB_LIMIT && <p className="small muted">and {list.length - THUMB_LIMIT} more</p>}
        </section>
      )}
    </div>
  );
}

export function AnnotatePage() {
  const pid = useProjectId();
  const qc = useQueryClient();
  const project = useQuery({ queryKey: ["project", pid], queryFn: () => api.project(pid) });
  const status = useQuery({ queryKey: ["ls", pid], queryFn: () => api.lsStatus(pid), refetchInterval: 15000, retry: false });
  const send = useMutation({
    mutationFn: () => api.lsSync(pid),
    onSuccess: (data) => {
      qc.setQueryData(["ls", pid], data);
      qc.invalidateQueries({ queryKey: ["images", pid] });
      qc.invalidateQueries({ queryKey: ["project", pid] });
    },
  });
  const s = status.data;
  const segment = project.data?.task_type === "segment";
  const lsDown = useLabelStudioServer().data?.reachable === false;

  return (
    <div className="stack">
      <PageHead title="Annotate">
        <p>Labeling happens in Label Studio. Open it in a new tab, label your images, then come back here.</p>
      </PageHead>
      <LabelStudioServerNotice />
      {!lsDown && <ErrorNotice error={status.error} />}
      {s && !s.ls_project_id && (
        <div className="panel stack">
          <p>
            {s.images
              ? `${s.images} images are ready. Send them to Label Studio to start labeling.`
              : "Upload some images first."}
          </p>
          <div className="row">
            {s.images ? (
              <button className="btn primary" onClick={() => send.mutate()} disabled={send.isPending}>
                {send.isPending ? "Sending…" : "Send to Label Studio"}
              </button>
            ) : (
              <Link className="btn primary" to="../upload">
                Upload images
              </Link>
            )}
          </div>
          <ErrorNotice error={send.error} />
          {project.data && <LinkExisting project={project.data} />}
        </div>
      )}
      {s?.ls_project_id && (
        <section className="panel stack">
          <div className="stat-row">
            <div className="stat">
              <b>{s.annotated}</b>
              <span>labeled</span>
            </div>
            <div className="stat">
              <b>{s.tasks}</b>
              <span>in Label Studio</span>
            </div>
            {s.pending_upload > 0 && (
              <div className="stat">
                <b>{s.pending_upload}</b>
                <span>not sent yet</span>
              </div>
            )}
          </div>
          <ProgressBar value={s.tasks ? s.annotated / s.tasks : 0} />
          <div className="row">
            {s.ls_url && (
              <a className="btn primary" href={s.ls_url} target="_blank" rel="noreferrer">
                Open Label Studio
              </a>
            )}
            {s.pending_upload > 0 && (
              <button className="btn" onClick={() => send.mutate()} disabled={send.isPending}>
                Send {s.pending_upload} new image{s.pending_upload === 1 ? "" : "s"}
              </button>
            )}
            <button className="btn" onClick={() => status.refetch()} disabled={status.isFetching}>
              Refresh count
            </button>
            {s.annotated > 0 && (
              <Link className="btn" to="../dataset">
                Build a dataset
              </Link>
            )}
          </div>
          <ErrorNotice error={send.error} />
        </section>
      )}
      <section className="panel sunk stack">
        <h3>Labeling tips</h3>
        <ul style={{ margin: 0, paddingLeft: "1.2rem" }}>
          {segment ? (
            <>
              <li>Use the brush to paint over each object, or the polygon tool to click around its outline.</li>
              <li>Paint each separate object as its own region; touching objects can share one region but count as one.</li>
            </>
          ) : (
            <li>Draw one tight rectangle per object. Leave no gap between the box and the object's edges.</li>
          )}
          <li>Pick the label first, then draw. Press Submit to save each image.</li>
          <li>Images you skip are left out of training. Label at least 20–50 images before your first run.</li>
        </ul>
      </section>
    </div>
  );
}

function LinkExisting({ project }: { project: Project }) {
  const qc = useQueryClient();
  const [lsId, setLsId] = useState<number | null>(null);
  const link = useMutation({
    mutationFn: () => api.linkLabelStudio(project.id, lsId!),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["ls", project.id] });
      qc.invalidateQueries({ queryKey: ["project", project.id] });
    },
  });
  // mirrors the server's rules so incompatible projects are explained up front
  const blocked = (p: LabelStudioProject) => {
    if (project.task_type === "segment" && p.task_type === "detect") return "Only has boxes; this project needs outlines.";
    const unknown = p.classes.filter((c) => !project.classes.includes(c));
    return unknown.length ? `Has labels this project doesn't: ${unknown.join(", ")}.` : null;
  };
  return (
    <details>
      <summary>Already labeled these images in Label Studio?</summary>
      <div className="stack" style={{ marginTop: "0.75rem" }}>
        <p className="small muted">
          Use that Label Studio project instead of creating a new one. Its labels must use this project's object types
          ({project.classes.join(", ")}).
        </p>
        <LabelStudioPicker value={lsId} onChange={setLsId} blocked={blocked} serverNotice={false} />
        <ErrorNotice error={link.error} />
        <div>
          <button className="btn primary" onClick={() => link.mutate()} disabled={lsId === null || link.isPending}>
            Use this Label Studio project
          </button>
        </div>
      </div>
    </details>
  );
}

function DatasetCard({ dataset }: { dataset: Dataset }) {
  const { splits, instances } = dataset.stats;
  const max = Math.max(1, ...Object.values(instances));
  const unlabeled = splits.train.unlabeled + splits.val.unlabeled;
  return (
    <section className="panel stack" style={{ ["--step" as string]: "0.9rem" }}>
      <div className="row spread">
        <div>
          <h3 style={{ marginBottom: 0 }}>{dataset.name}</h3>
          <span className="small muted">{dataset.source === "labelstudio" ? "From Label Studio" : "Uploaded zip"}</span>
        </div>
        <div className="row">
          <a className="btn small" href={fileUrl(`/datasets/${dataset.id}/download`)} download>
            Download .zip
          </a>
          <Link className="btn small" to={`../datasets/${dataset.id}`}>
            Preview labels
          </Link>
          <Link className="btn primary small" to={`../train?dataset=${dataset.id}`}>
            Train on this
          </Link>
        </div>
      </div>
      <div className="stat-row">
        <div className="stat">
          <b>{splits.train.images}</b>
          <span>training images</span>
        </div>
        <div className="stat">
          <b>{splits.val.images}</b>
          <span>validation images</span>
        </div>
      </div>
      <div className="class-bars">
        {Object.entries(instances).map(([name, n]) => (
          <div className="class-bar" key={name}>
            <span>{name}</span>
            <ProgressBar value={n / max} />
            <span className="num">{n}</span>
          </div>
        ))}
      </div>
      {Object.values(instances).some((n) => n === 0) && (
        <div className="notice run small">Some object types have no examples. The model can't learn to find them.</div>
      )}
      {unlabeled > 0 && <p className="small muted">{unlabeled} images have no objects; they teach the model what background looks like.</p>}
    </section>
  );
}

export function DatasetPage() {
  const pid = useProjectId();
  const qc = useQueryClient();
  const project = useQuery({ queryKey: ["project", pid], queryFn: () => api.project(pid) });
  const datasets = useQuery({ queryKey: ["datasets", pid], queryFn: () => api.datasets(pid) });
  const { prefs, save } = usePreferences();
  const [chosenPct, setValPct] = useState<number | null>(null);
  const valPct = chosenPct ?? prefs.dataset_val_pct ?? VAL_PCT_DEFAULT;
  const [exportJobId, setExportJobId] = useState<number | null>(null);
  const exportJob = useQuery({
    queryKey: ["job", exportJobId],
    queryFn: () => api.job(exportJobId!),
    enabled: exportJobId !== null,
    refetchInterval: (q) => (q.state.data && TERMINAL.includes(q.state.data.status) ? false : 1500),
  });
  const startExport = useMutation({
    mutationFn: () => {
      save({ dataset_val_pct: valPct });
      return api.exportDataset(pid, { val_ratio: valPct / 100 });
    },
    onSuccess: (job) => setExportJobId(job.id),
  });
  const [zipProgress, setZipProgress] = useState<number | null>(null);
  const [zipError, setZipError] = useState<unknown>(null);

  const job = exportJob.data;
  useEffect(() => {
    if (job?.status === "succeeded") void qc.invalidateQueries({ queryKey: ["datasets", pid] });
  }, [job?.status, pid, qc]);

  const uploadZip = async (files: File[]) => {
    const form = new FormData();
    form.append("file", files[0]);
    form.append("val_ratio", String(valPct / 100));
    save({ dataset_val_pct: valPct });
    setZipError(null);
    setZipProgress(0);
    try {
      await upload(`/projects/${pid}/datasets/upload`, form, setZipProgress);
      qc.invalidateQueries({ queryKey: ["datasets", pid] });
    } catch (e) {
      setZipError(e);
    } finally {
      setZipProgress(null);
    }
  };

  const busy = job && !TERMINAL.includes(job.status);
  return (
    <div className="stack">
      <PageHead title="Build dataset">
        <p>
          A dataset is a frozen copy of your labels, split into images the model learns from and images held back to
          check how well it does. Build a new one whenever you've labeled more.
        </p>
      </PageHead>
      <LabelStudioServerNotice />
      <section className="panel stack">
        <label className="field">
          Hold back {valPct}% of images for validation
          <span className="help">The model never trains on these, so they show how it does on images it hasn't seen. 15–25% is typical.</span>
          <input type="range" min={5} max={50} step={5} value={valPct} onChange={(e) => setValPct(Number(e.target.value))} />
        </label>
        {valPct !== VAL_PCT_DEFAULT && (
          <div>
            <button
              type="button"
              className="linkish small"
              onClick={() => {
                setValPct(VAL_PCT_DEFAULT);
                save({ dataset_val_pct: null });
              }}
            >
              Reset to default ({VAL_PCT_DEFAULT}%)
            </button>
          </div>
        )}
        <div className="row">
          <button
            className="btn primary"
            onClick={() => startExport.mutate()}
            disabled={!project.data?.ls_project_id || startExport.isPending || !!busy}
          >
            Build from Label Studio labels
          </button>
          {!project.data?.ls_project_id && <span className="small muted">Send images to Label Studio first.</span>}
        </div>
        {job && (
          <div className="row">
            <StatusBadge status={job.status} />
            {job.status === "queued" && <span className="small muted">Waiting for {job.queue_position} job(s) ahead.</span>}
            {job.status === "failed" && <span className="small">{job.error}</span>}
            {job.status === "succeeded" && (
              <span className="small">
                Dataset ready. <Link to={`../datasets/${job.result.dataset_id}`}>Preview the labels</Link> before training.
              </span>
            )}
          </div>
        )}
        <ErrorNotice error={startExport.error} />
        <details>
          <summary>Already have labels in YOLO format?</summary>
          <div className="stack" style={{ marginTop: "0.75rem" }}>
            <p className="small muted">
              Upload a .zip with images and matching .txt label files (same file name). If it has train/ and val/
              folders they're kept; otherwise the split above is used. Class numbers follow this project's object
              types in order: {project.data?.classes.map((c, i) => `${i} = ${c}`).join(", ")}.
            </p>
            <Dropzone
              accept=".zip"
              multiple={false}
              disabled={zipProgress !== null}
              onFiles={uploadZip}
              title={zipProgress === null ? "Drop a YOLO .zip" : "Uploading…"}
              hint="images + labels"
            />
            {zipProgress !== null && <ProgressBar value={zipProgress} />}
            <ErrorNotice error={zipError} />
          </div>
        </details>
      </section>
      {datasets.data?.map((d) => <DatasetCard key={d.id} dataset={d} />)}
    </div>
  );
}
