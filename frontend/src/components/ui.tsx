import { useEffect, useRef, useState, type ReactNode } from "react";
import { fileUrl, type Job, type JobStatus, type MetricRow, TERMINAL } from "../api";

const STATUS_TEXT: Record<JobStatus, string> = {
  uploading: "Uploading",
  queued: "Waiting",
  running: "Running",
  succeeded: "Done",
  failed: "Failed",
  cancelled: "Cancelled",
};

export function StatusBadge({ status }: { status: JobStatus }) {
  return <span className={`status ${status}`}>{STATUS_TEXT[status]}</span>;
}

export const JOB_KIND_TEXT: Record<Job["kind"], string> = {
  export: "Dataset export",
  train: "Training",
  predict: "Predictions",
  onnx: "ONNX export",
};

export function ProgressBar({ value, running }: { value: number; running?: boolean }) {
  const pct = Math.max(0, Math.min(1, value)) * 100;
  return (
    <div className={`bar${running ? " run" : ""}`} role="progressbar" aria-valuenow={Math.round(pct)} aria-valuemin={0} aria-valuemax={100}>
      <span style={{ width: `${pct}%` }} />
    </div>
  );
}

export function ErrorNotice({ error }: { error: unknown }) {
  if (!error) return null;
  const message = error instanceof Error ? error.message : String(error);
  return (
    <div className="notice error" role="alert">
      {message}
    </div>
  );
}

export function PageHead({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <header className="page-head">
      <h1>{title}</h1>
      {children}
    </header>
  );
}

export function Dropzone({
  accept,
  multiple = true,
  disabled,
  onFiles,
  title,
  hint,
}: {
  accept: string;
  multiple?: boolean;
  disabled?: boolean;
  onFiles: (files: File[]) => void;
  title: string;
  hint: string;
}) {
  const input = useRef<HTMLInputElement>(null);
  const [over, setOver] = useState(false);
  return (
    <div
      className={`dropzone${over ? " over" : ""}`}
      role="button"
      tabIndex={0}
      aria-disabled={disabled}
      onClick={() => !disabled && input.current?.click()}
      onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && !disabled && input.current?.click()}
      onDragOver={(e) => {
        e.preventDefault();
        setOver(true);
      }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setOver(false);
        if (!disabled && e.dataTransfer.files.length) onFiles(Array.from(e.dataTransfer.files));
      }}
    >
      <strong>{title}</strong>
      <span className="muted small">{hint}</span>
      <input
        ref={input}
        type="file"
        hidden
        accept={accept}
        multiple={multiple}
        onChange={(e) => {
          if (e.target.files?.length) onFiles(Array.from(e.target.files));
          e.target.value = "";
        }}
      />
    </div>
  );
}

export function ChipsInput({ value, onChange, placeholder }: { value: string[]; onChange: (v: string[]) => void; placeholder: string }) {
  const [draft, setDraft] = useState("");
  const add = () => {
    const names = draft
      .split(",")
      .map((s) => s.trim())
      .filter((s) => s && !value.includes(s));
    if (names.length) onChange([...value, ...names]);
    setDraft("");
  };
  return (
    <div className="chips">
      {value.map((c) => (
        <span className="chip" key={c}>
          {c}
          <button type="button" aria-label={`Remove ${c}`} onClick={() => onChange(value.filter((v) => v !== c))}>
            ×
          </button>
        </span>
      ))}
      <input
        type="text"
        value={draft}
        placeholder={placeholder}
        onChange={(e) => setDraft(e.target.value)}
        onBlur={add}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === ",") {
            e.preventDefault();
            add();
          } else if (e.key === "Backspace" && !draft && value.length) {
            onChange(value.slice(0, -1));
          }
        }}
      />
    </div>
  );
}

export function timeAgo(iso: string | null): string {
  if (!iso) return "";
  const seconds = (Date.now() - new Date(iso.endsWith("Z") || iso.includes("+") ? iso : `${iso}Z`).getTime()) / 1000;
  if (seconds < 60) return "just now";
  if (seconds < 3600) return `${Math.floor(seconds / 60)} min ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)} h ago`;
  return `${Math.floor(seconds / 86400)} d ago`;
}

export function duration(start: string | null, end: string | null): string {
  if (!start) return "";
  const parse = (s: string) => new Date(s.endsWith("Z") || s.includes("+") ? s : `${s}Z`).getTime();
  const secs = Math.max(0, ((end ? parse(end) : Date.now()) - parse(start)) / 1000);
  const m = Math.floor(secs / 60);
  return m ? `${m} min ${Math.round(secs % 60)} s` : `${Math.round(secs)} s`;
}

const MAX_LOG = 200_000;

/** Live job state over Server-Sent Events. */
export function useJobStream(jobId: number) {
  const [job, setJob] = useState<Job | null>(null);
  const [metrics, setMetrics] = useState<MetricRow[]>([]);
  const [log, setLog] = useState("");
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    setJob(null);
    setMetrics([]);
    setLog("");
    const source = new EventSource(fileUrl(`/jobs/${jobId}/events`));
    source.onopen = () => setConnected(true);
    source.addEventListener("job", (e) => setJob(JSON.parse((e as MessageEvent).data)));
    source.addEventListener("metric", (e) => {
      const row: MetricRow = JSON.parse((e as MessageEvent).data);
      setMetrics((prev) => [...prev.filter((m) => m.epoch !== row.epoch), row].sort((a, b) => a.epoch - b.epoch));
    });
    source.addEventListener("log", (e) => {
      const text: string = JSON.parse((e as MessageEvent).data);
      setLog((prev) => (prev + text).slice(-MAX_LOG));
    });
    source.addEventListener("end", () => {
      source.close();
      setConnected(false);
    });
    source.onerror = () => {
      setConnected(false);
      // EventSource retries by itself; stop once the job is finished
      setJob((j) => {
        if (j && TERMINAL.includes(j.status)) source.close();
        return j;
      });
    };
    return () => source.close();
  }, [jobId]);

  return { job, metrics, log, connected };
}
