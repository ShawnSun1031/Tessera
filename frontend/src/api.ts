// Typed client for the FastAPI backend (/api).

export type TaskType = "detect" | "segment";
export type JobStatus = "uploading" | "queued" | "running" | "succeeded" | "failed" | "cancelled";

export interface User {
  id: number;
  username: string;
  is_admin: boolean;
}

export interface Project {
  id: number;
  name: string;
  task_type: TaskType;
  classes: string[];
  owner_id: number;
  ls_project_id: number | null;
  ls_url: string | null;
}

export interface LabelStudioProject {
  id: number;
  title: string;
  tasks: number;
  annotated: number;
  task_type: TaskType | null;
  classes: string[];
  problem: string | null;
}

export interface ImageRow {
  id: number;
  filename: string;
  source_name: string;
  in_label_studio: boolean;
}

export interface LabelStudioStatus {
  ls_project_id: number | null;
  ls_url: string | null;
  images: number;
  pending_upload: number;
  tasks: number;
  annotated: number;
}

export interface DatasetStats {
  splits: Record<"train" | "val", { images: number; unlabeled: number }>;
  instances: Record<string, number>;
}

export interface Dataset {
  id: number;
  project_id: number;
  name: string;
  source: "labelstudio" | "upload";
  classes: string[];
  stats: DatasetStats;
  created_at: string;
}

export interface LabeledObject {
  class_id: number;
  kind: "box" | "polygon";
  points: [number, number][];
}

export interface DatasetItem {
  split: "train" | "val";
  name: string;
  objects: LabeledObject[];
}

export interface Job {
  id: number;
  project_id: number;
  username: string | null;
  kind: "export" | "train" | "predict" | "onnx";
  status: JobStatus;
  params: Record<string, any>;
  result: Record<string, any>;
  progress: Record<string, number>;
  error: string | null;
  cancel_requested: boolean;
  queue_position: number;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
}

export interface QueueEntry {
  id: number;
  kind: Job["kind"];
  status: JobStatus;
  username: string | null;
  mine: boolean;
  queue_position: number;
}

export interface MetricRow {
  epoch: number;
  values: Record<string, number>;
}

export interface Model {
  id: number;
  project_id: number;
  job_id: number;
  dataset_id: number;
  name: string;
  task_type: TaskType;
  architecture: string;
  classes: string[];
  metrics: Record<string, number>;
  created_at: string;
  onnx_available: boolean;
  plots: string[];
  imgsz: number;
}

export interface Architecture {
  family: string;
  size: string;
  note: string;
  weights: string;
}

export interface BaseWeights {
  name: string;
  size: number;
  uploaded_at: string;
}

export interface TrainRequest {
  dataset_id: number;
  base_model: string;
  name?: string;
  epochs: number;
  imgsz: number;
  batch: number;
  lr0: number;
  patience: number;
  augment: "none" | "light" | "default" | "strong";
}

const TOKEN_KEY = "bench.token";

export function getToken(): string | null {
  try {
    return localStorage.getItem(TOKEN_KEY);
  } catch {
    return null;
  }
}

export function setToken(token: string | null): void {
  try {
    if (token) localStorage.setItem(TOKEN_KEY, token);
    else localStorage.removeItem(TOKEN_KEY);
  } catch {
    // storage unavailable: the session just won't survive a reload
  }
}

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}

let onUnauthorized: () => void = () => {};
export function setUnauthorizedHandler(fn: () => void): void {
  onUnauthorized = fn;
}

function detailMessage(body: any, status: number): string {
  const detail = body?.detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail) && detail.length) {
    return detail
      .map((d: any) => `${(d.loc ?? []).filter((l: any) => l !== "body").join(".")}: ${d.msg}`)
      .join("; ");
  }
  return `Request failed (${status})`;
}

export async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers);
  const token = getToken();
  if (token) headers.set("Authorization", `Bearer ${token}`);
  if (init.body && !(init.body instanceof FormData) && !(init.body instanceof URLSearchParams)) {
    headers.set("Content-Type", "application/json");
  }
  const resp = await fetch(`/api${path}`, { ...init, headers });
  if (resp.status === 401 && !path.startsWith("/auth/login")) onUnauthorized();
  if (!resp.ok) {
    const body = await resp.json().catch(() => null);
    throw new ApiError(resp.status, detailMessage(body, resp.status));
  }
  if (resp.status === 204) return undefined as T;
  return resp.json();
}

const json = (body: unknown): RequestInit => ({ method: "POST", body: JSON.stringify(body) });

/** URL usable in <img src> / <a href> (adds the token as a query param). */
export function fileUrl(path: string): string {
  const token = getToken();
  const sep = path.includes("?") ? "&" : "?";
  return `/api${path}${token ? `${sep}access_token=${encodeURIComponent(token)}` : ""}`;
}

/** Upload with progress (fetch has no upload progress events). */
export function upload<T>(path: string, form: FormData, onProgress?: (fraction: number) => void): Promise<T> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `/api${path}`);
    const token = getToken();
    if (token) xhr.setRequestHeader("Authorization", `Bearer ${token}`);
    xhr.upload.onprogress = (e) => e.lengthComputable && onProgress?.(e.loaded / e.total);
    xhr.onload = () => {
      let body: any = null;
      try {
        body = JSON.parse(xhr.responseText);
      } catch {
        /* non-JSON error page */
      }
      if (xhr.status === 401) onUnauthorized();
      if (xhr.status >= 200 && xhr.status < 300) resolve(body as T);
      else reject(new ApiError(xhr.status, detailMessage(body, xhr.status)));
    };
    xhr.onerror = () => reject(new ApiError(0, "Network error: the upload did not reach the server"));
    xhr.send(form);
  });
}

const PART_FILES = 100;
const PART_BYTES = 200 * 1024 * 1024;
const RETRY_DELAYS_MS = [2000, 5000, 15000];

/** Groups of at most PART_FILES files / PART_BYTES; a bigger file goes alone. */
function uploadParts(files: File[]): File[][] {
  const parts: File[][] = [];
  let part: File[] = [];
  let bytes = 0;
  for (const f of files) {
    if (part.length && (part.length >= PART_FILES || bytes + f.size > PART_BYTES)) {
      parts.push(part);
      part = [];
      bytes = 0;
    }
    part.push(f);
    bytes += f.size;
  }
  if (part.length) parts.push(part);
  return parts;
}

export interface PartsProgress {
  sentBytes: number;
  totalBytes: number;
  sentFiles: number;
  totalFiles: number;
}

/** Send an inference job's files in parts, retrying a part that fails on the way.
 * Each part carries the index of its first file, so the server keeps the order and a
 * retried part replaces itself instead of adding duplicates. */
export async function uploadInputs(jobId: number, files: File[], onProgress: (p: PartsProgress) => void): Promise<void> {
  const totalBytes = files.reduce((n, f) => n + f.size, 0);
  let sentBytes = 0;
  let offset = 0;
  for (const part of uploadParts(files)) {
    const size = part.reduce((n, f) => n + f.size, 0);
    const form = new FormData();
    form.append("offset", String(offset));
    part.forEach((f) => form.append("files", f));
    for (let attempt = 0; ; attempt++) {
      try {
        await upload(`/jobs/${jobId}/inputs`, form, (fraction) =>
          onProgress({ sentBytes: sentBytes + fraction * size, totalBytes, sentFiles: offset, totalFiles: files.length }),
        );
        break;
      } catch (e) {
        // 4xx means the request itself is wrong; sending it again won't help
        const clientError = e instanceof ApiError && e.status >= 400 && e.status < 500;
        if (clientError || attempt >= RETRY_DELAYS_MS.length) throw e;
        await new Promise((r) => setTimeout(r, RETRY_DELAYS_MS[attempt]));
      }
    }
    sentBytes += size;
    offset += part.length;
    onProgress({ sentBytes, totalBytes, sentFiles: offset, totalFiles: files.length });
  }
}

export const api = {
  login: (username: string, password: string) =>
    request<{ access_token: string }>("/auth/login", {
      method: "POST",
      body: new URLSearchParams({ username, password }),
    }),
  me: () => request<User>("/auth/me"),
  users: () => request<User[]>("/auth/users"),
  createUser: (body: { username: string; password: string; is_admin: boolean }) =>
    request<User>("/auth/users", json(body)),

  projects: () => request<Project[]>("/projects"),
  project: (id: number) => request<Project>(`/projects/${id}`),
  createProject: (body: { name: string; task_type: TaskType; classes: string[] }) =>
    request<Project>("/projects", json(body)),
  addMember: (id: number, username: string) => request<void>(`/projects/${id}/members`, json({ username })),

  lsProjects: () => request<LabelStudioProject[]>("/labelstudio/projects"),
  createFromLabelStudio: (ls_project_id: number, name?: string) =>
    request<Project>("/projects/from-labelstudio", json({ ls_project_id, name: name || undefined })),
  linkLabelStudio: (id: number, ls_project_id: number) =>
    request<Project>(`/projects/${id}/labelstudio/link`, json({ ls_project_id })),

  images: (id: number) => request<ImageRow[]>(`/projects/${id}/images`),
  lsServer: () => request<LabelStudioServer>("/labelstudio/server"),
  startLsServer: () => request<LabelStudioServer>("/labelstudio/server/start", { method: "POST" }),
  lsStatus: (id: number) => request<LabelStudioStatus>(`/projects/${id}/labelstudio/status`),
  lsSync: (id: number) => request<LabelStudioStatus>(`/projects/${id}/labelstudio/sync`, { method: "POST" }),

  datasets: (id: number) => request<Dataset[]>(`/projects/${id}/datasets`),
  dataset: (id: number) => request<Dataset>(`/datasets/${id}`),
  datasetItems: (
    id: number,
    q: { split: "train" | "val"; class_id?: number; unlabeled?: boolean; offset: number; limit: number },
  ) => {
    const params = new URLSearchParams({ split: q.split, offset: String(q.offset), limit: String(q.limit) });
    if (q.class_id !== undefined) params.set("class_id", String(q.class_id));
    if (q.unlabeled) params.set("unlabeled", "true");
    return request<{ total: number; items: DatasetItem[] }>(`/datasets/${id}/items?${params}`);
  },
  exportDataset: (id: number, body: { name?: string; val_ratio: number }) =>
    request<Job>(`/projects/${id}/datasets/export`, json(body)),

  architectures: (id: number) =>
    request<{ architectures: Architecture[]; augment_presets: string[] }>(`/projects/${id}/architectures`),
  train: (id: number, body: TrainRequest) => request<Job>(`/projects/${id}/train`, json(body)),
  baseWeights: (id: number) => request<BaseWeights[]>(`/projects/${id}/base-weights`),
  uploadBaseWeights: (id: number, file: File, onProgress: (fraction: number) => void) => {
    const form = new FormData();
    form.append("file", file);
    return upload<BaseWeights>(`/projects/${id}/base-weights`, form, onProgress);
  },
  deleteBaseWeights: (id: number, name: string) =>
    request<void>(`/projects/${id}/base-weights/${encodeURIComponent(name)}`, { method: "DELETE" }),

  jobs: (id: number, kind?: Job["kind"]) => request<Job[]>(`/projects/${id}/jobs${kind ? `?kind=${kind}` : ""}`),
  job: (id: number) => request<Job>(`/jobs/${id}`),
  cancelJob: (id: number) => request<Job>(`/jobs/${id}/cancel`, { method: "POST" }),
  metrics: (id: number) => request<MetricRow[]>(`/jobs/${id}/metrics`),
  queue: () => request<QueueEntry[]>("/queue"),

  models: (id: number) => request<Model[]>(`/projects/${id}/models`),
  model: (id: number) => request<Model>(`/models/${id}`),
  exportOnnx: (id: number) => request<Job>(`/models/${id}/export-onnx`, json({})),
  createPredictJob: (id: number, conf: number) => {
    const form = new FormData();
    form.append("conf", String(conf));
    return request<Job>(`/models/${id}/predict`, { method: "POST", body: form });
  },
  createUploadedPredictJob: (projectId: number, weights: string, conf: number) => {
    const form = new FormData();
    form.append("conf", String(conf));
    return request<Job>(`/projects/${projectId}/base-weights/${encodeURIComponent(weights)}/predict`, { method: "POST", body: form });
  },
  startJob: (id: number) => request<Job>(`/jobs/${id}/start`, { method: "POST" }),
  sendPredictions: (jobId: number) =>
    request<SentToLabelStudio>(`/labelstudio/send-predictions/${jobId}`, { method: "POST" }),
  predictions: (jobId: number) => request<PredictionRecord[]>(`/jobs/${jobId}/files/output/predictions.json`),
};

export interface LabelStudioServer {
  url: string;
  reachable: boolean;
  can_start: boolean;
  starting: boolean;
  reason: string | null;
}

export interface SentToLabelStudio {
  ls_project_id: number;
  ls_url: string;
  images: number;
  objects: number;
  skipped_objects: number;
  sent_at: string;
}

export interface PredictionRecord {
  image: string;
  width: number;
  height: number;
  objects: { class: string; confidence: number }[];
}

export const TERMINAL: JobStatus[] = ["succeeded", "failed", "cancelled"];
