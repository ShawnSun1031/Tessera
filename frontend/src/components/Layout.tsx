import { useQuery } from "@tanstack/react-query";
import { Link, NavLink, Navigate, Outlet, useLocation, useParams } from "react-router-dom";
import { api } from "../api";
import { useAuth } from "../auth";

export function RequireAuth() {
  const { user, loading } = useAuth();
  if (loading) return <p className="page muted">Loading…</p>;
  if (!user) return <Navigate to="/login" replace />;
  return (
    <>
      <TopBar />
      <Outlet />
    </>
  );
}

function QueuePill() {
  const queue = useQuery({ queryKey: ["queue"], queryFn: api.queue, refetchInterval: 5000 });
  const jobs = queue.data ?? [];
  const running = jobs.filter((j) => j.status === "running");
  const waiting = jobs.length - running.length;
  if (!jobs.length) return <span className="queue-pill">GPU idle</span>;
  const who = running[0] ? (running[0].mine ? "your job" : `${running[0].username}'s job`) : "";
  return (
    <span className="queue-pill busy" title={jobs.map((j) => `#${j.id} ${j.kind} (${j.username}) ${j.status}`).join("\n")}>
      GPU busy{who && ` with ${who}`}
      {waiting > 0 && `, ${waiting} waiting`}
    </span>
  );
}

function TopBar() {
  const { user, logout } = useAuth();
  return (
    <header className="topbar">
      <Link to="/" className="brand">
        <span className="brand-mark" aria-hidden />
        Tessera
      </Link>
      <span className="spacer" />
      <QueuePill />
      {user?.is_admin && (
        <Link to="/admin/users" className="small">
          Users
        </Link>
      )}
      <span className="small muted who">{user?.username}</span>
      <button className="btn small" onClick={logout}>
        Sign out
      </button>
    </header>
  );
}

export function useProjectId(): number {
  const { pid } = useParams();
  return Number(pid);
}

/** Step completion is derived from what exists, so the rail doubles as a checklist. */
function useStepState(pid: number) {
  const images = useQuery({ queryKey: ["images", pid], queryFn: () => api.images(pid) });
  const ls = useQuery({ queryKey: ["ls", pid], queryFn: () => api.lsStatus(pid), retry: false });
  const datasets = useQuery({ queryKey: ["datasets", pid], queryFn: () => api.datasets(pid) });
  const models = useQuery({ queryKey: ["models", pid], queryFn: () => api.models(pid) });
  const predicts = useQuery({ queryKey: ["jobs", pid, "predict"], queryFn: () => api.jobs(pid, "predict") });
  return {
    // projects imported from Label Studio have their images there, not here
    upload: (images.data?.length ?? 0) > 0 || (ls.data?.tasks ?? 0) > 0,
    annotate: (ls.data?.annotated ?? 0) > 0,
    dataset: (datasets.data?.length ?? 0) > 0,
    train: (models.data?.length ?? 0) > 0,
    results: (models.data?.length ?? 0) > 0,
    predict: (predicts.data ?? []).some((j) => j.status === "succeeded"),
  };
}

const STEPS = [
  { to: "upload", label: "Upload images", hint: "TIF, PNG or JPG" },
  { to: "annotate", label: "Annotate", hint: "Draw labels in Label Studio" },
  { to: "dataset", label: "Build dataset", hint: "Export labels for training" },
  { to: "train", label: "Train", hint: "Pick a model and start" },
  { to: "models", label: "Review results", hint: "Accuracy, plots, downloads" },
  { to: "predict", label: "Predict", hint: "Run a model on new images" },
] as const;

export function ProjectShell() {
  const pid = useProjectId();
  const project = useQuery({ queryKey: ["project", pid], queryFn: () => api.project(pid) });
  const done = useStepState(pid);
  const previewing = useLocation().pathname.includes("/datasets/");
  const doneKey: Record<string, boolean> = {
    upload: done.upload,
    annotate: done.annotate,
    dataset: done.dataset,
    train: done.train,
    models: done.results,
    predict: done.predict,
  };
  if (project.error) return <p className="page">This project does not exist or you do not have access to it.</p>;
  const p = project.data;
  return (
    <div className="project-shell">
      <nav className="rail" aria-label="Project steps">
        <div className="rail-title">{p?.name ?? "…"}</div>
        <div className="rail-sub">
          {p ? `${p.task_type === "segment" ? "Segmentation" : "Detection"}: ${p.classes.join(", ")}` : ""}
        </div>
        <ol className="steps">
          {STEPS.map((s, i) => (
            <li key={s.to} className={`step${doneKey[s.to] ? " done" : ""}`}>
              <NavLink
                to={s.to}
                // dataset previews (datasets/:id) belong to the "Build dataset" step
                className={({ isActive }) => (isActive || (s.to === "dataset" && previewing) ? "active" : undefined)}
              >
                <span className="step-dot" aria-hidden>
                  {doneKey[s.to] ? "✓" : i + 1}
                </span>
                <span>
                  <span className="step-label">{s.label}</span>
                  <span className="step-hint">{s.hint}</span>
                </span>
              </NavLink>
            </li>
          ))}
        </ol>
        <div className="rail-extra">
          <NavLink to="activity">Activity</NavLink>
          <NavLink to="settings">Team access</NavLink>
          <Link to="/">All projects</Link>
        </div>
      </nav>
      <main className="main">
        <Outlet />
      </main>
    </div>
  );
}
