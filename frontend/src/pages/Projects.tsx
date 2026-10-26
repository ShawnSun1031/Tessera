import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { api, type TaskType } from "../api";
import { LabelStudioPicker } from "../components/LabelStudioPicker";
import { useProjectId } from "../components/Layout";
import { ChipsInput, duration, ErrorNotice, JOB_KIND_TEXT, PageHead, StatusBadge, timeAgo } from "../components/ui";

export function ProjectsPage() {
  const projects = useQuery({ queryKey: ["projects"], queryFn: api.projects });
  const [creating, setCreating] = useState(false);
  const list = projects.data ?? [];
  const showForm = creating || (projects.isSuccess && list.length === 0);

  return (
    <div className="page stack">
      <PageHead title="Projects">
        <p>A project holds one set of images, its labels, and the models trained on them.</p>
      </PageHead>
      {showForm ? (
        <NewProject onCancel={list.length ? () => setCreating(false) : undefined} />
      ) : (
        <div>
          <button className="btn primary" onClick={() => setCreating(true)}>
            New project
          </button>
        </div>
      )}
      <ErrorNotice error={projects.error} />
      <div className="project-list">
        {list.map((p) => (
          <Link key={p.id} to={`/p/${p.id}/upload`} className="project-link">
            <b>{p.name}</b>
            <span className="muted small">{p.task_type === "segment" ? "Segmentation" : "Detection"}</span>
            <span className="muted small">{p.classes.join(", ")}</span>
          </Link>
        ))}
      </div>
    </div>
  );
}

function NewProject({ onCancel }: { onCancel?: () => void }) {
  const [source, setSource] = useState<"scratch" | "labelstudio">("scratch");
  return (
    <section className="panel stack">
      <h2>New project</h2>
      <div className="grid-2" role="radiogroup" aria-label="Where are your labels?">
        <label className="choice">
          <input type="radio" name="source" checked={source === "scratch"} onChange={() => setSource("scratch")} />
          <span>
            <strong>Start fresh</strong>
            <span>Upload images here and label them in Label Studio.</span>
          </span>
        </label>
        <label className="choice">
          <input type="radio" name="source" checked={source === "labelstudio"} onChange={() => setSource("labelstudio")} />
          <span>
            <strong>Use labels already in Label Studio</strong>
            <span>Pick a project you've labeled there. Brush labels become YOLO outlines.</span>
          </span>
        </label>
      </div>
      {source === "scratch" ? <NewProjectForm onCancel={onCancel} /> : <ImportForm onCancel={onCancel} />}
    </section>
  );
}

function ImportForm({ onCancel }: { onCancel?: () => void }) {
  const navigate = useNavigate();
  const qc = useQueryClient();
  const [lsId, setLsId] = useState<number | null>(null);
  const [name, setName] = useState("");
  const create = useMutation({
    mutationFn: () => api.createFromLabelStudio(lsId!, name),
    onSuccess: (p) => {
      qc.invalidateQueries({ queryKey: ["projects"] });
      navigate(`/p/${p.id}/dataset`);
    },
  });
  return (
    <form
      className="stack"
      onSubmit={(e) => {
        e.preventDefault();
        create.mutate();
      }}
    >
      <p className="muted small">
        The project takes its object types and task from the Label Studio project. Your labels stay in Label Studio; you
        can keep labeling there and build a new dataset any time.
      </p>
      <LabelStudioPicker value={lsId} onChange={setLsId} />
      <label className="field">
        Name
        <span className="help">Optional. Defaults to the Label Studio project's title.</span>
        <input type="text" value={name} onChange={(e) => setName(e.target.value)} />
      </label>
      <ErrorNotice error={create.error} />
      <div className="row">
        <button className="btn primary" disabled={lsId === null || create.isPending}>
          Create project
        </button>
        {onCancel && (
          <button type="button" className="btn" onClick={onCancel}>
            Cancel
          </button>
        )}
      </div>
    </form>
  );
}

function NewProjectForm({ onCancel }: { onCancel?: () => void }) {
  const navigate = useNavigate();
  const qc = useQueryClient();
  const [name, setName] = useState("");
  const [taskType, setTaskType] = useState<TaskType>("segment");
  const [classes, setClasses] = useState<string[]>([]);
  const create = useMutation({
    mutationFn: () => api.createProject({ name, task_type: taskType, classes }),
    onSuccess: (p) => {
      qc.invalidateQueries({ queryKey: ["projects"] });
      navigate(`/p/${p.id}/upload`);
    },
  });

  return (
    <form
      className="stack"
      onSubmit={(e) => {
        e.preventDefault();
        create.mutate();
      }}
    >
      <label className="field">
        Name
        <input type="text" value={name} onChange={(e) => setName(e.target.value)} placeholder="Weld cracks, line 3" required />
      </label>
      <fieldset className="field" style={{ border: 0, padding: 0, margin: 0 }}>
        <legend style={{ fontWeight: 700, marginBottom: "0.3rem" }}>What should the model find?</legend>
        <div className="grid-2">
          <label className="choice">
            <input type="radio" name="task" checked={taskType === "segment"} onChange={() => setTaskType("segment")} />
            <span>
              <strong>Exact outlines (segmentation)</strong>
              <span>You paint the shape of each object. Best for cracks, holes and irregular shapes.</span>
            </span>
          </label>
          <label className="choice">
            <input type="radio" name="task" checked={taskType === "detect"} onChange={() => setTaskType("detect")} />
            <span>
              <strong>Boxes around objects (detection)</strong>
              <span>You draw a rectangle per object. Faster to label; enough for counting and locating.</span>
            </span>
          </label>
        </div>
      </fieldset>
      <label className="field">
        Object types
        <span className="help">Type a name and press Enter. You can't change these once labeling starts.</span>
        <ChipsInput value={classes} onChange={setClasses} placeholder={classes.length ? "Add another" : "e.g. Crack"} />
      </label>
      <ErrorNotice error={create.error} />
      <div className="row">
        <button className="btn primary" disabled={!name || !classes.length || create.isPending}>
          Create project
        </button>
        {onCancel && (
          <button type="button" className="btn" onClick={onCancel}>
            Cancel
          </button>
        )}
      </div>
    </form>
  );
}

export function ActivityPage() {
  const pid = useProjectId();
  const jobs = useQuery({ queryKey: ["jobs", pid], queryFn: () => api.jobs(pid), refetchInterval: 5000 });
  return (
    <div className="stack">
      <PageHead title="Activity">
        <p>Every export, training and prediction run in this project.</p>
      </PageHead>
      {jobs.data && !jobs.data.length && <p className="muted">Nothing has run yet.</p>}
      {!!jobs.data?.length && (
        <div className="panel table-wrap">
          <table className="data">
            <thead>
              <tr>
                <th>Run</th>
                <th>Status</th>
                <th>By</th>
                <th>Started</th>
                <th>Took</th>
              </tr>
            </thead>
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
                  <td>{j.username}</td>
                  <td className="muted">{timeAgo(j.started_at ?? j.created_at)}</td>
                  <td className="num muted">{duration(j.started_at, j.finished_at)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

export function SettingsPage() {
  const pid = useProjectId();
  const [username, setUsername] = useState("");
  const add = useMutation({ mutationFn: () => api.addMember(pid, username), onSuccess: () => setUsername("") });
  return (
    <div className="stack">
      <PageHead title="Team access">
        <p>Add teammates so they can see this project, label its images and train models.</p>
      </PageHead>
      <form
        className="panel stack"
        onSubmit={(e) => {
          e.preventDefault();
          add.mutate();
        }}
      >
        <label className="field">
          Username
          <input type="text" value={username} onChange={(e) => setUsername(e.target.value)} required />
        </label>
        <ErrorNotice error={add.error} />
        {add.isSuccess && <div className="notice ok">Added. They'll see the project next time they open the project list.</div>}
        <div>
          <button className="btn primary" disabled={add.isPending}>
            Add teammate
          </button>
        </div>
      </form>
    </div>
  );
}
