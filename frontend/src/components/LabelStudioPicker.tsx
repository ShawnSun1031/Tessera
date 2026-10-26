import { useQuery } from "@tanstack/react-query";
import { api, type LabelStudioProject } from "../api";
import { ErrorNotice } from "./ui";

/** Radio list of Label Studio projects; `blocked` returns why one can't be chosen. */
export function LabelStudioPicker({
  value,
  onChange,
  blocked,
}: {
  value: number | null;
  onChange: (id: number) => void;
  blocked?: (p: LabelStudioProject) => string | null;
}) {
  const projects = useQuery({ queryKey: ["ls-projects"], queryFn: api.lsProjects, retry: false });
  if (projects.isLoading) return <p className="muted small">Loading Label Studio projects…</p>;
  if (projects.error) return <ErrorNotice error={projects.error} />;
  const list = projects.data ?? [];
  if (!list.length) return <p className="muted">There are no projects in Label Studio yet.</p>;

  return (
    <div className="choice-list" role="radiogroup">
      {list.map((p) => {
        const reason = p.problem ?? blocked?.(p) ?? null;
        return (
          <label className="choice" key={p.id} style={reason ? { opacity: 0.6, cursor: "not-allowed" } : undefined}>
            <input type="radio" name="ls-project" disabled={!!reason} checked={value === p.id} onChange={() => onChange(p.id)} />
            <span>
              <strong>{p.title}</strong>
              <span>
                {p.annotated} of {p.tasks} images labeled
                {p.task_type && ` with ${p.task_type === "segment" ? "outlines" : "boxes"}`}
                {p.classes.length > 0 && ` (${p.classes.join(", ")})`}
              </span>
              {reason && <span style={{ display: "block", color: "var(--bad)" }}>{reason}</span>}
            </span>
          </label>
        );
      })}
    </div>
  );
}
