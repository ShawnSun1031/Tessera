import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect } from "react";
import { api } from "../api";
import { ErrorNotice } from "./ui";

/** Shared by every page that needs Label Studio; react-query dedupes the polling. */
export function useLabelStudioServer() {
  return useQuery({
    queryKey: ["ls-server"],
    queryFn: api.lsServer,
    refetchInterval: (q) => (q.state.data?.starting ? 2000 : q.state.data?.reachable ? 60000 : 10000),
  });
}

/** Shown when Label Studio is down; offers to start it when it runs on this server. */
export function LabelStudioServerNotice() {
  const qc = useQueryClient();
  const server = useLabelStudioServer();
  const start = useMutation({
    mutationFn: api.startLsServer,
    onSuccess: (data) => qc.setQueryData(["ls-server"], data),
  });
  const s = server.data;

  // once it is up, reload everything that failed while it was down
  useEffect(() => {
    if (!s?.reachable) return;
    for (const key of ["ls", "ls-projects", "project"]) void qc.invalidateQueries({ queryKey: [key] });
  }, [s?.reachable, qc]);

  if (!s || s.reachable) return null;
  if (s.starting || start.isPending) {
    return (
      <div className="notice run" role="status">
        Starting Label Studio… This usually takes 20 to 60 seconds. The page continues on its own when it's ready.
      </div>
    );
  }
  return (
    <div className="notice error stack" role="alert" style={{ ["--step" as string]: "0.5rem" }}>
      <span>
        <strong>Label Studio isn't running</strong> at {s.url}. Labeling, sending images and building datasets need it.
      </span>
      {s.reason && <pre className="small" style={{ margin: 0, whiteSpace: "pre-wrap" }}>{s.reason}</pre>}
      {s.can_start && (
        <div>
          <button className="btn primary" onClick={() => start.mutate()}>
            Start Label Studio
          </button>
        </div>
      )}
      <ErrorNotice error={start.error} />
    </div>
  );
}
