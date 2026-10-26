import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { api } from "../api";
import { useProjectId } from "./Layout";
import { Dropzone, ErrorNotice, ProgressBar } from "./ui";

/** Weights trained elsewhere (.pt), uploaded to the project to start training from. */
export function UploadedWeights({
  title,
  hint,
  selected,
  onSelect,
  fallback,
}: {
  title: string;
  hint: string;
  selected: string;
  onSelect: (base: string) => void;
  fallback: string;
}) {
  const pid = useProjectId();
  const qc = useQueryClient();
  const weights = useQuery({ queryKey: ["base-weights", pid], queryFn: () => api.baseWeights(pid) });
  const [progress, setProgress] = useState<number | null>(null);
  const send = useMutation({
    mutationFn: (file: File) => {
      setProgress(0);
      return api.uploadBaseWeights(pid, file, setProgress);
    },
    onSuccess: (w) => {
      qc.invalidateQueries({ queryKey: ["base-weights", pid] });
      onSelect(`upload:${w.name}`);
    },
    onSettled: () => setProgress(null),
  });
  const remove = useMutation({
    mutationFn: (name: string) => api.deleteBaseWeights(pid, name),
    onSuccess: (_, name) => {
      if (selected === `upload:${name}`) onSelect(fallback);
      qc.invalidateQueries({ queryKey: ["base-weights", pid] });
    },
  });

  return (
    <>
      <h3>{title}</h3>
      {!!weights.data?.length && (
        <div className="choice-grid">
          {weights.data.map((w) => (
            <label className="choice" key={w.name}>
              <input type="radio" name="base" checked={selected === `upload:${w.name}`} onChange={() => onSelect(`upload:${w.name}`)} />
              <span>
                <strong style={{ overflowWrap: "anywhere" }}>{w.name}</strong>
                <span className="num">{(w.size / 1024 / 1024).toFixed(1)} MB</span>{" "}
                <button
                  type="button"
                  className="linkish small"
                  disabled={remove.isPending}
                  onClick={(e) => {
                    e.preventDefault();
                    if (confirm(`Delete ${w.name}? Models and results made with it are kept.`)) remove.mutate(w.name);
                  }}
                >
                  Delete
                </button>
              </span>
            </label>
          ))}
        </div>
      )}
      <Dropzone
        accept=".pt"
        multiple={false}
        disabled={progress !== null}
        onFiles={([file]) => send.mutate(file)}
        title="Drop a .pt file here, or click to choose"
        hint={hint}
      />
      {progress !== null && <ProgressBar value={progress} running />}
      <ErrorNotice error={send.error ?? remove.error} />
    </>
  );
}
