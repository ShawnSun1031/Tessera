import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { request, type TaskType, type TrainRequest } from "./api";

/** Training options remembered between runs (not the dataset or run name). */
export type TrainSettings = Pick<TrainRequest, "epochs" | "imgsz" | "batch" | "lr0" | "patience" | "augment">;

export interface Preferences {
  train?: Partial<TrainSettings> & { base_model?: Partial<Record<TaskType, string>> };
  dataset_val_pct?: number;
  predict_conf?: number;
}

/** A null value removes that saved setting. */
export type PreferencesPatch = { [K in keyof Preferences]?: Preferences[K] | null };

export const VAL_PCT_DEFAULT = 20;
export const PREDICT_CONF_DEFAULT = 0.25;

export const TRAIN_DEFAULTS: TrainSettings = {
  epochs: 100,
  imgsz: 640,
  batch: -1,
  lr0: 0.01,
  patience: 50,
  augment: "default",
};

const fetchPreferences = () => request<Preferences>("/auth/me/preferences");

/** The signed-in user's saved settings, stored on the server so they follow them across browsers. */
export function usePreferences() {
  const qc = useQueryClient();
  const query = useQuery({ queryKey: ["prefs"], queryFn: fetchPreferences, staleTime: Infinity, retry: false });
  const save = useMutation({
    mutationFn: (patch: PreferencesPatch) =>
      request<Preferences>("/auth/me/preferences", { method: "PATCH", body: JSON.stringify(patch) }),
    onSuccess: (data) => qc.setQueryData(["prefs"], data),
  });
  return {
    prefs: query.data ?? {},
    // a failed load just means starting from defaults
    ready: !query.isLoading,
    save: (patch: PreferencesPatch) => save.mutate(patch),
  };
}
