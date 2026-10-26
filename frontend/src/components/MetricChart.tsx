import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { MetricRow } from "../api";

const COLORS = ["var(--chart-1)", "var(--chart-2)", "var(--chart-3)", "var(--chart-4)"];

export interface Series {
  key: string;
  label: string;
}

/** Plain-language names for Ultralytics metric keys. */
export const METRIC_LABELS: Record<string, string> = {
  "metrics/mAP50(B)": "Box mAP@50",
  "metrics/mAP50-95(B)": "Box mAP@50–95",
  "metrics/mAP50(M)": "Mask mAP@50",
  "metrics/mAP50-95(M)": "Mask mAP@50–95",
  "metrics/precision(B)": "Precision",
  "metrics/recall(B)": "Recall",
  "train/box_loss": "Box loss (train)",
  "train/seg_loss": "Mask loss (train)",
  "train/cls_loss": "Class loss (train)",
  "val/box_loss": "Box loss (val)",
  "val/seg_loss": "Mask loss (val)",
  "val/cls_loss": "Class loss (val)",
};

export function lossSeries(rows: MetricRow[]): Series[] {
  const keys = ["train/box_loss", "val/box_loss", "train/seg_loss", "val/seg_loss", "train/cls_loss", "val/cls_loss"];
  return pick(rows, keys);
}

export function accuracySeries(rows: MetricRow[]): Series[] {
  const keys = ["metrics/mAP50(M)", "metrics/mAP50-95(M)", "metrics/mAP50(B)", "metrics/mAP50-95(B)"];
  return pick(rows, keys);
}

function pick(rows: MetricRow[], keys: string[]): Series[] {
  const present = new Set(rows.flatMap((r) => Object.keys(r.values)));
  return keys.filter((k) => present.has(k)).slice(0, 4).map((k) => ({ key: k, label: METRIC_LABELS[k] ?? k }));
}

export function MetricChart({ rows, series, domain }: { rows: MetricRow[]; series: Series[]; domain?: [number, number] }) {
  const data = rows.map((r) => ({ epoch: r.epoch, ...r.values }));
  return (
    <div className="chart-box">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 8, right: 12, bottom: 4, left: -8 }}>
          <CartesianGrid stroke="var(--line)" strokeDasharray="2 4" />
          <XAxis dataKey="epoch" stroke="var(--muted)" fontSize={12} allowDecimals={false} label={{ value: "epoch", position: "insideBottomRight", offset: -2, fill: "var(--muted)", fontSize: 12 }} />
          <YAxis stroke="var(--muted)" fontSize={12} domain={domain ?? ["auto", "auto"]} tickFormatter={(v: number) => (Math.abs(v) >= 10 ? v.toFixed(0) : v.toFixed(2))} />
          <Tooltip
            contentStyle={{ background: "var(--surface)", border: "1px solid var(--line)", borderRadius: 6, color: "var(--ink)" }}
            formatter={(v: number) => v.toFixed(4)}
            labelFormatter={(e) => `Epoch ${e}`}
          />
          <Legend wrapperStyle={{ fontSize: 13 }} />
          {series.map((s, i) => (
            <Line
              key={s.key}
              type="monotone"
              dataKey={s.key}
              name={s.label}
              stroke={COLORS[i % COLORS.length]}
              strokeWidth={2}
              strokeDasharray={s.key.startsWith("val/") || s.key.includes("50-95") ? "5 3" : undefined}
              dot={rows.length < 30}
              isAnimationActive={false}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
