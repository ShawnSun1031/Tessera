import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { api, fileUrl, type DatasetItem } from "../api";
import { ErrorNotice, PageHead } from "../components/ui";

const PAGE_SIZE = 24;

// Saturated hues that stay visible on both dark scans and bright photos.
const CLASS_COLORS = ["#ff4d6d", "#2ec4ff", "#ffbe0b", "#3ddc84", "#c77dff", "#ff8c42", "#00f5d4", "#f15bb5"];
export const classColor = (i: number) => CLASS_COLORS[i % CLASS_COLORS.length];

type Filter = { kind: "all" } | { kind: "class"; id: number } | { kind: "unlabeled" };

function Overlay({ item, show }: { item: DatasetItem; show: boolean }) {
  if (!show) return null;
  return (
    <svg viewBox="0 0 1 1" preserveAspectRatio="none" aria-hidden>
      {item.objects.map((o, i) => (
        <polygon
          key={i}
          points={o.points.map(([x, y]) => `${x},${y}`).join(" ")}
          fill={classColor(o.class_id)}
          fillOpacity={o.kind === "polygon" ? 0.28 : 0.08}
          stroke={classColor(o.class_id)}
          strokeWidth={2}
          vectorEffect="non-scaling-stroke"
          strokeLinejoin="round"
        />
      ))}
    </svg>
  );
}

function summary(item: DatasetItem, classes: string[]): string {
  if (!item.objects.length) return "no objects";
  const counts = new Map<number, number>();
  item.objects.forEach((o) => counts.set(o.class_id, (counts.get(o.class_id) ?? 0) + 1));
  return [...counts].map(([id, n]) => `${n} ${classes[id] ?? `class ${id}`}`).join(", ");
}

export function DatasetPreviewPage() {
  const { datasetId } = useParams();
  const id = Number(datasetId);
  const dataset = useQuery({ queryKey: ["dataset", id], queryFn: () => api.dataset(id) });
  const [split, setSplit] = useState<"train" | "val">("train");
  const [filter, setFilter] = useState<Filter>({ kind: "all" });
  const [page, setPage] = useState(0);
  const [showLabels, setShowLabels] = useState(true);
  const [open, setOpen] = useState<number | null>(null);

  const items = useQuery({
    queryKey: ["dataset-items", id, split, filter, page],
    queryFn: () =>
      api.datasetItems(id, {
        split,
        class_id: filter.kind === "class" ? filter.id : undefined,
        unlabeled: filter.kind === "unlabeled",
        offset: page * PAGE_SIZE,
        limit: PAGE_SIZE,
      }),
    placeholderData: keepPreviousData,
  });

  const chooseSplit = (s: "train" | "val") => {
    setSplit(s);
    setPage(0);
  };
  const chooseFilter = (f: Filter) => {
    setFilter(f);
    setPage(0);
  };

  const d = dataset.data;
  if (dataset.error) return <ErrorNotice error={dataset.error} />;
  if (!d) return <p className="muted">Loading…</p>;
  const total = items.data?.total ?? 0;
  const list = items.data?.items ?? [];
  const pages = Math.max(1, Math.ceil(total / PAGE_SIZE));
  const isFilter = (f: Filter) => JSON.stringify(f) === JSON.stringify(filter);

  return (
    <div className="stack">
      <PageHead title={`Preview: ${d.name}`}>
        <p>
          Check the labels before you train. Look for objects that were missed, boxes or outlines that don't fit, and
          images with the wrong object type. Fix those in Label Studio, then build the dataset again.
        </p>
      </PageHead>

      <section className="panel stack" style={{ ["--step" as string]: "0.9rem" }}>
        <div className="row spread">
          <div className="segmented" role="group" aria-label="Which images">
            <button aria-pressed={split === "train"} onClick={() => chooseSplit("train")}>
              Training ({d.stats.splits.train.images})
            </button>
            <button aria-pressed={split === "val"} onClick={() => chooseSplit("val")}>
              Validation ({d.stats.splits.val.images})
            </button>
          </div>
          <label className="row small">
            <input type="checkbox" checked={showLabels} onChange={(e) => setShowLabels(e.target.checked)} />
            Show labels
          </label>
        </div>
        <div className="row" role="group" aria-label="Filter by object type">
          <button className="filter-chip" aria-pressed={isFilter({ kind: "all" })} onClick={() => chooseFilter({ kind: "all" })}>
            All images
          </button>
          {d.classes.map((name, i) => (
            <button
              key={name}
              className="filter-chip"
              aria-pressed={isFilter({ kind: "class", id: i })}
              onClick={() => chooseFilter({ kind: "class", id: i })}
            >
              <span className="swatch" style={{ background: classColor(i) }} />
              {name} <span className="muted num">{d.stats.instances[name] ?? 0}</span>
            </button>
          ))}
          <button className="filter-chip" aria-pressed={isFilter({ kind: "unlabeled" })} onClick={() => chooseFilter({ kind: "unlabeled" })}>
            No objects
          </button>
        </div>
      </section>

      <ErrorNotice error={items.error} />
      {items.isSuccess && !total && <p className="muted">No images match this filter.</p>}
      <div className="label-grid" style={{ opacity: items.isPlaceholderData ? 0.6 : 1 }}>
        {list.map((item, i) => (
          <figure className="labeled" key={`${item.split}/${item.name}`}>
            <button onClick={() => setOpen(i)} aria-label={`Open ${item.name}`}>
              <div className="labeled-frame">
                <img src={fileUrl(`/datasets/${id}/preview/${item.split}/${encodeURIComponent(item.name)}?size=512`)} alt="" loading="lazy" />
                <Overlay item={item} show={showLabels} />
              </div>
            </button>
            <figcaption>
              <span title={item.name}>{item.name}</span>
              <span className="num">{summary(item, d.classes)}</span>
            </figcaption>
          </figure>
        ))}
      </div>

      {total > PAGE_SIZE && (
        <div className="row spread">
          <span className="small muted num">
            {page * PAGE_SIZE + 1}–{Math.min(total, (page + 1) * PAGE_SIZE)} of {total}
          </span>
          <div className="row">
            <button className="btn small" disabled={page === 0} onClick={() => setPage(page - 1)}>
              Previous page
            </button>
            <button className="btn small" disabled={page >= pages - 1} onClick={() => setPage(page + 1)}>
              Next page
            </button>
          </div>
        </div>
      )}

      <div className="row">
        <Link className="btn primary" to={`../train?dataset=${d.id}`}>
          Train on this dataset
        </Link>
        <Link className="btn" to="../dataset">
          Back to datasets
        </Link>
      </div>

      {open !== null && list[open] && (
        <Viewer
          datasetId={id}
          item={list[open]}
          classes={d.classes}
          showLabels={showLabels}
          position={`${page * PAGE_SIZE + open + 1} of ${total}`}
          onPrev={open > 0 ? () => setOpen(open - 1) : undefined}
          onNext={open < list.length - 1 ? () => setOpen(open + 1) : undefined}
          onClose={() => setOpen(null)}
        />
      )}
    </div>
  );
}

function Viewer({
  datasetId,
  item,
  classes,
  showLabels,
  position,
  onPrev,
  onNext,
  onClose,
}: {
  datasetId: number;
  item: DatasetItem;
  classes: string[];
  showLabels: boolean;
  position: string;
  onPrev?: () => void;
  onNext?: () => void;
  onClose: () => void;
}) {
  const ref = useRef<HTMLDialogElement>(null);
  const [labels, setLabels] = useState(showLabels);

  useEffect(() => {
    ref.current?.showModal();
  }, []);
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "ArrowLeft") onPrev?.();
      if (e.key === "ArrowRight") onNext?.();
      if (e.key === "l") setLabels((v) => !v);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onPrev, onNext]);

  return (
    <dialog ref={ref} className="viewer stack" style={{ ["--step" as string]: "0.75rem" }} onClose={onClose} aria-label={item.name}>
      <div className="row spread">
        <div>
          <strong>{item.name}</strong>
          <div className="small muted">
            {position}: {summary(item, classes)}
          </div>
        </div>
        <div className="row">
          <label className="row small">
            <input type="checkbox" checked={labels} onChange={(e) => setLabels(e.target.checked)} />
            Show labels (L)
          </label>
          <button className="btn small" onClick={onPrev} disabled={!onPrev} aria-label="Previous image">
            ←
          </button>
          <button className="btn small" onClick={onNext} disabled={!onNext} aria-label="Next image">
            →
          </button>
          <button className="btn small" onClick={() => ref.current?.close()}>
            Close
          </button>
        </div>
      </div>
      <div className="labeled-frame">
        <div>
          <img src={fileUrl(`/datasets/${datasetId}/preview/${item.split}/${encodeURIComponent(item.name)}?size=2048`)} alt={item.name} />
          <Overlay item={item} show={labels} />
        </div>
      </div>
      <div className="row small">
        {classes.map((name, i) => (
          <span className="row" key={name} style={{ gap: "0.35rem" }}>
            <span className="swatch" style={{ background: classColor(i) }} />
            {name}
          </span>
        ))}
      </div>
    </dialog>
  );
}
