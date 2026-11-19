import { AnimatePresence, motion } from "framer-motion";
import { useState } from "react";
import { type Segment, gb, layout, residentBytes } from "../../shared/data/strip";
import type { GpuFrame } from "../../shared/data/types";
import { units } from "../../shared/theme/tokens";
import s from "./fleet.module.css";

const classFor: Record<Segment["kind"], string> = {
  weights: s.weights,
  cache: s.cache,
  small: s.small,
  free: "",
};

function describe(seg: Segment): string {
  switch (seg.kind) {
    case "weights":
      return "Model weights";
    case "cache":
      return `Context ${seg.id}: ${seg.rows?.toLocaleString() ?? "?"} rows, ${seg.features ?? "?"} features, tenant ${seg.tenant ?? "?"}, ${gb(seg.bytes)}`;
    case "small":
      return `${seg.count} small contexts, ${gb(seg.bytes)} together`;
    default:
      return "Free";
  }
}

export function MemoryStrip({ gpu, widthShare }: { gpu: GpuFrame; widthShare: number }) {
  const { total, segments } = layout(gpu, units.bytesPerCell, units.weightsBytes);
  const [hover, setHover] = useState<{ text: string; left: number } | null>(null);
  const cells: { cls: string; seg: Segment; start: number }[] = [];
  let at = 0;
  for (const seg of segments) {
    const cls = seg.kind === "cache" && seg.alt ? s.cacheAlt : classFor[seg.kind];
    for (let i = 0; i < seg.cells; i += 1) cells.push({ cls, seg, start: at });
    at += seg.cells;
  }
  const pct = (bytes: number) => `${Math.min(100, (bytes / gpu.hbm_bytes) * 100)}%`;
  const resident = residentBytes(gpu);
  return (
    <>
      <div className={s.track} style={{ width: `${widthShare * 100}%` }}>
        <div
          className={s.strip}
          style={{ gridTemplateColumns: `repeat(${total}, 1fr)` }}
          role="img"
          aria-label={`${gpu.name}: ${gb(resident)} of caches resident of ${gb(gpu.hbm_bytes)}`}
          onMouseLeave={() => setHover(null)}
        >
          {cells.map((c, i) => (
            <div
              // biome-ignore lint/suspicious/noArrayIndexKey: a cell is its position on the card
              key={i}
              className={`${s.cell} ${c.cls}`}
              onMouseEnter={() =>
                setHover(
                  c.seg.kind === "free" ? null : { text: describe(c.seg), left: (c.start / total) * 100 },
                )
              }
            />
          ))}
        </div>
        <span
          className={s.limit}
          style={{ left: pct(gpu.true_capacity) }}
          title="What the card really holds"
        />
        <span
          className={s.plan}
          style={{ left: pct(gpu.capacity_estimate) }}
          title="What the plane plans on"
        />
        <AnimatePresence>
          {hover ? (
            <motion.div
              key={hover.text}
              className={s.tip}
              style={{ left: `${Math.min(hover.left, 55)}%` }}
              initial={{ opacity: 0, y: -4 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.12 }}
            >
              {hover.text}
            </motion.div>
          ) : null}
        </AnimatePresence>
      </div>
      <div className={s.side}>
        <strong>{gb(resident)}</strong> in {gpu.caches.length}{" "}
        {gpu.caches.length === 1 ? "context" : "contexts"}
        <br />
        {gpu.host_caches > 0 ? `${gpu.host_caches} in host RAM (${gb(gpu.host_bytes)})` : "none in host RAM"}
        {gpu.queued > 0 ? `, ${gpu.queued} queued` : ""}
      </div>
    </>
  );
}
