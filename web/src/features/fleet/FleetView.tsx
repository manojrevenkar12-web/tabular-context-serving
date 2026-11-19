import { motion, useReducedMotion } from "framer-motion";
import { useEffect, useMemo, useState } from "react";
import { clock, residentBytes } from "../../shared/data/strip";
import type { FleetDay } from "../../shared/data/types";
import { useJson } from "../../shared/data/useJson";
import { motion as m } from "../../shared/theme/tokens";
import { Panel } from "../../shared/ui/Panel";
import { EmptyState, ErrorState, Loading } from "../../shared/ui/States";
import uis from "../../shared/ui/ui.module.css";
import { MemoryStrip } from "./MemoryStrip";
import s from "./fleet.module.css";

function busiest(day: FleetDay): number {
  let best = 0;
  let most = -1;
  day.frames.forEach((f, i) => {
    const b = f.gpus.reduce((a, g) => a + residentBytes(g), 0);
    if (b > most) {
      most = b;
      best = i;
    }
  });
  return best;
}

function Fleet({ day }: { day: FleetDay }) {
  const [index, setIndex] = useState(() => busiest(day));
  const [playing, setPlaying] = useState(false);
  const reduce = useReducedMotion();
  useEffect(() => {
    if (!playing) return;
    const id = window.setInterval(() => setIndex((i) => (i + 1) % day.frames.length), m.playStepMs);
    return () => window.clearInterval(id);
  }, [playing, day.frames.length]);
  const frame = day.frames[index];
  const win = day.windows[index];
  const widest = useMemo(() => Math.max(...day.frames[0].gpus.map((g) => g.hbm_bytes)), [day]);
  return (
    <>
      <div className={s.controls}>
        <span className={s.time} aria-live="polite">
          {clock(frame.t)}
        </span>
        <span className={s.timeNote}>into the simulated day</span>
        <input
          className={s.range}
          type="range"
          min={0}
          max={day.frames.length - 1}
          value={index}
          onChange={(e) => {
            setPlaying(false);
            setIndex(Number(e.target.value));
          }}
          aria-label="Time of day"
        />
        <button type="button" className={uis.button} onClick={() => setPlaying((p) => !p)}>
          {playing ? "Pause" : "Play the day"}
        </button>
      </div>
      <motion.div
        className={s.rows}
        initial="hidden"
        animate="shown"
        variants={{ shown: { transition: { staggerChildren: reduce ? 0 : m.revealStaggerS } } }}
      >
        {frame.gpus.map((g) => (
          <motion.div
            key={g.name}
            className={s.row}
            variants={{ hidden: { opacity: 0, x: reduce ? 0 : -12 }, shown: { opacity: 1, x: 0 } }}
            transition={{ duration: reduce ? 0 : m.revealS, ease: m.ease }}
          >
            <span className={s.name}>
              {g.name}
              <span className={s.kind}>
                {g.kind.toUpperCase()} {Math.round(g.hbm_bytes / 1e9)} GB{g.busy ? ", running" : ""}
              </span>
            </span>
            <MemoryStrip gpu={g} widthShare={g.hbm_bytes / widest} />
          </motion.div>
        ))}
      </motion.div>
      <div className={s.legend}>
        <span className={s.key}>
          <span className={s.swatch} style={{ background: "var(--weights)" }} /> Model weights
        </span>
        <span className={s.key}>
          <span className={s.swatch} style={{ background: "var(--cache)" }} /> A context's KV cache (1 cell =
          1 GB)
        </span>
        <span className={s.key}>
          <span className={s.swatch} style={{ background: "var(--cache-small)" }} /> Contexts under 1 GB
        </span>
        <span className={s.key}>
          <span className={s.swatch} style={{ background: "var(--limit-true)", width: 3 }} /> What the card
          really holds
        </span>
        <span className={s.key}>
          <span className={s.swatch} style={{ borderLeft: "2px dashed var(--limit-plan)", width: 3 }} /> What
          the plane plans on
        </span>
      </div>
      {win ? (
        <div className={s.stats}>
          <Stat value={win.requests.toLocaleString()} label="requests in these 10 minutes" />
          <Stat
            value={
              win.requests
                ? `${((win.in_slo / Math.max(1, win.requests - win.throttled)) * 100).toFixed(1)}%`
                : "none"
            }
            label="answered within SLO"
          />
          <Stat value={win.hits.toLocaleString()} label="answered from a cache" />
          <Stat value={win.fits.toLocaleString()} label="contexts fitted" />
          <Stat value={win.throttled.toLocaleString()} label="throttled by quota" />
          <Stat value={win.oom.toLocaleString()} label="out-of-memory errors" />
        </div>
      ) : null}
    </>
  );
}

function Stat({ value, label }: { value: string; label: string }) {
  return (
    <div className={s.stat}>
      <span className={s.statValue}>{value}</span>
      <span className={s.statLabel}>{label}</span>
    </div>
  );
}

export function FleetView() {
  const [load, retry] = useJson<FleetDay>("data/fleet.json");
  return (
    <Panel
      id="fleet"
      title="What each GPU holds"
      lead="One simulated day of the platform on eight GPUs. Each cell is a gigabyte of GPU memory; blue runs are contexts kept for their next predict. The plane plans on the dashed line and learns it from out-of-memory errors; the solid line is the truth it never sees."
    >
      {load.state === "loading" ? <Loading rows={8} /> : null}
      {load.state === "error" ? (
        <ErrorState what="the fleet day" message={load.message} onRetry={retry} />
      ) : null}
      {load.state === "ready" && load.data.frames.length === 0 ? (
        <EmptyState
          title="No frames in this day"
          text="Run python3 -m evals.console_data to record a day of the platform."
        />
      ) : null}
      {load.state === "ready" && load.data.frames.length > 0 ? <Fleet day={load.data} /> : null}
    </Panel>
  );
}
