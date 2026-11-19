import type { Designs } from "../../shared/data/types";
import { useJson } from "../../shared/data/useJson";
import { Panel } from "../../shared/ui/Panel";
import { EmptyState, ErrorState, Loading } from "../../shared/ui/States";
import s from "./designs.module.css";

const ORDER = [
  "platform",
  "baseline",
  "llm_style",
  "no_cache_routing",
  "fixed_chunks",
  "no_learning",
  "no_batching",
  "fifo_no_quota",
  "lru_no_host",
];

function n(v: number | null | undefined, digits = 0): string {
  return v === null || v === undefined
    ? "n/a"
    : v.toLocaleString(undefined, { maximumFractionDigits: digits, minimumFractionDigits: digits });
}

function Table({ data }: { data: Designs }) {
  const rows = [...data.designs].sort((a, b) => ORDER.indexOf(a.name) - ORDER.indexOf(b.name));
  return (
    <>
      <div className={s.wrap}>
        <table className={s.table}>
          <thead>
            <tr>
              <th scope="col">Design</th>
              <th scope="col">Requests in SLO</th>
              <th scope="col">Interactive p95</th>
              <th scope="col">Out-of-memory errors a day</th>
              <th scope="col">EUR per million rows</th>
              <th scope="col">Cache hit rate</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((d) => {
              const slo = d.summary.slo_attainment ?? 0;
              return (
                <tr key={d.name} className={d.name === "platform" ? s.mine : undefined}>
                  <td>
                    {d.name.replaceAll("_", " ")}
                    <span className={s.what}>{d.description}</span>
                  </td>
                  <td>
                    {n(slo * 100, 1)}%
                    <span
                      className={s.bar}
                      style={{ width: `${Math.max(0, (slo - 0.4) / 0.6) * 64}px` }}
                      aria-hidden="true"
                    />
                  </td>
                  <td>{n(d.summary.interactive_p95_s, 1)} s</td>
                  <td>{n(d.summary.oom_events)}</td>
                  <td>{n(d.summary.eur_per_million_rows, 2)}</td>
                  <td>{n((d.summary.cache_hit_rate ?? 0) * 100, 0)}%</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <div className={s.gates}>
        {data.gates.map((g) => (
          <span key={g.gate}>
            {g.gate.replaceAll("_", " ")}:{" "}
            <span className={g.pass ? s.pass : s.fail}>{g.pass ? "pass" : "fail"}</span>
          </span>
        ))}
      </div>
    </>
  );
}

export function DesignsView() {
  const [load, retry] = useJson<Designs>("data/designs.json");
  return (
    <Panel
      id="designs"
      title="Designs on the same traffic"
      lead={
        load.state === "ready"
          ? `Mean over ${load.data.seeds.length} seeds of ${load.data.hours} simulated hours each. Every design sees the same requests and the same GPUs; each ablation switches off one part of the plane.`
          : undefined
      }
    >
      {load.state === "loading" ? <Loading rows={9} height={28} /> : null}
      {load.state === "error" ? (
        <ErrorState what="the design comparison" message={load.message} onRetry={retry} />
      ) : null}
      {load.state === "ready" && load.data.designs.length === 0 ? (
        <EmptyState
          title="No designs compared yet"
          text="Run python3 -m evals.run, then python3 -m evals.console_data."
        />
      ) : null}
      {load.state === "ready" && load.data.designs.length > 0 ? <Table data={load.data} /> : null}
    </Panel>
  );
}
