import { clock, gb } from "../../shared/data/strip";
import type { FleetDay } from "../../shared/data/types";
import { useJson } from "../../shared/data/useJson";
import { Panel } from "../../shared/ui/Panel";
import { EmptyState, ErrorState, Loading } from "../../shared/ui/States";
import s from "../designs/designs.module.css";

export function ErasureView() {
  const [load, retry] = useJson<FleetDay>("data/fleet.json");
  return (
    <Panel
      id="erasure"
      title="Erasure certificates"
      lead="When a tenant asks to be erased the plane deletes its objects, drops its queued work, purges its caches from every GPU and host tier, then asks every worker what it still holds. Fifteen minutes later the simulation audits the bucket and every card."
    >
      {load.state === "loading" ? <Loading rows={3} height={24} /> : null}
      {load.state === "error" ? (
        <ErrorState what="the erasures" message={load.message} onRetry={retry} />
      ) : null}
      {load.state === "ready" && load.data.erasures.length === 0 ? (
        <EmptyState title="No erasures that day" text="Tenants asked for nothing to be erased in this run." />
      ) : null}
      {load.state === "ready" && load.data.erasures.length > 0 ? (
        <div className={s.wrap}>
          <table className={s.table}>
            <thead>
              <tr>
                <th scope="col">Tenant</th>
                <th scope="col">Asked at</th>
                <th scope="col">Datasets</th>
                <th scope="col">Verified with workers</th>
                <th scope="col">Left after 15 minutes</th>
              </tr>
            </thead>
            <tbody>
              {load.data.erasures.map((e) => {
                const audit = load.data.audits.find((a) => a.tenant === e.tenant);
                return (
                  <tr key={e.tenant}>
                    <td>{e.tenant}</td>
                    <td>{clock(e.requested_s)}</td>
                    <td>{e.datasets}</td>
                    <td>{e.verified ? `yes, after ${(e.seconds ?? 0).toFixed(1)} s` : "not yet"}</td>
                    <td>
                      {audit
                        ? audit.residual_bytes > 0
                          ? gb(audit.residual_bytes)
                          : "nothing"
                        : "not audited"}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      ) : null}
    </Panel>
  );
}
