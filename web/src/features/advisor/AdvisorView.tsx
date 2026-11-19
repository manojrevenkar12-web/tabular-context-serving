import type { Advisor } from "../../shared/data/types";
import { useJson } from "../../shared/data/useJson";
import { Panel } from "../../shared/ui/Panel";
import { EmptyState, ErrorState, Loading } from "../../shared/ui/States";
import s from "../designs/designs.module.css";

const label = (k: string, c: number) => `${c} x ${k.toUpperCase()}`;

export function AdvisorView() {
  const [load, retry] = useJson<Advisor>("data/advisor.json");
  return (
    <Panel
      id="advisor"
      title="Sizing a customer's own pool"
      lead="For VPC and self-hosted customers: the advisor replays a customer's request log through this plane on each candidate pool and picks the cheapest that meets the target. Each pick was then run in the simulated world to see what really happens."
    >
      {load.state === "loading" ? <Loading rows={5} height={24} /> : null}
      {load.state === "error" ? (
        <ErrorState what="the advisor results" message={load.message} onRetry={retry} />
      ) : null}
      {load.state === "ready" && load.data.profiles.length === 0 ? (
        <EmptyState title="No customer workloads scored" text="Run python3 -m evals.advisor_eval." />
      ) : null}
      {load.state === "ready" && load.data.profiles.length > 0 ? (
        <div className={s.wrap}>
          <table className={s.table}>
            <thead>
              <tr>
                <th scope="col">Workload</th>
                <th scope="col">Advisor's pick</th>
                <th scope="col">Really in SLO</th>
                <th scope="col">Cheapest pool that really meets it</th>
                <th scope="col">Two H100s, really in SLO</th>
              </tr>
            </thead>
            <tbody>
              {load.data.profiles.map((p) => {
                const a = p.choices.advisor;
                const h = p.choices.two_h100;
                return (
                  <tr key={p.profile}>
                    <td>
                      {p.profile.replaceAll("_", " ")}
                      <span className={s.what}>{p.requests.toLocaleString()} requests in two hours</span>
                    </td>
                    <td>
                      {label(a.kind, a.count)}, EUR {a.eur_per_hour.toFixed(2)}/h
                    </td>
                    <td className={a.meets_target ? s.pass : s.fail}>{(a.true_slo * 100).toFixed(1)}%</td>
                    <td>
                      {p.oracle
                        ? `${label(p.oracle.kind, p.oracle.count)}, EUR ${p.oracle.eur_per_hour.toFixed(2)}/h`
                        : "none in range"}
                    </td>
                    <td className={h.meets_target ? s.pass : s.fail}>{(h.true_slo * 100).toFixed(1)}%</td>
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
