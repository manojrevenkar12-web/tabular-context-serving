import { AdvisorView } from "../features/advisor/AdvisorView";
import { DesignsView } from "../features/designs/DesignsView";
import { ErasureView } from "../features/erasure/ErasureView";
import { FleetView } from "../features/fleet/FleetView";
import s from "./app.module.css";

export function App() {
  return (
    <main className={s.page}>
      <header className={s.head}>
        <div>
          <h1 className={s.title}>tabular-context-serving</h1>
          <p className={s.sub}>
            Where each in-context request runs, with what chunks, and which contexts stay in GPU memory for
            the next predict.
          </p>
        </div>
        <nav className={s.nav} aria-label="Sections">
          <a href="#fleet">Fleet</a>
          <a href="#designs">Designs</a>
          <a href="#erasure">Erasure</a>
          <a href="#advisor">Sizing</a>
        </nav>
      </header>
      <FleetView />
      <DesignsView />
      <ErasureView />
      <AdvisorView />
      <p className={s.foot}>
        Generated from reports/ by python3 -m evals.console_data. The fleet and the traffic are simulated; the
        plane is the code that serves the API.
      </p>
    </main>
  );
}
