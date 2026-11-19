import type { ReactNode } from "react";
import s from "./ui.module.css";

export function Panel({
  title,
  lead,
  children,
  id,
}: { title: string; lead?: string; children: ReactNode; id?: string }) {
  return (
    <section className={s.panel} aria-labelledby={id}>
      <h2 className={s.panelTitle} id={id}>
        {title}
      </h2>
      {lead ? <p className={s.panelLead}>{lead}</p> : null}
      {children}
    </section>
  );
}
