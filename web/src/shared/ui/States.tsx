import s from "./ui.module.css";

export function Skeleton({ height, width = "100%" }: { height: number; width?: string }) {
  return <div className={s.skeleton} style={{ height, width }} aria-hidden="true" />;
}

export function Loading({ rows = 4, height = 18 }: { rows?: number; height?: number }) {
  return (
    <div aria-busy="true" aria-label="Loading">
      {Array.from({ length: rows }, (_, i) => (
        // biome-ignore lint/suspicious/noArrayIndexKey: placeholders have no identity
        <Skeleton key={i} height={height} />
      ))}
    </div>
  );
}

export function ErrorState({
  what,
  message,
  onRetry,
}: { what: string; message: string; onRetry: () => void }) {
  return (
    <div className={s.state} role="alert">
      <p className={s.stateTitle}>Could not load {what}</p>
      <p className={s.stateText}>
        {message}. The console reads the files <code>python3 -m evals.console_data</code> writes to
        web/public/data; generate them, then load again.
      </p>
      <button type="button" className={s.button} onClick={onRetry}>
        Load again
      </button>
    </div>
  );
}

export function EmptyState({ title, text }: { title: string; text: string }) {
  return (
    <div className={s.state}>
      <p className={s.stateTitle}>{title}</p>
      <p className={s.stateText}>{text}</p>
    </div>
  );
}
