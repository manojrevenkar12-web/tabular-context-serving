import { useCallback, useEffect, useState } from "react";

export type Load<T> =
  | { state: "loading" }
  | { state: "error"; message: string }
  | { state: "ready"; data: T };

/** Fetch a JSON document with explicit loading and error states, and a retry. */
export function useJson<T>(url: string): [Load<T>, () => void] {
  const [load, setLoad] = useState<Load<T>>({ state: "loading" });
  const [attempt, setAttempt] = useState(0);
  const retry = useCallback(() => setAttempt((a) => a + 1), []);

  useEffect(() => {
    let live = true;
    setLoad({ state: "loading" });
    // attempt is a dependency so that retry fetches again
    void attempt;
    fetch(url)
      .then((r) => {
        if (!r.ok) throw new Error(`${url} answered ${r.status}`);
        return r.json() as Promise<T>;
      })
      .then((data) => live && setLoad({ state: "ready", data }))
      .catch((e: Error) => live && setLoad({ state: "error", message: e.message }));
    return () => {
      live = false;
    };
  }, [url, attempt]);

  return [load, retry];
}
