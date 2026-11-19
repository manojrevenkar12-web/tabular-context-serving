"""Measure the real PyTorch model on this machine's CPU and write reports/bench.json.

    python3 scripts/bench.py

* exactness: cached and chunked predictions against the uncached, unchunked forward pass;
* the cache: exact bytes per training row, fit time against predict-from-cache time;
* chunking: peak resident memory of a fit with whole-context attention against 512-row chunks (fresh process each);
* accuracy: the small model against logistic regression and k-nearest neighbours, to show it really learns in
  the forward pass (it is a toy next to TabPFN, and the numbers say so).
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tcs.model.icl import fit, kv_bytes_per_row, load, predict_proba, predict_reference  # noqa: E402
from tcs.model.prior import task  # noqa: E402

MODEL = str(ROOT / "models" / "tiny_icl.pt")


def peak_child(rows: int, chunk: int) -> float:
    """Peak growth of resident memory during one fit, sampled from /proc every millisecond in a fresh process."""
    code = """
import os, sys, threading, time
import numpy as np
sys.path.insert(0, %r)
from tcs.model.icl import load, fit
from tcs.model.prior import task
page = os.sysconf("SC_PAGE_SIZE")
def rss():
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * page
m = load(%r)
x, y = task(np.random.default_rng(0), %d, 16, 3)
fit(m, x[:64], y[:64], train_chunk=64)
base, peak, done = rss(), [0], [False]
def watch():
    while not done[0]:
        peak[0] = max(peak[0], rss())
        time.sleep(0.001)
th = threading.Thread(target=watch); th.start()
fit(m, x, y, train_chunk=%d)
done[0] = True; th.join()
print(max(0, peak[0] - base))
""" % (str(ROOT), MODEL, rows, chunk)  # noqa: UP031
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    return float(out.strip())


def timed(f, repeat: int = 3) -> float:
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        f()
        best = min(best, time.perf_counter() - t0)
    return best


def accuracy() -> dict:
    from sklearn.datasets import load_breast_cancer, load_iris, load_wine
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    m = load(MODEL)
    sets = {"iris": load_iris(return_X_y=True), "wine": load_wine(return_X_y=True), "breast_cancer": load_breast_cancer(return_X_y=True)}
    rng = np.random.default_rng(7)
    for i in range(5):
        x, y = task(rng, 400, int(rng.integers(4, 30)), int(rng.integers(2, 5)))
        sets[f"prior_task_{i}"] = (np.nan_to_num(x), y)
    out = {}
    for name, (x, y) in sets.items():
        scores = {"tiny_icl": [], "logistic_regression": [], "knn_5": []}
        for split in range(5):
            xtr, xte, ytr, yte = train_test_split(x, y, test_size=0.3, random_state=split, stratify=y)
            p = predict_proba(m, fit(m, xtr, ytr, 1024), xte, 1024)
            classes = np.unique(ytr)
            scores["tiny_icl"].append(float(np.mean(classes[p.argmax(1)] == yte)))
            for key, est in (("logistic_regression", LogisticRegression(max_iter=2000)), ("knn_5", KNeighborsClassifier(5))):
                scores[key].append(float(make_pipeline(StandardScaler(), est).fit(xtr, ytr).score(xte, yte)))
        out[name] = {k: round(float(np.mean(v)), 4) for k, v in scores.items()}
    return out


def main() -> None:
    m = load(MODEL)
    rng = np.random.default_rng(11)
    worst_cache, worst_chunk = 0.0, 0.0
    for _ in range(6):
        x, y = task(rng, 700, int(rng.integers(3, 60)), int(rng.integers(2, 6)))
        ref = predict_reference(m, x[:600], y[:600], x[600:])
        ctx = fit(m, x[:600], y[:600], train_chunk=10**9)
        worst_cache = max(worst_cache, float(np.abs(predict_proba(m, ctx, x[600:], 10**9) - ref).max()))
        for tc, sc in ((1, 1), (7, 13), (128, 64)):
            c = fit(m, x[:600], y[:600], train_chunk=tc)
            worst_chunk = max(worst_chunk, float(np.abs(predict_proba(m, c, x[600:], sc) - ref).max()))
    speed = []
    for rows in (1000, 4000, 16000):
        x, y = task(np.random.default_rng(rows), rows + 100, 20, 3)
        ctx = fit(m, x[:rows], y[:rows], 1024)
        t_fit = timed(lambda x=x, y=y, rows=rows: fit(m, x[:rows], y[:rows], 1024))
        t_pred = timed(lambda x=x, ctx=ctx, rows=rows: predict_proba(m, ctx, x[rows:], 1024))
        speed.append(
            {
                "train_rows": rows,
                "test_rows": 100,
                "fit_and_predict_s": round(t_fit + t_pred, 4),
                "predict_from_cache_s": round(t_pred, 4),
                "speedup": round((t_fit + t_pred) / t_pred, 1),
                "cache_bytes": ctx.kv_bytes,
            }
        )
    rows = 8000
    mem = {"train_rows": rows, "whole_context_peak_bytes": peak_child(rows, 10**9), "chunked_512_peak_bytes": peak_child(rows, 512)}
    mem["reduction"] = round(mem["whole_context_peak_bytes"] / max(1.0, mem["chunked_512_peak_bytes"]), 1)
    report = {
        "machine": {"cpu": platform.processor() or platform.machine(), "python": platform.python_version()},
        "parameters": sum(p.numel() for p in m.parameters()),
        "kv_bytes_per_row": kv_bytes_per_row(m.spec),
        "max_abs_diff_cached_vs_reference": worst_cache,
        "max_abs_diff_chunked_vs_reference": worst_chunk,
        "speed": speed,
        "memory": mem,
        "accuracy": accuracy(),
    }
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "bench.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "accuracy"}, indent=1))
    print(json.dumps(report["accuracy"], indent=1))


if __name__ == "__main__":
    main()
