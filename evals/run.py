"""Run every design on the same seeds and write reports/dev.json (or, once, reports/heldout.json).

    python3 -m evals.run                    # dev campaign -> reports/dev.json
    python3 -m evals.run --hours 1          # quick look, nothing written
    python3 -m evals.run --heldout          # once, behind reports/heldout.lock -> reports/heldout.json

Same luck: every design runs on the same seeds; traffic and job noise come from streams keyed by the seed and by
what the job is, so the designs differ only in what they decide.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import tomllib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from evals.operate import operate, world_config
from evals.truth import score
from tcs import config

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365}


def commit() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def _one(args: tuple) -> dict:
    design, world, seed, hours = args
    t0 = time.time()
    s = score(operate(design, world, seed, hours))
    s.update(design=design, seed=seed, seconds=round(time.time() - t0, 1))
    return {k: (float(v) if isinstance(v, np.floating) else v) for k, v in s.items()}


def ci(xs: list[float]) -> dict:
    xs = [x for x in xs if x is not None]
    if not xs:
        return {"mean": None, "ci95": None, "n": 0}
    m = float(np.mean(xs))
    if len(xs) < 2:
        return {"mean": m, "ci95": None, "n": len(xs)}
    half = T95.get(len(xs) - 1, 1.96) * float(np.std(xs, ddof=1)) / len(xs) ** 0.5
    return {"mean": m, "ci95": half, "n": len(xs)}


def aggregate(runs: list[dict]) -> dict:
    keys = [k for k in runs[0] if k not in ("design", "seed", "seconds")]
    return {k: ci([r[k] for r in runs]) for k in keys}


def paired(a: list[dict], b: list[dict], key: str) -> dict:
    by = {r["seed"]: r[key] for r in b}
    return ci([r[key] - by[r["seed"]] for r in a if r[key] is not None and by.get(r["seed"]) is not None])


def gates(summary: dict, runs: dict) -> list[dict]:
    g = tomllib.loads((ROOT / "config" / "gates.toml").read_text())["gates"]
    p = {k: v["mean"] for k, v in summary["platform"].items()}
    llm = {k: v["mean"] for k, v in summary["llm_style"].items()}
    fifo = {k: v["mean"] for k, v in summary["fifo_no_quota"].items()}
    per_1000 = p["oom_events"] / max(1.0, p["requests"]) * 1000
    out = [
        ("slo_attainment", p["slo_attainment"], p["slo_attainment"] >= g["min_slo_attainment"]),
        (
            "slo_gain_over_llm_style",
            p["slo_attainment"] - llm["slo_attainment"],
            p["slo_attainment"] - llm["slo_attainment"] >= g["min_slo_gain_over_llm_style"],
        ),
        ("oom_per_1000_requests", per_1000, per_1000 <= g["max_oom_per_1000_requests"]),
        (
            "cost_share_of_llm_style",
            p["eur_per_million_rows"] / llm["eur_per_million_rows"],
            p["eur_per_million_rows"] / llm["eur_per_million_rows"] <= g["max_cost_share_of_llm_style"],
        ),
        ("double_charges", max(r["double_charges"] for r in runs["platform"]), max(r["double_charges"] for r in runs["platform"]) <= g["max_double_charges"]),
        (
            "served_after_erasure",
            max(r["served_after_erasure"] for r in runs["platform"]),
            max(r["served_after_erasure"] for r in runs["platform"]) <= g["max_served_after_erasure"],
        ),
        (
            "erasure_residual_bytes",
            max(r["erasure_residual_bytes"] for r in runs["platform"]),
            max(r["erasure_residual_bytes"] for r in runs["platform"]) <= g["max_erasure_residual_bytes"],
        ),
        (
            "flood_p95_share_of_fifo",
            p["interactive_p95_during_flood_s"] / fifo["interactive_p95_during_flood_s"],
            p["interactive_p95_during_flood_s"] / fifo["interactive_p95_during_flood_s"] <= g["max_flood_p95_share_of_fifo"],
        ),
    ]
    return [{"gate": n, "value": float(v), "pass": bool(ok)} for n, v, ok in out]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--heldout", action="store_true")
    ap.add_argument("--hours", type=float, default=None)
    ap.add_argument("--designs", default="")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    args = ap.parse_args(argv)
    world = "heldout" if args.heldout else "world"
    wcfg = world_config(world)
    seeds = [int(s) for s in wcfg["campaign"]["dev_seeds"]]
    names = [n for n in config.designs() if not args.designs or n in args.designs.split(",")]
    if args.heldout:
        if (REPORTS / "heldout.json").exists():
            print("reports/heldout.json exists: the held-out campaign runs once. See docs/heldout.md.")
            return 2
        dirty = subprocess.run(["git", "status", "--porcelain", "tcs", "config", "evals", "world"], cwd=ROOT, capture_output=True, text=True).stdout
        if dirty.strip():
            print("commit the platform before the held-out run; uncommitted changes:\n" + dirty)
            return 2
        REPORTS.mkdir(exist_ok=True)
        (REPORTS / "heldout.lock").write_text(json.dumps({"commit": commit(), "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=2) + "\n")
    jobs = [(n, world, s, args.hours) for n in names for s in seeds]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(_one, jobs))
    runs: dict[str, list[dict]] = {}
    for r in rows:
        runs.setdefault(r["design"], []).append(r)
    summary = {n: aggregate(runs[n]) for n in names}
    full = {"platform", "llm_style", "fifo_no_quota"} <= set(names)
    report = {
        "world": world,
        "seeds": seeds,
        "hours": args.hours or float(wcfg["campaign"]["hours"]),
        "commit": commit(),
        "designs": {n: {"description": config.design(n).description, "summary": summary[n], "runs": runs[n]} for n in names},
        "paired_vs_baseline": {
            n: {k: paired(runs[n], runs["baseline"], k) for k in ("slo_attainment", "eur_per_million_rows", "oom_events")}
            for n in names
            if n != "baseline" and "baseline" in runs
        },
        "gates": gates(summary, runs) if full else [],
        "seconds": round(time.time() - t0, 1),
    }
    for n in names:
        s = summary[n]
        print(
            f"{n:18s} slo {s['slo_attainment']['mean']:.3f}  int p95 {s['interactive_p95_s']['mean'] or float('nan'):8.2f}s  "
            f"oom {s['oom_events']['mean']:8.1f}  EUR/M rows {s['eur_per_million_rows']['mean']:.3f}  hit {s['cache_hit_rate']['mean']:.3f}"
        )
    for g in report["gates"]:
        print(f"gate {g['gate']:28s} {g['value']:.4f} {'pass' if g['pass'] else 'FAIL'}")
    if args.hours is None or args.heldout:
        REPORTS.mkdir(exist_ok=True)
        out = REPORTS / ("heldout.json" if args.heldout else "dev.json")
        out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
        print("wrote", out.relative_to(ROOT))
    return 0 if all(g["pass"] for g in report["gates"]) or args.heldout else 1


if __name__ == "__main__":
    raise SystemExit(main())
