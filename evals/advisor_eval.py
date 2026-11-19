"""Score the sizing advisor against the truth: does its pick really meet the target, and what does it overpay?

For each customer profile the advisor sees only the request log. The truth runs every candidate pool in the
simulated world with the platform design and finds the cheapest pool that really meets the target. Two rules a
person might use instead are scored the same way: "two H100s" and "the cheapest card that holds the largest
context, one of it".

    python3 -m evals.advisor_eval        -> reports/advisor.json
"""

from __future__ import annotations

import copy
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from evals.operate import operate, world_config
from evals.truth import score
from tcs import config
from tcs.advisor.sizing import LoggedRequest, advise, calibration_from, check, draft, pool
from world.traffic import Predict, generate

ROOT = Path(__file__).resolve().parent.parent
PROFILES = [
    {"name": "small_interactive", "sessions_per_hour": 40.0, "rows_pareto_alpha": 1.2, "interactive_share": 0.9},
    {"name": "mixed", "sessions_per_hour": 90.0, "rows_pareto_alpha": 0.7, "interactive_share": 0.6},
    {"name": "large_contexts", "sessions_per_hour": 25.0, "rows_pareto_alpha": 0.45, "interactive_share": 0.4},
    {"name": "busy_batch", "sessions_per_hour": 160.0, "rows_pareto_alpha": 0.9, "interactive_share": 0.2},
    {"name": "busy_interactive", "sessions_per_hour": 200.0, "rows_pareto_alpha": 1.0, "interactive_share": 0.85},
]
KINDS = ["l4", "a100", "h100"]
MAX_COUNT = 6
TARGET = 0.97
SEED = 77


def customer_world(profile: dict) -> dict:
    w = copy.deepcopy(world_config())
    w["campaign"]["hours"] = 2.0
    t = w["traffic"]
    t.update(tenants=6, flood_tenant_share=0.0, erasure_tenants=0, retry_share=0.0)
    t.update({k: v for k, v in profile.items() if k != "name"})
    return w


def world_pool(w: dict, kind: str, count: int) -> list[dict]:
    template = next(g for g in w["gpu"] if g["kind"] == kind)
    return [dict(template, name=f"{kind}-{i}") for i in range(count)]


def truth(args: tuple) -> tuple:
    profile, kind, count = args
    w = customer_world(profile)
    w["gpu"] = world_pool(w, kind, count)
    p = copy.deepcopy(config.platform())
    p["fleet"] = pool(p, kind, count)
    s = score(operate("platform", "world", SEED, wcfg=w, pcfg=p))
    return profile["name"], kind, count, s["slo_attainment"], s["failed"]


def main() -> None:
    pcfg = config.platform()
    design = config.design("platform")
    grid = [(pr, k, c) for pr in PROFILES for k in KINDS for c in range(1, MAX_COUNT + 1)]
    with ProcessPoolExecutor() as ex:
        rows = list(ex.map(truth, grid))
    real = {(n, k, c): (float(slo), int(failed)) for n, k, c, slo, failed in rows}
    # What the managed API's plane has learned from its own telemetry (two hours of the dev world).
    cal = calibration_from(operate("platform", "world", 101, hours=2.0)["plane"])
    price = {k: next(g for g in pcfg["fleet"] if g["kind"] == k)["eur_per_hour"] for k in KINDS}
    out = []
    for pr in PROFILES:
        w = customer_world(pr)
        log = [LoggedRequest(e.t, e.rows, e.features, e.test_rows, e.interactive, e.slo_s, e.digest) for e in generate(w, SEED) if isinstance(e, Predict)]
        plan = advise(design, pcfg, log, TARGET, KINDS, MAX_COUNT, cal)
        text = draft(plan)
        feasible = [(price[k] * c, k, c) for k in KINDS for c in range(1, MAX_COUNT + 1) if real[(pr["name"], k, c)][0] >= TARGET]
        oracle = min(feasible) if feasible else None
        largest = max(e.rows * (4000 + 16 * e.features) for e in log) * 8
        holds = next(k for k in KINDS if next(g for g in pcfg["fleet"] if g["kind"] == k)["hbm_bytes"] * 0.8 > largest + 2.2e9) if log else "h100"
        rules = {"two_h100": ("h100", 2), "one_card_that_holds_the_largest": (holds, 1), "advisor": (plan.kind, plan.count)}
        entry = {
            "profile": pr["name"],
            "requests": len(log),
            "plan": {"kind": plan.kind, "count": plan.count, "eur_per_hour": plan.eur_per_hour, "predicted_slo": plan.slo_attainment},
            "draft": text,
            "draft_unbacked_numbers": check(text, plan),
            "oracle": {"kind": oracle[1], "count": oracle[2], "eur_per_hour": oracle[0]} if oracle else None,
            "choices": {},
        }
        for name, (k, c) in rules.items():
            slo, failed = real[(pr["name"], k, c)]
            entry["choices"][name] = {
                "kind": k,
                "count": c,
                "eur_per_hour": price[k] * c,
                "true_slo": slo,
                "meets_target": bool(slo >= TARGET),
                "overpay_eur_per_hour": (price[k] * c - oracle[0]) if oracle else None,
            }
        out.append(entry)
        print(pr["name"], {n: (v["kind"], v["count"], round(v["true_slo"], 3), v["meets_target"]) for n, v in entry["choices"].items()}, "oracle", oracle)
    summary = {}
    for name in ("advisor", "two_h100", "one_card_that_holds_the_largest"):
        cs = [e["choices"][name] for e in out]
        summary[name] = {
            "meets_target": sum(c["meets_target"] for c in cs),
            "profiles": len(cs),
            "mean_eur_per_hour": sum(c["eur_per_hour"] for c in cs) / len(cs),
            "mean_overpay_eur_per_hour": sum(c["overpay_eur_per_hour"] or 0 for c in cs) / len(cs),
        }
    report = {"target": TARGET, "seed": SEED, "profiles": out, "summary": summary, "truth_grid": [[n, k, c, float(slo), int(f)] for n, k, c, slo, f in rows]}
    (ROOT / "reports" / "advisor.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
