"""Score one run against what really happened: the traffic as sent, the fleet's execution log, the bucket and caches.

The bar for each request is the latency its client expected (its SLO). A request counts once per idempotency key:
its retries are the same request. The flood tenant is scored apart, so throttling it is not hidden in the totals.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from world.traffic import FLOOD_TENANT, Erase, Predict


def _q(xs: list[float], q: float) -> float | None:
    return float(np.quantile(xs, q)) if xs else None


def score(run: dict) -> dict:
    events = run["events"]
    predicts = [e for e in events if isinstance(e, Predict)]
    flood = FLOOD_TENANT
    erased_at = {e.tenant: e.t for e in events if isinstance(e, Erase)}
    by_rid = {p.request_id: p for p in predicts}
    first: dict[str, object] = {}
    for o in run["outcomes"]:
        if o.request_id in by_rid and not by_rid[o.request_id].is_retry and o.request_id not in first:
            first[o.request_id] = o
    originals = [p for p in predicts if not p.is_retry and not p.after_erasure]
    normal = [p for p in originals if p.tenant != flood]
    flood_reqs = [p for p in originals if p.tenant == flood]

    def ok(p: Predict) -> bool:
        o = first.get(p.request_id)
        return o is not None and o.status == "ok"

    def in_slo(p: Predict) -> bool:
        o = first.get(p.request_id)
        return ok(p) and o.latency <= p.slo_s

    lat_i = [first[p.request_id].latency for p in normal if p.interactive and ok(p)]
    lat_b = [first[p.request_id].latency for p in normal if not p.interactive and ok(p)]
    statuses = Counter(first[p.request_id].status if p.request_id in first else "unfinished" for p in normal)

    log = run["log"]
    eur = {g: v["eur_per_hour"] / 3600.0 for g, v in run["gpus"].items()}
    busy_eur = sum(x.seconds * eur[x.gpu] for x in log)
    oom_eur = sum(x.seconds * eur[x.gpu] for x in log if x.oom)
    served_rows = sum(p.test_rows for p in originals if ok(p))
    retry_ids = {p.request_id for p in predicts if p.is_retry}
    dup_exec_s = sum(x.seconds for x in log if x.ok and x.requests and all(r in retry_ids for r in x.requests))
    key_of = {p.request_id: p.idempotency_key for p in predicts}
    charged = Counter(key_of.get(k, k) for k, _ in run["charges"])
    ok_exec = [x for x in log if x.ok]
    hits = sum(1 for x in ok_exec if x.kind in ("predict", "reload_predict"))

    after = []
    for o in run["outcomes"]:
        p = by_rid.get(o.request_id)
        if p is not None and p.tenant in erased_at and o.status == "ok" and o.finished > erased_at[p.tenant]:
            after.append(o.request_id)
    certs = run["certificates"]
    done = [c["completed"] - c["requested"] for c in certs if c.get("completed") is not None and c.get("verified")]

    in_burst_lat = []
    flood_times = sorted(p.t for p in flood_reqs)
    if flood_times:
        lo, hi = flood_times[0], flood_times[-1]
        in_burst_lat = [first[p.request_id].latency for p in normal if p.interactive and ok(p) and lo <= p.t <= hi]

    hours = run["hours"]
    return {
        "requests": len(normal),
        "served_share": sum(ok(p) for p in normal) / max(1, len(normal)),
        "slo_attainment": sum(in_slo(p) for p in normal) / max(1, len(normal)),
        "slo_attainment_interactive": sum(in_slo(p) for p in normal if p.interactive) / max(1, sum(p.interactive for p in normal)),
        "slo_attainment_batch": sum(in_slo(p) for p in normal if not p.interactive) / max(1, sum(not p.interactive for p in normal)),
        "interactive_p50_s": _q(lat_i, 0.5),
        "interactive_p95_s": _q(lat_i, 0.95),
        "batch_p95_s": _q(lat_b, 0.95),
        "interactive_p95_during_flood_s": _q(in_burst_lat, 0.95),
        "failed": statuses.get("failed", 0) + statuses.get("too_large", 0) + statuses.get("unfinished", 0),
        "oom_events": sum(1 for x in log if x.oom),
        "oom_eur": oom_eur,
        "gpu_busy_eur_per_hour": busy_eur / hours,
        "eur_per_million_rows": busy_eur / max(1, served_rows) * 1e6,
        "served_rows": served_rows,
        "cache_hit_rate": hits / max(1, len(ok_exec)),
        "fits": sum(1 for x in ok_exec if x.kind == "fit_predict"),
        "duplicate_gpu_seconds": dup_exec_s,
        "double_charges": sum(1 for n in charged.values() if n > 1),
        "flood_throttled_share": sum(1 for p in flood_reqs if first.get(p.request_id) is not None and first[p.request_id].status == "throttled")
        / max(1, len(flood_reqs)),
        "served_after_erasure": len(after),
        "erasure_residual_bytes": sum(a["residual_bytes"] for a in run["audits"]),
        "erasures": len(certs),
        "erasure_verified": sum(1 for c in certs if c.get("verified")),
        "erasure_seconds_max": max(done) if done else None,
    }
