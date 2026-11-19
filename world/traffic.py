"""The traffic: tenants upload a dataset, then send predicts against it for a while.

Context sizes are heavy-tailed (most small, a few at a million rows), a fraction of predicts are interactive (the
client waits), one tenant floods in bursts, some clients resend a request with the same idempotency key, and a few
tenants ask for their data to be erased and later still send requests for it. Each request carries the latency the
client expects (its SLO), which the platform sees as a hint and the scoring uses as the bar.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from world.luck import stream

FLOOD_TENANT = "t00"  # the tenant that floods in bursts (when the config gives it a share)


@dataclass(frozen=True)
class Upload:
    t: float
    tenant: str
    digest: str
    rows: int
    features: int
    nbytes: float


@dataclass(frozen=True)
class Predict:
    t: float
    request_id: str
    idempotency_key: str
    tenant: str
    digest: str
    rows: int
    features: int
    test_rows: int
    interactive: bool
    slo_s: float
    after_erasure: bool = False
    is_retry: bool = False


@dataclass(frozen=True)
class Erase:
    t: float
    tenant: str


def slo(cfg: dict, rows: int, test_rows: int, interactive: bool) -> float:
    s = cfg["slo"]
    if interactive:
        return s["interactive_s"] + s["interactive_s_per_100k_train_rows"] * rows / 1e5
    return s["batch_s_base"] + s["batch_s_per_k_test_rows"] * test_rows / 1000 + s["batch_s_per_100k_train_rows"] * rows / 1e5


def _log_uniform(rng, lo: float, hi: float) -> float:
    return math.exp(rng.uniform(math.log(lo), math.log(hi)))


def generate(cfg: dict, seed: int) -> list:
    tc = cfg["traffic"]
    horizon = float(cfg["campaign"]["hours"]) * 3600.0
    rng = stream(seed, "sessions")
    events: list = []
    tenants = [f"t{i:02d}" for i in range(int(tc["tenants"]))]
    flood = FLOOD_TENANT
    rate = tc["sessions_per_hour"] / 3600.0
    t = 0.0
    bursts = []
    brng = stream(seed, "bursts")
    while len(bursts) < 3:
        b0 = brng.uniform(0.1, 0.85) * horizon
        bursts.append((b0, b0 + tc["flood_burst_minutes"] * 60.0))
    flood_rate = rate * tc["flood_tenant_share"] * horizon / sum(b1 - b0 for b0, b1 in bursts)
    if flood_rate <= 0:
        bursts = []
    starts: list[tuple[float, str]] = []
    while True:
        t += rng.exponential(1.0 / rate)
        if t > horizon:
            break
        starts.append((t, tenants[1 + int(rng.integers(len(tenants) - 1))]))
    frng = stream(seed, "flood")
    for b0, b1 in bursts:
        t = b0
        while True:
            t += frng.exponential(1.0 / flood_rate)
            if t > b1:
                break
            starts.append((t, flood))
    starts.sort()
    for session, (t0, tenant) in enumerate(starts):
        srng = stream(seed, "session", session)
        u = srng.random()
        rows = int(min(tc["rows_max"], tc["rows_min"] * u ** (-1.0 / tc["rows_pareto_alpha"])))
        features = int(round(_log_uniform(srng, tc["features_min"], tc["features_max"])))
        digest = f"{int(srng.integers(2**62)):016x}"
        events.append(Upload(t0, tenant, digest, rows, features, rows * features * 4.0))
        flooding = tenant == flood
        mean = tc["predicts_per_session_mean"] * (tc["flood_predicts_multiplier"] if flooding else 1.0)
        n_pred = 1 + int(srng.geometric(1.0 / mean))
        span = srng.exponential(tc["session_minutes_mean"] * 60.0)
        times = sorted(t0 + 1.0 + srng.random(n_pred) * span)
        for k, tp in enumerate(times):
            if tp > horizon:
                break
            interactive = srng.random() < tc["interactive_share"] and not flooding
            m = int(_log_uniform(srng, 1, 500)) if interactive else int(_log_uniform(srng, 500, tc["test_rows_max"]))
            rid = f"r{session:05d}-{k:02d}"
            p = Predict(tp, rid, rid, tenant, digest, rows, features, m, interactive, slo(cfg, rows, m, interactive))
            events.append(p)
            if srng.random() < tc["retry_share"]:
                events.append(Predict(**{**p.__dict__, "t": tp + tc["retry_delay_s"], "request_id": rid + "-retry", "is_retry": True}))
    erng = stream(seed, "erasure")
    pool = sorted({e.tenant for e in events if isinstance(e, Upload) and e.tenant != flood})
    chosen = [pool[int(i)] for i in erng.choice(len(pool), size=int(tc["erasure_tenants"]), replace=False)]
    out = []
    for tenant in chosen:
        te = erng.uniform(0.3, 0.75) * horizon
        out.append(Erase(te, tenant))
        theirs = [e for e in events if isinstance(e, Upload) and e.tenant == tenant and e.t < te]
        later = [e for e in events if isinstance(e, Predict) and e.tenant == tenant and e.t > te]
        for e in later:  # the tenant's later work is gone with the erasure; only the stale clients below remain
            events.remove(e)
        events[:] = [e for e in events if not (isinstance(e, Upload) and e.tenant == tenant and e.t > te)]
        for k in range(int(tc["after_erasure_requests"])):
            if not theirs:
                break
            u = theirs[int(erng.integers(len(theirs)))]
            tp = te + erng.uniform(30.0, 1800.0)
            rid = f"e-{tenant}-{k}"
            out.append(Predict(tp, rid, rid, tenant, u.digest, u.rows, u.features, 10, True, slo(cfg, u.rows, 10, True), True))
    events += out
    events.sort(key=lambda e: (e.t, type(e).__name__))
    return [e for e in events if e.t <= horizon]
