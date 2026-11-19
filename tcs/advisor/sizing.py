"""Sizing advisor for customer VPC and self-hosted deployments.

Customers running the open weights in their own cloud ask the same question before every deployment: which GPUs,
and how many, for our workload and our latency targets? The advisor answers from the customer's own request log
(context rows, features, test rows, timestamps, whether a person was waiting). It replays that log through this
plane, on each candidate pool, using only the plane's estimates, and recommends the cheapest pool that meets the
target. A model may write the recommendation for the customer; `check` refuses any number the plan does not hold.

The model reads, code decides: the pick is computed; the wording is drafted and checked.
"""

from __future__ import annotations

import copy
import heapq
import itertools
import math
import re
from dataclasses import dataclass

from tcs.config import Design
from tcs.plane.estimates import JobShape
from tcs.plane.plane import Plane, Request


@dataclass(frozen=True)
class LoggedRequest:
    t: float
    rows: int
    features: int
    test_rows: int
    interactive: bool
    slo_s: float
    context: str  # the customer's dataset id, as logged


class EstimateBackend:
    """A worker that behaves exactly as the plane believes a worker behaves."""

    def __init__(self, plane: Plane) -> None:
        self.plane = plane
        self.hbm: dict[str, dict[str, float]] = {}
        self.host: dict[str, dict[str, float]] = {}

    def run(self, gpu: str, job: dict) -> dict:
        g = self.plane.gpus[gpu]
        s = JobShape(job["kind"], job["rows"], job["features"], job["test_rows"], job["train_chunk"], job["test_chunk"])
        cache = self.plane.mem.cache_bytes(job["rows"], job["features"]) if job["kind"] == "fit_predict" else 0.0
        reload = self.host.get(gpu, {}).pop(job["digest"], 0.0) if job["kind"] == "reload_predict" else 0.0
        if job["kind"] == "reload_predict":
            cache = reload
        if job["kind"] != "predict" and job["keep_cache"] or job["kind"] == "reload_predict":
            self.hbm.setdefault(gpu, {})[job["digest"]] = cache
        seconds = self.plane.time.seconds(s, g.kind, g.speed, reload)
        return {
            "ok": True,
            "oom": False,
            "seconds": seconds,
            "peak_bytes": self.plane.mem.transient(s) + cache,
            "cache_bytes": cache or self.hbm.get(gpu, {}).get(job["digest"], 0.0),
            "error": "",
        }

    def evict(self, gpu: str, digest: str, to_host: bool) -> bool:
        b = self.hbm.get(gpu, {}).pop(digest, None)
        if b is not None and to_host:
            self.host.setdefault(gpu, {})[digest] = b
        return b is not None

    def purge(self, gpu: str, digest: str) -> None:
        self.hbm.get(gpu, {}).pop(digest, None)
        self.host.get(gpu, {}).pop(digest, None)

    def residents(self, gpu: str) -> dict:
        return {"hbm": dict(self.hbm.get(gpu, {})), "host": dict(self.host.get(gpu, {}))}

    def put(self, tenant: str, digest: str, nbytes: float) -> None: ...

    def delete(self, tenant: str, digest: str) -> None: ...


def pool(cfg: dict, kind: str, count: int) -> list[dict]:
    template = next(g for g in cfg["fleet"] if g["kind"] == kind)
    return [dict(template, name=f"{kind}-{i}") for i in range(count)]


def calibration_from(plane: Plane) -> dict:
    """What a serving plane has learned from its own telemetry, to plan a new pool with: the time corrections per
    GPU kind and the memory model's measured/predicted samples. Usable capacity is left out on purpose: it is learned
    per card from that card's own errors, and a busy managed fleet's estimate would make a new pool look smaller."""
    return {
        "time_ratio": dict(plane.time.ratio),
        "memory_ratios": {k: list(v) for k, v in plane.mem.ratios.items()},
        "cache_ratio": list(plane.mem.cache_ratio),
    }


def _calibrate(plane: Plane, cal: dict | None) -> None:
    if not cal:
        return
    plane.time.ratio.update(cal["time_ratio"])
    for k, v in cal["memory_ratios"].items():
        plane.mem.ratios[k].extend(v)
    plane.mem.cache_ratio.extend(cal["cache_ratio"])
    plane.mem._refresh()


def replay(design: Design, cfg: dict, fleet: list[dict], log: list[LoggedRequest], cal: dict | None = None) -> dict:
    cfg = copy.deepcopy(cfg)
    cfg["fleet"] = fleet
    plane = Plane(design, cfg, backend=None)  # type: ignore[arg-type]
    _calibrate(plane, cal)
    backend = EstimateBackend(plane)
    plane.backend = backend
    seen = set()
    seq = itertools.count()
    heap = [(r.t, next(seq), "req", r) for r in log]
    heapq.heapify(heap)
    by_id: dict[str, LoggedRequest] = {}
    while heap:
        t, _, kind, payload = heapq.heappop(heap)
        if kind == "req":
            if payload.context not in seen:
                plane.upload("customer", payload.context, payload.rows, payload.features, 0.0)
                seen.add(payload.context)
            rid = f"q{len(by_id)}"
            by_id[rid] = payload
            plane.arrive(Request(rid, rid, "customer", payload.context, payload.test_rows, t, payload.slo_s, payload.interactive), t)
        else:
            gpu, job, report = payload
            plane.complete(gpu, job, report, t)
        for gpu, job in plane.dispatch(t):
            report = backend.run(gpu, job.to_wire())
            heapq.heappush(heap, (t + report["seconds"], next(seq), "done", (gpu, job, report)))
    met = sum(1 for o in plane.outcomes if o.status == "ok" and o.latency <= by_id[o.request_id].slo_s)
    refused = sum(1 for o in plane.outcomes if o.status == "too_large")
    return {"slo_attainment": met / max(1, len(log)), "too_large": refused}


@dataclass(frozen=True)
class Plan:
    kind: str
    count: int
    eur_per_hour: float
    slo_attainment: float
    target: float
    considered: tuple[tuple[str, int, float, float], ...]  # kind, count, EUR/h, predicted SLO attainment


def advise(design: Design, cfg: dict, log: list[LoggedRequest], target: float, kinds: list[str], max_count: int, cal: dict | None = None) -> Plan:
    """The cheapest homogeneous pool whose replay meets the target plus the planning margin."""
    bar = min(1.0, target + float(cfg["advisor"]["planning_margin"]))
    considered = []
    for kind in kinds:
        for count in range(1, max_count + 1):
            fleet = pool(cfg, kind, count)
            r = replay(design, cfg, fleet, log, cal)
            price = sum(g["eur_per_hour"] for g in fleet)
            considered.append((kind, count, price, r["slo_attainment"]))
            if r["slo_attainment"] >= bar:
                break  # more of this kind only costs more
    ok = [c for c in considered if c[3] >= bar]
    pick = min(ok, key=lambda c: c[2]) if ok else max(considered, key=lambda c: (c[3], -c[2]))
    return Plan(pick[0], pick[1], pick[2], pick[3], target, tuple(considered))


def draft(plan: Plan) -> str:
    """The template drafter. A language model can replace it; `check` holds either to the plan's numbers."""
    alt = sorted((c for c in plan.considered if c[3] >= plan.target and (c[0], c[1]) != (plan.kind, plan.count)), key=lambda c: c[2])
    text = (
        f"Recommended pool: {plan.count} x {plan.kind.upper()} at EUR {plan.eur_per_hour:.2f} per hour, "
        f"expected to answer {plan.slo_attainment * 100:.1f}% of your requests within their latency target "
        f"(target {plan.target * 100:.1f}%)."
    )
    if alt:
        a = alt[0]
        text += f" Next cheapest that also meets it: {a[1]} x {a[0].upper()} at EUR {a[2]:.2f} per hour."
    return text


NUMBER = re.compile(r"\d+(?:\.\d+)?")


def check(text: str, plan: Plan) -> list[str]:
    """Every number in the text must be one the plan holds (counts, prices, percentages); returns the strays."""
    allowed = {f"{plan.target * 100:.1f}"}
    for kind, count, price, slo in plan.considered:
        allowed |= {str(count), f"{price:.2f}", f"{slo * 100:.1f}"}
        allowed |= set(NUMBER.findall(kind))
    return [n for n in NUMBER.findall(text) if n not in allowed and not math.isclose(float(n), 0.0)]
