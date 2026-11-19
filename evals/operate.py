"""Run one design on one simulated day: traffic in, the plane decides, the fleet executes, everything is logged.

This is the only place that holds both the plane and the world. The plane sees the fleet only through
`FleetBackend`, the same worker protocol the PyTorch workers implement.
"""

from __future__ import annotations

import heapq
import itertools
import tomllib
from pathlib import Path

from tcs import config
from tcs.plane.plane import Plane, Request
from world.fleet import Fleet
from world.traffic import Erase, Predict, Upload, generate

ROOT = Path(__file__).resolve().parent.parent


def world_config(name: str = "world") -> dict:
    return tomllib.loads((ROOT / "config" / f"{name}.toml").read_text())


class FleetBackend:
    def __init__(self, fleet: Fleet) -> None:
        self.fleet = fleet

    def evict(self, gpu: str, digest: str, to_host: bool) -> bool:
        return self.fleet.evict(gpu, digest, to_host)

    def purge(self, gpu: str, digest: str) -> None:
        self.fleet.purge(gpu, digest)

    def residents(self, gpu: str) -> dict:
        return self.fleet.residents(gpu)

    def put(self, tenant: str, digest: str, nbytes: float) -> None:
        self.fleet.put(tenant, digest, nbytes)

    def delete(self, tenant: str, digest: str) -> None:
        self.fleet.delete(tenant, digest)


def operate(
    design: str,
    world: str,
    seed: int,
    hours: float | None = None,
    wcfg: dict | None = None,
    pcfg: dict | None = None,
    events: list | None = None,
    snapshot_every: float | None = None,
) -> dict:
    wcfg = wcfg or world_config(world)
    if hours is not None:
        wcfg["campaign"]["hours"] = hours
    fleet = Fleet(wcfg, seed)
    events = events if events is not None else generate(wcfg, seed)
    plane = Plane(config.design(design), pcfg or config.platform(), FleetBackend(fleet))
    seq = itertools.count()
    heap: list = [(e.t, next(seq), "event", e) for e in events]
    heapq.heapify(heap)
    deadline = float(wcfg["campaign"]["erasure_deadline_s"])
    audits: list[dict] = []
    tenant_digests: dict[str, set[str]] = {}
    for e in events:
        if isinstance(e, Upload):
            tenant_digests.setdefault(e.tenant, set()).add(e.digest)
    snapshots: list[dict] = []
    next_snap = 0.0
    while heap:
        t, _, kind, payload = heapq.heappop(heap)
        while snapshot_every and t >= next_snap:
            snapshots.append({"t": next_snap, "plane": plane.snapshot(), "capacity": {n: fleet.capacity(g) for n, g in fleet.gpus.items()}})
            next_snap += snapshot_every
        if kind == "event":
            if isinstance(payload, Upload):
                plane.upload(payload.tenant, payload.digest, payload.rows, payload.features, payload.nbytes)
            elif isinstance(payload, Predict):
                plane.arrive(
                    Request(
                        payload.request_id, payload.idempotency_key, payload.tenant, payload.digest, payload.test_rows, t, payload.slo_s, payload.interactive
                    ),
                    t,
                )
            elif isinstance(payload, Erase):
                plane.erase(payload.tenant, t)
                heapq.heappush(heap, (t + deadline, next(seq), "audit", payload))
        elif kind == "done":
            gpu, job, report = payload
            plane.complete(gpu, job, report, t)
        elif kind == "audit":
            digests = {d for d in tenant_digests.get(payload.tenant, set())}
            audits.append({"tenant": payload.tenant, "requested": payload.t, "residual_bytes": fleet.residual_bytes(payload.tenant, digests)})
        for gpu, job in plane.dispatch(t):
            report = fleet.run(gpu, job.to_wire(), t)
            heapq.heappush(heap, (t + report["seconds"], next(seq), "done", (gpu, job, report)))
    return {
        "design": design,
        "seed": seed,
        "world": world,
        "events": events,
        "outcomes": plane.outcomes,
        "log": fleet.log,
        "gpus": {g.name: {"kind": g.kind, "eur_per_hour": g.eur_per_hour, "busy_s": g.busy_s} for g in fleet.gpus.values()},
        "audits": audits,
        "certificates": plane.certificates,
        "charges": plane.ledger.charged_keys(),
        "hours": float(wcfg["campaign"]["hours"]),
        "plane": plane,
        "snapshots": snapshots,
    }
