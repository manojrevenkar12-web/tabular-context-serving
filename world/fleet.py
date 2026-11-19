"""The GPU fleet and the dataset bucket, as they really are.

The platform talks to this only through the worker protocol (docs/icd/worker.md): run a job, evict or purge a
cache, list what is resident, put or delete a dataset. It never sees `reserved_share`, the fragmentation or the true
cost coefficients; it sees what a real worker reports: the job's measured peak memory, the size of the cache it
kept, the time it took, and an out-of-memory error when the allocation fails.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from world.luck import unit_normal


@dataclass
class Gpu:
    name: str
    kind: str
    hbm_bytes: float
    speed: float
    eur_per_hour: float
    reserved_share: float
    host_bytes: float
    hbm: dict[str, float] = field(default_factory=dict)  # digest -> true cache bytes resident in HBM
    host: dict[str, float] = field(default_factory=dict)  # digest -> bytes spilled to host RAM
    owner: dict[str, str] = field(default_factory=dict)  # digest -> tenant
    busy_s: float = 0.0


@dataclass
class Execution:
    gpu: str
    start: float
    seconds: float
    ok: bool
    oom: bool
    kind: str
    tenant: str
    digest: str
    test_rows: int
    requests: tuple[str, ...]
    error: str = ""


class Fleet:
    def __init__(self, cfg: dict, seed: int) -> None:
        self.cfg = cfg
        self.p = cfg["physics"]
        self.estimators = int(cfg["model"]["estimators"])
        self.seed = seed
        self.gpus = {g["name"]: Gpu(**{k: g[k] for k in Gpu.__dataclass_fields__ if k in g}) for g in cfg["gpu"]}
        self.bucket: dict[tuple[str, str], float] = {}
        self.log: list[Execution] = []

    # -- truth ---------------------------------------------------------------------------------------------------
    def _noise(self, *key: object) -> float:
        return math.exp(self.p["noise_sigma"] * unit_normal(self.seed, *key))

    def cache_bytes(self, rows: int, features: int) -> float:
        return self.estimators * rows * (self.p["cache_bytes_per_row"] + self.p["cache_bytes_per_cell"] * features)

    def capacity(self, g: Gpu) -> float:
        lost = g.reserved_share + self.p["fragmentation_per_cache"] * len(g.hbm)
        return g.hbm_bytes * (1.0 - lost)

    def _transient(self, job: dict) -> float:
        n, f, e = job["rows"], job["features"], self.estimators
        root = math.sqrt(n)
        b = 0.0
        if job["kind"] == "fit_predict":
            b = self.p["fit_bytes_per_cell"] * n * f + self.p["fit_bytes_per_chunk_unit"] * min(job["train_chunk"], n) * root * e
        pred = self.p["predict_bytes_per_unit"] * min(job["test_chunk"], max(1, job["test_rows"])) * root * e
        return max(b, pred)

    def _seconds(self, g: Gpu, job: dict) -> float:
        n, f, m, e = job["rows"], job["features"], job["test_rows"], self.estimators
        s = self.p["job_overhead_s"]
        if job["kind"] == "fit_predict":
            s += (self.p["fit_s_per_cell"] * n * f + self.p["fit_s_per_row_pow"] * n ** self.p["row_exponent"]) / g.speed
            s += math.ceil(n / max(1, job["train_chunk"])) * e * self.p["chunk_overhead_s"]
        if job["kind"] == "reload_predict":
            s += g.host.get(job["digest"], 0.0) / self.p["pcie_bytes_per_s"]
        s += self.p["predict_s_per_unit"] * m * math.sqrt(n) / g.speed
        s += math.ceil(max(1, m) / max(1, job["test_chunk"])) * e * self.p["chunk_overhead_s"]
        return s * self._noise("time", job["digest"], job["kind"], m)

    # -- worker protocol -----------------------------------------------------------------------------------------
    def run(self, gpu: str, job: dict, now: float) -> dict:
        """Run one job. Kinds: fit_predict (needs the dataset in the bucket), predict (needs the cache in HBM),
        reload_predict (needs the cache in host RAM, brings it back to HBM first)."""
        g = self.gpus[gpu]
        kind, digest, tenant = job["kind"], job["digest"], job["tenant"]
        reqs = tuple(job.get("requests", ()))
        if kind == "fit_predict" and (tenant, digest) not in self.bucket:
            self.log.append(Execution(gpu, now, 0.0, False, False, kind, tenant, digest, job["test_rows"], reqs, "missing_dataset"))
            return {"ok": False, "oom": False, "seconds": 0.0, "error": "missing_dataset"}
        if kind == "predict" and digest not in g.hbm:
            return {"ok": False, "oom": False, "seconds": 0.0, "error": "cache_miss"}
        if kind == "reload_predict" and digest not in g.host:
            return {"ok": False, "oom": False, "seconds": 0.0, "error": "cache_miss"}
        new_cache = self.cache_bytes(job["rows"], job["features"]) if kind != "predict" else 0.0
        if kind == "reload_predict":
            new_cache = g.host[digest]
        transient = self._transient(job) * self._noise("mem", digest, kind, job["test_rows"], job["test_chunk"])
        resident = sum(g.hbm.values())
        seconds = self._seconds(g, job)
        peak = self.p["weights_bytes"] + resident + new_cache + transient
        if peak > self.capacity(g):
            wasted = 0.3 * seconds
            g.busy_s += wasted
            self.log.append(Execution(gpu, now, wasted, False, True, kind, tenant, digest, job["test_rows"], reqs, "oom"))
            return {"ok": False, "oom": True, "seconds": wasted, "error": "CUDA out of memory"}
        g.busy_s += seconds
        if kind == "reload_predict":
            g.host.pop(digest)
        if kind != "predict" and job.get("keep_cache"):
            g.hbm[digest] = new_cache
            g.owner[digest] = tenant
        self.log.append(Execution(gpu, now, seconds, True, False, kind, tenant, digest, job["test_rows"], reqs))
        return {
            "ok": True,
            "oom": False,
            "seconds": seconds,
            "peak_bytes": transient + (new_cache if kind == "fit_predict" else 0.0),
            "cache_bytes": new_cache if kind != "predict" else g.hbm.get(digest, 0.0),
            "error": "",
        }

    def evict(self, gpu: str, digest: str, to_host: bool) -> bool:
        g = self.gpus[gpu]
        b = g.hbm.pop(digest, None)
        if b is None:
            return False
        if to_host and sum(g.host.values()) + b <= g.host_bytes:
            g.host[digest] = b
        return True

    def purge(self, gpu: str, digest: str) -> None:
        g = self.gpus[gpu]
        g.hbm.pop(digest, None)
        g.host.pop(digest, None)

    def residents(self, gpu: str) -> dict:
        g = self.gpus[gpu]
        return {"hbm": dict(g.hbm), "host": dict(g.host)}

    def put(self, tenant: str, digest: str, nbytes: float) -> None:
        self.bucket[(tenant, digest)] = nbytes

    def delete(self, tenant: str, digest: str) -> None:
        self.bucket.pop((tenant, digest), None)

    # -- audit (evals only) --------------------------------------------------------------------------------------
    def residual_bytes(self, tenant: str, digests: set[str]) -> float:
        total = sum(b for (t, d), b in self.bucket.items() if t == tenant)
        for g in self.gpus.values():
            total += sum(b for d, b in g.hbm.items() if d in digests)
            total += sum(b for d, b in g.host.items() if d in digests)
        return total
