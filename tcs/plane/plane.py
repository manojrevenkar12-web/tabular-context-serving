"""The serving plane: where each request runs, in what order, with what chunks, and what stays in GPU memory.

The plane makes decisions; a driver carries them out. `arrive`, `erase` and `upload` take requests in;
`dispatch(now)` hands each idle GPU its next job; `complete` takes the worker's report back. The same code runs
behind the HTTP API (with PyTorch workers) and inside the evaluation (with the simulated fleet), through the worker
protocol in docs/icd/worker.md.

Decisions, each switched by a design flag (config/designs.toml):

* routing: round robin, least loaded, or cache-aware (expected completion counting queue, fit, reload and predict
  time, plus single-flight: a predict for a context being fitted waits for that fit instead of fitting again);
* admission: fixed whole-context chunks, or chunk sizes chosen so the predicted peak fits the GPU, evicting caches
  if needed, from a memory model corrected by telemetry and out-of-memory errors;
* caching: none, LRU, or value-based (refit seconds saved per byte, GreedyDual-Size-Frequency) with a host-RAM tier;
* batching: queued requests for the same context run as one job;
* queueing: FIFO or fair (least recent GPU time per tenant) plus per-tenant quotas on GPU seconds;
* idempotency: one execution and one charge per idempotency key;
* erasure: delete the stored dataset only, or also purge every cache tier and verify with the workers.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Protocol

from tcs.config import Design
from tcs.plane.estimates import JobShape, MemoryModel, TimeModel
from tcs.plane.ledger import Ledger


class Backend(Protocol):
    def evict(self, gpu: str, digest: str, to_host: bool) -> bool: ...
    def purge(self, gpu: str, digest: str) -> None: ...
    def residents(self, gpu: str) -> dict: ...
    def put(self, tenant: str, digest: str, nbytes: float) -> None: ...
    def delete(self, tenant: str, digest: str) -> None: ...


@dataclass
class Request:
    request_id: str
    key: str
    tenant: str
    digest: str
    test_rows: int
    arrived: float
    slo_s: float
    interactive: bool
    payload: Any = None


@dataclass
class Job:
    job_id: str
    gpu: str
    kind: str
    tenant: str
    digest: str
    rows: int
    features: int
    requests: list[Request]
    created: float
    keep_cache: bool = False
    train_chunk: int = 10**9
    test_chunk: int = 10**9
    attempts: int = 0
    step_down: int = 0  # ladder rungs skipped after an out-of-memory error
    est_s: float = 0.0
    reload_bytes: float = 0.0

    @property
    def test_rows(self) -> int:
        return sum(r.test_rows for r in self.requests)

    def shape(self) -> JobShape:
        return JobShape(self.kind, self.rows, self.features, self.test_rows, self.train_chunk, self.test_chunk)

    def to_wire(self) -> dict:
        return {
            "job_id": self.job_id,
            "kind": self.kind,
            "tenant": self.tenant,
            "digest": self.digest,
            "rows": self.rows,
            "features": self.features,
            "test_rows": self.test_rows,
            "train_chunk": self.train_chunk,
            "test_chunk": self.test_chunk,
            "keep_cache": self.keep_cache,
            "requests": [r.request_id for r in self.requests],
            "payloads": [r.payload for r in self.requests],
        }


@dataclass
class Outcome:
    request_id: str
    key: str
    tenant: str
    status: str  # ok | throttled | gone | not_found | too_large | failed
    arrived: float
    finished: float
    gpu: str = ""
    duplicate: bool = False
    charged: bool = False
    result: Any = None

    @property
    def latency(self) -> float:
        return self.finished - self.arrived


@dataclass
class Entry:
    tenant: str
    nbytes: float
    refit_s: float
    last_used: float
    hits: int = 1
    priority: float = 0.0


@dataclass
class Gpu:
    name: str
    kind: str
    hbm_bytes: float
    speed: float
    eur_per_hour: float
    host_bytes: float
    capacity: float
    proven: float = 0.0  # the largest occupancy a job on this card has finished with (shown, not planned on)
    hbm: dict[str, Entry] = field(default_factory=dict)
    host: dict[str, Entry] = field(default_factory=dict)
    queue: list[Job] = field(default_factory=list)
    running: Job | None = None
    running_until: float = 0.0
    clock: float = 0.0  # GreedyDual inflation value


class Plane:
    def __init__(self, design: Design, cfg: dict, backend: Backend, ledger: Ledger | None = None) -> None:
        self.d = design
        self.cfg = cfg
        self.backend = backend
        self.ledger = ledger or Ledger()
        e = int(cfg["model"]["estimators"])
        self.mem = MemoryModel(cfg["prior"], cfg["memory"], e, design.learn)
        self.time = TimeModel(cfg["prior"], cfg["timing"], e, design.learn)
        head = float(cfg["memory"]["headroom_share"])
        self.gpus = {
            g["name"]: Gpu(g["name"], g["kind"], g["hbm_bytes"], g["speed"], g["eur_per_hour"], g["host_bytes"], g["hbm_bytes"] * head) for g in cfg["fleet"]
        }
        self.order = list(self.gpus)
        self.rr = itertools.cycle(self.order)
        self.ids = itertools.count()
        self.datasets: dict[tuple[str, str], tuple[int, int]] = {}
        self.owned: dict[str, set[str]] = {}
        self.erased: dict[str, float] = {}
        self.refused: dict[str, set[str]] = {}  # tenant -> erased dataset ids, answered 410 Gone
        self.pending_fit: dict[str, Job] = {}
        self.keys: dict[str, Outcome | list[Request]] = {}
        self.outcomes: list[Outcome] = []
        self.usage: dict[str, tuple[float, float]] = {}  # tenant -> (decayed GPU seconds, at)
        self.bucket: dict[str, tuple[float, float]] = {}  # tenant -> (tokens, at)
        self.certificates: list[dict] = []
        self.awaiting_purge: dict[str, set[str]] = {}  # gpu -> digests to purge when its running job ends
        self.ladders = (list(cfg["chunking"]["train_ladder"]), list(cfg["chunking"]["test_ladder"]))

    # -- intake ----------------------------------------------------------------------------------------------------
    def upload(self, tenant: str, digest: str, rows: int, features: int, nbytes: float) -> None:
        if tenant in self.erased:
            self.erased.pop(tenant)  # a new upload after erasure is new data
        self.refused.get(tenant, set()).discard(digest)
        self.backend.put(tenant, digest, nbytes)
        self.datasets[(tenant, digest)] = (rows, features)
        self.owned.setdefault(tenant, set()).add(digest)

    def arrive(self, req: Request, now: float) -> None:
        if self.d.idempotency and req.key in self.keys:
            seen = self.keys[req.key]
            if isinstance(seen, Outcome):
                self._finish(req, seen.status, now, seen.gpu, duplicate=True, result=seen.result)
            else:
                seen.append(req)
            return
        if req.digest in self.refused.get(req.tenant, ()):
            self._finish(req, "gone", now)
            return
        meta = self.datasets.get((req.tenant, req.digest))
        if meta is None:
            self._finish(req, "not_found", now)
            return
        if self.d.quota and not self._take_quota(req, meta, now):
            self._finish(req, "throttled", now)
            return
        if self.d.idempotency:
            self.keys[req.key] = []
        job = Job(f"j{next(self.ids)}", "", "fit_predict", req.tenant, req.digest, meta[0], meta[1], [req], now)
        self._route(job, now)

    # -- quota and fairness ----------------------------------------------------------------------------------------
    def _norm_seconds(self, kind: str, rows: int, features: int, test_rows: int) -> float:
        return self.time.seconds(JobShape(kind, rows, features, test_rows, 10**9, 10**9), "h100", 1.0)

    def _take_quota(self, req: Request, meta: tuple[int, int], now: float) -> bool:
        q = self.cfg["quota"]
        cap, rate = float(q["burst_gpu_seconds"]), float(q["gpu_seconds_per_hour"]) / 3600.0
        tokens, at = self.bucket.get(req.tenant, (cap, now))
        tokens = min(cap, tokens + (now - at) * rate)
        cached = any(req.digest in g.hbm or req.digest in g.host for g in self.gpus.values()) or req.digest in self.pending_fit
        cost = self._norm_seconds("predict" if cached else "fit_predict", meta[0], meta[1], req.test_rows)
        if cost > tokens and tokens < cap:
            self.bucket[req.tenant] = (tokens, now)
            return False
        self.bucket[req.tenant] = (tokens - cost, now)
        return True

    def _served(self, tenant: str, now: float) -> float:
        s, at = self.usage.get(tenant, (0.0, now))
        return s * 0.5 ** ((now - at) / float(self.cfg["queue"]["fairness_halflife_s"]))

    # -- estimates -------------------------------------------------------------------------------------------------
    def _resident(self, g: Gpu) -> float:
        return sum(e.nbytes for e in g.hbm.values())

    def _kind_on(self, g: Gpu, digest: str) -> str:
        if self.d.cache == "none":
            return "fit_predict"
        if digest in g.hbm:
            return "predict"
        if digest in g.host:
            return "reload_predict"
        return "fit_predict"

    def _est(self, job: Job, g: Gpu) -> float:
        reload = g.host[job.digest].nbytes if job.kind == "reload_predict" and job.digest in g.host else 0.0
        return self.time.seconds(job.shape(), g.kind, g.speed, reload)

    def _queue_wait(self, g: Gpu, now: float) -> float:
        return max(0.0, g.running_until - now) + sum(j.est_s for j in g.queue)

    def _min_transient(self, job: Job) -> float:
        s = JobShape(job.kind, job.rows, job.features, job.test_rows, self.ladders[0][-1], self.ladders[1][-1])
        return self.mem.transient(s)

    def _feasible(self, job: Job, g: Gpu, kind: str) -> bool:
        if self.d.admission == "fixed":
            return True
        new = self.mem.cache_bytes(job.rows, job.features) if kind == "fit_predict" else 0.0
        if kind == "reload_predict":
            new = g.host[job.digest].nbytes
        shape = JobShape(kind, job.rows, job.features, job.test_rows, self.ladders[0][-1], self.ladders[1][-1])
        return self.cfg["prior"]["weights_bytes"] + new + self.mem.transient(shape) <= g.capacity

    # -- routing ---------------------------------------------------------------------------------------------------
    def _route(self, job: Job, now: float) -> None:
        candidates = [g for g in self.gpus.values()]
        if self.d.routing == "round_robin":
            g = self.gpus[next(self.rr)]
            self._enqueue(job, g, self._kind_on(g, job.digest), now)
            return
        options: list[tuple[float, float, Gpu, str]] = []  # (expected completion, cost, gpu, kind)
        for g in candidates:
            kind = self._kind_on(g, job.digest)
            if (
                self.d.routing == "cache_aware"
                and kind == "fit_predict"
                and self.pending_fit.get(job.digest) is not None
                and self.pending_fit[job.digest].gpu == g.name
            ):
                fit = self.pending_fit[job.digest]
                if g.running is fit:
                    done = g.running_until
                elif fit in g.queue:
                    done = max(g.running_until, now) + sum(j.est_s for j in g.queue[: g.queue.index(fit) + 1])
                else:
                    done = now
                run = self.time.seconds(JobShape("predict", job.rows, job.features, job.test_rows, 10**9, 10**9), g.kind, g.speed)
                t = max(done, now) - now + run
                kind = "after_fit"
            else:
                if not self._feasible(job, g, kind):
                    continue
                job.kind = kind
                run = self._est(job, g)
                t = self._queue_wait(g, now) + (run if self.d.routing == "cache_aware" else 0.0)
            options.append((t, run * g.eur_per_hour / 3600.0, g, kind))
        if not options:
            self._fail(job, "too_large", now)
            return
        best_t, _, best, best_kind = min(options, key=lambda o: o[0])
        if self.d.routing == "cache_aware":
            # Deadline first, then money: the cheapest GPU expected to finish well inside the client's SLO.
            left = min(r.slo_s - (now - r.arrived) for r in job.requests) * float(self.cfg["routing"]["slo_margin"])
            in_time = [o for o in options if o[0] <= left]
            if in_time:
                best_t, _, best, best_kind = min(in_time, key=lambda o: (o[1], o[0]))
        if best_kind == "after_fit":
            fit = self.pending_fit[job.digest]
            if self.d.batching and fit in best.queue and fit.test_rows + job.test_rows <= int(self.cfg["batching"]["max_test_rows"]):
                fit.requests += job.requests
                fit.est_s = self._est(fit, best)
                return
            self._enqueue(job, best, "predict", now)
            return
        self._enqueue(job, best, best_kind, now)

    def _enqueue(self, job: Job, g: Gpu, kind: str, now: float) -> None:
        job.gpu, job.kind = g.name, kind
        job.keep_cache = kind == "fit_predict" and self.d.cache != "none"
        if kind == "fit_predict" and job.keep_cache and self.d.routing == "cache_aware":
            self.pending_fit[job.digest] = job
        job.est_s = self._est(job, g)
        g.queue.append(job)

    # -- dispatch --------------------------------------------------------------------------------------------------
    def dispatch(self, now: float) -> list[tuple[str, Job]]:
        out: list[tuple[str, Job]] = []
        moved = True
        while moved:  # a job admission sends elsewhere may land on a GPU already passed
            moved = False
            for g in self.gpus.values():
                while g.running is None and g.queue:
                    job = self._pick(g, now)
                    if job is None:
                        break
                    g.queue.remove(job)
                    if not self._admit(job, g, now):
                        moved = True
                        continue
                    g.running, g.running_until = job, now + job.est_s
                    out.append((g.name, job))
        return out

    def _ready(self, job: Job, g: Gpu) -> bool:
        if job.kind == "predict" and job.digest not in g.hbm:
            pending = self.pending_fit.get(job.digest)
            if pending is not None and pending.gpu == g.name:
                return False  # waits for the fit on this GPU
            job.kind = "reload_predict" if job.digest in g.host else "fit_predict"
            job.keep_cache = job.kind == "fit_predict" and self.d.cache != "none"
        if job.kind == "reload_predict" and job.digest not in g.host:
            job.kind = "predict" if job.digest in g.hbm else "fit_predict"
            job.keep_cache = job.kind == "fit_predict" and self.d.cache != "none"
        return True

    def _pick(self, g: Gpu, now: float) -> Job | None:
        ready = [j for j in g.queue if self._ready(j, g)]
        if not ready:
            return None
        if self.d.queue == "fair":
            job = min(ready, key=lambda j: (self._served(j.tenant, now), j.created))
        else:
            job = min(ready, key=lambda j: j.created)
        if self.d.batching:
            limit = int(self.cfg["batching"]["max_test_rows"])
            for other in [j for j in g.queue if j is not job and j.digest == job.digest and j.tenant == job.tenant]:
                if job.test_rows + other.test_rows > limit:
                    break
                job.requests += other.requests
                g.queue.remove(other)
                if self.pending_fit.get(other.digest) is other:
                    self.pending_fit[other.digest] = job
        return job

    def _victims(self, g: Gpu, keep: str) -> list[str]:
        names = [d for d in g.hbm if d != keep]
        if self.d.cache == "lru":
            return sorted(names, key=lambda d: g.hbm[d].last_used)
        return sorted(names, key=lambda d: g.hbm[d].priority)

    def _evict(self, g: Gpu, digest: str) -> None:
        e = g.hbm.pop(digest)
        if self.d.cache == "value":
            g.clock = max(g.clock, e.priority)
        to_host = self.d.host_tier and sum(x.nbytes for x in g.host.values()) + e.nbytes <= g.host_bytes
        self.backend.evict(g.name, digest, to_host)
        if to_host:
            g.host[digest] = e

    def _admit(self, job: Job, g: Gpu, now: float) -> bool:
        """Choose chunk sizes so the job fits, evicting caches if it must. False if the job went elsewhere."""
        weights = float(self.cfg["prior"]["weights_bytes"])
        new = 0.0
        if job.kind == "fit_predict":
            new = self.mem.cache_bytes(job.rows, job.features)
        elif job.kind == "reload_predict":
            new = g.host[job.digest].nbytes
            job.reload_bytes = new
        if job.keep_cache:
            budget = g.capacity * float(self.cfg["memory"]["cache_budget_share"])
            if new > budget:
                job.keep_cache = False
                if self.pending_fit.get(job.digest) is job:
                    self.pending_fit.pop(job.digest)
                    self._release_waiters(job.digest, g, now)
            else:
                for d in self._victims(g, job.digest):
                    if self._resident(g) + new <= budget:
                        break
                    self._evict(g, d)
        if self.d.admission == "fixed":
            job.train_chunk, job.test_chunk = self.ladders[0][0], self.ladders[1][0]
            job.est_s = self._est(job, g)
            return True
        trains, tests = (lad[min(job.step_down, len(lad) - 1) :] for lad in self.ladders)

        def fits(tc: int, sc: int) -> bool:
            s = JobShape(job.kind, job.rows, job.features, job.test_rows, tc, sc)
            return weights + self._resident(g) + new + self.mem.transient(s) <= g.capacity

        victims = self._victims(g, job.digest)
        while not fits(trains[-1], tests[-1]) and victims:
            self._evict(g, victims.pop(0))
        if not fits(trains[-1], tests[-1]):
            others = [o for o in self.gpus.values() if o is not g and self._feasible(job, o, "fit_predict")]
            job.attempts += 1
            if job.kind == "fit_predict" and others and job.attempts < int(self.cfg["retry"]["max_attempts"]):
                if self.pending_fit.get(job.digest) is job:
                    self.pending_fit.pop(job.digest)
                self._route(job, now)
            else:
                self._fail(job, "too_large", now)
            return False
        job.train_chunk = next(c for c in trains if fits(c, tests[-1]))
        job.test_chunk = next(c for c in tests if fits(job.train_chunk, c))
        job.est_s = self._est(job, g)
        return True

    def _release_waiters(self, digest: str, g: Gpu, now: float) -> None:
        """A fit that will not keep its cache: queued predicts waiting for it fit for themselves."""
        for j in g.queue:
            if j.digest == digest and j.kind == "predict":
                j.kind, j.keep_cache = "fit_predict", False

    # -- completion ------------------------------------------------------------------------------------------------
    def complete(self, gpu: str, job: Job, report: dict, now: float) -> None:
        g = self.gpus[gpu]
        g.running, g.running_until = None, now
        if self.pending_fit.get(job.digest) is job:
            self.pending_fit.pop(job.digest)
        shape = job.shape()
        if report["ok"]:
            self.mem.observe(shape, report.get("peak_bytes", 0.0), report.get("cache_bytes", 0.0))
            self.time.observe(shape, g.kind, g.speed, report["seconds"], job.reload_bytes)
            s, at = self._served(job.tenant, now), now
            self.usage[job.tenant] = (s + report["seconds"] * g.speed, at)
            occupied = float(self.cfg["prior"]["weights_bytes"]) + self._resident(g) + report.get("peak_bytes", 0.0)
            g.proven = max(g.proven, occupied)
            gone = self.d.erasure == "purge_verify" and job.tenant in self.erased
            if gone:
                pass  # the tenant was erased while this ran: nothing joins the directory, the purge follows below
            elif job.kind == "fit_predict" and job.keep_cache:
                refit = report["seconds"]
                g.hbm[job.digest] = Entry(job.tenant, report["cache_bytes"], refit, now, 1, g.clock + refit / max(report["cache_bytes"], 1.0) * 1e9)
            elif job.kind == "reload_predict" and job.digest in g.host:
                e = g.host.pop(job.digest)
                e.last_used, e.hits = now, e.hits + 1
                g.hbm[job.digest] = e
            elif job.kind == "predict" and job.digest in g.hbm:
                e = g.hbm[job.digest]
                e.last_used, e.hits = now, e.hits + 1
                e.priority = g.clock + e.hits * e.refit_s / max(e.nbytes, 1.0) * 1e9
            for r in job.requests:
                share = report["seconds"] * r.test_rows / max(1, job.test_rows)
                res = report.get("results", {}).get(r.request_id)
                self._finish(r, "gone" if gone else "ok", now, gpu, gpu_s=share, result=None if gone else res)
        elif report.get("oom"):
            if self.d.admission == "memory_model" and self.d.learn:
                # The card holds less than this job needed: plan for a little less, never below the sanity floor.
                floor = g.hbm_bytes * float(self.cfg["memory"]["min_capacity_share"])
                g.capacity = max(floor, g.capacity * float(self.cfg["memory"]["oom_backoff"]))
            job.attempts += 1
            job.step_down += 1
            if job.attempts >= int(self.cfg["retry"]["max_attempts"]):
                self._fail(job, "failed", now)
            elif self.d.admission == "fixed":
                job.kind = "fit_predict"
                self._route(job, now)
            else:
                g.queue.insert(0, job)
        elif report.get("error") == "cache_miss":
            job.kind, job.keep_cache = "fit_predict", self.d.cache != "none"
            g.hbm.pop(job.digest, None)
            g.host.pop(job.digest, None)
            g.queue.insert(0, job)
        else:
            self._fail(job, "gone" if report.get("error") == "missing_dataset" else "failed", now)
        for d in self.awaiting_purge.pop(gpu, set()):
            self._purge_everywhere(d)
        self._verify_pending(now)

    def _fail(self, job: Job, status: str, now: float) -> None:
        for r in job.requests:
            self._finish(r, status, now, job.gpu)

    def _finish(self, req: Request, status: str, now: float, gpu: str = "", duplicate: bool = False, gpu_s: float = 0.0, result: Any = None) -> None:
        charged = False
        if status == "ok" and not duplicate:
            charged = self.ledger.charge(req.key if self.d.idempotency else req.request_id, req.tenant, req.test_rows, gpu_s, now)
        out = Outcome(req.request_id, req.key, req.tenant, status, req.arrived, now, gpu, duplicate, charged, result)
        self.outcomes.append(out)
        if self.d.idempotency and not duplicate:
            waiting = self.keys.get(req.key)
            self.keys[req.key] = out
            if isinstance(waiting, list):
                for w in waiting:
                    self._finish(w, status, now, gpu, duplicate=True, result=result)

    # -- erasure ---------------------------------------------------------------------------------------------------
    def erase(self, tenant: str, now: float) -> dict:
        """Delete everything the tenant gave us. purge_verify also drops queued work and purges every cache tier,
        then asks every worker what it holds before calling the erasure done."""
        digests = set(self.owned.pop(tenant, set()))
        self.erased[tenant] = now
        self.refused.setdefault(tenant, set()).update(digests)
        for d in digests:
            self.backend.delete(tenant, d)
            self.datasets.pop((tenant, d), None)
        cert = {"tenant": tenant, "requested": now, "digests": sorted(digests), "completed": None, "verified": False}
        self.certificates.append(cert)
        if self.d.erasure != "purge_verify":
            cert["completed"] = now
            return cert
        for g in self.gpus.values():
            for j in [j for j in g.queue if j.tenant == tenant]:
                g.queue.remove(j)
                self._fail(j, "gone", now)
            if g.running is not None and g.running.tenant == tenant:
                self.awaiting_purge.setdefault(g.name, set()).update(digests)
        for d in digests:
            self.pending_fit.pop(d, None)
            self._purge_everywhere(d)
        self._verify_pending(now)
        return cert

    def _purge_everywhere(self, digest: str) -> None:
        for g in self.gpus.values():
            g.hbm.pop(digest, None)
            g.host.pop(digest, None)
            self.backend.purge(g.name, digest)

    def _verify_pending(self, now: float) -> None:
        if self.d.erasure != "purge_verify":
            return
        for cert in self.certificates:
            if cert["verified"]:
                continue
            if any(g.running is not None and g.running.tenant == cert["tenant"] for g in self.gpus.values()):
                continue
            left = set(cert["digests"])
            held = set()
            for g in self.gpus.values():
                r = self.backend.residents(g.name)
                held |= (set(r["hbm"]) | set(r["host"])) & left
            for d in held:
                self._purge_everywhere(d)
            cert["verified"] = not held
            cert["completed"] = now if not held else None

    # -- views -----------------------------------------------------------------------------------------------------
    def snapshot(self) -> dict:
        return {
            g.name: {
                "kind": g.kind,
                "capacity_estimate": g.capacity,
                "hbm": {d: e.nbytes for d, e in g.hbm.items()},
                "host": {d: e.nbytes for d, e in g.host.items()},
                "queued": len(g.queue),
                "running": g.running.job_id if g.running else None,
            }
            for g in self.gpus.values()
        }
