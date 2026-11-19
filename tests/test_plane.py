"""The plane's decisions, one at a time, against an in-memory worker."""

from __future__ import annotations

import dataclasses

import pytest

from tcs import config
from tcs.plane.estimates import JobShape
from tcs.plane.plane import Plane, Request

FAST = {"name": "fast", "kind": "h100", "hbm_bytes": 8e10, "speed": 1.0, "eur_per_hour": 10.0, "host_bytes": 2e11}
CHEAP = {"name": "cheap", "kind": "l4", "hbm_bytes": 2.4e10, "speed": 0.18, "eur_per_hour": 0.85, "host_bytes": 6.4e10}


def make(cfg, backend, design="platform", fleet=(FAST,), **flags):
    cfg["fleet"] = [dict(g) for g in fleet]
    d = dataclasses.replace(config.design(design), **flags)
    return Plane(d, cfg, backend)


def req(rid, digest="d1", tenant="a", rows=100, key=None, t=0.0, slo=10.0, interactive=True):
    return Request(rid, key or rid, tenant, digest, rows, t, slo, interactive)


def drain(plane, backend, now=0.0, **kw):
    while True:
        started = plane.dispatch(now)
        if not started:
            return
        for gpu, job in started:
            plane.complete(gpu, job, backend.run(gpu, job.to_wire(), now, **kw), now)


def test_a_retried_request_runs_once_and_is_charged_once(cfg, backend):
    p = make(cfg, backend)
    p.upload("a", "d1", 10_000, 10, 1e6)
    p.arrive(req("r1", key="k"), 0.0)
    p.arrive(req("r1-retry", key="k"), 0.0)  # resent while the first is queued
    drain(p, backend)
    p.arrive(req("r1-late", key="k"), 1.0)  # resent after it finished
    assert len(backend.runs) == 1
    assert [o.status for o in p.outcomes] == ["ok", "ok", "ok"]
    assert sum(o.charged for o in p.outcomes) == 1
    assert p.ledger.usage("a")["requests"] == 1


def test_without_idempotency_a_retry_is_charged_twice(cfg, backend):
    p = make(cfg, backend, design="baseline")
    p.upload("a", "d1", 10_000, 10, 1e6)
    p.arrive(req("r1", key="k"), 0.0)
    p.arrive(req("r2", key="k"), 0.0)
    drain(p, backend)
    assert sum(o.charged for o in p.outcomes) == 2


def test_single_flight_one_fit_for_many_predicts(cfg, backend):
    p = make(cfg, backend, batching=False)
    p.upload("a", "d1", 50_000, 10, 1e6)
    for i in range(4):
        p.arrive(req(f"r{i}"), 0.0)
    drain(p, backend)
    kinds = [job["kind"] for _, job in backend.runs]
    assert kinds.count("fit_predict") == 1 and kinds.count("predict") == 3


def test_batching_runs_queued_requests_for_one_context_together(cfg, backend):
    p = make(cfg, backend)
    p.upload("a", "d1", 50_000, 10, 1e6)
    for i in range(4):
        p.arrive(req(f"r{i}", rows=50), 0.0)
    drain(p, backend)
    assert len(backend.runs) == 1 and backend.runs[0][1]["test_rows"] == 200
    assert {o.status for o in p.outcomes} == {"ok"}


def test_chunks_are_chosen_to_fit_memory(cfg, backend):
    p = make(cfg, backend)
    p.upload("a", "big", 1_000_000, 50, 2e8)
    p.arrive(req("r", digest="big", rows=20_000, interactive=False, slo=600), 0.0)
    gpu, job = p.dispatch(0.0)[0]
    assert job.train_chunk < 1_000_000
    planned = cfg["prior"]["weights_bytes"] + p.mem.cache_bytes(1_000_000, 50) + p.mem.transient(job.shape())
    assert planned <= p.gpus[gpu].capacity
    whole = JobShape("fit_predict", 1_000_000, 50, 20_000, 10**9, 10**9)
    assert cfg["prior"]["weights_bytes"] + p.mem.cache_bytes(1_000_000, 50) + p.mem.transient(whole) > p.gpus[gpu].capacity


def test_out_of_memory_lowers_the_capacity_estimate_and_steps_chunks_down(cfg, backend):
    p = make(cfg, backend)
    p.upload("a", "big", 1_000_000, 50, 2e8)
    p.arrive(req("r", digest="big", rows=20_000, interactive=False, slo=600), 0.0)
    before = p.gpus["fast"].capacity
    backend.oom_next = 1
    drain(p, backend)
    first, second = backend.runs[0][1], backend.runs[1][1]
    assert p.gpus["fast"].capacity < before
    assert second["train_chunk"] <= first["train_chunk"] and second["test_chunk"] < first["test_chunk"]
    assert p.outcomes[-1].status == "ok"


def test_a_context_that_fits_nowhere_is_refused_not_crashed(cfg, backend):
    p = make(cfg, backend, fleet=(CHEAP,))
    p.upload("a", "huge", 1_000_000, 200, 8e8)
    p.arrive(req("r", digest="huge"), 0.0)
    drain(p, backend)
    assert p.outcomes[0].status == "too_large" and not backend.runs


def test_value_eviction_keeps_the_context_that_is_expensive_to_refit(cfg, backend):
    p = make(cfg, backend)
    g = p.gpus["fast"]
    from tcs.plane.plane import Entry

    g.hbm["cheap_to_refit"] = Entry("a", 1e10, refit_s=0.5, last_used=10.0, priority=0.5 / 1e10 * 1e9)
    g.hbm["dear_to_refit"] = Entry("a", 1e10, refit_s=60.0, last_used=0.0, priority=60.0 / 1e10 * 1e9)
    assert p._victims(g, keep="x")[0] == "cheap_to_refit"
    lru = make(cfg, backend, cache="lru")
    lru.gpus["fast"].hbm.update(g.hbm)
    assert lru._victims(lru.gpus["fast"], keep="x")[0] == "dear_to_refit"


def test_eviction_spills_to_host_and_comes_back_by_reload(cfg, backend):
    p = make(cfg, backend)
    p.upload("a", "d1", 50_000, 10, 1e6)
    p.arrive(req("r1"), 0.0)
    drain(p, backend)
    p._evict(p.gpus["fast"], "d1")
    assert "d1" in p.gpus["fast"].host and "d1" in backend.host["fast"]
    p.arrive(req("r2"), 1.0)
    drain(p, backend, now=1.0)
    assert backend.runs[-1][1]["kind"] == "reload_predict" and "d1" in p.gpus["fast"].hbm


def test_deadline_first_then_the_cheapest_gpu(cfg, backend):
    p = make(cfg, backend, fleet=(FAST, CHEAP))
    p.upload("a", "d1", 20_000, 10, 1e6)
    p.arrive(req("batch", rows=2000, slo=600.0, interactive=False), 0.0)
    p.arrive(req("urgent", digest="d1", rows=2000, slo=0.3, key="u", tenant="a"), 0.0)
    jobs = {j.requests[0].request_id: g for g, j in p.dispatch(0.0)}
    assert jobs.get("batch") == "cheap"
    assert p.gpus["fast"].queue or jobs.get("urgent") == "fast"


def test_fair_queue_serves_the_quiet_tenant_first(cfg, backend):
    p = make(cfg, backend, quota=False)
    p.usage["loud"] = (5000.0, 0.0)
    for t, d in (("loud", "d1"), ("quiet", "d2")):
        p.upload(t, d, 10_000, 10, 1e6)
    p.gpus["fast"].running = object()  # busy, so both queue
    p.arrive(req("l1", digest="d1", tenant="loud"), 0.0)
    p.arrive(req("q1", digest="d2", tenant="quiet"), 0.1)
    p.gpus["fast"].running = None
    gpu, job = p.dispatch(1.0)[0]
    assert job.tenant == "quiet"


def test_quota_throttles_a_flood(cfg, backend):
    cfg["quota"]["burst_gpu_seconds"] = 5.0
    p = make(cfg, backend)
    p.upload("a", "d1", 300_000, 100, 1e8)
    statuses = []
    for i in range(30):
        p.arrive(req(f"r{i}", rows=20_000, key=f"k{i}", interactive=False, slo=900), float(i))
        statuses.append(p.outcomes[-1].status if p.outcomes and p.outcomes[-1].request_id == f"r{i}" else "queued")
    assert "throttled" in statuses


def test_erasure_purges_every_tier_and_verifies(cfg, backend):
    p = make(cfg, backend, fleet=(FAST, CHEAP))
    for d in ("d1", "d2"):
        p.upload("a", d, 50_000, 10, 1e6)
        p.arrive(req(f"r-{d}", digest=d), 0.0)
    drain(p, backend)
    p._evict(p.gpus[backend.runs[0][0]], backend.runs[0][1]["digest"])  # one in host RAM
    p.gpus["fast"].running = object()
    p.arrive(req("queued", digest="d2", key="q"), 1.0)
    p.gpus["fast"].running = None
    cert = p.erase("a", 2.0)
    assert cert["verified"] and not backend.bucket
    assert all(not backend.residents(g)["hbm"] and not backend.residents(g)["host"] for g in ("fast", "cheap"))
    assert [o.status for o in p.outcomes if o.request_id == "queued"] == ["gone"]
    p.arrive(req("after", digest="d1", key="after"), 3.0)
    assert p.outcomes[-1].status == "gone"


def test_bucket_only_erasure_leaves_caches_behind(cfg, backend):
    p = make(cfg, backend, design="llm_style")
    p.upload("a", "d1", 50_000, 10, 1e6)
    p.arrive(req("r1"), 0.0)
    drain(p, backend)
    p.erase("a", 1.0)
    assert not backend.bucket and backend.residents("fast")["hbm"]


def test_reupload_after_erasure_is_served(cfg, backend):
    p = make(cfg, backend)
    p.upload("a", "d1", 1000, 5, 1e4)
    p.erase("a", 0.0)
    p.upload("a", "d1", 1000, 5, 1e4)
    p.arrive(req("r", key="new"), 1.0)
    drain(p, backend, now=1.0)
    assert p.outcomes[-1].status == "ok"


@pytest.mark.parametrize("kind", ["fit_predict", "predict"])
def test_memory_model_learns_an_upper_quantile_from_telemetry(cfg, kind):
    from tcs.plane.estimates import MemoryModel

    m = MemoryModel(cfg["prior"], cfg["memory"], 8, learn=True)
    s = JobShape(kind, 100_000, 20, 1000, 4096, 1024)
    prior = m.transient(s)
    cache = m.cache_bytes(100_000, 20) if kind == "fit_predict" else 0.0
    for i in range(50):
        m.observe(s, prior * (1.1 + 0.002 * i) + cache, cache)
    assert m.transient(s) >= prior * 1.19
