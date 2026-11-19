"""The world is fair to every design, and the evaluation scores what really happened."""

from __future__ import annotations

import numpy as np
import pytest

from evals.operate import operate, world_config
from evals.truth import score
from tcs.plane.store import LocalBucket, digest
from world.fleet import Fleet
from world.traffic import Erase, Predict, Upload, generate


def test_traffic_is_the_same_whatever_the_design():
    cfg = world_config()
    a, b = generate(cfg, 7), generate(cfg, 7)
    assert a == b and generate(cfg, 8) != a
    kinds = {type(e) for e in a}
    assert {Upload, Predict, Erase} <= kinds
    assert any(isinstance(e, Predict) and e.is_retry for e in a)


def test_job_noise_is_keyed_by_the_job_not_the_design():
    cfg = world_config()
    f1, f2 = Fleet(cfg, 3), Fleet(cfg, 3)
    f2.put("t", "d", 1.0)
    f1.put("t", "d", 1.0)
    job = {
        "kind": "fit_predict",
        "tenant": "t",
        "digest": "d",
        "rows": 10_000,
        "features": 10,
        "test_rows": 50,
        "train_chunk": 2048,
        "test_chunk": 2048,
        "keep_cache": True,
    }
    f2.run("h100-1", dict(job, digest="d"), 0.0)  # f2 did other work first on another GPU
    assert f1.run("h100-0", job, 5.0)["seconds"] == pytest.approx(f2.run("h100-0", job, 99.0)["seconds"])


def test_the_fleet_runs_out_of_memory_like_a_gpu():
    cfg = world_config()
    f = Fleet(cfg, 1)
    f.put("t", "d", 1.0)
    job = {
        "kind": "fit_predict",
        "tenant": "t",
        "digest": "d",
        "rows": 1_000_000,
        "features": 50,
        "test_rows": 10,
        "train_chunk": 10**9,
        "test_chunk": 10**9,
        "keep_cache": True,
    }
    assert f.run("h100-0", job, 0.0)["oom"]
    assert f.run("h100-0", dict(job, train_chunk=2048), 0.0)["ok"]
    assert f.residents("h100-0")["hbm"]


def test_content_address_is_stable_and_bucket_paths_are_safe(tmp_path):
    x = np.arange(12, dtype=np.float64).reshape(4, 3)
    y = np.array([0, 1, 0, 1])
    assert digest(x, y) == digest(x.astype(np.float32), y.astype(np.int32))
    assert digest(x, y) != digest(x[::-1], y)
    b = LocalBucket(tmp_path)
    with pytest.raises(ValueError):
        b.put("../x", "abc", b"")
    with pytest.raises(ValueError):
        b.put("acme", "../../etc", b"")


@pytest.fixture(scope="module")
def short_runs():
    return {d: score(operate(d, "world", 101, hours=1.0)) for d in ("baseline", "llm_style", "platform")}


def test_platform_beats_the_status_quo_on_a_short_day(short_runs):
    p, b = short_runs["platform"], short_runs["baseline"]
    assert p["slo_attainment"] > b["slo_attainment"]
    assert p["oom_events"] < b["oom_events"]
    assert p["cache_hit_rate"] > 0.3 and b["cache_hit_rate"] == 0.0


def test_money_and_privacy_invariants_hold_for_the_platform(short_runs):
    p = short_runs["platform"]
    assert p["double_charges"] == 0 and p["served_after_erasure"] == 0 and p["erasure_residual_bytes"] == 0
    assert short_runs["baseline"]["double_charges"] > 0
