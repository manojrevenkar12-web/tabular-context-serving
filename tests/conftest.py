from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tcs import config  # noqa: E402
from tcs.model.icl import load  # noqa: E402


@pytest.fixture(scope="session")
def model():
    return load(str(ROOT / "models" / "tiny_icl.pt"))


class FakeBackend:
    """An in-memory worker for plane unit tests: jobs take `seconds` and fail with OOM when told to."""

    def __init__(self) -> None:
        self.hbm: dict[str, dict[str, float]] = {}
        self.host: dict[str, dict[str, float]] = {}
        self.bucket: dict[tuple[str, str], float] = {}
        self.runs: list[tuple[str, dict]] = []
        self.oom_next = 0

    def run(self, gpu: str, job: dict, now: float, seconds: float = 1.0, cache_bytes: float = 1e9) -> dict:
        self.runs.append((gpu, job))
        if self.oom_next:
            self.oom_next -= 1
            return {"ok": False, "oom": True, "seconds": 0.1, "error": "CUDA out of memory"}
        hbm = self.hbm.setdefault(gpu, {})
        if job["kind"] == "predict" and job["digest"] not in hbm:
            return {"ok": False, "oom": False, "seconds": 0.0, "error": "cache_miss"}
        if job["kind"] == "reload_predict":
            hbm[job["digest"]] = self.host[gpu].pop(job["digest"])
        if job["kind"] == "fit_predict" and job["keep_cache"]:
            hbm[job["digest"]] = cache_bytes
        return {
            "ok": True,
            "oom": False,
            "seconds": seconds,
            "peak_bytes": 1e8 + (cache_bytes if job["kind"] == "fit_predict" else 0),
            "cache_bytes": cache_bytes,
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

    def put(self, tenant: str, digest: str, nbytes: float) -> None:
        self.bucket[(tenant, digest)] = nbytes

    def delete(self, tenant: str, digest: str) -> None:
        self.bucket.pop((tenant, digest), None)


@pytest.fixture
def backend():
    return FakeBackend()


@pytest.fixture
def cfg():
    return copy.deepcopy(config.platform())
