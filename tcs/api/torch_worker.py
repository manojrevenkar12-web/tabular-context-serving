"""A worker that runs the real PyTorch model, behind the same protocol as the simulated fleet.

Caches are real: a fitted context's keys and values stay in the worker's memory ("hbm") or are moved to a host copy
("host"). Peak memory is reported from the tensors the job holds at its widest point (on a GPU worker this is
`torch.cuda.max_memory_allocated`); the plane learns from it the same way it does in the simulation.
"""

from __future__ import annotations

import time

import numpy as np

from tcs.model.icl import FittedContext, TinyICL, fit, predict_proba
from tcs.plane.store import Bucket, decode


def _attention_bytes(spec, rows: int, query_rows: int) -> int:
    return 4 * spec.heads * min(query_rows, max(1, rows)) * rows


class TorchWorker:
    def __init__(self, model: TinyICL, bucket: Bucket, device: str = "cpu-0") -> None:
        self.model = model
        self.bucket = bucket
        self.device = device
        self.hbm: dict[str, FittedContext] = {}
        self.host: dict[str, FittedContext] = {}
        self.owner: dict[str, str] = {}

    def run(self, gpu: str, job: dict, now: float) -> dict:
        t0 = time.perf_counter()
        digest, kind = job["digest"], job["kind"]
        spec = self.model.spec
        peak = 0
        if kind == "fit_predict":
            blob = self.bucket.get(job["tenant"], digest)
            if blob is None:
                return {"ok": False, "oom": False, "seconds": 0.0, "error": "missing_dataset"}
            x, y = decode(blob)
            ctx = fit(self.model, x, y, train_chunk=job["train_chunk"])
            peak = ctx.kv_bytes + _attention_bytes(spec, ctx.rows, job["train_chunk"])
            if job["keep_cache"]:
                self.hbm[digest] = ctx
                self.owner[digest] = job["tenant"]
        elif kind == "reload_predict":
            if digest not in self.host:
                return {"ok": False, "oom": False, "seconds": 0.0, "error": "cache_miss"}
            ctx = self.host.pop(digest)
            self.hbm[digest] = ctx
        else:
            if digest not in self.hbm:
                return {"ok": False, "oom": False, "seconds": 0.0, "error": "cache_miss"}
            ctx = self.hbm[digest]
        results = {}
        for rid, x_test in zip(job["requests"], job["payloads"], strict=True):
            probs = predict_proba(self.model, ctx, np.asarray(x_test, dtype=np.float32), test_chunk=job["test_chunk"])
            results[rid] = {"classes": ctx.classes.tolist(), "probabilities": probs.round(6).tolist()}
        peak = max(peak, _attention_bytes(spec, ctx.rows, min(job["test_chunk"], max(1, job["test_rows"]))))
        return {
            "ok": True,
            "oom": False,
            "seconds": time.perf_counter() - t0,
            "peak_bytes": float(peak),
            "cache_bytes": float(ctx.kv_bytes),
            "results": results,
            "error": "",
        }

    # -- worker protocol ---------------------------------------------------------------------------------------------
    def evict(self, gpu: str, digest: str, to_host: bool) -> bool:
        ctx = self.hbm.pop(digest, None)
        if ctx is None:
            return False
        if to_host:
            ctx.kv = [(k.clone(), v.clone()) for k, v in ctx.kv]  # a separate copy, as a host-RAM tier would hold
            self.host[digest] = ctx
        return True

    def purge(self, gpu: str, digest: str) -> None:
        for tier in (self.hbm, self.host):
            ctx = tier.pop(digest, None)
            if ctx is not None:
                for k, v in ctx.kv:
                    k.zero_()
                    v.zero_()
        self.owner.pop(digest, None)

    def residents(self, gpu: str) -> dict:
        return {"hbm": {d: c.kv_bytes for d, c in self.hbm.items()}, "host": {d: c.kv_bytes for d, c in self.host.items()}}

    def put(self, tenant: str, digest: str, nbytes: float) -> None:
        """The API writes the object itself before telling the plane; nothing to do here."""

    def delete(self, tenant: str, digest: str) -> None:
        self.bucket.delete(tenant, digest)
