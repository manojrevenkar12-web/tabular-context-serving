"""What the plane believes a job will cost, corrected from what workers report.

Every job's cost has a known shape (the architecture fixes it: caches grow with rows, attention with chunk rows
times the square root of the context after row compression). The coefficients start at the team's benchmark prior
and are corrected from telemetry: the measured peak memory and seconds of every finished job, and every
out-of-memory error. Memory is planned at an upper quantile of measured/predicted, time at the running mean.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class JobShape:
    kind: str  # fit_predict | predict | reload_predict
    rows: int
    features: int
    test_rows: int
    train_chunk: int
    test_chunk: int


class MemoryModel:
    def __init__(self, prior: dict, cfg: dict, estimators: int, learn: bool) -> None:
        self.p = prior
        self.cfg = cfg
        self.e = estimators
        self.learn = learn
        self.window = int(cfg["window"])
        self.q = float(cfg["safety_quantile"])
        self.min_samples = int(cfg["min_samples"])
        self.ratios: dict[str, deque] = {"fit_predict": deque(maxlen=self.window), "predict": deque(maxlen=self.window)}
        self.cache_ratio: deque = deque(maxlen=self.window)
        self._safety = {"fit_predict": 1.0, "predict": 1.0}
        self._cache_scale = 1.0

    def cache_bytes(self, rows: int, features: int) -> float:
        return self.e * rows * (self.p["cache_bytes_per_row"] + self.p["cache_bytes_per_cell"] * features) * self._cache_scale

    def _transient_prior(self, s: JobShape) -> float:
        root = math.sqrt(s.rows)
        pred = self.p["predict_bytes_per_unit"] * min(s.test_chunk, max(1, s.test_rows)) * root * self.e
        if s.kind != "fit_predict":
            return pred
        fit = self.p["fit_bytes_per_cell"] * s.rows * s.features + self.p["fit_bytes_per_chunk_unit"] * min(s.train_chunk, s.rows) * root * self.e
        return max(fit, pred)

    def safety(self, kind: str) -> float:
        return self._safety["fit_predict" if kind == "fit_predict" else "predict"]

    def _refresh(self) -> None:
        for kind, r in self.ratios.items():
            self._safety[kind] = max(1.0, float(np.quantile(r, self.q))) if self.learn and len(r) >= self.min_samples else 1.0
        self._cache_scale = float(np.median(self.cache_ratio)) if self.learn and len(self.cache_ratio) >= self.min_samples else 1.0

    def transient(self, s: JobShape) -> float:
        return self._transient_prior(s) * self.safety(s.kind)

    def observe(self, s: JobShape, peak_bytes: float, cache_bytes: float) -> None:
        """A finished job's measured peak (its transient, plus the cache it built) and the cache it kept."""
        transient = peak_bytes - (cache_bytes if s.kind == "fit_predict" else 0.0)
        prior = self._transient_prior(s)
        if prior > 0 and transient > 0:
            self.ratios["fit_predict" if s.kind == "fit_predict" else "predict"].append(transient / prior)
        if s.kind == "fit_predict" and cache_bytes > 0:
            base = self.e * s.rows * (self.p["cache_bytes_per_row"] + self.p["cache_bytes_per_cell"] * s.features)
            self.cache_ratio.append(cache_bytes / base)
        self._refresh()


class TimeModel:
    def __init__(self, prior: dict, cfg: dict, estimators: int, learn: bool) -> None:
        self.p = prior
        self.e = estimators
        self.learn = learn
        self.lr = float(cfg["learning_rate"])
        self.ratio: dict[tuple[str, str], float] = {}

    def _prior(self, s: JobShape, speed: float, reload_bytes: float = 0.0) -> float:
        p, n, m = self.p, s.rows, s.test_rows
        t = p["job_overhead_s"]
        if s.kind == "fit_predict":
            t += (p["fit_s_per_cell"] * n * s.features + p["fit_s_per_row_pow"] * n ** p["row_exponent"]) / speed
            t += math.ceil(n / max(1, s.train_chunk)) * self.e * p["chunk_overhead_s"]
        if s.kind == "reload_predict":
            t += reload_bytes / p["pcie_bytes_per_s"]
        t += p["predict_s_per_unit"] * m * math.sqrt(n) / speed
        t += math.ceil(max(1, m) / max(1, s.test_chunk)) * self.e * p["chunk_overhead_s"]
        return t

    def seconds(self, s: JobShape, gpu_kind: str, speed: float, reload_bytes: float = 0.0) -> float:
        return self._prior(s, speed, reload_bytes) * self.ratio.get((gpu_kind, s.kind), 1.0)

    def observe(self, s: JobShape, gpu_kind: str, speed: float, seconds: float, reload_bytes: float = 0.0) -> None:
        if not self.learn or seconds <= 0:
            return
        r = seconds / self._prior(s, speed, reload_bytes)
        key = (gpu_kind, s.kind)
        old = self.ratio.get(key, 1.0)
        self.ratio[key] = math.exp((1 - self.lr) * math.log(old) + self.lr * math.log(r))
