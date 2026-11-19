"""Same luck for every design: every random draw comes from a stream keyed by the seed and a name.

A job's noise is keyed by what the job is (dataset, kind, rows), not by when or why a design ran it, so two designs
that run the same work see the same noise.
"""

from __future__ import annotations

import zlib

import numpy as np


def stream(seed: int, *names: object) -> np.random.Generator:
    key = [zlib.crc32(str(n).encode()) for n in names]
    return np.random.default_rng(np.random.SeedSequence([seed, *key]))


def unit_normal(seed: int, *names: object) -> float:
    return float(stream(seed, *names).standard_normal())
