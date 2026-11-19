"""Settings: config/platform.toml for the plane, config/designs.toml for the designs compared."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from functools import cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@cache
def platform() -> dict:
    return tomllib.loads((ROOT / "config" / "platform.toml").read_text())


@dataclass(frozen=True)
class Design:
    name: str
    description: str
    cache: str  # none | lru | value
    routing: str  # round_robin | least_loaded | cache_aware
    admission: str  # fixed | memory_model
    batching: bool
    queue: str  # fifo | fair
    idempotency: bool
    quota: bool
    host_tier: bool
    learn: bool
    erasure: str  # bucket_only | purge_verify


@cache
def designs() -> dict[str, Design]:
    raw = tomllib.loads((ROOT / "config" / "designs.toml").read_text())
    return {name: Design(name=name, **d) for name, d in raw.items()}


def design(name: str) -> Design:
    return designs()[name]
