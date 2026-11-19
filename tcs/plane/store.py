"""Content-addressed dataset uploads.

A dataset's id is the SHA-256 of its canonical bytes (shape, float32 features in C order, int64 labels), so the same
upload twice is stored once and a cache built for it is found again. Addresses are per tenant: two tenants who
upload the same table get two objects. Sharing across tenants would save bytes but would tell a tenant, by timing,
that someone else holds the same data, and would tie one tenant's erasure to another's object (ADR 0004).
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import Protocol

import numpy as np


class Bucket(Protocol):
    def put(self, tenant: str, digest: str, blob: bytes) -> None: ...
    def get(self, tenant: str, digest: str) -> bytes | None: ...
    def delete(self, tenant: str, digest: str) -> None: ...
    def list(self, tenant: str) -> list[str]: ...


def canonical(x: np.ndarray, y: np.ndarray) -> bytes:
    x = np.ascontiguousarray(np.asarray(x, dtype="<f4"))
    y = np.ascontiguousarray(np.asarray(y, dtype="<i8"))
    if x.ndim != 2 or y.ndim != 1 or x.shape[0] != y.shape[0]:
        raise ValueError(f"expected X (rows, features) and y (rows,), got {x.shape} and {y.shape}")
    head = np.asarray(x.shape, dtype="<i8").tobytes()
    return head + x.tobytes() + y.tobytes()


def digest(x: np.ndarray, y: np.ndarray) -> str:
    return hashlib.sha256(canonical(x, y)).hexdigest()


def encode(x: np.ndarray, y: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.savez(buf, x=np.asarray(x, dtype=np.float32), y=np.asarray(y, dtype=np.int64))
    return buf.getvalue()


def decode(blob: bytes) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(io.BytesIO(blob), allow_pickle=False)
    return z["x"], z["y"]


class LocalBucket:
    """A directory per tenant; GCS in production (deploy/terraform: CMEK, versioning off, uniform access)."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def _path(self, tenant: str, digest: str) -> Path:
        if not tenant.isalnum() or not all(c in "0123456789abcdef" for c in digest):
            raise ValueError("bad tenant or digest")
        return self.root / tenant / f"{digest}.npz"

    def put(self, tenant: str, digest: str, blob: bytes) -> None:
        p = self._path(tenant, digest)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_bytes(blob)
        tmp.replace(p)

    def get(self, tenant: str, digest: str) -> bytes | None:
        p = self._path(tenant, digest)
        return p.read_bytes() if p.exists() else None

    def delete(self, tenant: str, digest: str) -> None:
        self._path(tenant, digest).unlink(missing_ok=True)

    def list(self, tenant: str) -> list[str]:
        d = self.root / tenant
        return sorted(p.stem for p in d.glob("*.npz")) if d.exists() else []
