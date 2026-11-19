"""A small in-context tabular model with the serving properties of TabPFN.

Learning happens in the forward pass: training rows (features plus label) and test rows (features only) are tokens.
Training rows attend to each other; test rows attend to the training rows and never to each other. Two things
follow, and the serving plane is built on both:

* The training side does not depend on the test rows, so "fit" can run the training rows through every layer once
  and keep each layer's keys and values (the KV cache). A later predict runs only the test rows against the cache.
* Test rows are independent of each other, so predicting them in chunks of any size gives the same answer, and the
  training side's attention can also be computed in query chunks. Chunk sizes trade peak memory for time, never
  accuracy.

This is not TabPFN. It has about 140k parameters, trained for minutes on a CPU on a synthetic prior
(`scripts/pretrain.py`), so the serving code can run and be tested against a real PyTorch model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class ModelSpec:
    d_model: int = 64
    heads: int = 4
    layers: int = 4
    ff: int = 128
    max_features: int = 32
    max_classes: int = 10


class Block(nn.Module):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.heads = spec.heads
        self.dh = spec.d_model // spec.heads
        self.ln1 = nn.LayerNorm(spec.d_model)
        self.ln2 = nn.LayerNorm(spec.d_model)
        self.q = nn.Linear(spec.d_model, spec.d_model)
        self.kv = nn.Linear(spec.d_model, 2 * spec.d_model)
        self.o = nn.Linear(spec.d_model, spec.d_model)
        self.ffn = nn.Sequential(nn.Linear(spec.d_model, spec.ff), nn.GELU(), nn.Linear(spec.ff, spec.d_model))

    def _split(self, x: torch.Tensor) -> torch.Tensor:  # (..., n, d) -> (..., h, n, dh)
        return x.unflatten(-1, (self.heads, self.dh)).transpose(-3, -2)

    def keys_values(self, h_train: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        k, v = self.kv(self.ln1(h_train)).chunk(2, dim=-1)
        return self._split(k), self._split(v)

    def attend(self, h: torch.Tensor, k: torch.Tensor, v: torch.Tensor, chunk: int) -> torch.Tensor:
        """Rows of h attend to (k, v); queries are processed `chunk` rows at a time (exact, bounded memory)."""
        out = []
        for s in range(0, h.shape[-2], max(1, chunk)):
            hs = h[..., s : s + chunk, :]
            q = self._split(self.q(self.ln1(hs)))
            a = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(self.dh), dim=-1) @ v
            hs = hs + self.o(a.transpose(-3, -2).flatten(-2))
            out.append(hs + self.ffn(self.ln2(hs)))
        return torch.cat(out, dim=-2)


class TinyICL(nn.Module):
    def __init__(self, spec: ModelSpec | None = None) -> None:
        super().__init__()
        self.spec = spec or ModelSpec()
        s = self.spec
        self.x_proj = nn.Linear(s.max_features, s.d_model)
        self.y_emb = nn.Embedding(s.max_classes, s.d_model)
        self.blocks = nn.ModuleList(Block(s) for _ in range(s.layers))
        self.head = nn.Linear(s.d_model, s.max_classes)

    # Reference path: one forward over training and test rows together, no cache, no chunking.
    def forward(self, x_train: torch.Tensor, y_train: torch.Tensor, x_test: torch.Tensor) -> torch.Tensor:
        """x_* are (batch, rows, max_features), y_train (batch, rows) -> logits (batch, test rows, max_classes)."""
        h_tr = self.x_proj(x_train) + self.y_emb(y_train)
        h_te = self.x_proj(x_test)
        for b in self.blocks:
            k, v = b.keys_values(h_tr)
            n_tr = h_tr.shape[-2]
            both = b.attend(torch.cat([h_tr, h_te], dim=-2), k, v, chunk=10**9)
            h_tr, h_te = both[..., :n_tr, :], both[..., n_tr:, :]
        return self.head(h_te)


@dataclass
class FittedContext:
    """What "fit" leaves behind: preprocessing state and every layer's keys and values for the training rows."""

    mean: np.ndarray
    std: np.ndarray
    projection: np.ndarray | None
    classes: np.ndarray
    kv: list[tuple[torch.Tensor, torch.Tensor]]
    rows: int

    @property
    def kv_bytes(self) -> int:
        return sum(k.nbytes + v.nbytes for k, v in self.kv)


def _projection(features: int, width: int) -> np.ndarray | None:
    """Datasets wider than the model see a fixed random orthonormal projection (deterministic in the width)."""
    if features <= width:
        return None
    rng = np.random.default_rng(features)
    q, _ = np.linalg.qr(rng.standard_normal((features, width)))
    return q.astype(np.float32)


def _prepare(x: np.ndarray, mean: np.ndarray, std: np.ndarray, proj: np.ndarray | None, width: int) -> torch.Tensor:
    z = (np.asarray(x, dtype=np.float32) - mean) / std
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    if proj is not None:
        z = z @ proj
    out = np.zeros((z.shape[0], width), dtype=np.float32)
    out[:, : z.shape[1]] = z
    return torch.from_numpy(out)


@torch.no_grad()
def fit(model: TinyICL, x: np.ndarray, y: np.ndarray, train_chunk: int) -> FittedContext:
    """Run the training rows through every layer once and keep the keys and values."""
    width = model.spec.max_features
    x = np.asarray(x, dtype=np.float32)
    mean = np.nanmean(x, axis=0)
    std = np.nanstd(x, axis=0)
    mean = np.nan_to_num(mean)
    std = np.where(np.nan_to_num(std) > 0, np.nan_to_num(std), 1.0).astype(np.float32)
    proj = _projection(x.shape[1], width)
    classes, codes = np.unique(np.asarray(y), return_inverse=True)
    if len(classes) > model.spec.max_classes:
        raise ValueError(f"at most {model.spec.max_classes} classes, got {len(classes)}")
    h = model.x_proj(_prepare(x, mean, std, proj, width)) + model.y_emb(torch.from_numpy(codes.astype(np.int64)))
    kv = []
    for b in model.blocks:
        k, v = b.keys_values(h)
        kv.append((k.contiguous(), v.contiguous()))
        h = b.attend(h, k, v, chunk=train_chunk)
    return FittedContext(mean.astype(np.float32), std, proj, classes, kv, int(x.shape[0]))


@torch.no_grad()
def predict_proba(model: TinyICL, ctx: FittedContext, x_test: np.ndarray, test_chunk: int) -> np.ndarray:
    """Test rows against the cached keys and values, `test_chunk` rows at a time."""
    width = model.spec.max_features
    h = model.x_proj(_prepare(x_test, ctx.mean, ctx.std, ctx.projection, width))
    out = []
    for s in range(0, h.shape[0], max(1, test_chunk)):
        hs = h[s : s + test_chunk]
        for b, (k, v) in zip(model.blocks, ctx.kv, strict=True):
            hs = b.attend(hs, k, v, chunk=test_chunk)
        logits = model.head(hs)[:, : len(ctx.classes)]
        out.append(torch.softmax(logits, dim=-1))
    return torch.cat(out).numpy()


@torch.no_grad()
def predict_reference(model: TinyICL, x: np.ndarray, y: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    """The uncached, unchunked forward pass: what every cached or chunked path must reproduce."""
    ctx = fit(model, x, y, train_chunk=10**9)  # only for the preprocessing state
    width = model.spec.max_features
    xt = _prepare(x, ctx.mean, ctx.std, ctx.projection, width)[None]
    xs = _prepare(x_test, ctx.mean, ctx.std, ctx.projection, width)[None]
    _, codes = np.unique(np.asarray(y), return_inverse=True)
    logits = model(xt, torch.from_numpy(codes.astype(np.int64))[None], xs)[0, :, : len(ctx.classes)]
    return torch.softmax(logits, dim=-1).numpy()


def kv_bytes_per_row(spec: ModelSpec) -> int:
    """Exact cache size: keys and values, every layer, float32."""
    return spec.layers * 2 * spec.d_model * 4


def load(path: str) -> TinyICL:
    blob = torch.load(path, map_location="cpu", weights_only=True)
    model = TinyICL(ModelSpec(**blob["spec"]))
    model.load_state_dict(blob["state"])
    model.eval()
    return model
