"""The model's serving properties: the cache and chunking never change an answer."""

from __future__ import annotations

import numpy as np
import pytest

from tcs.model.icl import fit, kv_bytes_per_row, predict_proba, predict_reference
from tcs.model.prior import task


def test_cached_prediction_equals_the_full_forward_pass(model):
    x, y = task(np.random.default_rng(1), 260, 9, 3)
    ref = predict_reference(model, x[:200], y[:200], x[200:])
    ctx = fit(model, x[:200], y[:200], train_chunk=10**9)
    assert np.abs(predict_proba(model, ctx, x[200:], 10**9) - ref).max() < 1e-5


@pytest.mark.parametrize("train_chunk,test_chunk", [(1, 1), (5, 3), (64, 17), (199, 60)])
def test_chunk_sizes_change_memory_not_answers(model, train_chunk, test_chunk):
    x, y = task(np.random.default_rng(2), 260, 14, 4)
    ref = predict_reference(model, x[:200], y[:200], x[200:])
    ctx = fit(model, x[:200], y[:200], train_chunk=train_chunk)
    assert np.abs(predict_proba(model, ctx, x[200:], test_chunk) - ref).max() < 1e-5


def test_test_rows_are_independent(model):
    """A row's prediction does not depend on which other rows came in the same request (so batching is safe)."""
    x, y = task(np.random.default_rng(3), 300, 6, 2)
    ctx = fit(model, x[:200], y[:200], 64)
    alone = predict_proba(model, ctx, x[200:201], 1)
    together = predict_proba(model, ctx, x[200:], 32)[:1]
    assert np.abs(alone - together).max() < 1e-6


def test_cache_size_is_exact(model):
    x, y = task(np.random.default_rng(4), 150, 5, 2)
    assert fit(model, x, y, 32).kv_bytes == 150 * kv_bytes_per_row(model.spec)


def test_wide_tables_and_missing_values(model):
    rng = np.random.default_rng(5)
    x, y = task(rng, 120, 80, 3)  # wider than the model: projected
    x[rng.random(x.shape) < 0.1] = np.nan
    p = predict_proba(model, fit(model, x[:100], y[:100], 16), x[100:], 8)
    assert p.shape == (20, len(np.unique(y[:100]))) and np.allclose(p.sum(1), 1, atol=1e-5)


def test_too_many_classes_is_refused(model):
    x = np.random.default_rng(6).standard_normal((40, 3)).astype(np.float32)
    with pytest.raises(ValueError):
        fit(model, x, np.arange(40) % 12, 8)


def test_the_model_learns_in_the_forward_pass(model):
    """Trained only on the synthetic prior, it beats chance clearly on an unseen task."""
    x, y = task(np.random.default_rng(42), 600, 8, 2)
    p = predict_proba(model, fit(model, x[:400], y[:400], 128), x[400:], 128)
    chance = max(np.mean(y[400:] == c) for c in np.unique(y))
    assert np.mean(np.unique(y[:400])[p.argmax(1)] == y[400:]) > chance + 0.1
