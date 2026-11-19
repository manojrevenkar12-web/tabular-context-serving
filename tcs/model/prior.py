"""The synthetic prior the small model is trained on: random classification tasks from three generator families.

Each task draws correlated features, some of them irrelevant, and labels from a random linear map, a random
one-hidden-layer network or nearest-centroid clusters, with label noise. The model never sees a real dataset in
training; it learns to classify from the rows it is given, which is the TabPFN idea at toy scale.
"""

from __future__ import annotations

import numpy as np


def task(rng: np.random.Generator, rows: int, features: int, classes: int) -> tuple[np.ndarray, np.ndarray]:
    a = rng.standard_normal((features, features)) * rng.uniform(0.0, 0.7)
    x = rng.standard_normal((rows, features)) @ (np.eye(features) + a)
    relevant = rng.random(features) < rng.uniform(0.4, 1.0)
    relevant[rng.integers(features)] = True
    xr = x * relevant
    kind = rng.integers(3)
    if kind == 0:
        logits = xr @ rng.standard_normal((features, classes))
    elif kind == 1:
        h = np.tanh(xr @ rng.standard_normal((features, 16)) * rng.uniform(0.5, 2.0))
        logits = h @ rng.standard_normal((16, classes)) * 2.0
    else:
        centres = rng.standard_normal((classes, features)) * rng.uniform(0.8, 2.0)
        logits = -(((xr[:, None, :] - centres[None] * relevant) ** 2).sum(-1))
    temp = rng.uniform(0.05, 0.6) * logits.std()
    y = np.argmax(logits + rng.gumbel(size=logits.shape) * temp, axis=1)
    x = x * rng.uniform(0.2, 5.0, size=features) + rng.normal(0, 3, size=features)
    missing = rng.random(x.shape) < rng.choice([0.0, 0.0, 0.05])
    x[missing] = np.nan
    return x.astype(np.float32), y.astype(np.int64)
