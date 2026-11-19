"""Train the small in-context model on the synthetic prior and save it to models/tiny_icl.pt.

    python3 scripts/pretrain.py --steps 4000

CPU only, a few minutes. The weights are committed so tests and the API do not need to train.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tcs.model.icl import ModelSpec, TinyICL, _prepare  # noqa: E402
from tcs.model.prior import task  # noqa: E402


def batch(rng: np.random.Generator, spec: ModelSpec, size: int):
    n_train = int(rng.integers(24, 257))
    n_test = 64
    features = int(rng.integers(2, spec.max_features + 1))
    classes = int(rng.integers(2, 6))
    xs, ys, xt, yt = [], [], [], []
    for _ in range(size):
        x, y = task(rng, n_train + n_test, features, classes)
        mean = np.nan_to_num(np.nanmean(x[:n_train], 0))
        std = np.nan_to_num(np.nanstd(x[:n_train], 0))
        std = np.where(std > 0, std, 1.0)
        xs.append(_prepare(x[:n_train], mean, std, None, spec.max_features))
        xt.append(_prepare(x[n_train:], mean, std, None, spec.max_features))
        ys.append(torch.from_numpy(y[:n_train]))
        yt.append(torch.from_numpy(y[n_train:]))
    return torch.stack(xs), torch.stack(ys), torch.stack(xt), torch.stack(yt), classes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "models" / "tiny_icl.pt"))
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    spec = ModelSpec()
    model = TinyICL(spec)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=args.steps, pct_start=0.1)
    t0, running = time.time(), None
    for step in range(args.steps):
        xs, ys, xt, yt, classes = batch(rng, spec, 16)
        logits = model(xs, ys, xt)[..., :classes]
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, classes), yt.reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        running = loss.item() if running is None else 0.98 * running + 0.02 * loss.item()
        if step % 250 == 0 or step == args.steps - 1:
            print(f"step {step:5d} loss {running:.3f} {time.time() - t0:6.0f}s", flush=True)
    Path(args.out).parent.mkdir(exist_ok=True)
    torch.save({"spec": spec.__dict__, "state": model.state_dict(), "steps": args.steps}, args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
