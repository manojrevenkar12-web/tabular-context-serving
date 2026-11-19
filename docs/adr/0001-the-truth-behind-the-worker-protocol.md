# 0001 The truth behind the worker protocol

## Context
The plane's quality is measured by what happens on GPUs it does not control: how much memory a card really has free
once the driver and allocator take theirs, how long a fit really takes, when an allocation fails. If the plane could
read those numbers, the evaluation would measure nothing.

## Decision
`world/` holds the truth (hidden reserved memory, fragmentation, true coefficients, noise). The plane reaches it only
through the worker protocol (`docs/icd/worker.md`), which carries what a real worker reports: measured peak memory
(`torch.cuda.max_memory_allocated` on a GPU), the cache it kept, seconds, and out-of-memory errors. The PyTorch
worker behind the API implements the same protocol. Only `evals/` sees both sides.

## Consequences
The plane must learn capacity and cost from telemetry, as it would in production. An import test keeps `tcs/` away
from `world/`, and the world never reads the plane's settings.
