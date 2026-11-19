# 0003 Memory is planned, and learned from errors

## Context
With whole-context attention, a million-row context runs a card out of memory long before its KV cache does.
Chunking the attention (train-side query chunks, test-side row chunks) bounds the transient memory without changing
an answer. Smaller chunks cost time. Benchmarks give a prior for the memory model, but a card's usable memory depends
on the driver, the allocator and what else is resident, and no prior knows that.

## Decision
Before a job starts, the plane picks the largest chunks from a ladder for which weights, resident caches, the new
cache and the transient fit the card's usable-capacity estimate. If nothing fits even at the smallest chunks, it
evicts caches. Transient memory is planned at the 98th percentile of measured/predicted over recent jobs. An
out-of-memory error lowers that card's capacity estimate by 3%, never below 60% of nominal, and the retried job steps
down one rung of the ladder.

## Consequences
The plane has a few out-of-memory errors a day instead of thousands (`fixed_chunks`, `no_learning`). A capacity that
only ever goes down can under-use a card after a burst of unlucky errors. The floor bounds that; raising it again
from evidence is future work.
