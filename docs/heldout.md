# Held-out world

`config/heldout.toml` was written before the held-out run and was not used for tuning. Its GPU inventory is the same
as the dev world's, because the plane is configured with that inventory. Everything else hidden is different:

| | Dev world | Held-out world |
|---|---|---|
| Fit cost growth with context rows | rows^1.30 | rows^1.36 |
| KV cache bytes per row (per estimator) | 4300 | 4700 |
| Train-side attention memory coefficient | 64 | 71 |
| Test-side attention memory coefficient | 104 | 98 |
| Per-job noise (lognormal sigma) | 0.08 | 0.12 |
| Memory lost per resident cache (fragmentation) | 0.2% | 0.3% |
| H100 driver and allocator overhead | 7% | 8.5% |
| One L4's speed | 0.18 | 0.14 (thermally throttled) |
| Tenants | 36 | 50 |
| Sessions per hour | 340 | 300 |
| Share of large contexts | Pareto alpha 0.55 | 0.45 (more large ones) |
| Interactive share | 60% | 70% |
| Flooding tenant's share of sessions | 28% | 35% |
| Client retries | 3% | 5% |
| Seeds | 101, 202, 303, 404 | 9011, 9022, 9033, 9044 |

The run is locked: `reports/heldout.lock` records the commit, and the runner refuses a second run. Results are in the
README, as they came out.

## Result (run once at the commit in reports/heldout.lock)

The platform still leads every other design, but it **fails the SLO gate**: 94.4% of requests within their SLO,
against a gate of 97%. The cost, out-of-memory, idempotency, erasure and fairness gates pass.

- **What fell:** interactive requests. 92.5% of them met their SLO, against 98.8% in the dev world, and their p95
  went from 2.9 s to 24.9 s. Batch requests held up (98.8%).
- **What changed:** the fleet did about twice the work (busy GPU cost EUR 14.0 per hour against EUR 6.7), because
  of more large contexts, slower fits (rows^1.36) and a throttled L4.
- **Likely cause (a hypothesis, not yet tested):** a GPU runs one job at a time and nothing preempts. An
  interactive request routed to a card that has just started fitting a large context waits for that whole fit. The
  fair queue orders tenants, not urgency.
- **Candidate fixes, for a new held-out world:**
  - a priority lane for interactive work;
  - keeping one card per kind free of large fits;
  - yielding between attention chunks, which the chunked fit already makes possible.

Two ablations matter more here than in the dev world, which is useful in itself: without same-context batching the
SLO share drops to 91.3%, and with FIFO queues and no quotas to 77.5%.

Nothing in the plane was changed after this run. Per AGENTS.md, any fix needs a new held-out world before anything
is claimed about it.
