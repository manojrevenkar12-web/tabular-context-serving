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
