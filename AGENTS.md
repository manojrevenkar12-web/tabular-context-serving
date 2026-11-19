# Working on this repository (people and coding agents)

## Fast feedback

```bash
scripts/check.sh                       # ruff, then every test (about two minutes)
python3 -m evals.run --hours 1         # every design, one simulated hour, nothing written
python3 -m evals.run                   # the dev campaign -> reports/dev.json (about five minutes on two cores)
python3 -m evals.charts && python3 -m evals.readme && python3 -m evals.console_data
cd web && npm run typecheck && npm run lint && npm test
```

## Invariants (each has a test or a gate)

- **The truth stays in the world.** `world/` holds the GPUs' real memory and speed, the traffic and the bucket.
  `tcs/` reaches it only through the worker protocol in `docs/icd/worker.md` and never imports `world/` or
  `evals/` (`tests/test_invariants.py`). Only `evals/` holds both sides, to score a run.
- **The world does not read the product's settings.** `world/` reads `config/world.toml` or
  `config/heldout.toml`, never `config/platform.toml`, `config/designs.toml` or `config/api.toml`.
- **Same luck for every design.** Every random draw comes from a stream keyed by the seed and a name; a job's
  noise is keyed by what the job is, not by when or why a design ran it. Every design runs on the same seeds.
- **The model reads, code decides.** Routing, chunking, eviction, quotas and erasure are code. The advisor's pool
  is computed by replay; a model may phrase the recommendation and `tcs.advisor.sizing.check` refuses any number
  the plan does not hold.
- **Chunking and caching never change an answer.** Cached and chunked predictions equal the uncached forward
  pass (`tests/test_model.py`); batching is exact because test rows never attend to each other.
- **One execution and one charge per idempotency key.** The ledger's primary key is the key; a retry is answered
  from the first result (`tests/test_plane.py`, gate `double_charges`).
- **Erasure is verified with the workers.** Objects, queued work and every cache tier are purged, then every
  worker is asked what it holds; later requests get 410 (gates `served_after_erasure`, `erasure_residual_bytes`).
- **Settings live in config.** Thresholds, priors, ladders and quotas are in `config/platform.toml`; a test fails
  on a setting nobody reads and on a decimal literal in `tcs/plane/plane.py`.
- **Reports are generated.** `reports/*.json` come from `evals/` and `scripts/bench.py`; the README's results
  block comes from `evals/readme.py`. Never edit either by hand; CI checks the README is current.

## Held-out rule

`config/heldout.toml` is a world the plane was not tuned on: the same GPU inventory with different hidden memory
and speed coefficients, allocator overhead, noise, a throttled L4, and different traffic. `python3 -m evals.run
--heldout` runs it once: it refuses to start with uncommitted changes under `tcs/ config/ evals/ world/`, writes
`reports/heldout.lock` with the commit, and refuses to run again once `reports/heldout.json` exists. Never change
the plane to pass the held-out campaign. If you study its failures, say so in `docs/heldout.md` and write a new
held-out world before claiming anything about a fix.

## Where things are

| Path | What |
|---|---|
| `tcs/model/` | the small in-context model and its synthetic prior |
| `tcs/plane/` | the plane, memory and time models, ledger, content-addressed store |
| `tcs/api/` | FastAPI and the PyTorch worker |
| `tcs/advisor/` | the sizing advisor |
| `world/` | simulated GPUs, bucket, traffic, luck (the truth) |
| `evals/` | operate, score, gates, held-out lock, charts, README block, console data, advisor scoring |
| `web/` | operator console (React, TypeScript, Framer Motion), feature folders |
| `deploy/` | Terraform (GKE, GCS, KMS, Workload Identity) and the Helm chart |
| `docs/adr/` | decisions and why |
