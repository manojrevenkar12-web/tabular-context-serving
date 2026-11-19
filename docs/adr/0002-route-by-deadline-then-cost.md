# 0002 Route by deadline, then by cost

## Context
A request can run where its context is cached (predict only), where it sits in host RAM (reload, then predict), or
anywhere else (fit, then predict). Least-loaded routing ignores the caches; fastest-completion routing sends
everything to the H100s, which cost about twice an L4's price per unit of work.

## Decision
For each GPU the plane estimates completion as queue wait plus the job's own time on that card. Among GPUs expected to
finish within half the time left on the client's SLO, it picks the cheapest in euros; if none qualifies, the fastest.
A predict for a context whose fit is already queued or running waits for that fit (single-flight) instead of
fitting a second copy elsewhere.

## Consequences
Batch work with slack drifts to cheap cards and interactive work to fast ones. The ablation `no_cache_routing`
shows the cost of ignoring caches. The SLO margin is a setting (`[routing] slo_margin`).
