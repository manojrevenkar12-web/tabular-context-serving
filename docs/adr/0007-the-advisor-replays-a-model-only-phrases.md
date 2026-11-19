# 0007 The sizing advisor replays; a model only phrases

## Context
Customers running the open weights in their own cloud ask which GPUs to buy, and how many. A language model asked
that directly will produce a confident and unverifiable number.

## Decision
The advisor replays the customer's request log through this plane on each candidate pool. Each worker in the replay
behaves as the plane's own estimates say, calibrated by the time and memory corrections the managed fleet learned
from telemetry. Capacity is left out of the calibration: it is learned per card. The advisor picks the cheapest pool
whose replay meets the target plus a planning margin. A model may write the recommendation, and `check` refuses any
number the plan does not hold.

## Consequences
`evals/advisor_eval.py` runs every candidate pool in the simulated world to find what really meets the target, and
scores the advisor against two rules a person might use instead.
