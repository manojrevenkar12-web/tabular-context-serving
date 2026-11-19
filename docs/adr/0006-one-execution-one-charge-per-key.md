# 0006 One execution and one charge per idempotency key

## Context
Clients retry when a request is slow. Without keys, a retry of a large predict runs a second time and is charged a
second time.

## Decision
The plane keeps, per idempotency key, either the waiting requests or the finished outcome. A retry joins the first or
receives its result. The ledger's primary key is the key, so a second insert is ignored. API keys are scoped to
tenants (`tenant:key`).

## Consequences
The status quo double-charges every retry in the evaluation; the plane charges none. The demo ledger is SQLite;
production would use Cloud SQL with the same primary-key rule.
