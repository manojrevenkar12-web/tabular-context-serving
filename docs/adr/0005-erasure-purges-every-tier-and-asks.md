# 0005 Erasure purges every tier and asks the workers

## Context
Deleting a tenant's objects from the bucket is not erasure if its KV caches stay in GPU memory or host RAM. A cache
is a function of the data. Queued work can also bring the data back. GCS keeps soft-deleted objects for seven days
by default.

## Decision
`Plane.erase`:
1. Deletes the objects.
2. Drops the tenant's queued jobs.
3. Purges its caches from HBM and host RAM on every worker; a running job's caches are purged when it ends.
4. Asks every worker what it still holds. The erasure counts as complete only when no worker holds anything.

Later requests for those datasets get 410. The bucket has soft delete and versioning off (`deploy/terraform`).

## Consequences
The gates `served_after_erasure` and `erasure_residual_bytes` must be zero. The LLM-style design shows what is left
behind otherwise.
