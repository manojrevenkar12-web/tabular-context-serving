# 0004 Content addresses per tenant

## Context
Customers upload the same table many times. Addressing a dataset by the hash of its canonical bytes deduplicates
the upload and lets a predict find the cache built from it. Sharing one address across tenants would save more.

## Decision
The id is SHA-256 over shape, little-endian float32 features and int64 labels, and the object lives under the
tenant. Two tenants with the same table get two objects and two caches.

## Consequences
No tenant can learn, from a fast first predict, that someone else holds the same data, and one tenant's erasure never
touches another's object. The cost is duplicate bytes when two customers share a public dataset.
