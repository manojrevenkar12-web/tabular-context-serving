# Worker protocol

The plane drives workers through five calls. The PyTorch worker (`tcs/api/torch_worker.py`) and the simulated fleet
(`world/fleet.py`) both implement them; a GPU worker in a pod implements them over gRPC or HTTP.

| Call | Arguments | Returns |
|---|---|---|
| `run(gpu, job, now)` | job: `job_id, kind, tenant, digest, rows, features, test_rows, train_chunk, test_chunk, keep_cache, requests, payloads` | `ok, oom, seconds, peak_bytes, cache_bytes, error, results` |
| `evict(gpu, digest, to_host)` | | whether a cache was evicted; with `to_host` it is copied to host RAM if room |
| `purge(gpu, digest)` | | removes the cache from HBM and host RAM, zeroing tensors |
| `residents(gpu)` | | `{"hbm": {digest: bytes}, "host": {digest: bytes}}` |
| `put(tenant, digest, nbytes)`, `delete(tenant, digest)` | | dataset objects in the bucket |

Job kinds:
- `fit_predict` reads the dataset, runs the training rows through every layer, predicts, and keeps the KV cache if
  `keep_cache` is set.
- `predict` needs the cache resident in HBM.
- `reload_predict` brings a cache back from host RAM over PCIe, then predicts.

A missing cache answers `cache_miss`. A missing dataset answers `missing_dataset`. An allocation failure answers
`oom: true` with the seconds it wasted.

`peak_bytes` is the job's own peak: its transient memory, plus the cache it built for a fit. The plane learns its
memory model from that value and from out-of-memory errors.
