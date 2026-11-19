# Repeatable tasks

## Change a routing, admission or eviction rule
1. Change `tcs/plane/plane.py`; put any new number in `config/platform.toml`.
2. Add or change a unit test in `tests/test_plane.py` that pins the decision.
3. `scripts/check.sh`, then `python3 -m evals.run --hours 1` to look, then the full dev campaign.
4. `python3 -m evals.charts && python3 -m evals.readme && python3 -m evals.console_data`; commit reports with code.

## Add a design or an ablation
1. Add a block to `config/designs.toml` (every flag set).
2. If it needs a new flag, add it to `tcs.config.Design` and read it in the plane.
3. Run the dev campaign; the README table and the console pick it up.

## Calibrate the prior after a model release
1. Run the new model on each GPU kind with `fit_mode` caching and chunk ladders; record peak memory and seconds.
2. Update `[prior]` in `config/platform.toml`. The plane still corrects it from telemetry; the prior only decides
   how many out-of-memory errors happen before it has learned.

## Size a customer's VPC pool
1. Export their request log (rows, features, test rows, timestamps, interactive or not, latency target).
2. `tcs.advisor.sizing.advise(design, cfg, log, target, kinds, max_count, calibration_from(live_plane))`.
3. Send `draft(plan)` or a model's wording only after `check(text, plan)` returns nothing.

## Run the held-out campaign
1. Commit everything. 2. `python3 -m evals.run --heldout`. 3. Commit `reports/heldout.*`, regenerate the README.
Never tune on it; see AGENTS.md.

## Deploy
1. `cd deploy/terraform && terraform init && terraform apply -var project_id=...`.
2. Build and push the image to the `image_repository` output.
3. `helm install tcs-l4 deploy/helm/tcs --set pool=l4 --set serviceAccount.gcpServiceAccount=<api_service_account>`;
   one release per pool.
