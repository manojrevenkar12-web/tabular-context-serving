"""HTTP API: upload a table once, predict against it many times, erase it.

    uvicorn tcs.api.app:app --port 8000

POST   /v1/datasets         {"X": [[...]], "y": [...]}            -> dataset id (content address), rows, features
POST   /v1/predict          {"dataset_id": ..., "X": [[...]]}     -> classes, probabilities, cache hit or miss
                            header Idempotency-Key: one execution and one charge per key
DELETE /v1/data             erase every dataset and cache of the caller's tenant, with a verified certificate
GET    /v1/usage            the caller's charged requests, rows and compute seconds
GET    /healthz, /metrics   liveness and Prometheus counters

The tenant comes from the API key. Every predict goes through the same plane the evaluation measures, here with
one PyTorch worker (`TorchWorker`).
"""

from __future__ import annotations

import copy
import hashlib
import os
import tempfile
import threading
import time
import tomllib
import uuid
from pathlib import Path

import numpy as np
from fastapi import Depends, FastAPI, Header, HTTPException, Response
from pydantic import BaseModel, Field

from tcs import config
from tcs.api.torch_worker import TorchWorker
from tcs.model.icl import load
from tcs.plane.ledger import Ledger
from tcs.plane.plane import Plane, Request
from tcs.plane.store import LocalBucket, digest, encode

ROOT = Path(__file__).resolve().parents[2]


def api_config() -> dict:
    return tomllib.loads((ROOT / "config" / "api.toml").read_text())


def plane_config(api: dict) -> dict:
    cfg = copy.deepcopy(config.platform())
    prior = dict(api["prior"])
    cfg["model"]["estimators"] = int(prior.pop("estimators"))
    cfg["prior"] = prior
    cfg["fleet"] = [api["fleet"]]
    return cfg


class Upload(BaseModel):
    X: list[list[float | None]] = Field(..., description="training rows")
    y: list[int] = Field(..., description="class labels")


class PredictBody(BaseModel):
    dataset_id: str
    X: list[list[float | None]]


class Service:
    def __init__(self, data_dir: str | None = None) -> None:
        self.api = api_config()
        self.keys = {k["sha256"]: k["tenant"] for k in self.api["keys"]}
        self.bucket = LocalBucket(data_dir or os.environ.get("TCS_DATA_DIR") or tempfile.mkdtemp(prefix="tcs-"))
        self.worker = TorchWorker(load(str(ROOT / "models" / "tiny_icl.pt")), self.bucket)
        self.ledger = Ledger(os.environ.get("TCS_LEDGER", ":memory:"))
        self.plane = Plane(config.design("platform"), plane_config(self.api), self.worker, self.ledger)
        self.lock = threading.Lock()
        self.metrics = {"requests": 0, "cache_hits": 0, "fits": 0, "oom": 0, "throttled": 0, "erasures": 0}

    def tenant(self, authorization: str | None) -> str:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(401, "missing bearer key")
        t = self.keys.get(hashlib.sha256(authorization[7:].encode()).hexdigest())
        if t is None:
            raise HTTPException(401, "unknown key")
        return t

    def upload(self, tenant: str, body: Upload) -> dict:
        lim = self.api["limits"]
        x = np.asarray([[np.nan if v is None else v for v in row] for row in body.X], dtype=np.float32)
        y = np.asarray(body.y, dtype=np.int64)
        if x.ndim != 2 or x.shape[0] != y.shape[0] or x.shape[0] < 2:
            raise HTTPException(422, "X must be a table with one label per row")
        if x.shape[0] > lim["max_rows"] or x.shape[1] > lim["max_features"]:
            raise HTTPException(413, "context too large for this deployment; subsample rows")
        d = digest(x, y)
        with self.lock:
            known = (tenant, d) in self.plane.datasets
            if not known:
                self.bucket.put(tenant, d, encode(x, y))
                self.plane.upload(tenant, d, int(x.shape[0]), int(x.shape[1]), float(x.nbytes))
        return {"dataset_id": d, "rows": int(x.shape[0]), "features": int(x.shape[1]), "deduplicated": known}

    def predict(self, tenant: str, body: PredictBody, key: str | None) -> dict:
        x = np.asarray([[np.nan if v is None else v for v in row] for row in body.X], dtype=np.float32)
        if x.ndim != 2 or x.shape[0] == 0 or x.shape[0] > self.api["limits"]["max_test_rows"]:
            raise HTTPException(422, "X must be a non-empty table within the test-row limit")
        meta = self.plane.datasets.get((tenant, body.dataset_id))
        if meta is not None and x.shape[1] != meta[1]:
            raise HTTPException(422, f"expected {meta[1]} features, got {x.shape[1]}")
        rid = uuid.uuid4().hex
        with self.lock:
            now = time.monotonic()
            hits_before = sum(1 for g in self.plane.gpus.values() if body.dataset_id in g.hbm or body.dataset_id in g.host)
            req = Request(rid, f"{tenant}:{key or rid}", tenant, body.dataset_id, int(x.shape[0]), now, 30.0, True, x)
            n = len(self.plane.outcomes)
            self.plane.arrive(req, now)
            while True:
                mine = [o for o in self.plane.outcomes[n:] if o.request_id == rid]
                if mine:
                    out = mine[0]
                    break
                started = self.plane.dispatch(time.monotonic())
                if not started:
                    raise HTTPException(503, "no worker could take the request")
                for gpu, job in started:
                    report = self.worker.run(gpu, job.to_wire(), time.monotonic())
                    self.metrics["fits"] += job.kind == "fit_predict" and report["ok"]
                    self.metrics["oom"] += bool(report.get("oom"))
                    self.plane.complete(gpu, job, report, time.monotonic())
        self.metrics["requests"] += 1
        self.metrics["cache_hits"] += hits_before > 0
        status = {"ok": 200, "throttled": 429, "gone": 410, "not_found": 404, "too_large": 413}.get(out.status, 500)
        if status != 200:
            self.metrics["throttled"] += status == 429
            raise HTTPException(status, out.status)
        return {
            "request_id": out.request_id,
            "dataset_id": body.dataset_id,
            "cache": "hit" if hits_before else "miss",
            "replayed": out.duplicate,
            **(out.result or {}),
        }

    def erase(self, tenant: str) -> dict:
        with self.lock:
            cert = self.plane.erase(tenant, time.monotonic())
            for d in self.bucket.list(tenant):
                self.bucket.delete(tenant, d)
        self.metrics["erasures"] += 1
        return {
            "tenant": tenant,
            "datasets": len(cert["digests"]),
            "verified": cert["verified"],
            "bucket_objects_left": len(self.bucket.list(tenant)),
            "caches_left": sum(len(set(self.worker.residents(g)["hbm"]) | set(self.worker.residents(g)["host"])) for g in self.plane.gpus),
        }


def create_app(data_dir: str | None = None) -> FastAPI:
    svc = Service(data_dir)
    app = FastAPI(title="tabular-context-serving", version="1.0")
    app.state.svc = svc

    def tenant(authorization: str | None = Header(default=None)) -> str:
        return svc.tenant(authorization)

    @app.post("/v1/datasets")
    def upload(body: Upload, t: str = Depends(tenant)) -> dict:
        return svc.upload(t, body)

    @app.post("/v1/predict")
    def predict(body: PredictBody, t: str = Depends(tenant), idempotency_key: str | None = Header(default=None)) -> dict:
        return svc.predict(t, body, idempotency_key)

    @app.delete("/v1/data")
    def erase(t: str = Depends(tenant)) -> dict:
        return svc.erase(t)

    @app.get("/v1/usage")
    def usage(t: str = Depends(tenant)) -> dict:
        return svc.ledger.usage(t)

    @app.get("/healthz")
    def healthz() -> dict:
        return {"ok": True}

    @app.get("/metrics")
    def metrics() -> Response:
        lines = []
        for k, v in svc.metrics.items():
            lines += [f"# TYPE tcs_{k}_total counter", f"tcs_{k}_total {v}"]
        resident = sum(sum(svc.worker.residents(g)["hbm"].values()) for g in svc.plane.gpus)
        lines += ["# TYPE tcs_cache_resident_bytes gauge", f"tcs_cache_resident_bytes {resident}"]
        return Response("\n".join(lines) + "\n", media_type="text/plain; version=0.0.4")

    return app


app = create_app()
