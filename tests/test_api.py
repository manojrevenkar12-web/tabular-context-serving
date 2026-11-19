"""The HTTP API end to end with the real PyTorch worker."""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from tcs.api.app import create_app
from tcs.model.prior import task

ACME = {"Authorization": "Bearer dev-key-acme"}
GLOBEX = {"Authorization": "Bearer dev-key-globex"}


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    return TestClient(create_app(str(tmp_path_factory.mktemp("bucket"))))


@pytest.fixture(scope="module")
def data():
    x, y = task(np.random.default_rng(9), 460, 7, 3)
    return np.nan_to_num(x), y


def upload(client, data, headers=ACME):
    x, y = data
    return client.post("/v1/datasets", json={"X": x[:400].tolist(), "y": y[:400].tolist()}, headers=headers)


def test_keys_decide_the_tenant(client, data):
    assert client.post("/v1/datasets", json={"X": [[1.0]], "y": [0]}).status_code == 401
    assert client.post("/v1/datasets", json={"X": [[1.0]], "y": [0]}, headers={"Authorization": "Bearer nope"}).status_code == 401


def test_upload_is_content_addressed(client, data):
    a, b = upload(client, data).json(), upload(client, data).json()
    assert a["dataset_id"] == b["dataset_id"] and not a["deduplicated"] and b["deduplicated"]
    assert (a["rows"], a["features"]) == (400, 7)


def test_predict_miss_then_hit_and_idempotent_replay(client, data):
    x, y = data
    ds = upload(client, data).json()["dataset_id"]
    body = {"dataset_id": ds, "X": x[400:].tolist()}
    first = client.post("/v1/predict", json=body, headers={**ACME, "Idempotency-Key": "abc"}).json()
    second = client.post("/v1/predict", json=body, headers=ACME).json()
    replay = client.post("/v1/predict", json=body, headers={**ACME, "Idempotency-Key": "abc"}).json()
    assert first["cache"] == "miss" and second["cache"] == "hit" and replay["replayed"]
    assert np.allclose(first["probabilities"], second["probabilities"], atol=1e-6)
    assert replay["probabilities"] == first["probabilities"]
    acc = np.mean(np.asarray(first["classes"])[np.argmax(first["probabilities"], 1)] == y[400:])
    assert acc > 0.5
    usage = client.get("/v1/usage", headers=ACME).json()
    assert usage["requests"] == 2  # the replay is not charged


def test_other_tenants_cannot_see_a_dataset(client, data):
    x, _ = data
    ds = upload(client, data).json()["dataset_id"]
    r = client.post("/v1/predict", json={"dataset_id": ds, "X": x[400:].tolist()}, headers=GLOBEX)
    assert r.status_code == 404


def test_feature_mismatch_is_rejected(client, data):
    ds = upload(client, data).json()["dataset_id"]
    r = client.post("/v1/predict", json={"dataset_id": ds, "X": [[1.0, 2.0]]}, headers=ACME)
    assert r.status_code == 422


def test_erasure_removes_data_and_caches_and_later_requests_get_410(client, data):
    x, _ = data
    ds = upload(client, data, GLOBEX).json()["dataset_id"]
    client.post("/v1/predict", json={"dataset_id": ds, "X": x[400:].tolist()}, headers=GLOBEX)
    cert = client.delete("/v1/data", headers=GLOBEX).json()
    assert cert["verified"] and cert["bucket_objects_left"] == 0 and cert["caches_left"] == 0
    assert client.post("/v1/predict", json={"dataset_id": ds, "X": x[400:].tolist()}, headers=GLOBEX).status_code == 410


def test_metrics_and_health(client):
    assert client.get("/healthz").json() == {"ok": True}
    text = client.get("/metrics").text
    assert "tcs_requests_total" in text and "tcs_cache_resident_bytes" in text
