"""API tests using FastAPI's TestClient and the session-trained model."""

from __future__ import annotations

import copy

import pytest
from fastapi.testclient import TestClient

import credit_risk.api as api_module
from credit_risk.api import app


@pytest.fixture
def client(model, monkeypatch):
    # Point the API at the model trained into a temp dir by the `model` fixture.
    monkeypatch.setattr(api_module, "get_model", lambda: model)
    return TestClient(app)


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_model_info(client):
    resp = client.get("/model/info")
    assert resp.status_code == 200
    body = resp.json()
    assert "metrics" in body and "features" in body


def test_predict(client, sample_application):
    resp = client.post("/predict", json=sample_application)
    assert resp.status_code == 200
    body = resp.json()
    assert 0.0 <= body["default_probability"] <= 1.0
    assert body["risk_band"] in {"low", "medium", "high", "very_high"}


def test_predict_validation_error(client, sample_application):
    bad = dict(sample_application, home_ownership="spaceship")
    resp = client.post("/predict", json=bad)
    assert resp.status_code == 422


def test_predict_batch(client, sample_application):
    resp = client.post(
        "/predict/batch",
        json={"applications": [sample_application, sample_application]},
    )
    assert resp.status_code == 200
    assert len(resp.json()["predictions"]) == 2


def test_root(client):
    resp = client.get("/")
    assert resp.status_code == 200
    body = resp.json()
    assert body["service"] == "Credit Risk Scoring API"
    assert "/predict" in body["endpoints"]
    assert "/model/importance" in body["endpoints"]


def test_model_importance(client):
    resp = client.get("/model/importance")
    assert resp.status_code == 200
    body = resp.json()
    assert body["method"] == "permutation_importance"
    imps = body["feature_importances"]
    assert imps, "expected a non-empty importance list"
    # Ranked high to low, and every entry is fully described.
    means = [row["importance"] for row in imps]
    assert means == sorted(means, reverse=True)
    assert all({"feature", "importance", "std"} <= row.keys() for row in imps)


def _batch(sample_application: dict, n: int = 40) -> list[dict]:
    """A batch large enough to satisfy the endpoint's minimum sample size."""
    return [dict(sample_application) for _ in range(n)]


def test_drift_endpoint_reports_per_feature_psi(client, sample_application):
    resp = client.post("/monitor/drift", json={"applications": _batch(sample_application)})
    assert resp.status_code == 200
    body = resp.json()
    assert body["n_rows"] == 40
    assert body["verdict"] in {"stable", "moderate", "significant"}
    assert {f["feature"] for f in body["features"]}
    # Forty identical applications are a degenerate distribution by
    # construction, so the monitor had better say so.
    assert body["max_psi"] > 0.25


def test_drift_endpoint_rejects_a_batch_too_small_to_be_meaningful(client, sample_application):
    resp = client.post("/monitor/drift", json={"applications": _batch(sample_application, n=5)})
    assert resp.status_code == 422


def test_drift_endpoint_503s_on_an_artifact_without_a_reference_profile(
    model, monkeypatch, sample_application
):
    """Older artifacts predate the reference profile; say so rather than 500."""
    legacy = copy.copy(model)
    legacy.manifest = {k: v for k, v in model.manifest.items() if k != "reference_profile"}
    monkeypatch.setattr(api_module, "get_model", lambda: legacy)
    client = TestClient(app)
    resp = client.post("/monitor/drift", json={"applications": _batch(sample_application)})
    assert resp.status_code == 503
    assert "retrain" in resp.json()["detail"]
