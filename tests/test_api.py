"""API tests using FastAPI's TestClient and the session-trained model."""

from __future__ import annotations

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
