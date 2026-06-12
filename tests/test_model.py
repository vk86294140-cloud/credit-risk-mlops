"""Tests for the serving-time model wrapper."""

from __future__ import annotations

import pytest

from credit_risk.model import CreditRiskModel, ModelNotFoundError


def test_predict_one_shape(model, sample_application):
    out = model.predict_one(sample_application)
    assert 0.0 <= out["default_probability"] <= 1.0
    assert out["risk_band"] in {"low", "medium", "high", "very_high"}
    assert out["model_id"].startswith("credit_risk_")


def test_risk_increases_with_bad_credit(model, sample_application):
    safe = dict(sample_application, debt_to_income=0.05, credit_utilization=0.05,
                num_delinquencies=0)
    risky = dict(sample_application, debt_to_income=0.7, credit_utilization=0.95,
                 num_delinquencies=6)
    p_safe = model.predict_one(safe)["default_probability"]
    p_risky = model.predict_one(risky)["default_probability"]
    assert p_risky > p_safe


def test_missing_model_raises(tmp_path):
    with pytest.raises(ModelNotFoundError):
        CreditRiskModel.load_latest(model_dir=tmp_path)
