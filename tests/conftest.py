"""Shared fixtures: a trained model in a temp dir, and a sample application."""

from __future__ import annotations

import pytest

from credit_risk.model import CreditRiskModel
from credit_risk.train import train


@pytest.fixture(scope="session")
def trained_manifest(tmp_path_factory):
    model_dir = tmp_path_factory.mktemp("models")
    # Small, fast training run for the test suite.
    return train(n_rows=3000, model_dir=model_dir), model_dir


@pytest.fixture(scope="session")
def model(trained_manifest):
    _, model_dir = trained_manifest
    return CreditRiskModel.load_latest(model_dir=model_dir)


@pytest.fixture
def sample_application() -> dict:
    return {
        "age": 35,
        "annual_income": 72000,
        "loan_amount": 15000,
        "loan_term_months": 36,
        "interest_rate": 12.5,
        "debt_to_income": 0.28,
        "credit_utilization": 0.45,
        "num_open_accounts": 6,
        "num_delinquencies": 0,
        "months_employed": 48,
        "home_ownership": "mortgage",
        "loan_purpose": "debt_consolidation",
    }
