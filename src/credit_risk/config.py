"""Central configuration: paths and reproducibility settings."""

from __future__ import annotations

import os
from pathlib import Path

# Repo root = two levels up from this file (src/credit_risk/config.py).
ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = Path(os.getenv("CREDIT_RISK_DATA_DIR", ROOT / "data"))
MODEL_DIR = Path(os.getenv("CREDIT_RISK_MODEL_DIR", ROOT / "models"))

# A single global seed keeps data generation, splits, and training reproducible.
RANDOM_SEED = int(os.getenv("CREDIT_RISK_SEED", "42"))

# The feature columns the model consumes, declared once and reused by the
# data generator, the training pipeline, and the API request schema.
NUMERIC_FEATURES = [
    "age",
    "annual_income",
    "loan_amount",
    "loan_term_months",
    "interest_rate",
    "debt_to_income",
    "credit_utilization",
    "num_open_accounts",
    "num_delinquencies",
    "months_employed",
]
CATEGORICAL_FEATURES = [
    "home_ownership",   # rent | own | mortgage
    "loan_purpose",     # debt_consolidation | credit_card | home_improvement | other
]
TARGET = "defaulted"

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
