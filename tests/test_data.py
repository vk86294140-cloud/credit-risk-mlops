"""Tests for reproducible data generation."""

from __future__ import annotations

import pandas as pd

from credit_risk.config import ALL_FEATURES, TARGET
from credit_risk.data import generate_dataset


def test_schema_and_size():
    df = generate_dataset(n_rows=500)
    assert len(df) == 500
    for col in ALL_FEATURES + [TARGET]:
        assert col in df.columns


def test_target_is_binary_and_imbalanced():
    df = generate_dataset(n_rows=5000)
    assert set(df[TARGET].unique()) <= {0, 1}
    rate = df[TARGET].mean()
    # Realistic default rate, not 0% or 50%.
    assert 0.02 < rate < 0.45


def test_generation_is_reproducible():
    a = generate_dataset(n_rows=1000, seed=7)
    b = generate_dataset(n_rows=1000, seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_different_seeds_differ():
    a = generate_dataset(n_rows=1000, seed=1)
    b = generate_dataset(n_rows=1000, seed=2)
    assert not a.equals(b)
