"""Tests for the PSI drift monitor.

The properties that matter: a batch drawn from the training distribution must
score as stable (or the monitor cries wolf and gets ignored), and a genuine
shift must clear the significance threshold (or it is decorative).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from credit_risk.config import RANDOM_SEED
from credit_risk.data import load_dataset
from credit_risk.drift import (
    build_reference,
    population_stability_index,
    verdict,
)


@pytest.fixture(scope="module")
def frames():
    df = load_dataset(n_rows=4000)
    reference_frame = df.iloc[:2000].reset_index(drop=True)
    holdout = df.iloc[2000:].reset_index(drop=True)
    return build_reference(reference_frame), holdout


def test_verdict_bands():
    assert verdict(0.02) == "stable"
    assert verdict(0.15) == "moderate"
    assert verdict(0.40) == "significant"


def test_reference_profile_is_json_shaped_and_covers_every_feature(frames):
    reference, _ = frames
    assert reference["n_rows"] == 2000
    assert set(reference["numeric"]) and set(reference["categorical"])
    for profile in reference["numeric"].values():
        assert len(profile["edges"]) >= 2
        assert pytest.approx(sum(profile["proportions"]), abs=1e-6) == 1.0


def test_same_distribution_reads_as_stable(frames):
    reference, holdout = frames
    report = population_stability_index(reference, holdout)
    assert report["verdict"] == "stable"
    assert report["drifted_features"] == []


def test_shifted_numeric_feature_is_flagged(frames):
    reference, holdout = frames
    shifted = holdout.copy()
    # A credit-utilisation shift of this size is the kind of change a policy
    # or channel change produces - it must not pass as noise.
    shifted["credit_utilization"] = np.clip(shifted["credit_utilization"] + 0.35, 0, 1)

    report = population_stability_index(reference, shifted)
    assert report["verdict"] == "significant"
    assert "credit_utilization" in report["drifted_features"]
    assert report["features"][0]["feature"] == "credit_utilization"


def test_unseen_category_is_reported_rather_than_ignored(frames):
    reference, holdout = frames
    shifted = holdout.copy()
    shifted.loc[shifted.index[:800], "home_ownership"] = "co_op"

    report = population_stability_index(reference, shifted)
    assert "home_ownership" in report["drifted_features"]


def test_values_outside_the_training_range_still_land_in_a_bin(frames):
    reference, holdout = frames
    shifted = holdout.copy()
    shifted["annual_income"] = shifted["annual_income"] * 100  # far beyond training max

    report = population_stability_index(reference, shifted)
    entry = next(f for f in report["features"] if f["feature"] == "annual_income")
    assert entry["verdict"] == "significant"


def test_rollup_uses_the_worst_feature_not_the_average(frames):
    reference, holdout = frames
    shifted = holdout.copy()
    shifted["num_delinquencies"] = shifted["num_delinquencies"] + 4

    report = population_stability_index(reference, shifted)
    worst = max(f["psi"] for f in report["features"])
    assert report["max_psi"] == worst


def test_reference_is_deterministic():
    left = build_reference(load_dataset(n_rows=1500))
    right = build_reference(load_dataset(n_rows=1500))
    assert left == right, f"seed {RANDOM_SEED} should make profiles reproducible"


def test_missing_columns_are_skipped_not_fatal(frames):
    reference, holdout = frames
    partial = pd.DataFrame({"age": holdout["age"]})
    report = population_stability_index(reference, partial)
    assert [f["feature"] for f in report["features"]] == ["age"]
