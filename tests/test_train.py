"""Tests for training, evaluation, and artifact versioning."""

from __future__ import annotations

import json


def test_training_produces_skillful_model(trained_manifest):
    manifest, _ = trained_manifest
    metrics = manifest["metrics"]
    # The target is a noisy function of the features, so a real model must beat
    # chance comfortably even on the small (3k-row) training subset used here.
    # The pipeline is seeded, so this bound is deterministic.
    assert metrics["roc_auc"] > 0.65
    assert metrics["pr_auc"] > metrics["default_rate"]  # better than base rate
    assert 0.0 <= metrics["brier"] <= 0.25


def test_artifact_and_pointer_written(trained_manifest):
    manifest, model_dir = trained_manifest
    assert (model_dir / manifest["artifact"]).exists()
    latest = json.loads((model_dir / "latest.json").read_text())
    assert latest["model_id"] == manifest["model_id"]
    assert latest["features"] == manifest["features"]
