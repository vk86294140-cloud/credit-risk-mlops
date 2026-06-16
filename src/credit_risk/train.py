"""Train, evaluate, and version a credit-default model.

Produces a self-describing, timestamped artifact (the fitted sklearn pipeline)
plus a metrics JSON, and updates a `latest` pointer the API loads from. This is
a lightweight model-registry pattern: every training run is reproducible and
auditable without pulling in a heavy MLOps platform.
"""

from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from . import __version__
from .config import ALL_FEATURES, MODEL_DIR, RANDOM_SEED, TARGET, ensure_dirs
from .data import load_dataset
from .features import build_preprocessor


@dataclass
class Metrics:
    roc_auc: float
    pr_auc: float
    precision: float
    recall: float
    f1: float
    brier: float
    n_train: int
    n_test: int
    default_rate: float


def build_model() -> Pipeline:
    """Preprocessing + gradient-boosted trees, as one fit/serve unit."""
    classifier = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.06,
        max_depth=6,
        l2_regularization=1.0,
        random_state=RANDOM_SEED,
    )
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            ("model", classifier),
        ]
    )


def evaluate(pipeline: Pipeline, X_test, y_test, n_train: int) -> Metrics:
    proba = pipeline.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)
    return Metrics(
        roc_auc=float(roc_auc_score(y_test, proba)),
        pr_auc=float(average_precision_score(y_test, proba)),
        precision=float(precision_score(y_test, preds, zero_division=0)),
        recall=float(recall_score(y_test, preds, zero_division=0)),
        f1=float(f1_score(y_test, preds, zero_division=0)),
        brier=float(brier_score_loss(y_test, proba)),
        n_train=int(n_train),
        n_test=int(len(y_test)),
        default_rate=float(np.mean(y_test)),
    )


def feature_importances(
    pipeline: Pipeline, X_test, y_test, n_repeats: int = 5
) -> list[dict[str, float]]:
    """Permutation importance per *original* input feature, ranked high to low.

    Permuting the raw input columns (rather than the post-one-hot columns) keeps
    the attribution interpretable for an auditor: it answers "how much does the
    model's ROC-AUC drop when this application field is shuffled?". HistGradient-
    BoostingClassifier exposes no native `feature_importances_`, so this is the
    model-agnostic way to surface what actually drives the score.
    """
    result = permutation_importance(
        pipeline,
        X_test,
        y_test,
        n_repeats=n_repeats,
        random_state=RANDOM_SEED,
        scoring="roc_auc",
    )
    ranked = sorted(
        zip(ALL_FEATURES, result.importances_mean, result.importances_std),
        key=lambda kv: kv[1],
        reverse=True,
    )
    return [
        {"feature": f, "importance": round(float(mean), 5), "std": round(float(std), 5)}
        for f, mean, std in ranked
    ]


def train(
    data_path: str | Path | None = None,
    n_rows: int = 20_000,
    model_dir: Path = MODEL_DIR,
) -> Dict[str, Any]:
    """Run the full training pipeline and persist a versioned artifact."""

    ensure_dirs()
    df = load_dataset(path=data_path, n_rows=n_rows)

    X = df[ALL_FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    pipeline = build_model()
    pipeline.fit(X_train, y_train)
    metrics = evaluate(pipeline, X_test, y_test, n_train=len(X_train))
    importances = feature_importances(pipeline, X_test, y_test)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = model_dir / f"credit_risk_{timestamp}.joblib"
    joblib.dump(pipeline, artifact_path)

    manifest = {
        "model_id": f"credit_risk_{timestamp}",
        "created_utc": timestamp,
        "package_version": __version__,
        "python": platform.python_version(),
        "features": ALL_FEATURES,
        "artifact": artifact_path.name,
        "metrics": asdict(metrics),
        "feature_importances": importances,
    }

    # Persist run metrics next to the artifact, and update the `latest` pointer
    # the serving layer reads.
    (model_dir / f"credit_risk_{timestamp}.metrics.json").write_text(
        json.dumps(manifest, indent=2)
    )
    (model_dir / "latest.json").write_text(json.dumps(manifest, indent=2))

    return manifest


if __name__ == "__main__":
    result = train()
    print(json.dumps(result["metrics"], indent=2))
    print(f"\nSaved artifact: {result['artifact']}")
