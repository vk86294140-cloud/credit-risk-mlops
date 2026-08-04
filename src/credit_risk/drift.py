"""Data-drift monitoring against the training distribution.

A model does not degrade with a stack trace. It degrades because the traffic
stopped looking like the data it was fitted on - a new acquisition channel
shifts the income mix, a policy change moves utilisation, a lending partner
sends a segment that was never in the training set. Accuracy metrics can't
catch that in production, because the labels (did this loan actually default?)
arrive months later, if ever.

What *is* available immediately is the input distribution. This module captures
a compact profile of the training data at fit time, then scores live batches
against it with the Population Stability Index:

    PSI = sum over bins of  (p_live - p_ref) * ln(p_live / p_ref)

The conventional reading, which `verdict()` applies:

    PSI < 0.10          stable        no action
    0.10 <= PSI < 0.25  moderate      investigate
    PSI >= 0.25         significant   retraining is likely overdue

Numeric features are bucketed on quantile edges taken from the training data,
so each reference bin starts equally populated and PSI reacts to shape rather
than to an arbitrary choice of bin width. Categorical features compare category
proportions directly, with unseen categories collected into one bucket - their
appearance is itself drift worth reporting.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

from .config import CATEGORICAL_FEATURES, NUMERIC_FEATURES

# Number of quantile buckets for numeric features. Ten is the convention in
# credit-risk monitoring and keeps the profile small enough to sit in the
# model manifest.
N_BINS = 10

# Floor for empty bins. PSI takes a log of a ratio, so a bin that is empty on
# either side would send the term to infinity and one missing category would
# dominate the whole score.
EPSILON = 1e-6

STABLE_BELOW = 0.10
SIGNIFICANT_AT = 0.25

UNSEEN_CATEGORY = "__unseen__"


def verdict(psi: float) -> str:
    """Interpretation band for a PSI value."""
    if psi < STABLE_BELOW:
        return "stable"
    if psi < SIGNIFICANT_AT:
        return "moderate"
    return "significant"


def build_reference(frame: pd.DataFrame) -> Dict[str, Any]:
    """Summarise a training frame into a drift reference profile.

    Deliberately small and JSON-serialisable: bin edges and proportions only,
    no raw rows. It ships inside the model manifest, so monitoring a batch
    needs the artifact and nothing else - no access to the training data at
    serving time.
    """
    numeric: Dict[str, Any] = {}
    for feature in NUMERIC_FEATURES:
        values = frame[feature].astype(float).to_numpy()
        # Deduplicate: a feature with heavy mass on one value (many zero
        # delinquencies) produces repeated quantile edges, which would make
        # empty bins and inflate PSI for no reason.
        edges = np.unique(np.quantile(values, np.linspace(0, 1, N_BINS + 1)))
        counts, _ = np.histogram(values, bins=_open_ended(edges))
        numeric[feature] = {
            "edges": [float(e) for e in edges],
            "proportions": _proportions(counts),
        }

    categorical: Dict[str, Any] = {}
    for feature in CATEGORICAL_FEATURES:
        shares = frame[feature].astype(str).value_counts(normalize=True)
        categorical[feature] = {
            "proportions": {str(k): float(v) for k, v in shares.items()},
        }

    return {
        "n_rows": int(len(frame)),
        "n_bins": N_BINS,
        "numeric": numeric,
        "categorical": categorical,
    }


def population_stability_index(reference: Dict[str, Any], frame: pd.DataFrame) -> Dict[str, Any]:
    """Score a batch against a reference profile.

    Returns per-feature PSI plus a roll-up. The roll-up uses the maximum, not
    the mean: one badly shifted feature is a real problem, and averaging it
    against nine stable ones is how drift gets missed.
    """
    features: List[Dict[str, Any]] = []

    for feature, profile in reference["numeric"].items():
        if feature not in frame.columns:
            continue
        edges = _open_ended(np.array(profile["edges"], dtype=float))
        counts, _ = np.histogram(frame[feature].astype(float).to_numpy(), bins=edges)
        psi = _psi(profile["proportions"], _proportions(counts))
        features.append({"feature": feature, "kind": "numeric", "psi": psi, "verdict": verdict(psi)})

    for feature, profile in reference["categorical"].items():
        if feature not in frame.columns:
            continue
        ref_shares: Dict[str, float] = dict(profile["proportions"])
        observed = frame[feature].astype(str).value_counts(normalize=True).to_dict()

        # Anything the training data never saw is folded into one bucket, so a
        # long tail of new categories reads as a single, interpretable signal.
        live_shares = {key: 0.0 for key in ref_shares}
        unseen = 0.0
        for key, share in observed.items():
            if key in live_shares:
                live_shares[key] = float(share)
            else:
                unseen += float(share)
        if unseen:
            ref_shares[UNSEEN_CATEGORY] = 0.0
            live_shares[UNSEEN_CATEGORY] = unseen

        order = sorted(ref_shares)
        psi = _psi([ref_shares[k] for k in order], [live_shares[k] for k in order])
        features.append(
            {"feature": feature, "kind": "categorical", "psi": psi, "verdict": verdict(psi)}
        )

    features.sort(key=lambda item: -item["psi"])
    worst = features[0]["psi"] if features else 0.0
    return {
        "n_rows": int(len(frame)),
        "reference_rows": int(reference.get("n_rows", 0)),
        "max_psi": worst,
        "verdict": verdict(worst),
        "drifted_features": [f["feature"] for f in features if f["verdict"] != "stable"],
        "features": features,
    }


def _open_ended(edges: np.ndarray) -> np.ndarray:
    """Bin edges with the outermost bounds opened to +/- infinity.

    Live values below the training minimum or above its maximum are real and
    interesting; without this they'd fall outside every bin and silently
    vanish from the comparison, hiding exactly the shift worth catching.
    """
    interior = edges[1:-1] if edges.size > 2 else edges[:0]
    return np.concatenate([[-np.inf], interior, [np.inf]])


def _proportions(counts: np.ndarray) -> List[float]:
    total = float(counts.sum())
    if total == 0:
        return [0.0] * len(counts)
    return [float(c) / total for c in counts]


def _psi(reference: Sequence[float], live: Sequence[float]) -> float:
    total = 0.0
    for ref, obs in zip(reference, live):
        ref = max(float(ref), EPSILON)
        obs = max(float(obs), EPSILON)
        total += (obs - ref) * math.log(obs / ref)
    return round(total, 5)
