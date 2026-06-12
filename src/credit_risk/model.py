"""Load a trained artifact and score requests at serving time."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

import joblib
import pandas as pd

from .config import ALL_FEATURES, MODEL_DIR


class ModelNotFoundError(RuntimeError):
    """Raised when no trained model is available to serve."""


class CreditRiskModel:
    def __init__(self, pipeline, manifest: Dict[str, Any]):
        self.pipeline = pipeline
        self.manifest = manifest

    @classmethod
    def load_latest(cls, model_dir: Path = MODEL_DIR) -> "CreditRiskModel":
        latest = Path(model_dir) / "latest.json"
        if not latest.exists():
            raise ModelNotFoundError(
                f"No model found in {model_dir}. Run `python -m credit_risk.train` "
                f"(or `make train`) first."
            )
        manifest = json.loads(latest.read_text())
        artifact = Path(model_dir) / manifest["artifact"]
        if not artifact.exists():
            raise ModelNotFoundError(f"Manifest points to missing artifact: {artifact}")
        return cls(joblib.load(artifact), manifest)

    def predict_one(self, record: Dict[str, Any]) -> Dict[str, Any]:
        return self.predict_many([record])[0]

    def predict_many(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        frame = pd.DataFrame(records)[ALL_FEATURES]
        proba = self.pipeline.predict_proba(frame)[:, 1]
        out = []
        for p in proba:
            out.append(
                {
                    "default_probability": round(float(p), 4),
                    "risk_band": _risk_band(p),
                    "model_id": self.manifest["model_id"],
                }
            )
        return out


def _risk_band(p: float) -> str:
    if p < 0.10:
        return "low"
    if p < 0.30:
        return "medium"
    if p < 0.60:
        return "high"
    return "very_high"


@lru_cache(maxsize=1)
def get_model() -> CreditRiskModel:
    """Process-wide cached model handle for the API."""
    return CreditRiskModel.load_latest()
