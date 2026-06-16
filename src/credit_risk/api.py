"""FastAPI scoring service for the credit-default model."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException

from . import __version__
from .model import ModelNotFoundError, get_model
from .schema import (
    BatchRequest,
    BatchResponse,
    CreditApplication,
    PredictionResponse,
)

app = FastAPI(
    title="Credit Risk Scoring API",
    version=__version__,
    description="Serves a versioned credit-default risk model trained by the "
    "credit_risk pipeline.",
)


@app.get("/")
def root() -> dict:
    """Service metadata and a map of available endpoints."""
    return {
        "service": "Credit Risk Scoring API",
        "version": __version__,
        "docs": "/docs",
        "endpoints": [
            "/health",
            "/model/info",
            "/model/importance",
            "/predict",
            "/predict/batch",
        ],
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__}


@app.get("/model/info")
def model_info() -> dict:
    try:
        model = get_model()
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    return model.manifest


@app.get("/model/importance")
def model_importance() -> dict:
    """Permutation feature importance for the live model, ranked high to low.

    Surfaces which application fields drive the risk score — useful for model
    audits and adverse-action reasoning. Populated at training time; older
    artifacts without it return 503 until retrained.
    """
    try:
        model = get_model()
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    importances = model.manifest.get("feature_importances")
    if not importances:
        raise HTTPException(
            status_code=503,
            detail="This model artifact predates feature-importance tracking; "
            "retrain (`make train`) to populate it.",
        )
    return {
        "model_id": model.manifest["model_id"],
        "scoring": "roc_auc",
        "method": "permutation_importance",
        "feature_importances": importances,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(application: CreditApplication) -> PredictionResponse:
    try:
        model = get_model()
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    result = model.predict_one(application.model_dump())
    return PredictionResponse(**result)


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(request: BatchRequest) -> BatchResponse:
    try:
        model = get_model()
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    records = [a.model_dump() for a in request.applications]
    results = model.predict_many(records)
    return BatchResponse(predictions=[PredictionResponse(**r) for r in results])
