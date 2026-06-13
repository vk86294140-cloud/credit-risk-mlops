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
        "endpoints": ["/health", "/model/info", "/predict", "/predict/batch"],
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
