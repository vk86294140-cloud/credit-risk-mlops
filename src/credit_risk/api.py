"""FastAPI scoring service for the credit-default model."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

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

# Serve static files (frontend)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=FileResponse)
def root() -> FileResponse:
    """Serve the demo UI."""
    return FileResponse("static/index.html")

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
