"""Pydantic request/response contracts for the scoring API."""

from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, Field


class CreditApplication(BaseModel):
    age: float = Field(..., ge=18, le=100, examples=[35])
    annual_income: float = Field(..., ge=0, examples=[72000])
    loan_amount: float = Field(..., ge=0, examples=[15000])
    loan_term_months: int = Field(..., ge=1, le=120, examples=[36])
    interest_rate: float = Field(..., ge=0, le=100, examples=[12.5])
    debt_to_income: float = Field(..., ge=0, le=1, examples=[0.28])
    credit_utilization: float = Field(..., ge=0, le=1, examples=[0.45])
    num_open_accounts: int = Field(..., ge=0, examples=[6])
    num_delinquencies: int = Field(..., ge=0, examples=[0])
    months_employed: float = Field(..., ge=0, examples=[48])
    home_ownership: Literal["rent", "own", "mortgage"] = Field(..., examples=["mortgage"])
    loan_purpose: Literal[
        "debt_consolidation", "credit_card", "home_improvement", "other"
    ] = Field(..., examples=["debt_consolidation"])


class PredictionResponse(BaseModel):
    default_probability: float
    risk_band: Literal["low", "medium", "high", "very_high"]
    model_id: str


class BatchRequest(BaseModel):
    applications: List[CreditApplication]


class BatchResponse(BaseModel):
    predictions: List[PredictionResponse]
