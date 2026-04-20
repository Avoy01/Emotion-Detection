"""Pydantic schemas for the FastAPI interface."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Prediction request payload."""

    text: str = Field(..., description="Input text to classify.")


class PredictionResponse(BaseModel):
    """Prediction response payload."""

    input_text: str
    predicted_emotion: str
    top_emotion: str
    confidence: float
    confidence_threshold: float
    is_uncertain: bool
    probabilities: dict[str, float]


class HealthResponse(BaseModel):
    """Health check response payload."""

    status: str
    model_loaded: bool


class ClassesResponse(BaseModel):
    """Available class list response."""

    classes: list[str]


class ErrorResponse(BaseModel):
    """Sanitized error payload."""

    error: str
