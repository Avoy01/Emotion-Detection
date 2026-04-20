"""FastAPI application serving emotion predictions."""

from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from emotion_detector import config
from emotion_detector.api.schemas import (
    ClassesResponse,
    ErrorResponse,
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
)
from emotion_detector.inference.predictor import EmotionPredictor
from emotion_detector.utils.logger import configure_logging, get_logger

configure_logging()
LOGGER = get_logger(__name__)
app = FastAPI(title="Emotion Detection API", version="0.1.0")
_PREDICTOR: EmotionPredictor | None = None


@app.exception_handler(HTTPException)
async def http_exception_handler(_request, exc: HTTPException) -> JSONResponse:
    """Return SRS-compliant JSON error payloads for HTTP errors."""

    return JSONResponse(status_code=exc.status_code, content={"error": str(exc.detail)})


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_request, exc: RequestValidationError) -> JSONResponse:
    """Differentiate malformed JSON from input validation errors."""

    error_types = {error["type"] for error in exc.errors()}
    if "json_invalid" in error_types:
        return JSONResponse(status_code=422, content={"error": "malformed JSON"})
    return JSONResponse(
        status_code=400,
        content={"error": "text field is required and must be non-empty"},
    )


def get_predictor_service() -> EmotionPredictor:
    """Load and cache the predictor service."""

    global _PREDICTOR
    if _PREDICTOR is None:
        artefact_dir = config.resolve_artefact_dir(os.getenv("ARTEFACT_DIR"))
        _PREDICTOR = EmotionPredictor.from_artefacts(artefact_dir)
    return _PREDICTOR


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Return basic health information for the service."""

    try:
        loaded = get_predictor_service() is not None
    except Exception as exc:
        LOGGER.warning("Predictor not ready: %s", exc)
        loaded = False
    return HealthResponse(status="ok", model_loaded=loaded)


@app.get("/classes", response_model=ClassesResponse)
def list_classes() -> ClassesResponse:
    """Return the available emotion labels."""

    try:
        classes = get_predictor_service().classes
    except Exception:
        classes = list(config.LABEL_CLASSES)
    return ClassesResponse(classes=classes)


@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
def predict(payload: PredictionRequest) -> PredictionResponse:
    """Run single-text inference via REST."""

    if not payload.text or not payload.text.strip():
        raise HTTPException(
            status_code=400,
            detail="text field is required and must be non-empty",
        )

    try:
        result = get_predictor_service().predict(payload.text)
        return PredictionResponse(**result)
    except HTTPException:
        raise
    except Exception as exc:
        LOGGER.exception("Prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail="internal prediction error") from exc
