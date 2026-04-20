"""Unit tests for the FastAPI app."""

from __future__ import annotations

from fastapi.testclient import TestClient

from emotion_detector.api.app import app


class FakePredictor:
    """Simple predictor stub used by API tests."""

    classes = ["anger", "fear", "joy", "love", "sadness", "surprise"]

    def predict(self, text: str) -> dict:
        return {
            "input_text": text,
            "predicted_emotion": "joy",
            "top_emotion": "joy",
            "confidence": 0.9,
            "confidence_threshold": 0.45,
            "is_uncertain": False,
            "probabilities": {
                "anger": 0.01,
                "fear": 0.01,
                "joy": 0.9,
                "love": 0.03,
                "sadness": 0.03,
                "surprise": 0.02,
            },
        }


def test_api_predict_valid(monkeypatch) -> None:
    """A valid request should return a prediction payload."""

    monkeypatch.setattr("emotion_detector.api.app.get_predictor_service", lambda: FakePredictor())
    client = TestClient(app)

    response = client.post("/predict", json={"text": "I am happy"})
    assert response.status_code == 200
    assert response.json()["predicted_emotion"] == "joy"


def test_api_predict_missing_field(monkeypatch) -> None:
    """Missing text should return the SRS-required 400 response."""

    monkeypatch.setattr("emotion_detector.api.app.get_predictor_service", lambda: FakePredictor())
    client = TestClient(app)

    response = client.post("/predict", json={})
    assert response.status_code == 400
    assert response.json() == {"error": "text field is required and must be non-empty"}


def test_api_health(monkeypatch) -> None:
    """Health check should indicate whether the model is ready."""

    monkeypatch.setattr("emotion_detector.api.app.get_predictor_service", lambda: FakePredictor())
    client = TestClient(app)

    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}
