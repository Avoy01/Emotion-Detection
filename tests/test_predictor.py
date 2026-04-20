"""Unit tests for inference helpers."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer

from emotion_detector import config
from emotion_detector.inference.predictor import predict_emotion


class DummyModel:
    """Small prediction stub for unit tests."""

    def predict(self, _inputs, verbose: int = 0) -> np.ndarray:
        return np.array([[0.05, 0.05, 0.7, 0.1, 0.05, 0.05]], dtype=np.float32)


class LowConfidenceModel:
    """Prediction stub that never clears the uncertainty threshold."""

    def predict(self, _inputs, verbose: int = 0) -> np.ndarray:
        return np.array([[0.29, 0.28, 0.21, 0.09, 0.08, 0.05]], dtype=np.float32)


def _build_label_encoder() -> LabelEncoder:
    encoder = LabelEncoder()
    encoder.fit(["anger", "fear", "joy", "love", "sadness", "surprise"])
    return encoder


def test_predict_emotion_schema() -> None:
    """The prediction payload should match the public contract."""

    tokenizer = Tokenizer(num_words=100, oov_token="<OOV>")
    tokenizer.fit_on_texts(["i am happy today"])
    result = predict_emotion(
        text="I am happy today",
        model=DummyModel(),
        tokenizer=tokenizer,
        label_encoder=_build_label_encoder(),
        maxlen=10,
    )

    assert set(result.keys()) == {
        "input_text",
        "predicted_emotion",
        "top_emotion",
        "confidence",
        "confidence_threshold",
        "is_uncertain",
        "probabilities",
    }
    assert result["predicted_emotion"] == "joy"
    assert result["top_emotion"] == "joy"
    assert result["is_uncertain"] is False
    assert abs(sum(result["probabilities"].values()) - 1.0) < 1e-6


def test_predict_emotion_deterministic() -> None:
    """The pure predictor should be deterministic for fixed inputs."""

    tokenizer = Tokenizer(num_words=100, oov_token="<OOV>")
    tokenizer.fit_on_texts(["i am happy today"])
    label_encoder = _build_label_encoder()

    first = predict_emotion("steady input", DummyModel(), tokenizer, label_encoder, maxlen=10)
    second = predict_emotion("steady input", DummyModel(), tokenizer, label_encoder, maxlen=10)
    assert first == second


def test_predict_emotion_uncertain_threshold() -> None:
    """Low-confidence predictions should be surfaced as uncertain."""

    tokenizer = Tokenizer(num_words=100, oov_token="<OOV>")
    tokenizer.fit_on_texts(["i am happy today"])
    result = predict_emotion(
        text="I am so happy today",
        model=LowConfidenceModel(),
        tokenizer=tokenizer,
        label_encoder=_build_label_encoder(),
        maxlen=10,
        confidence_threshold=0.45,
    )

    assert result["predicted_emotion"] == "uncertain"
    assert result["top_emotion"] == "anger"
    assert result["is_uncertain"] is True


def test_predict_emotion_uses_environment_threshold(monkeypatch) -> None:
    """The env var should override the saved default threshold."""

    monkeypatch.setenv(config.INFERENCE_CONFIDENCE_THRESHOLD_ENV, "0.2")
    tokenizer = Tokenizer(num_words=100, oov_token="<OOV>")
    tokenizer.fit_on_texts(["i am happy today"])
    result = predict_emotion(
        text="I am so happy today",
        model=LowConfidenceModel(),
        tokenizer=tokenizer,
        label_encoder=_build_label_encoder(),
        maxlen=10,
    )

    assert result["predicted_emotion"] == "anger"
    assert result["is_uncertain"] is False
    assert result["confidence_threshold"] == 0.2


def test_predict_emotion_rejects_invalid_threshold(monkeypatch) -> None:
    """Invalid thresholds should fail fast with a clear error."""

    monkeypatch.setenv(config.INFERENCE_CONFIDENCE_THRESHOLD_ENV, "1.5")
    tokenizer = Tokenizer(num_words=100, oov_token="<OOV>")
    tokenizer.fit_on_texts(["i am happy today"])

    with pytest.raises(ValueError, match="between 0.0 and 1.0"):
        predict_emotion(
            text="I am so happy today",
            model=LowConfidenceModel(),
            tokenizer=tokenizer,
            label_encoder=_build_label_encoder(),
            maxlen=10,
        )
