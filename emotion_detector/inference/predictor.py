"""Inference utilities and predictor service."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer

from emotion_detector import config
from emotion_detector.data.preprocessor import clean_text, tokenize_and_pad
from emotion_detector.model.attention import AttentionLayer
from emotion_detector.utils.files import read_json


def predict_emotion(
    text: str,
    model: tf.keras.Model,
    tokenizer: Tokenizer,
    label_encoder: LabelEncoder,
    maxlen: int,
    confidence_threshold: float | None = None,
) -> dict:
    """Run pure end-to-end inference for a single text input."""

    resolved_threshold = config.resolve_confidence_threshold(override=confidence_threshold)
    cleaned_text = clean_text(text)
    padded = tokenize_and_pad([cleaned_text], tokenizer=tokenizer, maxlen=maxlen)
    probabilities = model.predict(padded, verbose=0)[0]
    class_probabilities = {
        str(class_name): float(probability)
        for class_name, probability in zip(label_encoder.classes_, probabilities)
    }
    predicted_index = int(np.argmax(probabilities))
    top_emotion = str(label_encoder.classes_[predicted_index])
    confidence = float(probabilities[predicted_index])
    is_uncertain = confidence < resolved_threshold
    predicted_emotion = "uncertain" if is_uncertain else top_emotion
    return {
        "input_text": text,
        "predicted_emotion": predicted_emotion,
        "top_emotion": top_emotion,
        "confidence": confidence,
        "confidence_threshold": resolved_threshold,
        "is_uncertain": is_uncertain,
        "probabilities": class_probabilities,
    }


@dataclass
class EmotionPredictor:
    """Service object that loads artifacts once and serves predictions."""

    model: tf.keras.Model
    tokenizer: Tokenizer
    label_encoder: LabelEncoder
    maxlen: int
    config_payload: dict

    @classmethod
    def from_artefacts(cls, artefact_dir: str | Path) -> "EmotionPredictor":
        """Load the model and preprocessing artefacts from disk."""

        resolved_artefact_dir = Path(artefact_dir).resolve()
        config_payload = read_json(resolved_artefact_dir / config.CONFIG_FILENAME)
        model = tf.keras.models.load_model(
            resolved_artefact_dir / config.BEST_MODEL_FILENAME,
            custom_objects={"AttentionLayer": AttentionLayer},
            compile=False,
        )
        tokenizer = joblib.load(resolved_artefact_dir / config.TOKENIZER_FILENAME)
        label_encoder = joblib.load(resolved_artefact_dir / config.LABEL_ENCODER_FILENAME)
        cls._validate_loaded_state(model, label_encoder, config_payload)
        return cls(
            model=model,
            tokenizer=tokenizer,
            label_encoder=label_encoder,
            maxlen=int(config_payload["max_len"]),
            config_payload=config_payload,
        )

    @staticmethod
    def _validate_loaded_state(
        model: tf.keras.Model,
        label_encoder: LabelEncoder,
        config_payload: dict,
    ) -> None:
        """Validate saved artefacts against the runtime config snapshot."""

        expected_classes = list(config_payload["label_classes"])
        if list(label_encoder.classes_) != expected_classes:
            raise ValueError(
                "Saved label encoder classes do not match config.json: "
                f"{list(label_encoder.classes_)} != {expected_classes}"
            )

        expected_input_length = int(config_payload["max_len"])
        if model.input_shape[1] != expected_input_length:
            raise ValueError(
                "Loaded model input length does not match config.json: "
                f"{model.input_shape[1]} != {expected_input_length}"
            )

        expected_output_units = int(config_payload["num_classes"])
        if model.output_shape[-1] != expected_output_units:
            raise ValueError(
                "Loaded model output shape does not match config.json: "
                f"{model.output_shape[-1]} != {expected_output_units}"
            )

    @property
    def classes(self) -> list[str]:
        """Return the label classes served by the predictor."""

        return [str(class_name) for class_name in self.label_encoder.classes_]

    def predict(self, text: str, confidence_threshold: float | None = None) -> dict:
        """Predict the emotion of a single text input."""

        return predict_emotion(
            text=text,
            model=self.model,
            tokenizer=self.tokenizer,
            label_encoder=self.label_encoder,
            maxlen=self.maxlen,
            confidence_threshold=config.resolve_confidence_threshold(
                override=confidence_threshold,
                config_payload=self.config_payload,
            ),
        )
