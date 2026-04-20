"""Central configuration for the emotion detector project."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
GLOVE_DIR = PROJECT_ROOT / "glove"
DEFAULT_ARTEFACT_DIR = PROJECT_ROOT / "artefacts"

VOCAB_SIZE = 20_000
MAX_LEN = 100
EMBEDDING_DIM = 100
LSTM_UNITS = 128
ATTENTION_DIM = 128
DENSE_UNITS = 64
DROPOUT_RATE = 0.5
NUM_CLASSES = 6
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-3
VALIDATION_SPLIT = 0.2
TEST_SIZE = 0.2
SEED = 42
MIN_TRAIN_SAMPLES_PER_CLASS = 100
INFERENCE_CONFIDENCE_THRESHOLD = 0.45
INFERENCE_CONFIDENCE_THRESHOLD_ENV = "INFERENCE_CONFIDENCE_THRESHOLD"

DATA_PATH = DATA_DIR / "emotion.csv"
GLOVE_PATH = GLOVE_DIR / "glove.6B.100d.txt"

BEST_MODEL_FILENAME = "best_model.keras"
TOKENIZER_FILENAME = "tokenizer.pkl"
LABEL_ENCODER_FILENAME = "label_encoder.pkl"
CONFIG_FILENAME = "config.json"
TRAINING_LOG_FILENAME = "training_log.csv"
EVALUATION_REPORT_FILENAME = "evaluation_report.txt"
ACCURACY_CURVE_FILENAME = "accuracy_curve.png"
LOSS_CURVE_FILENAME = "loss_curve.png"
CONFUSION_MATRIX_FILENAME = "confusion_matrix.png"

DATASET_DOWNLOAD_URL = "https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp"
GLOVE_DOWNLOAD_URL = "https://nlp.stanford.edu/projects/glove/"

LABEL_CLASSES = ("anger", "fear", "joy", "love", "sadness", "surprise")


def resolve_artefact_dir(override: str | os.PathLike[str] | None = None) -> Path:
    """Return the effective artefact directory."""

    if override:
        return Path(override).resolve()

    env_override = os.getenv("ARTEFACT_DIR")
    if env_override:
        return Path(env_override).resolve()

    return DEFAULT_ARTEFACT_DIR.resolve()


def _validate_threshold(value: float, source: str) -> float:
    """Validate that a confidence threshold is within [0.0, 1.0]."""

    if not 0.0 <= value <= 1.0:
        raise ValueError(
            f"{source} must be between 0.0 and 1.0 inclusive, received {value}."
        )
    return value


def resolve_confidence_threshold(
    override: float | None = None,
    config_payload: dict[str, Any] | None = None,
) -> float:
    """Resolve the runtime confidence threshold.

    Precedence:
    1. Explicit function override
    2. ``INFERENCE_CONFIDENCE_THRESHOLD`` environment variable
    3. Saved ``config.json`` payload
    4. Project default
    """

    if override is not None:
        return _validate_threshold(float(override), "CLI confidence threshold")

    env_override = os.getenv(INFERENCE_CONFIDENCE_THRESHOLD_ENV)
    if env_override:
        try:
            return _validate_threshold(
                float(env_override),
                f"Environment variable {INFERENCE_CONFIDENCE_THRESHOLD_ENV}",
            )
        except ValueError as exc:
            raise ValueError(str(exc)) from exc

    if config_payload and "inference_confidence_threshold" in config_payload:
        return _validate_threshold(
            float(config_payload["inference_confidence_threshold"]),
            "Saved config inference confidence threshold",
        )

    return _validate_threshold(
        INFERENCE_CONFIDENCE_THRESHOLD,
        "Default inference confidence threshold",
    )


def serializable_config(artefact_dir: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Build a JSON-serializable configuration snapshot."""

    resolved_artefact_dir = resolve_artefact_dir(artefact_dir)
    return {
        "project_root": str(PROJECT_ROOT),
        "data_path": str(DATA_PATH),
        "glove_path": str(GLOVE_PATH),
        "artefact_dir": str(resolved_artefact_dir),
        "vocab_size": VOCAB_SIZE,
        "max_len": MAX_LEN,
        "embedding_dim": EMBEDDING_DIM,
        "lstm_units": LSTM_UNITS,
        "attention_dim": ATTENTION_DIM,
        "dense_units": DENSE_UNITS,
        "dropout_rate": DROPOUT_RATE,
        "num_classes": NUM_CLASSES,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "validation_split": VALIDATION_SPLIT,
        "test_size": TEST_SIZE,
        "seed": SEED,
        "min_train_samples_per_class": MIN_TRAIN_SAMPLES_PER_CLASS,
        "inference_confidence_threshold": INFERENCE_CONFIDENCE_THRESHOLD,
        "label_classes": list(LABEL_CLASSES),
        "filenames": {
            "model": BEST_MODEL_FILENAME,
            "tokenizer": TOKENIZER_FILENAME,
            "label_encoder": LABEL_ENCODER_FILENAME,
            "config": CONFIG_FILENAME,
            "training_log": TRAINING_LOG_FILENAME,
            "evaluation_report": EVALUATION_REPORT_FILENAME,
            "accuracy_curve": ACCURACY_CURVE_FILENAME,
            "loss_curve": LOSS_CURVE_FILENAME,
            "confusion_matrix": CONFUSION_MATRIX_FILENAME,
        },
    }
