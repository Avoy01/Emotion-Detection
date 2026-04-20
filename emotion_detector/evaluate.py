"""CLI entry point for evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import tensorflow as tf

from emotion_detector import config
from emotion_detector.data.data_loader import load_dataset
from emotion_detector.data.preprocessor import (
    prepare_corpus,
    split_data,
    tokenize_and_pad,
    transform_labels,
)
from emotion_detector.evaluation.evaluator import evaluate, load_training_history
from emotion_detector.inference.predictor import EmotionPredictor
from emotion_detector.model.attention import AttentionLayer
from emotion_detector.utils.files import validate_required_assets
from emotion_detector.utils.logger import configure_logging, get_logger
from emotion_detector.utils.seed import set_global_seed

LOGGER = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the evaluation CLI parser."""

    parser = argparse.ArgumentParser(
        description="Evaluate a trained emotion detection model."
    )
    parser.add_argument("--data", default=str(config.DATA_PATH), help="Path to emotion.csv.")
    parser.add_argument(
        "--artefacts",
        default=str(config.DEFAULT_ARTEFACT_DIR),
        help="Directory containing saved model artefacts.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Python logging level, e.g. INFO or DEBUG.",
    )
    return parser


def main() -> int:
    """Run model evaluation against the deterministic test split."""

    parser = build_parser()
    args = parser.parse_args()
    configure_logging(args.log_level)
    set_global_seed(config.SEED)

    asset_errors = validate_required_assets(
        data_path=args.data,
        glove_path=config.GLOVE_PATH,
    )
    if asset_errors:
        for error in asset_errors:
            if "GloVe" not in error:
                LOGGER.error(error)
        if any("Dataset" in error for error in asset_errors):
            return 1

    artefact_dir = Path(args.artefacts).resolve()
    predictor = EmotionPredictor.from_artefacts(artefact_dir)
    label_encoder = joblib.load(artefact_dir / config.LABEL_ENCODER_FILENAME)
    tokenizer = predictor.tokenizer

    raw_texts, raw_labels = load_dataset(args.data)
    cleaned_texts, cleaned_labels = prepare_corpus(raw_texts, raw_labels)
    encoded_labels = transform_labels(cleaned_labels, label_encoder)
    _X_train_texts, X_test_texts, _y_train, y_test = split_data(
        cleaned_texts,
        encoded_labels,
        test_size=config.TEST_SIZE,
        seed=config.SEED,
    )

    X_test = tokenize_and_pad(
        X_test_texts,
        tokenizer=tokenizer,
        maxlen=predictor.maxlen,
    )
    history_df = load_training_history(artefact_dir / config.TRAINING_LOG_FILENAME)
    model = tf.keras.models.load_model(
        artefact_dir / config.BEST_MODEL_FILENAME,
        custom_objects={"AttentionLayer": AttentionLayer},
        compile=False,
    )
    metrics = evaluate(
        model=model,
        X_test=X_test,
        y_test=y_test,
        label_encoder=label_encoder,
        history_df=history_df,
        artefact_dir=str(artefact_dir),
    )

    LOGGER.info(
        "Evaluation complete: accuracy=%.4f weighted_f1=%.4f",
        metrics["accuracy"],
        metrics["weighted_f1"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
