"""CLI entry point for model training."""

from __future__ import annotations

import argparse
from pathlib import Path

from emotion_detector import config
from emotion_detector.data.data_loader import load_dataset
from emotion_detector.data.preprocessor import (
    compute_balanced_class_weights,
    encode_labels,
    fit_tokenizer,
    prepare_corpus,
    split_data,
    tokenize_and_pad,
    validate_minimum_class_samples,
)
from emotion_detector.embedding.embedding_builder import build_embedding_matrix
from emotion_detector.model.model_builder import build_model
from emotion_detector.training.trainer import train
from emotion_detector.utils.files import (
    ensure_directory,
    validate_required_assets,
    write_json,
)
from emotion_detector.utils.logger import configure_logging, get_logger
from emotion_detector.utils.seed import set_global_seed

LOGGER = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the training CLI parser."""

    parser = argparse.ArgumentParser(description="Train the emotion detection model.")
    parser.add_argument("--data", default=str(config.DATA_PATH), help="Path to emotion.csv.")
    parser.add_argument("--glove", default=str(config.GLOVE_PATH), help="Path to GloVe vectors.")
    parser.add_argument(
        "--output",
        default=str(config.DEFAULT_ARTEFACT_DIR),
        help="Directory where artefacts will be written.",
    )
    parser.add_argument("--epochs", type=int, default=config.EPOCHS, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE, help="Batch size.")
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Python logging level, e.g. INFO or DEBUG.",
    )
    return parser


def main() -> int:
    """Run the end-to-end training pipeline."""

    parser = build_parser()
    args = parser.parse_args()
    configure_logging(args.log_level)
    set_global_seed(config.SEED)

    asset_errors = validate_required_assets(data_path=args.data, glove_path=args.glove)
    if asset_errors:
        for error in asset_errors:
            LOGGER.error(error)
        return 1

    artefact_dir = ensure_directory(args.output)

    raw_texts, raw_labels = load_dataset(args.data)
    cleaned_texts, cleaned_labels = prepare_corpus(raw_texts, raw_labels)
    encoded_labels = encode_labels(
        cleaned_labels,
        save_path=str(Path(artefact_dir) / config.LABEL_ENCODER_FILENAME),
    )

    X_train_texts, _X_test_texts, y_train, _y_test = split_data(
        cleaned_texts,
        encoded_labels,
        test_size=config.TEST_SIZE,
        seed=config.SEED,
    )
    validate_minimum_class_samples(y_train)

    tokenizer = fit_tokenizer(
        X_train_texts,
        save_path=str(Path(artefact_dir) / config.TOKENIZER_FILENAME),
        num_words=config.VOCAB_SIZE,
    )
    X_train = tokenize_and_pad(X_train_texts, tokenizer=tokenizer, maxlen=config.MAX_LEN)

    embedding_matrix = build_embedding_matrix(
        glove_path=args.glove,
        word_index=tokenizer.word_index,
        vocab_size=config.VOCAB_SIZE,
        embedding_dim=config.EMBEDDING_DIM,
    )
    effective_vocab_size = embedding_matrix.shape[0] - 1

    model = build_model(
        vocab_size=effective_vocab_size,
        embedding_dim=config.EMBEDDING_DIM,
        embedding_matrix=embedding_matrix,
        maxlen=config.MAX_LEN,
        lstm_units=config.LSTM_UNITS,
        attention_dim=config.ATTENTION_DIM,
        dense_units=config.DENSE_UNITS,
        dropout_rate=config.DROPOUT_RATE,
        num_classes=config.NUM_CLASSES,
        learning_rate=config.LEARNING_RATE,
    )

    class_weights = compute_balanced_class_weights(y_train)
    history = train(
        model=model,
        X_train=X_train,
        y_train=y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        val_split=config.VALIDATION_SPLIT,
        artefact_dir=str(artefact_dir),
        class_weights=class_weights,
    )

    runtime_config = config.serializable_config(artefact_dir=artefact_dir)
    runtime_config.update(
        {
            "artefact_dir": str(artefact_dir),
            "data_path": str(Path(args.data).resolve()),
            "glove_path": str(Path(args.glove).resolve()),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "effective_vocab_size": effective_vocab_size,
            "label_classes": list(config.LABEL_CLASSES),
            "history": {
                key: [float(value) for value in values]
                for key, values in history.history.items()
            },
        }
    )
    write_json(runtime_config, Path(artefact_dir) / config.CONFIG_FILENAME)
    LOGGER.info(
        "Training complete. Best model saved to %s",
        Path(artefact_dir) / config.BEST_MODEL_FILENAME,
    )
    LOGGER.info(
        "Run evaluation next with: python -m emotion_detector.evaluate --data %s",
        args.data,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
