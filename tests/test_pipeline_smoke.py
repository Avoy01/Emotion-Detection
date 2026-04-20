"""Integration smoke test for the training pipeline."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from emotion_detector import config
from emotion_detector.data.data_loader import load_dataset
from emotion_detector.data.preprocessor import (
    compute_balanced_class_weights,
    encode_labels,
    fit_tokenizer,
    prepare_corpus,
    split_data,
    tokenize_and_pad,
    transform_labels,
)
from emotion_detector.embedding.embedding_builder import build_embedding_matrix
from emotion_detector.evaluation.evaluator import evaluate, load_training_history
from emotion_detector.inference.predictor import EmotionPredictor
from emotion_detector.model.model_builder import build_model
from emotion_detector.training.trainer import train
from emotion_detector.utils.files import ensure_directory, write_json
from emotion_detector.utils.seed import set_global_seed


def _write_small_glove_file(glove_path: Path, words: list[str], embedding_dim: int) -> None:
    lines = []
    for word_index, word in enumerate(words, start=1):
        vector = " ".join(f"{0.01 * word_index:.4f}" for _ in range(embedding_dim))
        lines.append(f"{word} {vector}")
    glove_path.write_text("\n".join(lines), encoding="utf-8")


@pytest.mark.slow
def test_training_pipeline_smoke(tmp_path) -> None:
    """A tiny end-to-end run should save reloadable artefacts."""

    set_global_seed(42)
    data_path = tmp_path / "emotion.csv"
    glove_path = tmp_path / "glove.6B.100d.txt"
    artefact_dir = ensure_directory(tmp_path / "artefacts")

    class_tokens = {
        "anger": "furious",
        "fear": "scared",
        "joy": "happy",
        "love": "adore",
        "sadness": "gloomy",
        "surprise": "astonished",
    }
    rows = []
    for label, token in class_tokens.items():
        for index in range(12):
            rows.append({"text": f"{token} feeling sample {index}", "label": label})
    pd.DataFrame(rows).to_csv(data_path, sep=";", index=False)
    _write_small_glove_file(
        glove_path,
        list(class_tokens.values()) + ["feeling", "sample"],
        8,
    )

    texts, labels = load_dataset(str(data_path))
    cleaned_texts, cleaned_labels = prepare_corpus(texts, labels)
    encoded_labels = encode_labels(
        cleaned_labels,
        str(artefact_dir / config.LABEL_ENCODER_FILENAME),
    )

    X_train_texts, X_test_texts, y_train, y_test = split_data(
        cleaned_texts,
        encoded_labels,
        0.2,
        42,
    )
    tokenizer = fit_tokenizer(
        X_train_texts,
        str(artefact_dir / config.TOKENIZER_FILENAME),
        num_words=100,
    )
    X_train = tokenize_and_pad(X_train_texts, tokenizer, maxlen=6)
    X_test = tokenize_and_pad(X_test_texts, tokenizer, maxlen=6)

    embedding_matrix = build_embedding_matrix(
        glove_path=str(glove_path),
        word_index=tokenizer.word_index,
        vocab_size=100,
        embedding_dim=8,
    )
    model = build_model(
        vocab_size=embedding_matrix.shape[0] - 1,
        embedding_dim=8,
        embedding_matrix=embedding_matrix,
        maxlen=6,
        lstm_units=4,
        attention_dim=4,
        dense_units=4,
        dropout_rate=0.2,
        num_classes=6,
        learning_rate=1e-3,
    )
    train(
        model=model,
        X_train=X_train,
        y_train=y_train,
        epochs=2,
        batch_size=8,
        val_split=0.2,
        artefact_dir=str(artefact_dir),
        class_weights=compute_balanced_class_weights(y_train),
    )
    write_json(
        {
            "max_len": 6,
            "num_classes": 6,
            "label_classes": list(config.LABEL_CLASSES),
        },
        artefact_dir / config.CONFIG_FILENAME,
    )

    assert (artefact_dir / config.BEST_MODEL_FILENAME).exists()
    assert (artefact_dir / config.TOKENIZER_FILENAME).exists()
    assert (artefact_dir / config.LABEL_ENCODER_FILENAME).exists()
    assert (artefact_dir / config.TRAINING_LOG_FILENAME).exists()

    history_df = load_training_history(
        artefact_dir / config.TRAINING_LOG_FILENAME
    )
    label_encoder = joblib.load(artefact_dir / config.LABEL_ENCODER_FILENAME)
    metrics = evaluate(
        model=model,
        X_test=X_test,
        y_test=y_test,
        label_encoder=label_encoder,
        history_df=history_df,
        artefact_dir=str(artefact_dir),
    )
    assert metrics["accuracy"] >= 0.0

    predictor = EmotionPredictor.from_artefacts(artefact_dir)
    original = predictor.predict("happy feeling sample")
    reloaded = predictor.predict("happy feeling sample")
    assert original == reloaded

    transformed_labels = transform_labels(cleaned_labels, label_encoder)
    assert isinstance(transformed_labels, np.ndarray)
