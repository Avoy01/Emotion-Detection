"""Unit tests for preprocessing helpers."""

from __future__ import annotations

import numpy as np

from emotion_detector.data.preprocessor import (
    clean_text,
    fit_tokenizer,
    split_data,
    tokenize_and_pad,
)


def test_clean_text() -> None:
    """The cleaning pipeline should normalize text deterministically."""

    raw = "Hello <b>WORLD</b>! Visit https://example.com now!!!"
    assert clean_text(raw) == "hello world visit now"


def test_clean_text_empty() -> None:
    """Empty input should stay empty."""

    assert clean_text("") == ""


def test_stratified_split() -> None:
    """Class distribution should remain close after stratified splitting."""

    texts = [f"sample {index}" for index in range(60)]
    labels = np.array([0] * 30 + [1] * 30)
    X_train, X_test, y_train, y_test = split_data(texts, labels, test_size=0.2, seed=42)

    train_ratio = y_train.mean()
    test_ratio = y_test.mean()
    assert len(X_train) == 48
    assert len(X_test) == 12
    assert abs(train_ratio - test_ratio) <= 0.02


def test_tokenizer_no_leakage(tmp_path) -> None:
    """Words seen only in test data should map to the OOV token."""

    tokenizer = fit_tokenizer(["happy joy"], save_path=str(tmp_path / "tokenizer.pkl"))
    sequences = tokenizer.texts_to_sequences(["mystery token"])
    assert sequences[0] == [1, 1]


def test_padding_shape(tmp_path) -> None:
    """Padded sequences should always use the configured max length."""

    tokenizer = fit_tokenizer(["happy joy", "sad day"], save_path=str(tmp_path / "tokenizer.pkl"))
    padded = tokenize_and_pad(["happy joy", "sad"], tokenizer=tokenizer, maxlen=5)
    assert padded.shape == (2, 5)
