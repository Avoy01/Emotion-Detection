"""Text preprocessing utilities."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from emotion_detector import config
from emotion_detector.utils.logger import get_logger

LOGGER = get_logger(__name__)

URL_PATTERN = re.compile(r"http[s]?://\S+")
HTML_PATTERN = re.compile(r"<.*?>")
NON_ALPHANUMERIC_PATTERN = re.compile(r"[^a-z0-9\s']")
WHITESPACE_PATTERN = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Apply the deterministic text cleaning pipeline from the SRS.

    Args:
        text: Raw input text.

    Returns:
        The cleaned text.
    """

    text = str(text).lower()
    text = URL_PATTERN.sub(" ", text)
    text = HTML_PATTERN.sub(" ", text)
    text = NON_ALPHANUMERIC_PATTERN.sub(" ", text)
    text = WHITESPACE_PATTERN.sub(" ", text)
    return text.strip()


def deduplicate_samples(texts: list[str], labels: list[str]) -> tuple[list[str], list[str], int]:
    """Remove duplicate text-label pairs while preserving order."""

    seen: set[tuple[str, str]] = set()
    deduped_texts: list[str] = []
    deduped_labels: list[str] = []
    dropped = 0

    for text, label in zip(texts, labels):
        key = (text, label)
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        deduped_texts.append(text)
        deduped_labels.append(label)

    return deduped_texts, deduped_labels, dropped


def drop_empty_samples(texts: list[str], labels: list[str]) -> tuple[list[str], list[str], int]:
    """Drop rows that are empty after cleaning."""

    filtered_texts: list[str] = []
    filtered_labels: list[str] = []
    dropped = 0

    for text, label in zip(texts, labels):
        if not text:
            dropped += 1
            continue
        filtered_texts.append(text)
        filtered_labels.append(label)

    return filtered_texts, filtered_labels, dropped


def prepare_corpus(texts: list[str], labels: list[str]) -> tuple[list[str], list[str]]:
    """Prepare raw dataset text and labels for downstream processing."""

    deduped_texts, deduped_labels, raw_duplicates = deduplicate_samples(texts, labels)
    if raw_duplicates:
        LOGGER.warning("Dropped %s duplicate raw text-label pairs.", raw_duplicates)

    cleaned_texts = [clean_text(text) for text in deduped_texts]
    cleaned_texts, cleaned_labels, empty_rows = drop_empty_samples(cleaned_texts, deduped_labels)
    if empty_rows:
        LOGGER.warning("Dropped %s rows that became empty after cleaning.", empty_rows)

    cleaned_texts, cleaned_labels, cleaned_duplicates = deduplicate_samples(
        cleaned_texts,
        cleaned_labels,
    )
    if cleaned_duplicates:
        LOGGER.warning("Dropped %s duplicate pairs after cleaning.", cleaned_duplicates)

    unique_labels = sorted(set(cleaned_labels))
    if set(unique_labels) != set(config.LABEL_CLASSES):
        raise ValueError(
            "Expected exactly these labels: "
            f"{list(config.LABEL_CLASSES)}. Found: {unique_labels}"
        )

    return cleaned_texts, cleaned_labels


def encode_labels(labels: list[str], save_path: str) -> np.ndarray:
    """Fit and persist a label encoder.

    Args:
        labels: Emotion labels as strings.
        save_path: File path where the encoder should be stored.

    Returns:
        Encoded integer labels.
    """

    encoder = LabelEncoder()
    encoded = encoder.fit_transform(labels)
    classes = list(encoder.classes_)

    if classes != list(config.LABEL_CLASSES):
        raise ValueError(
            "Expected label classes "
            f"{list(config.LABEL_CLASSES)}, but found {classes}"
        )

    if len(classes) != config.NUM_CLASSES:
        raise ValueError(f"Expected {config.NUM_CLASSES} classes, but found {classes}")

    target = Path(save_path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(encoder, target)
    LOGGER.info("Saved label encoder to %s", target)
    return encoded


def transform_labels(labels: list[str], encoder: LabelEncoder) -> np.ndarray:
    """Transform labels with an already-fitted label encoder."""

    transformed = encoder.transform(labels)
    if np.isnan(transformed).any():
        raise ValueError("Encoded labels contain NaN values.")
    return transformed


def split_data(
    X: Iterable[str],
    y: np.ndarray,
    test_size: float = config.TEST_SIZE,
    seed: int = config.SEED,
) -> tuple[list[str], list[str], np.ndarray, np.ndarray]:
    """Perform a stratified train-test split."""

    X_train, X_test, y_train, y_test = train_test_split(
        list(X),
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test


def fit_tokenizer(
    X_train: list[str],
    save_path: str,
    num_words: int = config.VOCAB_SIZE,
    oov_token: str = "<OOV>",
) -> Tokenizer:
    """Fit a tokenizer on the training corpus and persist it."""

    tokenizer = Tokenizer(num_words=num_words, oov_token=oov_token)
    tokenizer.fit_on_texts(X_train)

    target = Path(save_path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(tokenizer, target)
    LOGGER.info("Saved tokenizer to %s", target)
    return tokenizer


def tokenize_and_pad(
    X: list[str],
    tokenizer: Tokenizer,
    maxlen: int,
    padding: str = "post",
    truncating: str = "post",
) -> np.ndarray:
    """Convert text to padded integer sequences."""

    sequences = tokenizer.texts_to_sequences(X)
    return pad_sequences(sequences, maxlen=maxlen, padding=padding, truncating=truncating)


def compute_balanced_class_weights(y_train: np.ndarray) -> dict[int, float]:
    """Compute balanced class weights for training."""

    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weights = {int(label): float(weight) for label, weight in zip(classes, weights)}
    LOGGER.info("Computed class weights: %s", class_weights)
    return class_weights


def validate_minimum_class_samples(
    y_train: np.ndarray,
    min_samples: int = config.MIN_TRAIN_SAMPLES_PER_CLASS,
) -> None:
    """Ensure each class has a minimum number of training examples."""

    counts = Counter(y_train.tolist())
    insufficient = {int(label): count for label, count in counts.items() if count < min_samples}
    if insufficient:
        raise ValueError(
            "Training split does not satisfy the minimum samples per class "
            f"requirement of {min_samples}: {insufficient}"
        )
