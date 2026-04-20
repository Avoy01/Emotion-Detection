"""GloVe embedding matrix construction."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from emotion_detector.utils.logger import get_logger

LOGGER = get_logger(__name__)


def build_embedding_matrix(
    glove_path: str,
    word_index: dict[str, int],
    vocab_size: int,
    embedding_dim: int,
) -> np.ndarray:
    """Construct an embedding matrix from a GloVe file.

    Args:
        glove_path: Path to the GloVe text file.
        word_index: Tokenizer word index mapping.
        vocab_size: Vocabulary ceiling.
        embedding_dim: Expected embedding dimensionality.

    Returns:
        A NumPy array shaped ``(vocab_size + 1, embedding_dim)``.
    """

    resolved_glove_path = Path(glove_path).resolve()
    if not resolved_glove_path.exists():
        raise FileNotFoundError(f"GloVe file not found: {resolved_glove_path}")

    capped_word_index = {
        word: index for word, index in word_index.items() if index <= vocab_size
    }
    embedding_matrix = np.zeros((len(capped_word_index) + 1, embedding_dim), dtype=np.float32)

    found_words = 0
    with resolved_glove_path.open("r", encoding="utf-8") as glove_file:
        for line in glove_file:
            values = line.rstrip().split(" ")
            if len(values) != embedding_dim + 1:
                continue

            word = values[0]
            token_index = capped_word_index.get(word)
            if token_index is None:
                continue

            embedding_matrix[token_index] = np.asarray(values[1:], dtype=np.float32)
            found_words += 1

    total_words = len(capped_word_index)
    oov_words = total_words - found_words
    oov_rate = oov_words / max(total_words, 1)
    LOGGER.info(
        "Embedding coverage: vocab=%s found=%s oov=%s (%.2f%%)",
        total_words,
        found_words,
        oov_words,
        oov_rate * 100,
    )

    if oov_rate > 0.15:
        LOGGER.warning("OOV rate %.2f%% exceeds the expected 15%% threshold.", oov_rate * 100)

    return embedding_matrix
