"""Dataset loading utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from emotion_detector.utils.logger import get_logger

LOGGER = get_logger(__name__)


def load_dataset(filepath: str) -> tuple[list[str], list[str]]:
    """Load the semicolon-delimited emotion CSV file.

    Args:
        filepath: Path to the CSV file.

    Returns:
        A tuple containing raw texts and raw labels.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If the required columns are missing.
    """

    resolved_path = Path(filepath).resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {resolved_path}")

    try:
        dataframe = pd.read_csv(resolved_path, sep=";")
    except Exception as exc:
        raise ValueError(f"Failed to read dataset at {resolved_path}: {exc}") from exc

    required_columns = {"text", "label"}
    missing_columns = required_columns - set(dataframe.columns)
    if missing_columns:
        raise ValueError(
            "Dataset is missing required columns: "
            f"{sorted(missing_columns)}. Found columns: {list(dataframe.columns)}"
        )

    original_size = len(dataframe)
    dataframe = dataframe.dropna(subset=["text", "label"])
    dropped_rows = original_size - len(dataframe)
    if dropped_rows:
        LOGGER.warning("Dropped %s rows with null text/label values.", dropped_rows)

    texts = dataframe["text"].astype(str).tolist()
    labels = dataframe["label"].astype(str).tolist()
    LOGGER.info("Loaded %s rows from %s", len(texts), resolved_path)
    return texts, labels
