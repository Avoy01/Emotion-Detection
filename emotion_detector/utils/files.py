"""File and setup helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from emotion_detector import config


def ensure_directory(path: str | Path) -> Path:
    """Create a directory if it does not already exist."""

    resolved = Path(path).resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def write_json(payload: dict[str, Any], filepath: str | Path) -> Path:
    """Write JSON to disk with stable formatting."""

    target = Path(filepath).resolve()
    ensure_directory(target.parent)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return target


def read_json(filepath: str | Path) -> dict[str, Any]:
    """Read a JSON file from disk."""

    return json.loads(Path(filepath).resolve().read_text(encoding="utf-8"))


def validate_required_assets(
    data_path: str | Path = config.DATA_PATH,
    glove_path: str | Path = config.GLOVE_PATH,
) -> list[str]:
    """Return a list of setup errors for required external assets."""

    errors: list[str] = []
    resolved_data_path = Path(data_path).resolve()
    resolved_glove_path = Path(glove_path).resolve()

    if not resolved_data_path.exists():
        errors.append(
            "Dataset not found at "
            f"'{resolved_data_path}'. Download the Kaggle emotion dataset from "
            f"{config.DATASET_DOWNLOAD_URL} and place emotion.csv there."
        )

    if not resolved_glove_path.exists():
        errors.append(
            "GloVe vectors not found at "
            f"'{resolved_glove_path}'. Download glove.6B.zip from "
            f"{config.GLOVE_DOWNLOAD_URL} and extract glove.6B.100d.txt there."
        )

    return errors
