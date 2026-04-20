"""Validate local setup and external assets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from emotion_detector import config
from emotion_detector.utils.files import validate_required_assets
from emotion_detector.utils.logger import configure_logging, get_logger

LOGGER = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for setup validation."""

    parser = argparse.ArgumentParser(description="Validate environment assets for the project.")
    parser.add_argument("--data", default=str(config.DATA_PATH), help="Path to emotion.csv.")
    parser.add_argument(
        "--glove",
        default=str(config.GLOVE_PATH),
        help="Path to glove.6B.100d.txt.",
    )
    return parser


def main() -> int:
    """Run setup validation."""

    configure_logging()
    parser = build_parser()
    args = parser.parse_args()

    LOGGER.info("Python executable: %s", sys.executable)
    LOGGER.info("Expected dataset path: %s", Path(args.data).resolve())
    LOGGER.info("Expected GloVe path: %s", Path(args.glove).resolve())

    errors = validate_required_assets(data_path=args.data, glove_path=args.glove)
    if errors:
        for error in errors:
            LOGGER.error(error)
        return 1

    LOGGER.info("Required assets are present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
