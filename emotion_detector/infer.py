"""CLI entry point for single-text inference."""

from __future__ import annotations

import argparse
import json

from emotion_detector import config
from emotion_detector.inference.predictor import EmotionPredictor
from emotion_detector.utils.logger import configure_logging


def build_parser() -> argparse.ArgumentParser:
    """Build the inference CLI parser."""

    parser = argparse.ArgumentParser(description="Run single-text emotion inference.")
    parser.add_argument("--text", required=True, help="Text to classify.")
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
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=None,
        help=(
            "Optional uncertainty threshold in [0.0, 1.0]. "
            "Overrides the INFERENCE_CONFIDENCE_THRESHOLD environment variable."
        ),
    )
    return parser


def main() -> int:
    """Run inference and print the prediction as JSON."""

    parser = build_parser()
    args = parser.parse_args()
    configure_logging(args.log_level)
    predictor = EmotionPredictor.from_artefacts(args.artefacts)
    result = predictor.predict(
        args.text,
        confidence_threshold=args.confidence_threshold,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
