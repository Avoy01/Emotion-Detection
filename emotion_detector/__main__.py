"""Convenience entry point for the package."""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    """Dispatch to one of the package subcommands."""

    parser = argparse.ArgumentParser(description="Emotion detector command dispatcher.")
    parser.add_argument(
        "command",
        choices=["train", "evaluate", "infer", "check-setup"],
        help="Subcommand to run.",
    )
    args, remaining = parser.parse_known_args()

    if args.command == "train":
        from emotion_detector import train as command_module
    elif args.command == "evaluate":
        from emotion_detector import evaluate as command_module
    elif args.command == "infer":
        from emotion_detector import infer as command_module
    else:
        from emotion_detector import check_setup as command_module

    sys.argv = [sys.argv[0], *remaining]
    return command_module.main()


if __name__ == "__main__":
    raise SystemExit(main())
