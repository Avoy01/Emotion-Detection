"""Emotion detector package."""

from __future__ import annotations

import logging
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
logging.getLogger("tensorflow").setLevel(logging.ERROR)

__all__ = ["__version__"]

__version__ = "0.1.0"
