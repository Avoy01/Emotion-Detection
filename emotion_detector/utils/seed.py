"""Reproducibility utilities."""

from __future__ import annotations

import os
import random

import numpy as np
import tensorflow as tf

from emotion_detector.config import SEED


def set_global_seed(seed: int = SEED) -> None:
    """Set all supported random seeds for reproducible runs."""

    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
