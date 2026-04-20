"""Training loop helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf

from emotion_detector import config
from emotion_detector.utils.files import ensure_directory


def train(
    model: tf.keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int,
    batch_size: int,
    val_split: float,
    artefact_dir: str,
    class_weights: dict[int, float],
) -> tf.keras.callbacks.History:
    """Train the model with the required callbacks."""

    resolved_artefact_dir = ensure_directory(artefact_dir)
    checkpoint_path = Path(resolved_artefact_dir) / config.BEST_MODEL_FILENAME
    csv_log_path = Path(resolved_artefact_dir) / config.TRAINING_LOG_FILENAME

    callbacks: list[tf.keras.callbacks.Callback] = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(str(csv_log_path), append=False),
    ]

    return model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=val_split,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1,
    )
