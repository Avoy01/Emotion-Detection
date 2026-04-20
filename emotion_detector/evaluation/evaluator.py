"""Evaluation pipeline utilities."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder, label_binarize

from emotion_detector import config
from emotion_detector.utils.files import ensure_directory

plt.switch_backend("Agg")


def load_training_history(log_path: str | Path) -> pd.DataFrame:
    """Load the CSVLogger output into a DataFrame."""

    return pd.read_csv(Path(log_path).resolve())


def _plot_training_curves(history_df: pd.DataFrame, artefact_dir: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(history_df["accuracy"], label="train_accuracy")
    plt.plot(history_df["val_accuracy"], label="val_accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(artefact_dir / config.ACCURACY_CURVE_FILENAME, dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(history_df["loss"], label="train_loss")
    plt.plot(history_df["val_loss"], label="val_loss")
    plt.title("Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(artefact_dir / config.LOSS_CURVE_FILENAME, dpi=200)
    plt.close()


def _plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_encoder: LabelEncoder,
    artefact_dir: Path,
) -> np.ndarray:
    matrix = confusion_matrix(y_true, y_pred, normalize="true")
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Normalized Confusion Matrix")
    plt.tight_layout()
    plt.savefig(artefact_dir / config.CONFUSION_MATRIX_FILENAME, dpi=200)
    plt.close()
    return matrix


def _compute_per_class_auroc(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    label_encoder: LabelEncoder,
) -> dict[str, float]:
    classes = np.arange(len(label_encoder.classes_))
    y_true_binarized = label_binarize(y_true, classes=classes)

    auroc_scores: dict[str, float] = {}
    for index, class_name in enumerate(label_encoder.classes_):
        try:
            score = roc_auc_score(y_true_binarized[:, index], probabilities[:, index])
        except ValueError:
            score = float("nan")
        auroc_scores[str(class_name)] = float(score)
    return auroc_scores


def _write_evaluation_report(
    artefact_dir: Path,
    model_filename: str,
    num_samples: int,
    accuracy: float,
    macro_f1: float,
    weighted_f1: float,
    report_dict: dict,
    auroc_scores: dict[str, float],
    history_df: pd.DataFrame,
) -> Path:
    final_train_accuracy = float(history_df["accuracy"].iloc[-1])
    final_val_accuracy = float(history_df["val_accuracy"].iloc[-1])
    overfit_gap = final_train_accuracy - final_val_accuracy
    gap_status = "PASS" if overfit_gap <= 0.10 else "WARNING"

    lines = [
        "======================================================",
        "EMOTION DETECTION MODEL - EVALUATION REPORT",
        "======================================================",
        f"Date           : {datetime.utcnow().isoformat()}Z",
        f"Model          : {model_filename}",
        f"Test samples   : {num_samples}",
        "",
        "OVERALL METRICS",
        "---------------",
        f"Test Accuracy  : {accuracy:.4f}",
        f"Macro F1       : {macro_f1:.4f}",
        f"Weighted F1    : {weighted_f1:.4f}",
        "",
        "PER-CLASS METRICS",
        "-----------------",
        "Class       Precision  Recall  F1-Score  Support  AUROC",
    ]

    for class_name in config.LABEL_CLASSES:
        class_metrics = report_dict[class_name]
        lines.append(
            f"{class_name:<11}"
            f"{class_metrics['precision']:.4f}     "
            f"{class_metrics['recall']:.4f}  "
            f"{class_metrics['f1-score']:.4f}    "
            f"{int(class_metrics['support']):<8}"
            f"{auroc_scores[class_name]:.4f}"
        )

    lines.extend(
        [
            "",
            "OVERFITTING CHECK",
            "-----------------",
            f"Final train accuracy : {final_train_accuracy:.4f}",
            f"Final val accuracy   : {final_val_accuracy:.4f}",
            f"Gap                  : {overfit_gap:.4f}  [{gap_status}]",
            "",
            "ARTEFACTS SAVED",
            "---------------",
            str(artefact_dir / config.BEST_MODEL_FILENAME),
            str(artefact_dir / config.TOKENIZER_FILENAME),
            str(artefact_dir / config.LABEL_ENCODER_FILENAME),
            str(artefact_dir / config.ACCURACY_CURVE_FILENAME),
            str(artefact_dir / config.LOSS_CURVE_FILENAME),
            str(artefact_dir / config.CONFUSION_MATRIX_FILENAME),
            str(artefact_dir / config.TRAINING_LOG_FILENAME),
            "======================================================",
        ]
    )

    target = artefact_dir / config.EVALUATION_REPORT_FILENAME
    target.write_text("\n".join(lines), encoding="utf-8")
    return target


def evaluate(
    model: tf.keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    label_encoder: LabelEncoder,
    history_df: pd.DataFrame,
    artefact_dir: str,
) -> dict[str, float | dict[str, float]]:
    """Evaluate the model and save plots and reports."""

    resolved_artefact_dir = ensure_directory(artefact_dir)
    probabilities = model.predict(X_test, verbose=0)
    predictions = np.argmax(probabilities, axis=1)

    accuracy = float(accuracy_score(y_test, predictions))
    macro_f1 = float(f1_score(y_test, predictions, average="macro"))
    weighted_f1 = float(f1_score(y_test, predictions, average="weighted"))

    report_dict = classification_report(
        y_test,
        predictions,
        target_names=list(label_encoder.classes_),
        digits=4,
        output_dict=True,
        zero_division=0,
    )
    confusion = _plot_confusion_matrix(y_test, predictions, label_encoder, resolved_artefact_dir)
    _plot_training_curves(history_df, resolved_artefact_dir)
    auroc_scores = _compute_per_class_auroc(y_test, probabilities, label_encoder)
    _write_evaluation_report(
        artefact_dir=resolved_artefact_dir,
        model_filename=config.BEST_MODEL_FILENAME,
        num_samples=len(y_test),
        accuracy=accuracy,
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
        report_dict=report_dict,
        auroc_scores=auroc_scores,
        history_df=history_df,
    )

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "confusion_matrix": confusion.tolist(),
        "auroc_scores": auroc_scores,
    }
