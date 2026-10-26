"""Metrics and validation-threshold selection for multi-label predictions."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


def find_f1_thresholds(
    targets: np.ndarray,
    probabilities: np.ndarray,
    *,
    candidates: np.ndarray | None = None,
) -> np.ndarray:
    """Selects one F1-maximizing threshold per class on validation data."""

    if candidates is None:
        candidates = np.arange(0.05, 1.0, 0.05)

    thresholds = []
    for class_index in range(targets.shape[1]):
        class_targets = targets[:, class_index]
        class_probabilities = probabilities[:, class_index]
        scores = [
            f1_score(
                class_targets,
                class_probabilities >= threshold,
                zero_division=0,
            )
            for threshold in candidates
        ]
        thresholds.append(float(candidates[int(np.argmax(scores))]))
    return np.asarray(thresholds, dtype=np.float32)


def compute_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    *,
    thresholds: np.ndarray | None = None,
) -> dict[str, float]:
    """Computes the project's three macro-averaged evaluation metrics."""

    if thresholds is None:
        thresholds = np.full(targets.shape[1], 0.5, dtype=np.float32)
    predictions = probabilities >= thresholds.reshape(1, -1)
    return {
        "macro_auroc": float(
            roc_auc_score(targets, probabilities, average="macro")
        ),
        "macro_auprc": float(
            average_precision_score(targets, probabilities, average="macro")
        ),
        "macro_f1": float(
            f1_score(targets, predictions, average="macro", zero_division=0)
        ),
    }


def compute_per_class_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    *,
    thresholds: np.ndarray,
    class_names: Sequence[str],
) -> dict[str, dict[str, float | int]]:
    """Computes class-specific metrics at validation-selected thresholds."""

    if targets.shape != probabilities.shape:
        raise ValueError(
            "targets and probabilities must have the same shape; "
            f"got {targets.shape} and {probabilities.shape}"
        )
    if targets.shape[1] != len(class_names):
        raise ValueError(
            f"Expected {targets.shape[1]} class names; got {len(class_names)}"
        )
    if len(thresholds) != targets.shape[1]:
        raise ValueError(
            f"Expected {targets.shape[1]} thresholds; got {len(thresholds)}"
        )

    results = {}
    for index, class_name in enumerate(class_names):
        class_targets = targets[:, index]
        class_probabilities = probabilities[:, index]
        threshold = float(thresholds[index])
        predictions = class_probabilities >= threshold
        results[class_name] = {
            "auroc": float(roc_auc_score(class_targets, class_probabilities)),
            "auprc": float(
                average_precision_score(class_targets, class_probabilities)
            ),
            "f1": float(
                f1_score(class_targets, predictions, zero_division=0)
            ),
            "threshold": threshold,
            "positive_count": int(class_targets.sum()),
            "prevalence": float(class_targets.mean()),
        }
    return results
