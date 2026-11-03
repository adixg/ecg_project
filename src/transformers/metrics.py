"""Classwise AUROC, AP/AUPRC, F1, and validation-only threshold selection."""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, roc_auc_score

from .data import CLASSES


def classification_metrics(targets, probabilities, thresholds=None):
    """Return one row per class with AUROC, AP, and fixed/tuned F1.

    Thresholds must have been selected on validation data. With no thresholds,
    both F1 columns use 0.5. Inputs have shape (records, five classes).
    """
    thresholds = np.full(len(CLASSES), 0.5) if thresholds is None else np.asarray(thresholds)
    if targets.ndim != 2 or targets.shape != probabilities.shape or targets.shape[1] != len(CLASSES):
        raise ValueError("Targets and probabilities must have matching (N, 5) shapes")
    if thresholds.shape != (len(CLASSES),) or not np.isfinite(thresholds).all():
        raise ValueError("Provide five finite thresholds")
    if not np.isfinite(probabilities).all() or not ((probabilities >= 0) & (probabilities <= 1)).all():
        raise ValueError("Probabilities must be finite values between zero and one")
    rows = []
    tuned_predictions = probabilities >= thresholds.reshape(1, -1)
    fixed_predictions = probabilities >= 0.5
    for i, name in enumerate(CLASSES):
        valid = np.unique(targets[:, i]).size == 2
        rows.append({
            'class': name,
            'AUROC': roc_auc_score(targets[:, i], probabilities[:, i]) if valid else np.nan,
            'AUPRC': average_precision_score(targets[:, i], probabilities[:, i]) if valid else np.nan,
            'F1_at_0.5': f1_score(targets[:, i], fixed_predictions[:, i], zero_division=0),
            'F1_val_threshold': f1_score(targets[:, i], tuned_predictions[:, i], zero_division=0),
            'threshold': float(thresholds[i]),
            'positives': int(targets[:, i].sum()),
            'prevalence': float(targets[:, i].mean()),
        })
    return pd.DataFrame(rows).set_index('class')


def choose_validation_thresholds(targets, probabilities):
    """Maximize validation F1 per class; break ties closest to 0.5."""
    thresholds = []
    for i in range(len(CLASSES)):
        if np.unique(targets[:, i]).size != 2:
            thresholds.append(0.5)
            continue
        precision, recall, candidates = precision_recall_curve(targets[:, i], probabilities[:, i])
        # The final (precision=1, recall=0) point has no corresponding threshold.
        scores = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
        best = np.flatnonzero(np.isclose(scores, scores.max(), rtol=0, atol=1e-12))
        selected = best[np.argmin(np.abs(candidates[best] - 0.5))]
        thresholds.append(float(candidates[selected]))
    return np.asarray(thresholds, dtype=np.float64)


def summarize_metrics(targets, probabilities, table, thresholds):
    """Combine classwise metrics with micro scores across flattened labels."""
    return {
        'macro_AUROC': float(table.AUROC.mean()),
        'macro_AUPRC': float(table.AUPRC.mean()),
        'macro_F1_at_0.5': float(table['F1_at_0.5'].mean()),
        'macro_F1_val_threshold': float(table.F1_val_threshold.mean()),
        'micro_AUROC': float(roc_auc_score(targets.ravel(), probabilities.ravel())),
        'micro_AUPRC': float(average_precision_score(targets.ravel(), probabilities.ravel())),
        'micro_F1_at_0.5': float(f1_score(targets, probabilities >= 0.5, average='micro', zero_division=0)),
        'micro_F1_val_threshold': float(f1_score(targets, probabilities >= thresholds[None, :],
                                               average='micro', zero_division=0)),
    }


def save_predictions(path, frame, targets, probabilities):
    """Write ECG/patient IDs, targets, and probabilities without precision loss."""
    result = frame[['ecg_id', 'patient_id']].reset_index(drop=True).copy()
    for i, name in enumerate(CLASSES):
        result[f'{name}_target'] = targets[:, i].astype(int)
        result[f'{name}_probability'] = probabilities[:, i].astype(np.float64)
    result.to_csv(path, index=False, float_format="%.17g")
