import numpy as np
from sklearn.metrics import f1_score
from typing import Dict, List, Tuple


def tune_decision_thresholds(
    y_val_true: np.ndarray,
    y_val_probs: np.ndarray,
    label_names: List[str],
    min_thresh: float = 0.15,
    max_thresh: float = 0.85,
    step: float = 0.02
) -> Dict[str, float]:
    """
    Finds the optimal probability threshold per label on the validation set
    to maximize F1 score for that label.
    """
    thresholds_dict: Dict[str, float] = {}
    threshold_candidates = np.arange(min_thresh, max_thresh + step / 2, step)

    for i, label in enumerate(label_names):
        best_f1 = -1.0
        best_thresh = 0.50
        y_true = y_val_true[:, i]
        probs = y_val_probs[:, i]

        if y_true.sum() == 0:
            thresholds_dict[label] = 0.50
            continue

        for t in threshold_candidates:
            preds = (probs >= t).astype(int)
            score = f1_score(y_true, preds, zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_thresh = float(round(t, 3))

        thresholds_dict[label] = best_thresh

    return thresholds_dict


def apply_thresholds(
    y_probs: np.ndarray,
    thresholds: Dict[str, float],
    label_names: List[str]
) -> np.ndarray:
    """Apply label-specific thresholds to convert probability matrix to binary predictions."""
    y_preds = np.zeros_like(y_probs, dtype=int)
    for i, label in enumerate(label_names):
        t = thresholds.get(label, 0.50)
        y_preds[:, i] = (y_probs[:, i] >= t).astype(int)
    return y_preds
