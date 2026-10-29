import numpy as np
from typing import Dict, Any, List, Tuple
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    hamming_loss,
    accuracy_score,
    jaccard_score,
    confusion_matrix,
)

LABEL_NAMES = ["fork", "pin", "skewer", "discoveredAttack"]


def compute_multilabel_metrics(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    y_pred: np.ndarray,
    label_names: List[str] = LABEL_NAMES,
) -> Dict[str, Any]:
    """
    Computes complete set of multi-label evaluation metrics:
    - Global: micro/macro/weighted F1, Hamming loss, subset accuracy, Jaccard
    - Per-label: precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix
    """
    metrics: Dict[str, Any] = {}

    # Global multi-label metrics
    metrics["micro_f1"] = float(round(f1_score(y_true, y_pred, average="micro", zero_division=0), 4))
    metrics["macro_f1"] = float(round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4))
    metrics["weighted_f1"] = float(round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 4))
    metrics["hamming_loss"] = float(round(hamming_loss(y_true, y_pred), 4))
    metrics["subset_accuracy"] = float(round(accuracy_score(y_true, y_pred), 4))
    metrics["jaccard_score"] = float(round(jaccard_score(y_true, y_pred, average="macro", zero_division=0), 4))

    # Per-label metrics
    per_label = {}
    for i, label in enumerate(label_names):
        y_t = y_true[:, i]
        y_p = y_pred[:, i]
        probs = y_probs[:, i]

        prec = float(round(precision_score(y_t, y_p, zero_division=0), 4))
        rec = float(round(recall_score(y_t, y_p, zero_division=0), 4))
        f1 = float(round(f1_score(y_t, y_p, zero_division=0), 4))

        try:
            auc = float(round(roc_auc_score(y_t, probs), 4))
        except Exception:
            auc = 0.50

        try:
            pr_auc = float(round(average_precision_score(y_t, probs), 4))
        except Exception:
            pr_auc = float(round(y_t.mean(), 4))

        cm = confusion_matrix(y_t, y_p, labels=[0, 1])
        tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

        per_label[label] = {
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "roc_auc": auc,
            "pr_auc": pr_auc,
            "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        }

    metrics["per_label"] = per_label
    return metrics


def evaluate_rule_based_baseline(
    fens: List[str],
    y_true: np.ndarray,
    label_names: List[str] = LABEL_NAMES
) -> Dict[str, Any]:
    """Evaluates rule-based python-chess detector baseline on test FENs."""
    from src.features.rules import RuleBasedTacticsDetector
    import chess

    detector = RuleBasedTacticsDetector()
    y_pred = np.zeros_like(y_true, dtype=int)
    y_probs = np.zeros_like(y_true, dtype=float)

    for idx, fen in enumerate(fens):
        try:
            board = chess.Board(fen)
            res = detector.detect_all(board)
            flags = res["flags"]
            for i, label in enumerate(label_names):
                val = flags.get(label, 0)
                y_pred[idx, i] = val
                y_probs[idx, i] = float(val)
        except Exception:
            pass

    return compute_multilabel_metrics(y_true, y_probs, y_pred, label_names)


def evaluate_dummy_baseline(
    y_train: np.ndarray,
    y_test: np.ndarray,
    label_names: List[str] = LABEL_NAMES
) -> Dict[str, Any]:
    """Majority-class dummy baseline."""
    # Predict most frequent class for each label
    majority = (y_train.mean(axis=0) >= 0.5).astype(int)
    y_pred = np.tile(majority, (len(y_test), 1))
    # Predict empirical prior probability as confidence
    priors = y_train.mean(axis=0)
    y_probs = np.tile(priors, (len(y_test), 1))

    return compute_multilabel_metrics(y_test, y_probs, y_pred, label_names)
