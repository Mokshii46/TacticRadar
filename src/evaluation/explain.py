import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

LABEL_NAMES = ["fork", "pin", "skewer", "discoveredAttack"]


def run_shap_analysis(
    model: Any,
    X_sample: np.ndarray,
    feature_names: List[str],
    output_path: Path
):
    """Computes SHAP values on tabular sample and saves summary plot."""
    try:
        import shap
        logger.info("Computing SHAP values for tabular model...")
        
        # Extract underlying tree if OneVsRest or pipeline
        base = getattr(model, "estimator", model)
        if hasattr(base, "estimators_"):
            estimator = base.estimators_[0]
        else:
            estimator = base

        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(X_sample)

        # Handle list vs ndarray
        if isinstance(shap_values, list):
            sv = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        elif len(shap_values.shape) == 3:
            sv = shap_values[:, :, 0]
        else:
            sv = shap_values

        plt.figure(figsize=(10, 6))
        shap.summary_plot(sv, X_sample, feature_names=feature_names, show=False, max_display=15)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close()
        logger.info(f"Saved SHAP summary to {output_path}")

    except Exception as e:
        logger.warning(f"SHAP analysis skipped: {e}")


def explain_prediction_heuristic(
    features: Dict[str, float],
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Fast feature attribution for single-instance API explanation.
    Sorts prominent chess features that signal tactical motifs.
    """
    key_weights = {
        "rule_fork_flag": 3.0,
        "max_fork_targets_overall": 2.5,
        "max_fork_targets_knight": 2.2,
        "rule_pin_flag": 3.0,
        "total_absolute_pins": 2.5,
        "aligned_slider_targets": 2.0,
        "rule_skewer_flag": 3.0,
        "rule_discovered_flag": 3.0,
        "discovered_checks_count": 2.5,
        "white_hanging_pieces": 1.5,
        "black_hanging_pieces": 1.5,
        "white_attacked_by_lower": 1.8,
        "black_attacked_by_lower": 1.8,
        "is_check": 1.5,
    }

    scored_features = []
    for feat, val in features.items():
        if val != 0.0:
            weight = key_weights.get(feat, 1.0)
            importance = abs(val) * weight
            scored_features.append({
                "feature": feat,
                "value": round(val, 2),
                "importance": round(importance, 2),
                "direction": "positive" if val > 0 else "negative"
            })

    scored_features.sort(key=lambda x: x["importance"], reverse=True)
    return scored_features[:top_k]


def perform_error_analysis(
    test_df: pd.DataFrame,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path
) -> List[Dict[str, Any]]:
    """
    Identifies False Positives and False Negatives on the held-out test set
    for in-depth qualitative evaluation.
    """
    errors = []
    for i, row in test_df.iterrows():
        t_vec = y_test[i]
        p_vec = y_pred[i]
        
        # Check if mismatch
        if not np.array_equal(t_vec, p_vec):
            fp_motifs = [LABEL_NAMES[j] for j in range(4) if p_vec[j] == 1 and t_vec[j] == 0]
            fn_motifs = [LABEL_NAMES[j] for j in range(4) if p_vec[j] == 0 and t_vec[j] == 1]
            
            errors.append({
                "puzzle_id": row.get("PuzzleId", f"test_{i}"),
                "fen": row["FEN"],
                "true_labels": [LABEL_NAMES[j] for j in range(4) if t_vec[j] == 1],
                "pred_labels": [LABEL_NAMES[j] for j in range(4) if p_vec[j] == 1],
                "false_positives": fp_motifs,
                "false_negatives": fn_motifs,
                "rating": row.get("Rating", 0),
                "themes": row.get("Themes", ""),
            })

    # Pick representative samples
    logger.info(f"Identified {len(errors)} error instances on test set.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(errors[:50], f, indent=2)
    logger.info(f"Saved top error analysis samples to {output_path}")
    return errors[:50]
