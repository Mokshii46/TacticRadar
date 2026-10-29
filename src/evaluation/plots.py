import logging
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, List
from sklearn.metrics import roc_curve, precision_recall_curve

logger = logging.getLogger(__name__)

# Dark mode / modern aesthetic theme
plt.style.use("seaborn-v0_8-darkgrid" if "seaborn-v0_8-darkgrid" in plt.style.available else "default")
COLORS = ["#3b82f6", "#10b981", "#f59e0b", "#ec4899", "#8b5cf6", "#06b6d4", "#ef4444", "#6366f1", "#14b8a6"]


def plot_model_comparison(metrics_dict: Dict[str, Any], output_path: Path):
    """Bar chart comparing Macro F1, Micro F1, and Subset Accuracy across all models."""
    models = list(metrics_dict.keys())
    macro_f1s = [metrics_dict[m]["macro_f1"] for m in models]
    micro_f1s = [metrics_dict[m]["micro_f1"] for m in models]
    subset_accs = [metrics_dict[m]["subset_accuracy"] for m in models]

    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    r1 = ax.bar(x - width, macro_f1s, width, label="Macro F1", color="#3b82f6")
    r2 = ax.bar(x, micro_f1s, width, label="Micro F1", color="#10b981")
    r3 = ax.bar(x + width, subset_accs, width, label="Subset Accuracy", color="#8b5cf6")

    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("TacticRadar: Model Performance Comparison on Test Set", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=25, ha="right", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=True)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    logger.info(f"Saved {output_path}")


def plot_per_label_f1(metrics_dict: Dict[str, Any], label_names: List[str], output_path: Path):
    """Grouped bar chart of F1 score per tactical motif."""
    models = list(metrics_dict.keys())
    fig, ax = plt.subplots(figsize=(13, 6))
    
    x = np.arange(len(label_names))
    width = 0.8 / len(models)

    for idx, model in enumerate(models):
        f1_scores = [metrics_dict[model]["per_label"][lbl]["f1"] for lbl in label_names]
        offset = (idx - len(models) / 2) * width + width / 2
        ax.bar(x + offset, f1_scores, width, label=model, color=COLORS[idx % len(COLORS)])

    ax.set_ylabel("F1 Score", fontsize=12)
    ax.set_title("Per-Motif F1 Score by Model", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([lbl.upper() for lbl in label_names], fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    logger.info(f"Saved {output_path}")


def plot_roc_and_pr_curves(
    y_test: np.ndarray,
    probs_dict: Dict[str, np.ndarray],
    label_names: List[str],
    roc_path: Path,
    pr_path: Path
):
    """Plot ROC and PR curves for each label across models."""
    fig_roc, axes_roc = plt.subplots(2, 2, figsize=(12, 10))
    fig_pr, axes_pr = plt.subplots(2, 2, figsize=(12, 10))
    axes_roc = axes_roc.flatten()
    axes_pr = axes_pr.flatten()

    for i, label in enumerate(label_names):
        ax_r = axes_roc[i]
        ax_p = axes_pr[i]
        y_t = y_test[:, i]

        for m_idx, (model_name, probs) in enumerate(probs_dict.items()):
            color = COLORS[m_idx % len(COLORS)]
            # ROC
            fpr, tpr, _ = roc_curve(y_t, probs[:, i])
            ax_r.plot(fpr, tpr, label=model_name, color=color, linewidth=1.8)
            # PR
            prec, rec, _ = precision_recall_curve(y_t, probs[:, i])
            ax_p.plot(rec, prec, label=model_name, color=color, linewidth=1.8)

        # Baseline diagonal
        ax_r.plot([0, 1], [0, 1], "k--", alpha=0.4)
        ax_r.set_title(f"ROC: {label.upper()}", fontweight="bold")
        ax_r.set_xlabel("False Positive Rate")
        ax_r.set_ylabel("True Positive Rate")
        ax_r.set_ylim(-0.02, 1.02)
        if i == 0:
            ax_r.legend(fontsize=8, loc="lower right")

        prior = y_t.mean()
        ax_p.plot([0, 1], [prior, prior], "k--", alpha=0.4, label="Baseline")
        ax_p.set_title(f"Precision-Recall: {label.upper()}", fontweight="bold")
        ax_p.set_xlabel("Recall")
        ax_p.set_ylabel("Precision")
        ax_p.set_ylim(-0.02, 1.02)
        if i == 0:
            ax_p.legend(fontsize=8, loc="upper right")

    fig_roc.tight_layout()
    roc_path.parent.mkdir(parents=True, exist_ok=True)
    fig_roc.savefig(roc_path, dpi=200)
    plt.close(fig_roc)

    fig_pr.tight_layout()
    pr_path.parent.mkdir(parents=True, exist_ok=True)
    fig_pr.savefig(pr_path, dpi=200)
    plt.close(fig_pr)
    logger.info(f"Saved {roc_path} and {pr_path}")


def plot_confusion_matrices(
    per_label_metrics: Dict[str, Any],
    label_names: List[str],
    output_path: Path,
    model_name: str = "Best Model"
):
    """Plot 2x2 confusion matrix heatmaps for the 4 tactical motifs."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    for i, label in enumerate(label_names):
        cm_dict = per_label_metrics[label]["confusion_matrix"]
        cm_arr = np.array([
            [cm_dict["tn"], cm_dict["fp"]],
            [cm_dict["fn"], cm_dict["tp"]]
        ])
        ax = axes[i]
        sns.heatmap(
            cm_arr,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=ax,
            xticklabels=["Neg", "Pos"],
            yticklabels=["Neg", "Pos"],
        )
        f1 = per_label_metrics[label]["f1"]
        ax.set_title(f"{label.upper()}\n(F1 = {f1:.3f})", fontweight="bold")
        ax.set_ylabel("True Label")
        ax.set_xlabel("Predicted Label")

    fig.suptitle(f"Confusion Matrices: {model_name}", fontsize=14, fontweight="bold", y=1.05)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    logger.info(f"Saved {output_path}")


def plot_feature_importance(
    feature_names: List[str],
    importances: np.ndarray,
    output_path: Path,
    top_n: int = 20,
    model_name: str = "XGBoost"
):
    """Horizontal bar chart of top N important features."""
    indices = np.argsort(importances)[::-1][:top_n]
    top_feats = [feature_names[i] for i in indices][::-1]
    top_scores = importances[indices][::-1]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(top_feats, top_scores, color="#3b82f6")
    ax.set_xlabel("Importance (Gini / Gain)")
    ax.set_title(f"Top {top_n} Most Informative Features ({model_name})", fontsize=13, fontweight="bold")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    logger.info(f"Saved {output_path}")


def plot_label_cooccurrence(y: np.ndarray, label_names: List[str], output_path: Path):
    """Heatmap showing label co-occurrence frequencies."""
    matrix = np.zeros((len(label_names), len(label_names)), dtype=int)
    for i in range(len(label_names)):
        for j in range(len(label_names)):
            matrix[i, j] = int(((y[:, i] == 1) & (y[:, j] == 1)).sum())

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Purples",
        xticklabels=label_names,
        yticklabels=label_names,
        ax=ax,
    )
    ax.set_title("Tactical Motif Co-occurrence Matrix", fontsize=13, fontweight="bold")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    logger.info(f"Saved {output_path}")
