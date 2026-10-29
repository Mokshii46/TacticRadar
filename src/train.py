import os
import sys
import json
import yaml
import time
import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd

from src.data.loader import load_raw_dataset
from src.data.preprocess import preprocess_dataset
from src.data.split import split_dataset, LABEL_COLS
from src.features.extractor import extract_features_batch, get_feature_names
from src.features.tensor import fens_to_tensor_batch
from src.models.classical import build_classical_models
from src.models.neural import ChessTabularMLP, ChessResNetCNN, NeuralModelWrapper
from src.models.ensemble import MultiModelEnsemble
from src.models.thresholds import tune_decision_thresholds, apply_thresholds
from src.models.registry import ModelRegistry, LABEL_NAMES
from src.evaluation.metrics import (
    compute_multilabel_metrics,
    evaluate_rule_based_baseline,
    evaluate_dummy_baseline,
)
from src.evaluation.plots import (
    plot_model_comparison,
    plot_per_label_f1,
    plot_roc_and_pr_curves,
    plot_confusion_matrices,
    plot_feature_importance,
    plot_label_cooccurrence,
)
from src.evaluation.explain import run_shap_analysis, perform_error_analysis

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("src.train")


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate TacticRadar models.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config yaml")
    parser.add_argument("--quick", action="store_true", help="Quick mode for testing pipeline")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.quick:
        config["data"]["target_samples_per_label"] = 500
        config["data"]["max_total_samples"] = 2500
        config["models"]["mlp"]["epochs"] = 5
        config["models"]["cnn"]["epochs"] = 5

    logger.info("==================================================")
    logger.info("   TacticRadar: Tactical Motif Training Suite     ")
    logger.info("==================================================")

    # 1. Data Ingestion & Preprocessing
    raw_df = load_raw_dataset(config)
    processed_df = preprocess_dataset(raw_df, config)
    train_df, val_df, test_df = split_dataset(processed_df, config)

    # 2. Feature Extraction
    proc_dir = Path(config["data"]["processed_dir"])
    X_train_path = proc_dir / "X_train_tabular.parquet"
    X_val_path = proc_dir / "X_val_tabular.parquet"
    X_test_path = proc_dir / "X_test_tabular.parquet"

    n_workers = config["features"].get("num_workers", 4)
    if not (X_train_path.exists() and X_val_path.exists() and X_test_path.exists()):
        logger.info(f"Extracting tabular features for Train ({len(train_df)}), Val ({len(val_df)}), Test ({len(test_df)})...")
        X_train_df = extract_features_batch(train_df["FEN"].tolist(), num_workers=n_workers)
        X_val_df = extract_features_batch(val_df["FEN"].tolist(), num_workers=n_workers)
        X_test_df = extract_features_batch(test_df["FEN"].tolist(), num_workers=n_workers)

        X_train_df.to_parquet(X_train_path)
        X_val_df.to_parquet(X_val_path)
        X_test_df.to_parquet(X_test_path)
    else:
        logger.info("Loading cached tabular features from disk...")
        X_train_df = pd.read_parquet(X_train_path)
        X_val_df = pd.read_parquet(X_val_path)
        X_test_df = pd.read_parquet(X_test_path)

    feature_names = list(X_train_df.columns)
    logger.info(f"Total tabular features: {len(feature_names)}")

    X_train = X_train_df.values
    y_train = train_df[LABEL_COLS].values
    X_val = X_val_df.values
    y_val = val_df[LABEL_COLS].values
    X_test = X_test_df.values
    y_test = test_df[LABEL_COLS].values

    # Tensor representations for CNN
    T_train_path = proc_dir / "T_train_tensor.npy"
    T_val_path = proc_dir / "T_val_tensor.npy"
    T_test_path = proc_dir / "T_test_tensor.npy"

    if not (T_train_path.exists() and T_val_path.exists() and T_test_path.exists()):
        logger.info("Building 8x8x12 tensor representations for CNN...")
        T_train = fens_to_tensor_batch(train_df["FEN"].tolist())
        T_val = fens_to_tensor_batch(val_df["FEN"].tolist())
        T_test = fens_to_tensor_batch(test_df["FEN"].tolist())
        np.save(T_train_path, T_train)
        np.save(T_val_path, T_val)
        np.save(T_test_path, T_test)
    else:
        logger.info("Loading cached 8x8x12 tensor representations...")
        T_train = np.load(T_train_path)
        T_val = np.load(T_val_path)
        T_test = np.load(T_test_path)

    # 3. Model Training & Validation Threshold Tuning
    artifacts_dir = Path(config["artifacts"]["dir"])
    plots_dir = Path(config["artifacts"]["plots_dir"])
    models_dir = Path(config["artifacts"]["models_dir"])
    registry = ModelRegistry(str(models_dir))

    all_test_metrics = {}
    test_probs_dict = {}
    best_tabular_model = None
    best_macro_f1 = -1.0

    # A. Classical Models
    classical_models = build_classical_models(config)
    for key, model in classical_models.items():
        logger.info(f"\n--- Training {model.name} ({key}) ---")
        model.fit(X_train, y_train)

        # Tune thresholds on validation set
        val_probs = model.predict_proba(X_val)
        thresholds = tune_decision_thresholds(y_val, val_probs, LABEL_NAMES)
        logger.info(f"Tuned validation thresholds for {key}: {thresholds}")

        # Evaluate on Test set
        test_probs = model.predict_proba(X_test)
        test_preds = apply_thresholds(test_probs, thresholds, LABEL_NAMES)
        metrics = compute_multilabel_metrics(y_test, test_probs, test_preds, LABEL_NAMES)
        metrics["training_time_sec"] = round(model.training_time_sec, 2)
        metrics["inference_time_ms"] = round(model.inference_time_ms, 2)
        all_test_metrics[key] = metrics
        test_probs_dict[key] = test_probs

        logger.info(f"Test Macro F1: {metrics['macro_f1']:.4f} | Micro F1: {metrics['micro_f1']:.4f} | Subset Acc: {metrics['subset_accuracy']:.4f}")

        # Track best tabular model for SHAP
        if metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = metrics["macro_f1"]
            best_tabular_model = model

        # Save to registry
        registry.save_model(key, model, thresholds, metrics, feature_names, config["models"].get(key))

    # B. PyTorch MLP
    logger.info("\n--- Training PyTorch MLP (Tabular) ---")
    mlp_cfg = config["models"]["mlp"]
    mlp_net = ChessTabularMLP(
        input_dim=len(feature_names),
        hidden_dims=mlp_cfg.get("hidden_dims", [256, 128, 64]),
        dropout=mlp_cfg.get("dropout", 0.25)
    )
    mlp_wrapper = NeuralModelWrapper("PyTorch Tabular MLP", mlp_net)
    mlp_wrapper.fit(
        X_train, y_train, X_val, y_val,
        epochs=mlp_cfg.get("epochs", 25),
        batch_size=mlp_cfg.get("batch_size", 128),
        lr=mlp_cfg.get("lr", 0.001),
        weight_decay=mlp_cfg.get("weight_decay", 1e-4),
        patience=mlp_cfg.get("early_stopping_patience", 5)
    )

    mlp_val_probs = mlp_wrapper.predict_proba(X_val)
    mlp_thresh = tune_decision_thresholds(y_val, mlp_val_probs, LABEL_NAMES)
    mlp_test_probs = mlp_wrapper.predict_proba(X_test)
    mlp_test_preds = apply_thresholds(mlp_test_probs, mlp_thresh, LABEL_NAMES)
    mlp_metrics = compute_multilabel_metrics(y_test, mlp_test_probs, mlp_test_preds, LABEL_NAMES)
    mlp_metrics["training_time_sec"] = round(mlp_wrapper.training_time_sec, 2)
    mlp_metrics["inference_time_ms"] = round(mlp_wrapper.inference_time_ms, 2)
    all_test_metrics["mlp"] = mlp_metrics
    test_probs_dict["mlp"] = mlp_test_probs
    registry.save_model("mlp", mlp_wrapper, mlp_thresh, mlp_metrics, feature_names, mlp_cfg)

    # C. PyTorch CNN (8x8x12 ResNet)
    logger.info("\n--- Training PyTorch ResNet CNN (8x8x12 Tensor) ---")
    cnn_cfg = config["models"]["cnn"]
    cnn_net = ChessResNetCNN(in_channels=12, dropout=cnn_cfg.get("dropout", 0.3))
    cnn_wrapper = NeuralModelWrapper("PyTorch ResNet CNN", cnn_net)
    cnn_wrapper.fit(
        T_train, y_train, T_val, y_val,
        epochs=cnn_cfg.get("epochs", 25),
        batch_size=cnn_cfg.get("batch_size", 128),
        lr=cnn_cfg.get("lr", 0.0008),
        weight_decay=cnn_cfg.get("weight_decay", 1e-4),
        patience=cnn_cfg.get("early_stopping_patience", 5)
    )

    cnn_val_probs = cnn_wrapper.predict_proba(T_val)
    cnn_thresh = tune_decision_thresholds(y_val, cnn_val_probs, LABEL_NAMES)
    cnn_test_probs = cnn_wrapper.predict_proba(T_test)
    cnn_test_preds = apply_thresholds(cnn_test_probs, cnn_thresh, LABEL_NAMES)
    cnn_metrics = compute_multilabel_metrics(y_test, cnn_test_probs, cnn_test_preds, LABEL_NAMES)
    cnn_metrics["training_time_sec"] = round(cnn_wrapper.training_time_sec, 2)
    cnn_metrics["inference_time_ms"] = round(cnn_wrapper.inference_time_ms, 2)
    all_test_metrics["cnn"] = cnn_metrics
    test_probs_dict["cnn"] = cnn_test_probs
    registry.save_model("cnn", cnn_wrapper, cnn_thresh, cnn_metrics, ["8x8x12_tensor"], cnn_cfg)

    # D. Soft-Voting Ensemble
    logger.info("\n--- Evaluating Soft-Voting Ensemble ---")
    ens_models = {
        "xgboost": classical_models["xgboost"],
        "lightgbm": classical_models["lightgbm"],
        "random_forest": classical_models["random_forest"],
        "mlp": mlp_wrapper,
        "cnn": cnn_wrapper,
    }
    ens_weights = config["models"].get("ensemble", {}).get("weights", {
        "xgboost": 0.25, "lightgbm": 0.25, "random_forest": 0.20, "mlp": 0.15, "cnn": 0.15
    })
    ensemble = MultiModelEnsemble(ens_models, weights=ens_weights)
    ens_val_probs = ensemble.predict_proba(X_val, T_val)
    ens_thresh = tune_decision_thresholds(y_val, ens_val_probs, LABEL_NAMES)
    ens_test_probs = ensemble.predict_proba(X_test, T_test)
    ens_test_preds = apply_thresholds(ens_test_probs, ens_thresh, LABEL_NAMES)
    ens_metrics = compute_multilabel_metrics(y_test, ens_test_probs, ens_test_preds, LABEL_NAMES)
    ens_metrics["training_time_sec"] = 0.0
    ens_metrics["inference_time_ms"] = round(ensemble.inference_time_ms, 2)
    all_test_metrics["ensemble"] = ens_metrics
    test_probs_dict["ensemble"] = ens_test_probs
    registry.save_model("ensemble", ensemble, ens_thresh, ens_metrics, feature_names, ens_weights)

    # E. Baselines: Dummy Majority & Rule-Based Detector
    logger.info("\n--- Evaluating Baselines ---")
    dummy_metrics = evaluate_dummy_baseline(y_train, y_test, LABEL_NAMES)
    dummy_metrics["training_time_sec"] = 0.0
    dummy_metrics["inference_time_ms"] = 0.01
    all_test_metrics["baseline_dummy"] = dummy_metrics

    rule_metrics = evaluate_rule_based_baseline(test_df["FEN"].tolist(), y_test, LABEL_NAMES)
    rule_metrics["training_time_sec"] = 0.0
    rule_metrics["inference_time_ms"] = 0.50
    all_test_metrics["baseline_rules"] = rule_metrics

    logger.info(f"Rule Baseline Macro F1: {rule_metrics['macro_f1']:.4f}")
    logger.info(f"Dummy Baseline Macro F1: {dummy_metrics['macro_f1']:.4f}")

    # 4. Generate Visualizations & Diagnostic Artifacts
    logger.info("\n--- Generating Artifacts & Plots ---")
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Save full metrics.json
    metrics_file = Path(config["artifacts"]["metrics_file"])
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_file, "w") as f:
        json.dump(all_test_metrics, f, indent=2)
    logger.info(f"Wrote all test metrics to {metrics_file}")

    # Plot Model Comparison
    plot_model_comparison(all_test_metrics, plots_dir / "model_comparison_f1.png")
    plot_per_label_f1(all_test_metrics, LABEL_NAMES, plots_dir / "per_label_f1_comparison.png")
    plot_roc_and_pr_curves(
        y_test, test_probs_dict, LABEL_NAMES,
        plots_dir / "roc_curves.png", plots_dir / "pr_curves.png"
    )
    plot_label_cooccurrence(y_test, LABEL_NAMES, plots_dir / "label_cooccurrence_heatmap.png")

    # Confusion matrix for top model
    top_model_key = max(all_test_metrics.keys(), key=lambda k: all_test_metrics[k]["macro_f1"])
    plot_confusion_matrices(
        all_test_metrics[top_model_key]["per_label"],
        LABEL_NAMES,
        plots_dir / f"confusion_matrices_{top_model_key}.png",
        model_name=top_model_key
    )

    # Feature importances from XGBoost
    if "xgboost" in classical_models:
        try:
            xgb_est = classical_models["xgboost"].estimator.estimators_[0]
            imps = xgb_est.feature_importances_
            plot_feature_importance(feature_names, imps, plots_dir / "feature_importance.png", top_n=20)
        except Exception as e:
            logger.warning(f"Could not plot feature importance: {e}")

    # SHAP analysis on sample
    if best_tabular_model is not None:
        X_sample = X_test[:min(100, len(X_test))]
        run_shap_analysis(best_tabular_model, X_sample, feature_names, plots_dir / "shap_summary.png")

    # Error analysis
    perform_error_analysis(test_df, y_test, ens_test_preds, artifacts_dir / "error_analysis.json")

    logger.info("\n==================================================")
    logger.info("   Training & Evaluation Completed Successfully!  ")
    logger.info("==================================================")


if __name__ == "__main__":
    main()
