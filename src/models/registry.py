import os
import json
import joblib
import logging
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List

from src.models.classical import ClassicalMultiLabelModel
from src.models.neural import NeuralModelWrapper, ChessTabularMLP, ChessResNetCNN
from src.models.ensemble import MultiModelEnsemble
from src.features.extractor import extract_tabular_features
from src.features.tensor import fen_to_tensor
from src.models.thresholds import apply_thresholds

logger = logging.getLogger(__name__)

LABEL_NAMES = ["fork", "pin", "skewer", "discoveredAttack"]


class ModelRegistry:
    """Manages serialization, loading, and serving of all trained TacticRadar models."""

    def __init__(self, artifacts_dir: str = "artifacts/models"):
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.models: Dict[str, Any] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}
        self.feature_names: List[str] = []

    def save_model(
        self,
        model_key: str,
        model: Any,
        thresholds: Dict[str, float],
        metrics: Dict[str, Any],
        feature_names: List[str],
        config: Optional[Dict[str, Any]] = None
    ) -> Path:
        """Saves model weights/estimator and metadata JSON to disk."""
        meta = {
            "key": model_key,
            "name": getattr(model, "name", model_key),
            "created_at": datetime.utcnow().isoformat(),
            "thresholds": thresholds,
            "metrics": metrics,
            "training_time_sec": getattr(model, "training_time_sec", 0.0),
            "inference_time_ms": getattr(model, "inference_time_ms", 0.0),
            "feature_names": feature_names,
            "config": config or {},
        }

        meta_path = self.artifacts_dir / f"{model_key}_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        # Save model weights/estimator
        if isinstance(model, ClassicalMultiLabelModel):
            model_path = self.artifacts_dir / f"{model_key}.joblib"
            joblib.dump(model.estimator, model_path)
        elif isinstance(model, NeuralModelWrapper):
            model_path = self.artifacts_dir / f"{model_key}.pt"
            torch.save(model.model.state_dict(), model_path)
        elif isinstance(model, MultiModelEnsemble):
            model_path = self.artifacts_dir / f"{model_key}_weights.json"
            with open(model_path, "w") as f:
                json.dump(model.weights, f, indent=2)

        logger.info(f"Saved {model_key} and metadata to {self.artifacts_dir}")
        return meta_path

    def load_all(self, feature_names: List[str], device: str = "cpu") -> Dict[str, Any]:
        """Loads all available models and metadata from artifacts directory."""
        self.feature_names = feature_names
        meta_files = list(self.artifacts_dir.glob("*_metadata.json"))
        
        for meta_path in meta_files:
            try:
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                key = meta["key"]
                self.metadata[key] = meta

                if key == "mlp":
                    mlp_net = ChessTabularMLP(input_dim=len(feature_names))
                    pt_path = self.artifacts_dir / f"{key}.pt"
                    if pt_path.exists():
                        mlp_net.load_state_dict(torch.load(pt_path, map_location=device))
                    wrapper = NeuralModelWrapper("MLP (Tabular)", mlp_net, device=device)
                    wrapper.training_time_sec = meta.get("training_time_sec", 0.0)
                    wrapper.inference_time_ms = meta.get("inference_time_ms", 0.0)
                    self.models[key] = wrapper

                elif key == "cnn":
                    cnn_net = ChessResNetCNN(in_channels=12)
                    pt_path = self.artifacts_dir / f"{key}.pt"
                    if pt_path.exists():
                        cnn_net.load_state_dict(torch.load(pt_path, map_location=device))
                    wrapper = NeuralModelWrapper("ResNet CNN (8x8x12)", cnn_net, device=device)
                    wrapper.training_time_sec = meta.get("training_time_sec", 0.0)
                    wrapper.inference_time_ms = meta.get("inference_time_ms", 0.0)
                    self.models[key] = wrapper

                elif key == "ensemble":
                    # Ensemble loaded after individual models
                    pass

                else:
                    # Classical scikit-learn / XGB / LGBM
                    joblib_path = self.artifacts_dir / f"{key}.joblib"
                    if joblib_path.exists():
                        estimator = joblib.load(joblib_path)
                        wrapper = ClassicalMultiLabelModel(meta.get("name", key), estimator)
                        wrapper.training_time_sec = meta.get("training_time_sec", 0.0)
                        wrapper.inference_time_ms = meta.get("inference_time_ms", 0.0)
                        self.models[key] = wrapper

            except Exception as e:
                logger.error(f"Error loading model {meta_path}: {e}")

        # Load ensemble if weights exist
        ens_weights_path = self.artifacts_dir / "ensemble_weights.json"
        if ens_weights_path.exists() and "ensemble" in self.metadata:
            try:
                with open(ens_weights_path, "r") as f:
                    weights = json.load(f)
                ensemble = MultiModelEnsemble(self.models, weights=weights)
                self.models["ensemble"] = ensemble
            except Exception as e:
                logger.error(f"Error assembling ensemble: {e}")

        logger.info(f"Loaded {len(self.models)} models into registry: {list(self.models.keys())}")
        return self.models

    def predict_fen(self, model_key: str, fen: str) -> Dict[str, Any]:
        """Runs inference for a given FEN on specified model."""
        if model_key not in self.models:
            raise KeyError(f"Model '{model_key}' not found. Available: {list(self.models.keys())}")

        model = self.models[model_key]
        meta = self.metadata.get(model_key, {})
        thresholds = meta.get("thresholds", {l: 0.50 for l in LABEL_NAMES})

        # Extract features
        t0 = datetime.utcnow()
        if model_key == "cnn":
            tensor = fen_to_tensor(fen)
            X = np.expand_dims(tensor, axis=0)
            probs = model.predict_proba(X)[0]
        elif model_key == "ensemble":
            raw_feats = extract_tabular_features(fen)
            feature_names = meta.get("feature_names", self.feature_names)
            X_tab = np.array([[raw_feats.get(fn, 0.0) for fn in feature_names]])
            X_tensor = np.expand_dims(fen_to_tensor(fen), axis=0)
            probs = model.predict_proba(X_tab, X_tensor)[0]
        else:
            raw_feats = extract_tabular_features(fen)
            feature_names = meta.get("feature_names", self.feature_names)
            X = np.array([[raw_feats.get(fn, 0.0) for fn in feature_names]])
            probs = model.predict_proba(X)[0]

        latency_ms = (datetime.utcnow() - t0).total_seconds() * 1000.0

        # Apply thresholds
        probabilities = {label: float(round(probs[i], 4)) for i, label in enumerate(LABEL_NAMES)}
        detected_motifs = [
            label for i, label in enumerate(LABEL_NAMES) if probs[i] >= thresholds.get(label, 0.50)
        ]

        return {
            "model_key": model_key,
            "model_name": meta.get("name", model_key),
            "probabilities": probabilities,
            "thresholds": thresholds,
            "detected_motifs": detected_motifs,
            "latency_ms": round(latency_ms, 2),
        }
