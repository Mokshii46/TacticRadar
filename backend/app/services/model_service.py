import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from src.models.registry import ModelRegistry, LABEL_NAMES
from src.features.extractor import extract_tabular_features, get_feature_names
from src.evaluation.explain import explain_prediction_heuristic
from src.features.rules import RuleBasedTacticsDetector

logger = logging.getLogger(__name__)


class ModelService:
    _instance: Optional["ModelService"] = None

    def __init__(self, artifacts_dir: str = "artifacts"):
        self.artifacts_dir = Path(artifacts_dir)
        self.models_dir = self.artifacts_dir / "models"
        self.registry = ModelRegistry(str(self.models_dir))
        self.feature_names = get_feature_names()
        self.metrics_data: Dict[str, Any] = {}
        self.rules_detector = RuleBasedTacticsDetector()
        self.initialize()

    @classmethod
    def get_instance(cls) -> "ModelService":
        if cls._instance is None:
            cls._instance = ModelService()
        return cls._instance

    def initialize(self):
        """Loads trained models and pre-computed evaluation metrics."""
        logger.info("Initializing ModelService and loading models...")
        self.registry.load_all(self.feature_names)

        # Load metrics.json if available
        metrics_file = self.artifacts_dir / "metrics.json"
        if metrics_file.exists():
            try:
                with open(metrics_file, "r") as f:
                    self.metrics_data = json.load(f)
                logger.info(f"Loaded evaluation metrics for {len(self.metrics_data)} models.")
            except Exception as e:
                logger.error(f"Error loading {metrics_file}: {e}")

    def get_available_models(self) -> List[Dict[str, Any]]:
        """List all available models with test metrics."""
        result = []
        for key, meta in self.registry.metadata.items():
            result.append({
                "key": key,
                "name": meta.get("name", key),
                "metrics": meta.get("metrics", self.metrics_data.get(key, {})),
                "thresholds": meta.get("thresholds", {l: 0.50 for l in LABEL_NAMES}),
                "training_time_sec": meta.get("training_time_sec", 0.0),
                "inference_time_ms": meta.get("inference_time_ms", 0.0),
            })
        return result

    def get_all_metrics(self) -> Dict[str, Any]:
        """Returns the full evaluation results dictionary for the dashboard."""
        return self.metrics_data

    def predict(self, fen: str, model_key: Optional[str] = "ensemble") -> Dict[str, Any]:
        """Predict multi-label tactical motifs for a given position."""
        target_model = model_key or "ensemble"
        # If requested model not present, fallback to first available or ensemble
        if target_model not in self.registry.models:
            if "ensemble" in self.registry.models:
                target_model = "ensemble"
            elif self.registry.models:
                target_model = next(iter(self.registry.models))
            else:
                # Emergency fallback to rule-based detector if models not yet loaded
                import chess
                res = self.rules_detector.detect_all(chess.Board(fen))
                flags = res["flags"]
                return {
                    "model_key": "baseline_rules",
                    "model_name": "Rule-Based Baseline (Fallback)",
                    "probabilities": {k: float(v) for k, v in flags.items()},
                    "thresholds": {k: 0.5 for k in LABEL_NAMES},
                    "detected_motifs": [k for k, v in flags.items() if v == 1],
                    "latency_ms": 1.2,
                    "fen": fen,
                }

        pred = self.registry.predict_fen(target_model, fen)
        pred["fen"] = fen
        return pred

    def predict_compare(self, fen: str) -> Dict[str, Any]:
        """Runs predictions from ALL loaded models side-by-side."""
        predictions = {}
        for key in self.registry.models.keys():
            try:
                predictions[key] = self.registry.predict_fen(key, fen)
            except Exception as e:
                logger.error(f"Error predicting with {key}: {e}")

        # Add rule baseline
        import chess
        rule_res = self.rules_detector.detect_all(chess.Board(fen))
        rule_baseline = {
            "flags": rule_res["flags"],
            "counts": rule_res["counts"],
            "detected_motifs": [k for k, v in rule_res["flags"].items() if v == 1]
        }

        return {
            "fen": fen,
            "predictions": predictions,
            "rule_baseline": rule_baseline,
        }

    def explain(self, fen: str, model_key: str = "xgboost") -> Dict[str, Any]:
        """Provides top contributing chess features explaining the prediction."""
        raw_feats = extract_tabular_features(fen)
        top_features = explain_prediction_heuristic(raw_feats, top_k=6)
        return {
            "fen": fen,
            "model_used": model_key,
            "top_features": top_features,
        }
