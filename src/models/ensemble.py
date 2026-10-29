import time
import logging
import numpy as np
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class MultiModelEnsemble:
    """
    Weighted soft-voting ensemble combining top classical and neural models.
    Blends predicted probability distributions for robust multi-label detection.
    """

    def __init__(self, models: Dict[str, Any], weights: Optional[Dict[str, float]] = None):
        self.name = "Soft Voting Ensemble"
        self.models = models
        self.weights = weights or {}
        self.training_time_sec = 0.0
        self.inference_time_ms = 0.0

        # Normalize weights
        active_models = [m for m in self.models.keys() if m in self.weights]
        if not active_models:
            # Uniform weights
            self.weights = {m: 1.0 / len(self.models) for m in self.models.keys()}
        else:
            tot = sum(self.weights[m] for m in active_models)
            self.weights = {m: self.weights[m] / tot for m in active_models}

    def predict_proba(
        self,
        X_tabular: np.ndarray,
        X_tensor: Optional[np.ndarray] = None
    ) -> np.ndarray:
        t0 = time.time()
        combined_probs = None

        for name, model in self.models.items():
            w = self.weights.get(name, 0.0)
            if w <= 0.0:
                continue

            # Pass appropriate input type
            if name == "cnn" and X_tensor is not None:
                probs = model.predict_proba(X_tensor)
            else:
                probs = model.predict_proba(X_tabular)

            if combined_probs is None:
                combined_probs = w * probs
            else:
                combined_probs += w * probs

        if combined_probs is None:
            raise ValueError("No models contributed to ensemble prediction.")

        self.inference_time_ms = (time.time() - t0) * 1000.0 / max(1, len(X_tabular))
        return np.clip(combined_probs, 0.0, 1.0)
