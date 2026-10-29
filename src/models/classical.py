import time
import logging
import numpy as np
from typing import Dict, Any, Tuple, Optional
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import ClassifierChain
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

logger = logging.getLogger(__name__)


class ClassicalMultiLabelModel:
    """Wrapper around classical multi-label scikit-learn / XGBoost / LightGBM models."""

    def __init__(self, name: str, estimator: Any):
        self.name = name
        self.estimator = estimator
        self.training_time_sec = 0.0
        self.inference_time_ms = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ClassicalMultiLabelModel":
        t0 = time.time()
        self.estimator.fit(X, y)
        self.training_time_sec = time.time() - t0
        logger.info(f"Trained {self.name} in {self.training_time_sec:.2f}s")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        t0 = time.time()
        if hasattr(self.estimator, "predict_proba"):
            probs = self.estimator.predict_proba(X)
            # Handle native multi-output list of arrays from RandomForestClassifier
            if isinstance(probs, list):
                # list of (N, 2) arrays -> take prob of class 1 for each label
                probs = np.column_stack([p[:, 1] if p.shape[1] > 1 else np.zeros(len(X)) for p in probs])
        elif hasattr(self.estimator, "decision_function"):
            df = self.estimator.decision_function(X)
            # sigmoid of decision function
            probs = 1.0 / (1.0 + np.exp(-df))
        else:
            # fallback to binary predictions
            probs = self.estimator.predict(X).astype(float)

        elapsed_ms = (time.time() - t0) * 1000.0 / max(1, len(X))
        self.inference_time_ms = elapsed_ms
        return np.clip(probs, 0.0, 1.0)


def build_classical_models(config: Dict[str, Any]) -> Dict[str, ClassicalMultiLabelModel]:
    """Factory creating all required classical multi-label architectures."""
    cfg = config.get("models", {})
    seed = cfg.get("random_state", 42)

    models: Dict[str, ClassicalMultiLabelModel] = {}

    # 1. Logistic Regression (OneVsRest + StandardScaler)
    lr_cfg = cfg.get("logistic_regression", {})
    lr_base = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=lr_cfg.get("C", 1.0),
            max_iter=lr_cfg.get("max_iter", 1000),
            solver=lr_cfg.get("solver", "lbfgs"),
            random_state=seed
        ))
    ])
    models["logistic_regression"] = ClassicalMultiLabelModel(
        "Logistic Regression",
        OneVsRestClassifier(lr_base)
    )

    # 2. Linear SVM with Calibrated Probabilities (OneVsRest + StandardScaler)
    svm_cfg = cfg.get("svm", {})
    svm_base = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", CalibratedClassifierCV(
            estimator=LinearSVC(
                C=svm_cfg.get("C", 1.0),
                random_state=seed,
                dual=False,
                max_iter=2000
            ),
            cv=3
        ))
    ])
    models["svm"] = ClassicalMultiLabelModel(
        "Linear SVM (Calibrated)",
        OneVsRestClassifier(svm_base)
    )

    # 3. Decision Tree (native multi-output)
    dt_cfg = cfg.get("decision_tree", {})
    models["decision_tree"] = ClassicalMultiLabelModel(
        "Decision Tree",
        DecisionTreeClassifier(
            max_depth=dt_cfg.get("max_depth", 12),
            min_samples_leaf=dt_cfg.get("min_samples_leaf", 5),
            random_state=seed
        )
    )

    # 4. Random Forest (native multi-output)
    rf_cfg = cfg.get("random_forest", {})
    models["random_forest"] = ClassicalMultiLabelModel(
        "Random Forest",
        RandomForestClassifier(
            n_estimators=rf_cfg.get("n_estimators", 150),
            max_depth=rf_cfg.get("max_depth", 16),
            min_samples_split=rf_cfg.get("min_samples_split", 5),
            n_jobs=rf_cfg.get("n_jobs", -1),
            random_state=seed
        )
    )

    # 5. XGBoost (OneVsRest)
    xgb_cfg = cfg.get("xgboost", {})
    xgb_base = XGBClassifier(
        n_estimators=xgb_cfg.get("n_estimators", 120),
        max_depth=xgb_cfg.get("max_depth", 6),
        learning_rate=xgb_cfg.get("learning_rate", 0.1),
        subsample=xgb_cfg.get("subsample", 0.8),
        eval_metric="logloss",
        n_jobs=xgb_cfg.get("n_jobs", -1),
        random_state=seed
    )
    models["xgboost"] = ClassicalMultiLabelModel(
        "XGBoost",
        OneVsRestClassifier(xgb_base)
    )

    # 6. LightGBM (OneVsRest)
    lgb_cfg = cfg.get("lightgbm", {})
    lgb_base = LGBMClassifier(
        n_estimators=lgb_cfg.get("n_estimators", 120),
        max_depth=lgb_cfg.get("max_depth", 6),
        learning_rate=lgb_cfg.get("learning_rate", 0.1),
        n_jobs=lgb_cfg.get("n_jobs", -1),
        verbose=lgb_cfg.get("verbose", -1),
        random_state=seed
    )
    models["lightgbm"] = ClassicalMultiLabelModel(
        "LightGBM",
        OneVsRestClassifier(lgb_base)
    )

    # 7. k-NN (StandardScaler + KNeighborsClassifier)
    knn_cfg = cfg.get("knn", {})
    knn_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(
            n_neighbors=knn_cfg.get("n_neighbors", 7),
            weights=knn_cfg.get("weights", "distance"),
            n_jobs=knn_cfg.get("n_jobs", -1)
        ))
    ])
    models["knn"] = ClassicalMultiLabelModel(
        "k-Nearest Neighbors",
        knn_pipe
    )

    # 8. Classifier Chains (Random Forest base estimator to model label dependencies)
    cc_cfg = cfg.get("classifier_chains", {})
    rf_chain_base = RandomForestClassifier(
        n_estimators=cc_cfg.get("n_estimators", 100),
        max_depth=cc_cfg.get("max_depth", 12),
        n_jobs=cc_cfg.get("n_jobs", -1),
        random_state=seed
    )
    models["classifier_chains"] = ClassicalMultiLabelModel(
        "Classifier Chains (RF)",
        ClassifierChain(base_estimator=rf_chain_base, order="random", random_state=seed)
    )

    return models
