import pytest
import numpy as np
import torch
from src.models.classical import build_classical_models
from src.models.neural import ChessTabularMLP, ChessResNetCNN, NeuralModelWrapper
from src.models.thresholds import tune_decision_thresholds, apply_thresholds
from src.models.ensemble import MultiModelEnsemble


def test_classical_models_fit_and_predict():
    np.random.seed(42)
    N, D, K = 60, 20, 4
    X = np.random.randn(N, D)
    y = np.random.randint(0, 2, size=(N, K))

    config = {
        "models": {
            "random_state": 42,
            "logistic_regression": {"max_iter": 100},
            "decision_tree": {"max_depth": 3},
            "random_forest": {"n_estimators": 5, "max_depth": 3, "n_jobs": 1},
        }
    }
    models = build_classical_models(config)
    
    # Test Logistic Regression, Decision Tree, Random Forest
    for key in ["logistic_regression", "decision_tree", "random_forest"]:
        m = models[key]
        m.fit(X, y)
        probs = m.predict_proba(X)
        assert probs.shape == (N, K)
        assert (probs >= 0.0).all() and (probs <= 1.0).all()


def test_mlp_and_cnn_forward_shapes():
    # Tabular MLP
    batch_size = 8
    feat_dim = 30
    mlp = ChessTabularMLP(input_dim=feat_dim, hidden_dims=[32, 16], num_classes=4)
    x_tab = torch.randn(batch_size, feat_dim)
    logits_tab = mlp(x_tab)
    assert logits_tab.shape == (batch_size, 4)

    # ResNet CNN
    cnn = ChessResNetCNN(in_channels=12, num_classes=4)
    x_tensor = torch.randn(batch_size, 12, 8, 8)
    logits_cnn = cnn(x_tensor)
    assert logits_cnn.shape == (batch_size, 4)


def test_threshold_tuning_and_application():
    np.random.seed(42)
    N = 100
    y_true = np.random.randint(0, 2, size=(N, 4))
    y_probs = np.random.uniform(0.0, 1.0, size=(N, 4))
    label_names = ["fork", "pin", "skewer", "discoveredAttack"]

    thresholds = tune_decision_thresholds(y_true, y_probs, label_names)
    assert len(thresholds) == 4
    for l in label_names:
        assert 0.10 <= thresholds[l] <= 0.90

    y_pred = apply_thresholds(y_probs, thresholds, label_names)
    assert y_pred.shape == (N, 4)
    assert set(np.unique(y_pred)).issubset({0, 1})
