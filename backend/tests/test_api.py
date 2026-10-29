import pytest
from fastapi.testclient import TestClient
import chess
from backend.app.main import app

client = TestClient(app)

VALID_FEN = chess.STARTING_FEN
TACTICAL_FEN = "8/2q1k3/8/3N4/8/8/8/3K4 b - - 0 1"  # Knight fork (Black to move while checked)
INVALID_FEN = "invalid_fen_string_here"


def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "TacticRadar"


def test_predict_endpoint_valid():
    payload = {"fen": TACTICAL_FEN}
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "probabilities" in data
    assert "detected_motifs" in data
    assert "latency_ms" in data
    for motif in ["fork", "pin", "skewer", "discoveredAttack"]:
        assert motif in data["probabilities"]
        assert 0.0 <= data["probabilities"][motif] <= 1.0


def test_predict_endpoint_invalid_fen():
    payload = {"fen": INVALID_FEN}
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 422


def test_compare_endpoint_valid():
    payload = {"fen": TACTICAL_FEN}
    response = client.post("/api/predict/compare", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert "rule_baseline" in data


def test_models_endpoint():
    response = client.get("/api/models")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert "count" in data


def test_metrics_endpoint():
    response = client.get("/api/metrics")
    assert response.status_code == 200


def test_motif_squares_endpoint_valid():
    response = client.get(f"/api/motif-squares?fen={TACTICAL_FEN}")
    assert response.status_code == 200
    data = response.json()
    assert "motifs" in data
    assert "counts" in data
    assert "flags" in data
    assert data["flags"]["fork"] == 1


def test_motif_squares_endpoint_invalid():
    response = client.get(f"/api/motif-squares?fen={INVALID_FEN}")
    assert response.status_code == 422


def test_explain_endpoint_valid():
    response = client.get(f"/api/explain?fen={TACTICAL_FEN}&model=xgboost")
    assert response.status_code == 200
    data = response.json()
    assert "top_features" in data
    assert len(data["top_features"]) > 0


def test_explain_endpoint_invalid():
    response = client.get(f"/api/explain?fen={INVALID_FEN}")
    assert response.status_code == 422
