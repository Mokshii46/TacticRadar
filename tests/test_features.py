import pytest
import chess
import numpy as np
from src.features.extractor import extract_tabular_features, get_feature_names
from src.features.tensor import board_to_tensor, fen_to_tensor
from src.features.rules import RuleBasedTacticsDetector


def test_starting_position_features():
    features = extract_tabular_features(chess.STARTING_FEN)
    assert isinstance(features, dict)
    assert features["turn"] == 1.0  # White to move
    assert features["is_check"] == 0.0
    assert features["material_balance"] == 0.0
    assert features["count_white_pawn"] == 8.0
    assert features["count_black_pawn"] == 8.0
    assert features["count_white_queen"] == 1.0
    assert features["legal_moves_count"] == 20.0
    assert not np.isnan(list(features.values())).any()


def test_knight_fork_detection():
    # Knight on d5 forks black King on e7 and black Queen on c7
    # White: Kd1, Nd5. Black: Ke7, Qc7
    fen = "8/2q1k3/8/3N4/8/8/8/3K4 w - - 0 1"
    detector = RuleBasedTacticsDetector()
    board = chess.Board(fen)
    forks = detector.detect_forks(board)
    assert len(forks) > 0
    assert forks[0]["attacker_square"] == "d5"
    assert "e7" in forks[0]["target_squares"] or "c7" in forks[0]["target_squares"]

    features = extract_tabular_features(fen)
    assert features["max_fork_targets_knight"] >= 2.0
    assert features["rule_fork_flag"] == 1.0


def test_absolute_pin_detection():
    # White Bishop on g5 pins Black Knight on f6 against Black King on d8
    # White: Kg1, Bg5. Black: Kd8, Nf6
    fen = "3k4/8/5n2/6B1/8/8/8/6K1 w - - 0 1"
    detector = RuleBasedTacticsDetector()
    board = chess.Board(fen)
    pins = detector.detect_pins(board)
    assert len(pins) > 0
    assert any(p["pinned_square"] == "f6" and p["is_absolute"] for p in pins)

    features = extract_tabular_features(fen)
    assert features["black_absolute_pins"] >= 1.0
    assert features["rule_pin_flag"] == 1.0


def test_skewer_detection():
    # White Rook on e1 skewers Black King on e7, exposing Black Queen on e8
    fen = "4q3/4k3/8/8/8/8/8/4R1K1 w - - 0 1"
    detector = RuleBasedTacticsDetector()
    board = chess.Board(fen)
    skewers = detector.detect_skewers(board)
    assert len(skewers) > 0
    assert skewers[0]["front_square"] == "e7"
    assert skewers[0]["back_square"] == "e8"

    features = extract_tabular_features(fen)
    assert features["rule_skewer_flag"] == 1.0


def test_discovered_attack_detection():
    # White Rook on d1, White Knight on d4 masking line to Black Queen on d8
    fen = "3q4/8/8/8/3N4/8/8/3R2K1 w - - 0 1"
    detector = RuleBasedTacticsDetector()
    board = chess.Board(fen)
    disc = detector.detect_discovered_attacks(board)
    assert len(disc) > 0
    assert disc[0]["masking_square"] == "d4"
    assert disc[0]["slider_square"] == "d1"
    assert disc[0]["target_square"] == "d8"

    features = extract_tabular_features(fen)
    assert features["rule_discovered_flag"] == 1.0


def test_board_to_tensor_shape_and_values():
    board = chess.Board()
    tensor = board_to_tensor(board)
    assert tensor.shape == (12, 8, 8)
    assert tensor.dtype == np.float32
    # Exactly 32 pieces on starting board
    assert tensor.sum() == 32.0

    # Test FEN helper
    fen_tensor = fen_to_tensor(chess.STARTING_FEN)
    np.testing.assert_array_equal(tensor, fen_tensor)
