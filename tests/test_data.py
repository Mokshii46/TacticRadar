import pytest
import chess
import pandas as pd
import numpy as np
from src.data.preprocess import apply_fen_offset, parse_themes, extract_quiet_positions
from src.data.split import split_dataset


def test_fen_offset_fix():
    # Starting position FEN, first move 'e2e4'
    pre_fen = chess.STARTING_FEN
    real_fen = apply_fen_offset(pre_fen, "e2e4")
    assert real_fen != ""
    board = chess.Board(real_fen)
    # Turn should now be Black
    assert board.turn == chess.BLACK
    # e4 square should have a White Pawn
    assert board.piece_at(chess.E4) == chess.Piece(chess.PAWN, chess.WHITE)


def test_parse_themes():
    themes_str = "fork pin endgame advantage"
    themes_set = parse_themes(themes_str)
    assert "fork" in themes_set
    assert "pin" in themes_set
    assert "endgame" in themes_set
    assert "skewer" not in themes_set


def test_quiet_positions_generation():
    # Synthetic dataframe with non-tactical puzzles
    dummy_data = [
        {"PuzzleId": f"P{i}", "FEN": chess.STARTING_FEN, "Moves": "e2e4 e7e5", "Themes": "quietMove endgame", "Popularity": 90, "NbPlays": 100, "Rating": 1500}
        for i in range(10)
    ]
    df = pd.DataFrame(dummy_data)
    df["themes_set"] = df["Themes"].apply(parse_themes)
    
    target_motifs = ["fork", "pin", "skewer", "discoveredAttack"]
    quiet_df = extract_quiet_positions(df, num_quiet_needed=5, target_motifs=target_motifs, random_state=42)
    assert len(quiet_df) == 5
    assert (quiet_df["label_fork"] == 0).all()
    assert (quiet_df["label_pin"] == 0).all()
    assert (quiet_df["label_skewer"] == 0).all()
    assert (quiet_df["label_discoveredAttack"] == 0).all()
    assert (quiet_df["is_quiet"] == 1).all()


def test_stratified_split_no_leakage(tmp_path):
    # Create synthetic multi-label dataset
    np.random.seed(42)
    n = 100
    puzzles = [f"puz_{i:04d}" for i in range(n)]
    fens = [chess.STARTING_FEN] * n
    y = np.random.randint(0, 2, size=(n, 4))
    
    df = pd.DataFrame({
        "PuzzleId": puzzles,
        "FEN": fens,
        "label_fork": y[:, 0],
        "label_pin": y[:, 1],
        "label_skewer": y[:, 2],
        "label_discoveredAttack": y[:, 3],
    })
    
    config = {
        "data": {
            "processed_dir": str(tmp_path),
            "test_size": 0.15,
            "val_size": 0.15,
            "random_state": 42
        }
    }
    
    train_df, val_df, test_df = split_dataset(df, config)
    assert len(train_df) + len(val_df) + len(test_df) == n
    
    # Check no overlap
    train_ids = set(train_df["PuzzleId"])
    val_ids = set(val_df["PuzzleId"])
    test_ids = set(test_df["PuzzleId"])
    assert len(train_ids.intersection(val_ids)) == 0
    assert len(train_ids.intersection(test_ids)) == 0
    assert len(val_ids.intersection(test_ids)) == 0
