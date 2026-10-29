import chess
import numpy as np
from typing import List, Union

# 12 piece channels: White [P, N, B, R, Q, K], Black [p, n, b, r, q, k]
PIECE_TO_CHANNEL = {
    chess.Piece(chess.PAWN, chess.WHITE): 0,
    chess.Piece(chess.KNIGHT, chess.WHITE): 1,
    chess.Piece(chess.BISHOP, chess.WHITE): 2,
    chess.Piece(chess.ROOK, chess.WHITE): 3,
    chess.Piece(chess.QUEEN, chess.WHITE): 4,
    chess.Piece(chess.KING, chess.WHITE): 5,
    chess.Piece(chess.PAWN, chess.BLACK): 6,
    chess.Piece(chess.KNIGHT, chess.BLACK): 7,
    chess.Piece(chess.BISHOP, chess.BLACK): 8,
    chess.Piece(chess.ROOK, chess.BLACK): 9,
    chess.Piece(chess.QUEEN, chess.BLACK): 10,
    chess.Piece(chess.KING, chess.BLACK): 11,
}


def board_to_tensor(board: chess.Board) -> np.ndarray:
    """
    Converts a python-chess Board into an 8x8x12 tensor representation
    following Schwarzschild et al., NeurIPS 2021.
    Output shape: (12, 8, 8) with dtype float32.
    """
    tensor = np.zeros((12, 8, 8), dtype=np.float32)
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is not None and piece in PIECE_TO_CHANNEL:
            ch = PIECE_TO_CHANNEL[piece]
            rank = chess.square_rank(sq)
            file = chess.square_file(sq)
            # rank 7 is top (row 0), rank 0 is bottom (row 7)
            row = 7 - rank
            col = file
            tensor[ch, row, col] = 1.0

    return tensor


def fen_to_tensor(fen: str) -> np.ndarray:
    """Convenience function to convert FEN string directly to (12, 8, 8) tensor."""
    try:
        board = chess.Board(fen)
    except Exception:
        board = chess.Board()
    return board_to_tensor(board)


def fens_to_tensor_batch(fens: List[str]) -> np.ndarray:
    """Batch convert list of FEN strings to (N, 12, 8, 8) NumPy tensor."""
    tensors = [fen_to_tensor(f) for f in fens]
    return np.stack(tensors, axis=0)
