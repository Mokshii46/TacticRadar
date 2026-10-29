import chess
from typing import Dict, Any, List
from src.features.rules import RuleBasedTacticsDetector


class ChessService:
    def __init__(self):
        self.detector = RuleBasedTacticsDetector()

    def get_motif_squares(self, fen: str) -> Dict[str, Any]:
        """Detects motifs and extracts involved squares and piece coordinates for board UI."""
        board = chess.Board(fen)
        res = self.detector.detect_all(board)
        return {
            "fen": fen,
            "motifs": res["motifs"],
            "counts": res["counts"],
            "flags": res["flags"],
        }
