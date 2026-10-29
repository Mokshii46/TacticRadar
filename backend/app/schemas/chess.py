import chess
from pydantic import BaseModel, field_validator
from typing import Dict, List, Any, Optional


class FENRequest(BaseModel):
    fen: str

    @field_validator("fen")
    @classmethod
    def validate_fen(cls, v: str) -> str:
        clean_fen = v.strip()
        try:
            board = chess.Board(clean_fen)
            if not board.is_valid():
                # python-chess is_valid checks kings, pawns on back rank, etc.
                raise ValueError("Invalid chess position: King missing or invalid pawn ranks.")
        except Exception as e:
            raise ValueError(f"Invalid FEN string: {e}")
        return clean_fen


class PredictRequest(FENRequest):
    model: Optional[str] = "ensemble"


class MotifItem(BaseModel):
    motif: str
    side: str
    attacker_square: Optional[str] = None
    attacker_piece: Optional[str] = None
    target_squares: Optional[List[str]] = None
    pinning_square: Optional[str] = None
    pinned_square: Optional[str] = None
    target_square: Optional[str] = None
    is_absolute: Optional[bool] = None
    skewer_square: Optional[str] = None
    front_square: Optional[str] = None
    back_square: Optional[str] = None
    slider_square: Optional[str] = None
    masking_square: Optional[str] = None
    is_discovered_check: Optional[bool] = None
    description: str


class MotifSquaresResponse(BaseModel):
    fen: str
    motifs: Dict[str, List[Dict[str, Any]]]
    counts: Dict[str, int]
    flags: Dict[str, int]
