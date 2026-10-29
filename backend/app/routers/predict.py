import chess
from fastapi import APIRouter, HTTPException, Query, status
from backend.app.schemas.chess import PredictRequest, MotifSquaresResponse
from backend.app.schemas.prediction import PredictResponse, CompareResponse
from backend.app.services.model_service import ModelService
from backend.app.services.chess_service import ChessService

router = APIRouter(prefix="/api", tags=["Prediction"])


@router.post("/predict", response_model=PredictResponse)
def predict_tactics(req: PredictRequest):
    """Predicts multi-label tactical motifs (fork, pin, skewer, discovered attack) for a chess FEN."""
    service = ModelService.get_instance()
    try:
        res = service.predict(req.fen, req.model)
        return res
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}"
        )


@router.post("/predict/compare", response_model=CompareResponse)
def compare_all_models(req: PredictRequest):
    """Returns side-by-side predictions from all trained classical, neural, and ensemble models."""
    service = ModelService.get_instance()
    try:
        res = service.predict_compare(req.fen)
        return res
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Comparison error: {str(e)}"
        )


@router.get("/motif-squares", response_model=MotifSquaresResponse)
def get_motif_squares(fen: str = Query(..., description="Chess FEN string")):
    """Returns board squares and piece coordinates involved in each detected motif for board UI highlighting."""
    try:
        board = chess.Board(fen)
        if not board.is_valid():
            raise ValueError("Position is invalid.")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid FEN: {str(e)}"
        )

    service = ChessService()
    return service.get_motif_squares(fen)
