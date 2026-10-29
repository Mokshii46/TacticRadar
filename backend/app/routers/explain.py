import chess
from fastapi import APIRouter, HTTPException, Query, status
from backend.app.schemas.prediction import ExplainResponse
from backend.app.services.model_service import ModelService

router = APIRouter(prefix="/api", tags=["Explainability"])


@router.get("/explain", response_model=ExplainResponse)
def explain_tactics(
    fen: str = Query(..., description="Chess position FEN"),
    model: str = Query("xgboost", description="Model identifier to explain")
):
    """Returns top contributing tactical chess features driving the model prediction."""
    try:
        board = chess.Board(fen)
        if not board.is_valid():
            raise ValueError("Position is invalid.")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid FEN: {str(e)}"
        )

    service = ModelService.get_instance()
    return service.explain(fen, model)
