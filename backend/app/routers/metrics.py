from fastapi import APIRouter
from typing import Dict, Any, List
from backend.app.schemas.prediction import ModelsListResponse
from backend.app.services.model_service import ModelService

router = APIRouter(prefix="/api", tags=["Models & Metrics"])


@router.get("/models", response_model=ModelsListResponse)
def get_available_models():
    """Returns available trained models with their per-label thresholds and test evaluation metrics."""
    service = ModelService.get_instance()
    models = service.get_available_models()
    return {"models": models, "count": len(models)}


@router.get("/metrics")
def get_evaluation_metrics() -> Dict[str, Any]:
    """Returns complete evaluation results dictionary from artifacts/metrics.json for dashboard."""
    service = ModelService.get_instance()
    return service.get_all_metrics()
