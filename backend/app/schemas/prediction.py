from pydantic import BaseModel
from typing import Dict, List, Any, Optional


class SingleModelPrediction(BaseModel):
    model_key: str
    model_name: str
    probabilities: Dict[str, float]
    thresholds: Dict[str, float]
    detected_motifs: List[str]
    latency_ms: float


class PredictResponse(SingleModelPrediction):
    fen: str


class CompareResponse(BaseModel):
    fen: str
    predictions: Dict[str, SingleModelPrediction]
    rule_baseline: Dict[str, Any]


class ModelInfo(BaseModel):
    key: str
    name: str
    metrics: Dict[str, Any]
    thresholds: Dict[str, float]
    training_time_sec: float
    inference_time_ms: float


class ModelsListResponse(BaseModel):
    models: List[ModelInfo]
    count: int


class FeatureAttribution(BaseModel):
    feature: str
    value: float
    importance: float
    direction: str


class ExplainResponse(BaseModel):
    fen: str
    model_used: str
    top_features: List[FeatureAttribution]
