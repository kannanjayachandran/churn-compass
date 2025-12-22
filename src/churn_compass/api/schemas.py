"""
Churn Compass - API Schemas

Pydantic models for request/response validation.
"""

from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field, ConfigDict


# Input Schemas
class CustomerInput(BaseModel):
    """Customer features for prediction"""

    CreditScore: int = Field(ge=300, le=850)
    Geography: Literal["France", "Spain", "Germany"]
    Gender: Literal["Male", "Female"]
    Age: int = Field(ge=10, le=100)
    Tenure: int = Field(ge=0, le=10)
    Balance: float = Field(ge=0)
    NumOfProducts: int = Field(ge=1, le=4)
    HasCrCard: int = Field(ge=0, le=1)
    IsActiveMember: int = Field(ge=0, le=1)
    EstimatedSalary: float = Field(ge=0, le=200_000)
    CardType: Literal["SILVER", "GOLD", "DIAMOND", "PLATINUM"]

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "CreditScore": 650, 
                "Geography": "Germany", 
                "Gender": "Female", 
                "Age": 46, 
                "Tenure": 4, 
                "Balance": 100_000.0, 
                "NumOfProducts": 3, 
                "HasCrCard": 1, 
                "IsActiveMember": 0, 
                "EstimatedSalary": 80000.0, 
                "CardType": "DIAMOND",
            }
        }
    )


# Prediction Schemas
class PredictionResponse(BaseModel):
    """Single prediction response."""
    
    prediction: int
    probability: float
    risk_level: Literal["low", "medium", "high"]


class BatchPredictionRequest(BaseModel):
    """Batch prediction request."""

    customers: List[CustomerInput] = Field(min_length=1, max_length=1000)
    include_features: bool = False


class BatchPredictionResponse(BaseModel):
    """Batch prediction response."""

    predictions: List[PredictionResponse]
    summary: Dict[str, Any]


# SHAP Explanation Schema
class FeatureContribution(BaseModel):
    feature: str
    contribution: float
    impact: Literal["increase", "decrease"]


class SHAPExplanation(BaseModel):
    top_features: List[FeatureContribution]
    base_value: float
    prediction_value: float


class ExplanationResponse(BaseModel):
    prediction: int
    probability: float
    risk_level: Literal["low", "medium", "high"]
    explanation: SHAPExplanation


# Top-K Targeting
class TopKRequest(BaseModel):
    k: Optional[int] = Field(default=None, ge=1)
    k_percent: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class TopKResponse(BaseModel):
    customers: List[Dict[str, Any]]
    k: int
    k_percent: float


# System Schemas
class HealthResponse(BaseModel):
    status: Literal["healthy", "unhealthy"]
    model_loaded: bool
    timestamp: str 
    version: str


class VersionResponse(BaseModel):
    api_version: str
    model_name: str
    model_version: Optional[str] = None
    model_stage: Optional[str] = None
    mlflow_tracking_uri: str


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    timestamp: str
    