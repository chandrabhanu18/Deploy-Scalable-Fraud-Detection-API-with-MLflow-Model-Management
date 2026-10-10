"""
app/models/schemas.py
Pydantic schemas and validation models for Fraud Detection API requests and responses.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class FraudPredictionRequest(BaseModel):
    """Input payload for single transaction fraud prediction."""
    transaction_amount: float = Field(
        ...,
        gt=0,
        description="Amount of the transaction in USD",
        examples=[125.50]
    )
    time_of_day_hour: int = Field(
        ...,
        ge=0,
        le=23,
        description="Hour of the day (0-23) when transaction occurred",
        examples=[14]
    )
    num_transactions_1h: int = Field(
        ...,
        ge=0,
        description="Number of transactions initiated in the last hour",
        examples=[2]
    )
    avg_transaction_7d: float = Field(
        ...,
        gt=0,
        description="Average transaction amount in the last 7 days",
        examples=[98.20]
    )
    location_risk_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Risk score associated with the transaction location (0.0=safe, 1.0=very risky)",
        examples=[0.12]
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "transaction_amount": 4200.0,
                "time_of_day_hour": 3,
                "num_transactions_1h": 8,
                "avg_transaction_7d": 120.0,
                "location_risk_score": 0.88,
            }
        }
    )


class FraudPredictionResponse(BaseModel):
    """Prediction outcome for a single transaction."""
    is_fraud: bool = Field(
        ...,
        description="True if the transaction is classified as fraudulent, False otherwise"
    )
    fraud_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Predicted probability of fraud [0.0, 1.0]"
    )
    risk_level: RiskLevel = Field(
        ...,
        description="Categorized risk tier (LOW, MEDIUM, HIGH)"
    )
    model_name: str = Field(
        ...,
        description="Name of the model used for inference"
    )
    model_version: str = Field(
        ...,
        description="Version or stage of the model"
    )
    prediction_id: Optional[str] = Field(
        default=None,
        description="Unique identifier for the inference request"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp of the prediction (UTC)"
    )


class BatchFraudPredictionRequest(BaseModel):
    """Batch input payload for multiple transactions."""
    transactions: List[FraudPredictionRequest] = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="List of transaction items to evaluate"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "transactions": [
                    {
                        "transaction_amount": 45.0,
                        "time_of_day_hour": 12,
                        "num_transactions_1h": 1,
                        "avg_transaction_7d": 50.0,
                        "location_risk_score": 0.05,
                    },
                    {
                        "transaction_amount": 4800.0,
                        "time_of_day_hour": 2,
                        "num_transactions_1h": 7,
                        "avg_transaction_7d": 80.0,
                        "location_risk_score": 0.92,
                    }
                ]
            }
        }
    )


class BatchFraudPredictionResponse(BaseModel):
    """Batch prediction summary and individual results."""
    total_transactions: int = Field(..., description="Total number of evaluated transactions")
    fraud_detected_count: int = Field(..., description="Number of flagged fraudulent transactions")
    fraud_rate_percentage: float = Field(..., description="Percentage of flagged transactions")
    predictions: List[FraudPredictionResponse] = Field(..., description="List of individual prediction responses")


class HealthResponse(BaseModel):
    """Health check response schema."""
    status: str = Field(..., description="Overall service status ('healthy' or 'degraded')")
    service: str = Field(..., description="Service name")
    version: str = Field(..., description="API Version")
    model_status: str = Field(..., description="Model load status ('loaded', 'fallback_loaded', 'failed')")
    model_source: str = Field(..., description="Source of the loaded model ('mlflow_registry' or 'local_fallback')")
    model_name: str = Field(..., description="Model identifier name")
    model_version: str = Field(..., description="Model version")
    mlflow_tracking_uri: str = Field(..., description="Configured MLflow tracking URI")
    uptime_seconds: float = Field(..., description="Total uptime in seconds")


class ModelInfoResponse(BaseModel):
    """Detailed model metadata and training parameters schema."""
    model_name: str = Field(..., description="Model name")
    model_version: str = Field(..., description="Model version or tag")
    source: str = Field(..., description="Model source")
    model_type: str = Field(..., description="Underlying ML algorithm family")
    features: List[str] = Field(..., description="Expected feature columns in order")
    fraud_threshold: float = Field(..., description="Decision boundary threshold for fraud classification")
    metrics: Optional[Dict[str, float]] = Field(default=None, description="Model evaluation performance metrics")


class ErrorResponse(BaseModel):
    """Standardized error response body."""
    error: str = Field(..., description="Error message title")
    detail: Optional[Any] = Field(default=None, description="Detailed diagnostic error information")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp of the error occurrence"
    )
