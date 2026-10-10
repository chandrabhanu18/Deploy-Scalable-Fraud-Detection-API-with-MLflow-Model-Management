"""
app/api/v1/endpoints.py
API endpoints for Fraud Detection inference, batch scoring, model health, and metadata.
"""

import time
import uuid
import logging
from fastapi import APIRouter, HTTPException, status

from app.core.config import settings
from app.models.schemas import (
    BatchFraudPredictionRequest,
    BatchFraudPredictionResponse,
    FraudPredictionRequest,
    FraudPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
)
from app.services.fraud_service import fraud_service

logger = logging.getLogger(__name__)

router = APIRouter()

# Track start time for uptime calculation
_SERVICE_START_TIME = time.time()


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Health & Readiness Check",
    description="Returns the health status of the API, active MLflow model registration, and system uptime."
)
async def health_check():
    """
    Perform a comprehensive health check on the inference engine and ML model availability.
    """
    uptime = round(time.time() - _SERVICE_START_TIME, 2)
    model_info = fraud_service.get_info()

    is_healthy = model_info["is_loaded"]
    overall_status = "healthy" if is_healthy else "degraded"

    return HealthResponse(
        status=overall_status,
        service=settings.app_name,
        version=settings.app_version,
        model_status="loaded" if is_healthy else "failed",
        model_source=model_info["source"],
        model_name=model_info["model_name"],
        model_version=model_info["model_version"],
        mlflow_tracking_uri=settings.mlflow_tracking_uri,
        uptime_seconds=uptime,
    )


@router.post(
    "/predict",
    response_model=FraudPredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Evaluate Single Transaction",
    description="Analyzes transaction attributes and returns fraud classification, confidence probability, and risk tier."
)
async def predict_fraud(request: FraudPredictionRequest):
    """
    Execute real-time fraud scoring for an incoming financial transaction.
    """
    prediction_id = f"pred-{uuid.uuid4().hex[:10]}"
    try:
        response = fraud_service.predict_single(request, prediction_id=prediction_id)
        logger.info(
            "Fraud prediction completed",
            extra={
                "prediction_id": prediction_id,
                "is_fraud": response.is_fraud,
                "fraud_probability": response.fraud_probability,
                "risk_level": response.risk_level.value,
            }
        )
        return response
    except ValueError as ve:
        logger.warning(f"Data validation error during prediction: {ve}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid transaction features: {ve}"
        )
    except Exception as exc:
        logger.error(f"Inference error on prediction {prediction_id}: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference engine failure: {str(exc)}"
        )


@router.post(
    "/predict/batch",
    response_model=BatchFraudPredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Evaluate Batch of Transactions",
    description="Batch scores multiple financial transactions in a single vectorized inference call."
)
async def predict_fraud_batch(batch_request: BatchFraudPredictionRequest):
    """
    Batch scoring endpoint for high-volume offline or streaming transaction chunks.
    """
    try:
        predictions = fraud_service.predict_batch(batch_request.transactions)
        fraud_count = sum(1 for p in predictions if p.is_fraud)
        total = len(predictions)
        fraud_rate = round((fraud_count / total) * 100, 2) if total > 0 else 0.0

        return BatchFraudPredictionResponse(
            total_transactions=total,
            fraud_detected_count=fraud_count,
            fraud_rate_percentage=fraud_rate,
            predictions=predictions,
        )
    except Exception as exc:
        logger.error(f"Batch prediction failed: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch inference failure: {str(exc)}"
        )


@router.get(
    "/model-info",
    response_model=ModelInfoResponse,
    status_code=status.HTTP_200_OK,
    summary="Model Metadata & Details",
    description="Retrieves metadata about the loaded MLflow model, feature requirements, and decision thresholds."
)
async def get_model_metadata():
    """
    Return active model details and configuration.
    """
    info = fraud_service.get_info()
    return ModelInfoResponse(
        model_name=info["model_name"],
        model_version=info["model_version"],
        source=info["source"],
        model_type=info["model_type"],
        features=info["features"],
        fraud_threshold=info["fraud_threshold"],
        metrics=info.get("metrics"),
    )


@router.post(
    "/model/reload",
    status_code=status.HTTP_200_OK,
    summary="Hot-Reload Model",
    description="Triggers the service to reload the model from the MLflow Model Registry without downtime."
)
async def reload_model():
    """
    Hot reload model from MLflow registry.
    """
    success = fraud_service.load_model()
    if success:
        return {
            "status": "success",
            "message": "Model reloaded successfully",
            "model_source": fraud_service.model_source,
            "model_name": fraud_service.model_name,
            "model_version": fraud_service.model_version,
        }
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="Failed to hot-reload model from MLflow or local fallback."
    )
