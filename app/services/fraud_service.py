"""
app/services/fraud_service.py
Service handling MLflow model retrieval, local fallback caching, and high-performance inference.
"""

import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
import requests

from app.core.config import settings
from app.models.schemas import (
    FraudPredictionRequest,
    FraudPredictionResponse,
    RiskLevel,
)

logger = logging.getLogger(__name__)

FEATURE_ORDER = [
    "transaction_amount",
    "time_of_day_hour",
    "num_transactions_1h",
    "avg_transaction_7d",
    "location_risk_score",
]


class FraudDetectionService:
    """
    MLflow-integrated Fraud Detection Service with offline fallback resilience.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.model: Optional[Any] = None
        self.model_source: str = "uninitialized"
        self.model_name: str = settings.model_name
        self.model_version: str = str(settings.model_version)
        self.metrics: Dict[str, float] = {}
        self.feature_columns: List[str] = FEATURE_ORDER
        self.model_type: str = "RandomForestClassifier"
        self.is_loaded: bool = False

    def _is_mlflow_reachable(self, uri: str) -> bool:
        """Fast pre-check if remote HTTP MLflow tracking server is reachable."""
        if uri.startswith(("http://", "https://")):
            try:
                resp = requests.get(f"{uri.rstrip('/')}/health", timeout=0.8)
                return resp.status_code < 500
            except Exception:
                try:
                    resp = requests.get(uri.rstrip('/'), timeout=0.8)
                    return resp.status_code < 500
                except Exception:
                    return False
        return True

    def load_model(self) -> bool:
        """
        Load model from MLflow Model Registry, or fall back to local artifact.
        """
        with self._lock:
            # Ensure file store exception bypass is set if local file backend is used
            os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

            tracking_uri = settings.mlflow_tracking_uri

            # Attempt 1: Load from MLflow Model Registry / Tracking Server
            if self._is_mlflow_reachable(tracking_uri):
                try:
                    mlflow.set_tracking_uri(tracking_uri)
                    if settings.model_stage:
                        model_uri = f"models:/{settings.model_name}/{settings.model_stage}"
                    else:
                        model_uri = f"models:/{settings.model_name}/{settings.model_version}"

                    logger.info(f"Attempting to load model from MLflow URI: {model_uri}")
                    self.model = mlflow.pyfunc.load_model(model_uri)
                    self.model_source = "mlflow_registry"
                    self.is_loaded = True
                    logger.info(f"Successfully loaded model from MLflow registry: {model_uri}")
                    return True
                except Exception as mlflow_err:
                    logger.warning(
                        f"Could not load model from MLflow ({mlflow_err}). "
                        f"Falling back to local artifact: {settings.model_fallback_path}"
                    )
            else:
                logger.info(
                    f"MLflow server at '{tracking_uri}' is unreachable. Proceeding directly to local fallback."
                )

            # Attempt 2: Fallback to local serialized model artifact
            try:
                fallback_path = Path(settings.model_fallback_path)
                if fallback_path.exists():
                    saved_payload = joblib.load(fallback_path)
                    if isinstance(saved_payload, dict) and "model" in saved_payload:
                        self.model = saved_payload["model"]
                        self.metrics = saved_payload.get("metrics", {})
                        self.model_type = saved_payload.get("model_type", "RandomForestClassifier")
                        self.feature_columns = saved_payload.get("feature_columns", FEATURE_ORDER)
                    else:
                        self.model = saved_payload

                    self.model_source = "local_fallback"
                    self.is_loaded = True
                    logger.info(f"Successfully loaded fallback model from: {fallback_path}")
                    return True
                else:
                    logger.error(f"Fallback model file not found at: {fallback_path}")
            except Exception as fallback_err:
                logger.error(f"Failed to load fallback model: {fallback_err}", exc_info=True)

            self.is_loaded = False
            self.model_source = "failed"
            return False

    def _determine_risk_level(self, probability: float) -> RiskLevel:
        """Categorize numerical fraud probability into operational risk tiers."""
        if probability >= settings.high_risk_threshold:
            return RiskLevel.HIGH
        elif probability >= settings.medium_risk_threshold:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW

    def _prepare_dataframe(self, records: List[Dict[str, Any]]) -> pd.DataFrame:
        """Create structured DataFrame with strictly validated columns."""
        df = pd.DataFrame(records)
        for col in self.feature_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required feature: {col}")
        return df[self.feature_columns]

    def predict_single(
        self, request: FraudPredictionRequest, prediction_id: Optional[str] = None
    ) -> FraudPredictionResponse:
        """
        Execute prediction for a single transaction.
        """
        if not self.is_loaded or self.model is None:
            if not self.load_model():
                raise RuntimeError("Fraud detection model is not loaded and could not be initialized.")

        data_dict = request.model_dump()
        df = self._prepare_dataframe([data_dict])

        probability = self._compute_probability(df)[0]
        is_fraud = bool(probability >= settings.fraud_probability_threshold)
        risk_level = self._determine_risk_level(probability)

        return FraudPredictionResponse(
            is_fraud=is_fraud,
            fraud_probability=round(float(probability), 4),
            risk_level=risk_level,
            model_name=self.model_name,
            model_version=self.model_version,
            prediction_id=prediction_id,
        )

    def predict_batch(
        self, requests: List[FraudPredictionRequest]
    ) -> List[FraudPredictionResponse]:
        """
        Execute batch prediction for high throughput.
        """
        if not self.is_loaded or self.model is None:
            if not self.load_model():
                raise RuntimeError("Fraud detection model is not loaded and could not be initialized.")

        records = [req.model_dump() for req in requests]
        df = self._prepare_dataframe(records)

        probabilities = self._compute_probability(df)
        responses = []

        for i, prob in enumerate(probabilities):
            prob_float = float(prob)
            is_fraud = bool(prob_float >= settings.fraud_probability_threshold)
            risk_level = self._determine_risk_level(prob_float)
            responses.append(
                FraudPredictionResponse(
                    is_fraud=is_fraud,
                    fraud_probability=round(prob_float, 4),
                    risk_level=risk_level,
                    model_name=self.model_name,
                    model_version=self.model_version,
                    prediction_id=f"batch-{i + 1}",
                )
            )

        return responses

    def _compute_probability(self, df: pd.DataFrame) -> np.ndarray:
        """Extract fraud probability from MLflow PyFunc or native scikit-learn model."""
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(df)
            if probs.ndim == 2 and probs.shape[1] >= 2:
                return probs[:, 1]
            return probs.flatten()

        if hasattr(self.model, "predict"):
            preds = self.model.predict(df)
            if isinstance(preds, pd.DataFrame) or isinstance(preds, pd.Series):
                preds = preds.to_numpy()
            if isinstance(preds, np.ndarray):
                if preds.ndim == 2 and preds.shape[1] >= 2:
                    return preds[:, 1]
                elif preds.ndim == 1:
                    if np.issubdtype(preds.dtype, np.floating) and np.all((preds >= 0.0) & (preds <= 1.0)):
                        return preds
                    return preds.astype(float)
            return np.array([float(p) for p in preds])

        raise TypeError(f"Loaded model of type {type(self.model)} does not support predict or predict_proba.")

    def get_info(self) -> Dict[str, Any]:
        """Return model metadata and configuration."""
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "source": self.model_source,
            "model_type": self.model_type,
            "features": self.feature_columns,
            "fraud_threshold": settings.fraud_probability_threshold,
            "metrics": self.metrics,
            "is_loaded": self.is_loaded,
        }


# Singleton service instance
fraud_service = FraudDetectionService()
