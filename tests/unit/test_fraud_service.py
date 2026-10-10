"""
tests/unit/test_fraud_service.py
Unit tests for FraudDetectionService prediction logic, fallback handling, risk tiers, and feature mapping.
"""

import pytest
from app.models.schemas import FraudPredictionRequest, RiskLevel
from app.services.fraud_service import FraudDetectionService


def test_service_initialization():
    service = FraudDetectionService()
    assert service.model_name == "FraudDetectionModel"
    assert len(service.feature_columns) == 5
    assert "transaction_amount" in service.feature_columns


def test_service_load_model(ensure_model_available):
    service = FraudDetectionService()
    loaded = service.load_model()
    assert loaded is True
    assert service.is_loaded is True
    assert service.model_source in ["mlflow_registry", "local_fallback"]


def test_service_predict_single(ensure_model_available):
    service = FraudDetectionService()
    service.load_model()

    req = FraudPredictionRequest(
        transaction_amount=25.0,
        time_of_day_hour=13,
        num_transactions_1h=1,
        avg_transaction_7d=30.0,
        location_risk_score=0.05,
    )

    response = service.predict_single(req, prediction_id="test-unit-1")
    assert response.prediction_id == "test-unit-1"
    assert isinstance(response.is_fraud, bool)
    assert 0.0 <= response.fraud_probability <= 1.0
    assert response.risk_level in [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH]
    assert response.model_name == "FraudDetectionModel"


def test_service_predict_batch(ensure_model_available):
    service = FraudDetectionService()
    service.load_model()

    req1 = FraudPredictionRequest(
        transaction_amount=15.0,
        time_of_day_hour=10,
        num_transactions_1h=0,
        avg_transaction_7d=20.0,
        location_risk_score=0.02,
    )
    req2 = FraudPredictionRequest(
        transaction_amount=4800.0,
        time_of_day_hour=2,
        num_transactions_1h=8,
        avg_transaction_7d=70.0,
        location_risk_score=0.92,
    )

    responses = service.predict_batch([req1, req2])
    assert len(responses) == 2
    assert all(0.0 <= r.fraud_probability <= 1.0 for r in responses)


def test_service_risk_level_tiers():
    service = FraudDetectionService()
    assert service._determine_risk_level(0.85) == RiskLevel.HIGH
    assert service._determine_risk_level(0.55) == RiskLevel.MEDIUM
    assert service._determine_risk_level(0.20) == RiskLevel.LOW


def test_service_get_info(ensure_model_available):
    service = FraudDetectionService()
    service.load_model()
    info = service.get_info()

    assert info["model_name"] == "FraudDetectionModel"
    assert info["is_loaded"] is True
    assert "features" in info
    assert len(info["features"]) == 5
