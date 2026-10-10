"""
tests/integration/test_api.py
Integration tests for FastAPI endpoints: /health, /predict, /predict/batch, /model-info, /model/reload.
"""

import pytest
from fastapi.testclient import TestClient


def test_root_endpoint(client: TestClient):
    """Test root landing page returning service info."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "service" in data
    assert "docs" in data
    assert "health" in data


def test_health_check_endpoint(client: TestClient):
    """Test /api/v1/health endpoint returns 200 and healthy status."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_status"] == "loaded"
    assert data["model_name"] == "FraudDetectionModel"
    assert "uptime_seconds" in data
    assert "model_source" in data


def test_model_info_endpoint(client: TestClient):
    """Test /api/v1/model-info endpoint."""
    response = client.get("/api/v1/model-info")
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "FraudDetectionModel"
    assert isinstance(data["features"], list)
    assert len(data["features"]) == 5
    assert data["fraud_threshold"] == 0.50


def test_predict_legitimate_transaction(client: TestClient, legitimate_transaction_payload):
    """Test /api/v1/predict endpoint for normal transaction."""
    response = client.post("/api/v1/predict", json=legitimate_transaction_payload)
    assert response.status_code == 200
    data = response.json()
    assert "is_fraud" in data
    assert isinstance(data["is_fraud"], bool)
    assert "fraud_probability" in data
    assert 0.0 <= data["fraud_probability"] <= 1.0
    assert "risk_level" in data
    assert data["risk_level"] in ["LOW", "MEDIUM", "HIGH"]
    assert "prediction_id" in data
    assert "timestamp" in data
    # Verify custom headers
    assert "X-Request-ID" in response.headers
    assert "X-Process-Time-Ms" in response.headers


def test_predict_fraudulent_transaction(client: TestClient, fraudulent_transaction_payload):
    """Test /api/v1/predict endpoint for high-risk transaction."""
    response = client.post("/api/v1/predict", json=fraudulent_transaction_payload)
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data["is_fraud"], bool)
    assert 0.0 <= data["fraud_probability"] <= 1.0


def test_predict_validation_error_negative_amount(client: TestClient):
    """Test 422 Unprocessable Entity when transaction amount is negative."""
    invalid_payload = {
        "transaction_amount": -50.0,
        "time_of_day_hour": 14,
        "num_transactions_1h": 1,
        "avg_transaction_7d": 50.00,
        "location_risk_score": 0.1,
    }
    response = client.post("/api/v1/predict", json=invalid_payload)
    assert response.status_code == 422
    data = response.json()
    assert data["error"] == "Unprocessable Entity"
    assert "details" in data


def test_predict_validation_error_invalid_hour(client: TestClient):
    """Test 422 Unprocessable Entity when hour > 23."""
    invalid_payload = {
        "transaction_amount": 100.0,
        "time_of_day_hour": 25,  # Invalid hour
        "num_transactions_1h": 1,
        "avg_transaction_7d": 50.00,
        "location_risk_score": 0.1,
    }
    response = client.post("/api/v1/predict", json=invalid_payload)
    assert response.status_code == 422


def test_predict_validation_error_invalid_risk_score(client: TestClient):
    """Test 422 Unprocessable Entity when location_risk_score > 1.0."""
    invalid_payload = {
        "transaction_amount": 100.0,
        "time_of_day_hour": 12,
        "num_transactions_1h": 1,
        "avg_transaction_7d": 50.00,
        "location_risk_score": 1.5,  # Exceeds max 1.0
    }
    response = client.post("/api/v1/predict", json=invalid_payload)
    assert response.status_code == 422


def test_predict_validation_missing_fields(client: TestClient):
    """Test 422 Unprocessable Entity when fields are missing."""
    response = client.post("/api/v1/predict", json={"transaction_amount": 100.0})
    assert response.status_code == 422


def test_predict_batch_endpoint(
    client: TestClient, legitimate_transaction_payload, fraudulent_transaction_payload
):
    """Test /api/v1/predict/batch endpoint with multiple items."""
    batch_payload = {
        "transactions": [legitimate_transaction_payload, fraudulent_transaction_payload]
    }
    response = client.post("/api/v1/predict/batch", json=batch_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_transactions"] == 2
    assert "fraud_detected_count" in data
    assert "fraud_rate_percentage" in data
    assert len(data["predictions"]) == 2


def test_predict_batch_empty_list(client: TestClient):
    """Test 422 Unprocessable Entity when batch list is empty."""
    response = client.post("/api/v1/predict/batch", json={"transactions": []})
    assert response.status_code == 422


def test_model_reload_endpoint(client: TestClient):
    """Test /api/v1/model/reload endpoint."""
    response = client.post("/api/v1/model/reload")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "model_source" in data
