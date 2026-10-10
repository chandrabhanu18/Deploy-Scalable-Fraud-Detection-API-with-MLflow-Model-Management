"""
tests/conftest.py
Pytest test fixtures, mock setups, and test clients for FastAPI testing.
"""

import os
import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

# Ensure root workspace is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

# Configure test environment variables before importing app
os.environ["APP_ENV"] = "testing"
os.environ["MLFLOW_TRACKING_URI"] = "sqlite:///mlflow.db"
os.environ["MODEL_NAME"] = "FraudDetectionModel"
os.environ["MODEL_VERSION"] = "1"
os.environ["MODEL_FALLBACK_PATH"] = "data/fraud_model.pkl"
os.environ["STRUCTURED_LOGGING"] = "false"

from app.main import app
from app.services.fraud_service import fraud_service


@pytest.fixture(scope="session", autouse=True)
def ensure_model_available():
    """Ensure that either MLflow model or local fallback model is available for tests."""
    from train_model import train_and_register

    if not Path("data/fraud_model.pkl").exists():
        print("Training model for test suite...")
        train_and_register(
            data_path="data/raw_transactions.csv",
            tracking_uri="sqlite:///mlflow.db",
            registered_model_name="FraudDetectionModel",
            fallback_model_path="data/fraud_model.pkl",
            n_estimators=20,
            max_depth=4,
        )

    fraud_service.load_model()
    yield


@pytest.fixture(scope="module")
def client():
    """FastAPI TestClient fixture."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def legitimate_transaction_payload():
    """Standard payload representing a low-risk legitimate transaction."""
    return {
        "transaction_amount": 45.50,
        "time_of_day_hour": 14,
        "num_transactions_1h": 1,
        "avg_transaction_7d": 50.00,
        "location_risk_score": 0.08,
    }


@pytest.fixture
def fraudulent_transaction_payload():
    """Payload representing a high-risk fraudulent transaction."""
    return {
        "transaction_amount": 4500.00,
        "time_of_day_hour": 3,
        "num_transactions_1h": 9,
        "avg_transaction_7d": 65.00,
        "location_risk_score": 0.95,
    }
