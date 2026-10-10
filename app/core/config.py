"""
app/core/config.py
Pydantic BaseSettings management for environment variables and service configurations.
"""

from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Application Info
    app_name: str = "Fraud Detection API"
    app_version: str = "1.0.0"
    app_env: str = "production"
    api_v1_str: str = "/api/v1"
    docs_url: str = "/docs"
    redoc_url: str = "/redoc"
    openapi_url: str = "/openapi.json"

    # Server Settings
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"
    structured_logging: bool = True

    # MLflow Settings
    mlflow_tracking_uri: str = "http://localhost:5000"
    model_name: str = "FraudDetectionModel"
    model_version: str = "1"  # Or "latest"
    model_stage: Optional[str] = None  # e.g., "Production", "Staging", or None

    # Model Local Fallback Path (if MLflow server is temporarily unreachable)
    model_fallback_path: str = "data/fraud_model.pkl"

    # Prediction Logic Parameters
    fraud_probability_threshold: float = 0.50
    high_risk_threshold: float = 0.75
    medium_risk_threshold: float = 0.40

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )


settings = Settings()
