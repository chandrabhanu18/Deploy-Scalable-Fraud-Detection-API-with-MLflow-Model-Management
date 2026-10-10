"""
train_model.py
Trains a robust Fraud Detection classifier, logs experiment metrics and parameters to MLflow Tracking,
and registers the model in the MLflow Model Registry. Also saves a local fallback model artifact.
"""

import argparse
import os
from pathlib import Path
import joblib
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from generate_data import generate_synthetic_data

import sys

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# Ensure compatibility with file stores if used
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

FEATURE_COLUMNS = [
    "transaction_amount",
    "time_of_day_hour",
    "num_transactions_1h",
    "avg_transaction_7d",
    "location_risk_score",
]
TARGET_COLUMN = "is_fraud"


def load_or_generate_dataset(data_path: str) -> pd.DataFrame:
    """Load dataset from path or generate if missing."""
    file_path = Path(data_path)
    if not file_path.exists():
        print(f"Dataset not found at {data_path}. Generating synthetic dataset...")
        df = generate_synthetic_data(num_samples=5000, random_seed=42)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(file_path, index=False)
        print(f"Saved generated dataset to {data_path}")
    else:
        df = pd.read_csv(file_path)
        print(f"Loaded {len(df)} records from {data_path}")
    return df


def train_and_register(
    data_path: str = "data/raw_transactions.csv",
    tracking_uri: str = "sqlite:///mlflow.db",
    registered_model_name: str = "FraudDetectionModel",
    fallback_model_path: str = "data/fraud_model.pkl",
    n_estimators: int = 150,
    max_depth: int = 8,
    random_state: int = 42,
) -> dict:
    """
    Train RandomForest fraud classifier, log to MLflow, and export local fallback artifact.
    """
    # 1. Load Data
    df = load_or_generate_dataset(data_path)
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    # Stratified Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=random_state, stratify=y
    )
    print(f"Training samples: {len(X_train)} | Testing samples: {len(X_test)}")
    print(f"Fraud distribution - Train: {y_train.mean():.3f} | Test: {y_test.mean():.3f}")

    # 2. Configure MLflow
    mlflow.set_tracking_uri(tracking_uri)
    experiment_name = "Fraud Detection Training"
    mlflow.set_experiment(experiment_name)
    print(f"MLflow Tracking URI: {tracking_uri}")
    print(f"MLflow Experiment: {experiment_name}")

    # 3. Model Training
    print("Training RandomForestClassifier...")
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # 4. Evaluation
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = float(accuracy_score(y_test, y_pred))
    precision = float(precision_score(y_test, y_pred, zero_division=0))
    recall = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    roc_auc = float(roc_auc_score(y_test, y_prob))

    metrics = {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
    }

    print("\n--- Evaluation Metrics ---")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # 5. Log with MLflow Tracking & Register in Model Registry
    signature = infer_signature(X_train, model.predict(X_train))
    input_example = X_train.head(3).to_dict(orient="records")

    with mlflow.start_run(run_name=f"RandomForest_n{n_estimators}_d{max_depth}") as run:
        run_id = run.info.run_id
        print(f"MLflow Run ID: {run_id}")

        # Log parameters
        mlflow.log_params({
            "model_type": "RandomForestClassifier",
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "class_weight": "balanced_subsample",
            "random_state": random_state,
            "feature_columns": ",".join(FEATURE_COLUMNS),
            "num_features": len(FEATURE_COLUMNS),
        })

        # Log metrics
        mlflow.log_metrics(metrics)

        # Log model artifact and register
        try:
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="fraud_model",
                registered_model_name=registered_model_name,
                signature=signature,
                input_example=input_example,
                serialization_format="cloudpickle",
            )
            print(f"Model successfully logged and registered as '{registered_model_name}'")
        except Exception as e:
            print(f"Notice: Model registration with registry failed or not supported by backend ({e}). Logging model artifact without registry.")
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="fraud_model",
                signature=signature,
                input_example=input_example,
                serialization_format="cloudpickle",
            )

    # 6. Save Local Fallback Model Artifact
    fallback_path = Path(fallback_model_path)
    fallback_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "metrics": metrics,
        "model_type": "RandomForestClassifier",
    }, fallback_path)
    print(f"Saved local fallback model artifact to: {fallback_path}")

    return {
        "run_id": run_id,
        "metrics": metrics,
        "model_name": registered_model_name,
        "fallback_path": str(fallback_path),
    }


def main():
    parser = argparse.ArgumentParser(description="Train and register Fraud Detection Model in MLflow")
    parser.add_argument("--data-path", type=str, default="data/raw_transactions.csv", help="Path to CSV dataset")
    parser.add_argument(
        "--tracking-uri",
        type=str,
        default=os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"),
        help="MLflow tracking URI (e.g. http://localhost:5000 or sqlite:///mlflow.db)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=os.getenv("MODEL_NAME", "FraudDetectionModel"),
        help="Registered model name in MLflow",
    )
    parser.add_argument(
        "--fallback-path",
        type=str,
        default=os.getenv("MODEL_FALLBACK_PATH", "data/fraud_model.pkl"),
        help="Local fallback pickle path",
    )
    parser.add_argument("--n-estimators", type=int, default=150, help="Number of trees")
    parser.add_argument("--max-depth", type=int, default=8, help="Max tree depth")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    train_and_register(
        data_path=args.data_path,
        tracking_uri=args.tracking_uri,
        registered_model_name=args.model_name,
        fallback_model_path=args.fallback_path,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.seed,
    )


if __name__ == "__main__":
    main()
