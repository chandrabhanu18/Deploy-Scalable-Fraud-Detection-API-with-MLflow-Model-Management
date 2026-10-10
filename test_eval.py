import os
import requests
import json
import time

BASE_URL = os.getenv("BASE_URL", "http://localhost:8002")

def evaluate():
    print("=" * 70)
    print("  FRAUD DETECTION API & MLOps PIPELINE - COMPREHENSIVE EVALUATION")
    print("=" * 70)

    # 1. Root Endpoint
    print("\n[TEST 1] Root Endpoint (GET /)")
    r = requests.get(f"{BASE_URL}/")
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    print(f" -> Status: {r.status_code} OK")
    print(f" -> Payload: {json.dumps(r.json(), indent=2)}")

    # 2. Health Endpoint
    print("\n[TEST 2] Health Endpoint (GET /api/v1/health)")
    r = requests.get(f"{BASE_URL}/api/v1/health")
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    data = r.json()
    assert data["status"] == "healthy", f"Expected status healthy, got {data['status']}"
    assert data["model_status"] == "loaded", f"Expected model loaded, got {data['model_status']}"
    print(f" -> Status: {r.status_code} OK")
    print(f" -> Health: {json.dumps(data, indent=2)}")

    # 3. Model Info Endpoint
    print("\n[TEST 3] Model Metadata Endpoint (GET /api/v1/model-info)")
    r = requests.get(f"{BASE_URL}/api/v1/model-info")
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    data = r.json()
    assert "features" in data
    assert len(data["features"]) == 5
    print(f" -> Status: {r.status_code} OK")
    print(f" -> Features: {data['features']}")
    print(f" -> Threshold: Fraud={data.get('fraud_threshold')}, ModelType={data.get('model_type')}")

    # 4. Single Prediction - Low Risk Transaction
    print("\n[TEST 4] Single Prediction - Low Risk Legitimate (POST /api/v1/predict)")
    low_payload = {
        "transaction_amount": 35.50,
        "time_of_day_hour": 14,
        "num_transactions_1h": 1,
        "avg_transaction_7d": 40.00,
        "location_risk_score": 0.05
    }
    r = requests.post(f"{BASE_URL}/api/v1/predict", json=low_payload)
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    data = r.json()
    assert data["is_fraud"] is False
    assert data["risk_level"] == "LOW"
    assert "prediction_id" in data
    print(f" -> Status: {r.status_code} OK")
    print(f" -> Result: Fraud={data['is_fraud']}, Probability={data['fraud_probability']}, Risk={data['risk_level']}, ID={data['prediction_id']}")

    # 5. Single Prediction - High Risk Transaction
    print("\n[TEST 5] Single Prediction - High Risk Anomaly (POST /api/v1/predict)")
    high_payload = {
        "transaction_amount": 5500.00,
        "time_of_day_hour": 3,
        "num_transactions_1h": 14,
        "avg_transaction_7d": 35.00,
        "location_risk_score": 0.98
    }
    r = requests.post(f"{BASE_URL}/api/v1/predict", json=high_payload)
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    data = r.json()
    print(f" -> Status: {r.status_code} OK")
    print(f" -> Result: Fraud={data['is_fraud']}, Probability={data['fraud_probability']}, Risk={data['risk_level']}, ID={data['prediction_id']}")

    # 6. Batch Prediction
    print("\n[TEST 6] Batch Prediction Endpoint (POST /api/v1/predict/batch)")
    batch_payload = {"transactions": [low_payload, high_payload]}
    r = requests.post(f"{BASE_URL}/api/v1/predict/batch", json=batch_payload)
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    data = r.json()
    assert data["total_transactions"] == 2
    assert len(data["predictions"]) == 2
    print(f" -> Status: {r.status_code} OK")
    print(f" -> Total: {data['total_transactions']}, Fraud Count: {data['fraud_detected_count']}, Rate: {data['fraud_rate_percentage']}%")

    # 7. Input Validation & Boundaries (422 Unprocessable Entity)
    print("\n[TEST 7] Input Validation & Boundary Checks (Negative Amount, Hour > 23, Risk Score > 1.0)")
    invalid_cases = [
        {"desc": "Negative Amount", "payload": {**low_payload, "transaction_amount": -10.0}},
        {"desc": "Hour 25", "payload": {**low_payload, "time_of_day_hour": 25}},
        {"desc": "Negative Burst Count", "payload": {**low_payload, "num_transactions_1h": -1}},
        {"desc": "Risk Score > 1.0", "payload": {**low_payload, "location_risk_score": 1.5}},
    ]
    for case in invalid_cases:
        r = requests.post(f"{BASE_URL}/api/v1/predict", json=case["payload"])
        assert r.status_code == 422, f"Expected 422 for {case['desc']}, got {r.status_code}"
        print(f" -> {case['desc']}: 422 Unprocessable Entity as expected")

    # 8. Model Hot-Reload
    print("\n[TEST 8] Model Dynamic Hot-Reload Endpoint (POST /api/v1/model/reload)")
    r = requests.post(f"{BASE_URL}/api/v1/model/reload")
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    data = r.json()
    assert data["status"] == "success"
    print(f" -> Status: {r.status_code} OK")
    print(f" -> Reload Response: {json.dumps(data, indent=2)}")

    # 9. Concurrency & Performance
    print("\n[TEST 9] Concurrency / Load Check (30 Rapid Sequential Inferences)")
    start = time.perf_counter()
    for _ in range(30):
        r = requests.post(f"{BASE_URL}/api/v1/predict", json=low_payload)
        assert r.status_code == 200
    elapsed = (time.perf_counter() - start) * 1000
    print(f" -> 30 Requests finished in {elapsed:.2f}ms (Avg: {elapsed/30:.2f}ms/request)")

    # 10. Swagger Docs & OpenAPI Schema
    print("\n[TEST 10] OpenAPI & Swagger Documentation (GET /openapi.json & /docs)")
    r = requests.get(f"{BASE_URL}/openapi.json")
    assert r.status_code == 200
    assert "paths" in r.json()
    r = requests.get(f"{BASE_URL}/docs")
    assert r.status_code == 200
    print(" -> Swagger UI & OpenAPI Specification verified OK")

    print("\n" + "=" * 70)
    print("  ALL 10 EVALUATION SECTIONS PASSED WITH 100% SUCCESS!")
    print("=" * 70)

if __name__ == "__main__":
    evaluate()
