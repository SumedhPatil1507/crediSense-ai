"""Integration tests for the FastAPI endpoints."""
import sys
import pytest
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from fastapi.testclient import TestClient
from api.main import app, API_SECRET_KEY

client = TestClient(app)
HEADERS = {"x-api-key": API_SECRET_KEY}

VALID_PAYLOAD = {
    "income_lpa": 10.0,
    "age_years": 30,
    "experience_years": 5,
    "profession": "Engineer",
    "city": "Mumbai",
    "state": "Maharashtra",
    "house_ownership": "owned",
    "marital_status": "single",
    "car_ownership": "no",
    "current_job_years": 2,
    "current_house_years": 3,
}


class TestHealth:
    def test_health_returns_ok(self):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_health_has_db_info(self):
        r = client.get("/health")
        assert "db" in r.json()


class TestPredict:
    def test_predict_requires_api_key(self):
        r = client.post("/api/v1/predict", json=VALID_PAYLOAD)
        assert r.status_code == 422 or r.status_code == 401

    def test_predict_with_valid_key(self):
        r = client.post("/api/v1/predict", json=VALID_PAYLOAD, headers=HEADERS)
        assert r.status_code == 200
        data = r.json()
        assert "risk_probability" in data
        assert "decision" in data
        assert "confidence" in data
        assert "confidence_interval_95" in data
        assert 0.0 <= data["risk_probability"] <= 1.0

    def test_predict_invalid_age(self):
        payload = {**VALID_PAYLOAD, "age_years": 15}
        r = client.post("/api/v1/predict", json=payload, headers=HEADERS)
        assert r.status_code == 422

    def test_predict_invalid_income(self):
        payload = {**VALID_PAYLOAD, "income_lpa": -5}
        r = client.post("/api/v1/predict", json=payload, headers=HEADERS)
        assert r.status_code == 422

    def test_predict_returns_prediction_id(self):
        r = client.post("/api/v1/predict", json=VALID_PAYLOAD, headers=HEADERS)
        assert r.status_code == 200
        assert "prediction_id" in r.json()

    def test_predict_decision_values(self):
        r = client.post("/api/v1/predict", json=VALID_PAYLOAD, headers=HEADERS)
        assert r.json()["decision"] in ["Approve", "Manual Review", "Reject"]


class TestBatchPredict:
    def test_batch_predict(self):
        payload = {"applicants": [VALID_PAYLOAD, VALID_PAYLOAD]}
        r = client.post("/api/v1/predict/batch", json=payload, headers=HEADERS)
        assert r.status_code == 200
        data = r.json()
        assert data["count"] == 2
        assert len(data["results"]) == 2

    def test_batch_empty_list(self):
        r = client.post("/api/v1/predict/batch", json={"applicants": []}, headers=HEADERS)
        assert r.status_code in [200, 422]


class TestExplain:
    def test_explain_returns_features(self):
        r = client.post("/api/v1/explain", json=VALID_PAYLOAD, headers=HEADERS)
        assert r.status_code == 200
        data = r.json()
        assert "top_features" in data
        assert len(data["top_features"]) > 0
        assert "feature" in data["top_features"][0]
        assert "shap_value" in data["top_features"][0]


class TestFeedback:
    def test_feedback_submission(self):
        # First get a prediction ID
        r = client.post("/api/v1/predict", json=VALID_PAYLOAD, headers=HEADERS)
        pred_id = r.json()["prediction_id"]

        fb_payload = {
            "prediction_id": pred_id,
            "feedback": "correct",
            "corrected_label": "",
            "notes": "Test feedback"
        }
        r2 = client.post("/api/v1/feedback", json=fb_payload, headers=HEADERS)
        assert r2.status_code == 200
        assert r2.json()["status"] == "recorded"

    def test_feedback_invalid_type(self):
        fb_payload = {
            "prediction_id": "test123",
            "feedback": "wrong_value",
        }
        r = client.post("/api/v1/feedback", json=fb_payload, headers=HEADERS)
        assert r.status_code == 422


class TestMetrics:
    def test_metrics_endpoint(self):
        r = client.get("/api/v1/metrics", headers=HEADERS)
        assert r.status_code == 200
        data = r.json()
        assert "total_predictions" in data

    def test_metrics_requires_auth(self):
        r = client.get("/api/v1/metrics")
        assert r.status_code in [401, 422]
