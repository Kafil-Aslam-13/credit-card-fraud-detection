"""Tests the API layer using FastAPI's TestClient.

Only the /health endpoint is tested without mocking, since /predict
needs a real trained model+scaler on disk -- that's covered by an
integration test once artifacts exist (see README for how to run
end-to-end tests after `python train.py`).
"""

from fastapi.testclient import TestClient

from app.api.main import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"
    assert data["model"] == "credit-card-fraud-detection"
