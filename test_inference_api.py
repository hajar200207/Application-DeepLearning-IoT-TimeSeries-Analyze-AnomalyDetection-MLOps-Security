import pytest
from fastapi.testclient import TestClient
from app.inference_api import app

client = TestClient(app)

def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert "Inference API IoT" in response.text

def test_predict_lstm_format_error():
    payload = {
        "features": [1.0, 2.0, 3.0]  # format invalide, doit être 2D
    }
    response = client.post("/predict/lstm", json=payload)
    assert response.status_code == 400
    assert "Format invalide" in response.json()["detail"]

def test_predict_model_not_found():
    payload = {
        "features": [[0.1, 0.2], [0.3, 0.4]]
    }
    response = client.post("/predict/unknown_model", json=payload)
    assert response.status_code == 400
    assert response.json()["detail"] == "Unsupported model"

def test_mlflow_redirect_bad_host():
    headers = {"host": "localhost"}
    response = client.get("/mlflow", headers=headers)
    assert response.status_code == 400
    assert response.json()["detail"] == "Host non autorisé"
