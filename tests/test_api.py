from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_predict_valid_input():
    response = client.post("/predict", json={
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    })
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "class_name" in data
    assert "confidence" in data
    assert 0.0 <= data["confidence"] <= 1.0

def test_predict_invalid_input():
    response = client.post("/predict", json={
        "sepal_length": -1,  # Invalid — must be > 0
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    })
    assert response.status_code == 422  # Validation error

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200