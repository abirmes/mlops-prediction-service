import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import pandas as pd
from main import app, PredictionInput

client = TestClient(app)


@patch('main.model')
def test_predict_success(mock_model):
    # Mock du modèle
    mock_model.predict.return_value = [1.0]
    
    payload = {
        "Pregnancies": 6,
        "Glucose": 148,
        "BloodPressure": 72,
        "SkinThickness": 35,
        "Insulin": 0,
        "BMI": 33.6,
        "DiabetesPedigreeFunction": 0.627,
        "Age": 50
    }
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "prediction" in response.json()

def test_predict_model_not_loaded():
    with patch('main.model', None):
        payload = {
            "Pregnancies": 6,
            "Glucose": 148,
            "BloodPressure": 72,
            "SkinThickness": 35,
            "Insulin": 0,
            "BMI": 33.6,
            "DiabetesPedigreeFunction": 0.627,
            "Age": 50
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 503
