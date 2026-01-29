from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest
from fastapi.responses import Response
import mlflow.pyfunc
import time

# Créer l'app FastAPI
app = FastAPI(
    title="MLOps API",
    description="API de prédiction avec monitoring",
    version="1.0.0"
)

# Variables globales
model = None
MODEL_NAME = "diabetes_logistic_regression"
MLFLOW_URI = "http://mlflow:5000"  # tracking server MLflow

# Métriques Prometheus
predictions_total = Counter('predictions_total', 'Nombre total de prédictions')
prediction_duration = Histogram('prediction_duration_seconds', 'Temps de prédiction en secondes')
errors_total = Counter('errors_total', 'Nombre total d\'erreurs')

# Chargement du modèle au démarrage
@app.on_event("startup")
def load_model():
    global model
    try:
        client = MlflowClient("http://mlflow:5000")
        latest_versions = client.get_latest_versions("diabetes_logistic_regression", stages=["Production"])
        version = latest_versions[0].version
        model = mlflow.pyfunc.load_model(f"models:/diabetes_logistic_regression/{version}")
    except Exception as e:
        model = None

class PredictionInput(BaseModel):
    feature1: float
    feature2: float
    feature3: float
    feature4: float

    class Config:
        schema_extra = {
            "example": {
                "feature1": 5.1,
                "feature2": 3.5,
                "feature3": 1.4,
                "feature4": 0.2
            }
        }

class PredictionOutput(BaseModel):
    prediction: float
    duration_ms: float
    model_version: str

# Endpoints
@app.get("/")
def root():
    return {
        "message": "Bienvenue sur l'API MLOps",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "metrics": "/metrics",
            "docs": "/docs"
        }
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "mlflow_uri": MLFLOW_URI
    }

@app.post("/predict", response_model=PredictionOutput)
def predict(data: PredictionInput):
    start_time = time.time()
    
    if model is None:
        errors_total.inc()
        raise HTTPException(
            status_code=503,
            detail="Le modèle n'est pas chargé. Vérifiez MLflow."
        )
    
    try:
        input_data = [[data.feature1, data.feature2, data.feature3, data.feature4]]
        prediction = model.predict(input_data)
        duration = time.time() - start_time

        predictions_total.inc()
        prediction_duration.observe(duration)

        return PredictionOutput(
            prediction=float(prediction[0]),
            duration_ms=round(duration * 1000, 2),
            model_version=MODEL_NAME
        )
    except Exception as e:
        errors_total.inc()
        duration = time.time() - start_time
        prediction_duration.observe(duration)
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la prédiction: {str(e)}"
        )

@app.get("/metrics")
def metrics():
    return Response(
        content=generate_latest(),
        media_type="text/plain"
    )
