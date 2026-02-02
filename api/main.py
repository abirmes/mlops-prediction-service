from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from fastapi.responses import Response
import mlflow.pyfunc
import pandas as pd
from mlflow.tracking import MlflowClient
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
MLFLOW_URI = "http://mlflow:5000"
stage = "Production"


predictions_total = Counter(
    'predictions_total',
    'Nombre total de prédictions'
)

prediction_duration = Histogram(
    'prediction_duration_seconds',
    'Temps de prédiction en secondes',
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
)

errors_total = Counter(
    'errors_total',
    'Nombre total d\'erreurs'
)

api_requests_total = Counter(
    'api_requests_total',
    'Nombre total de requêtes API',
    ['endpoint', 'method', 'status']
)

model_info = Gauge(
    'model_info',
    'Informations sur le modèle',
    ['model_name', 'stage']
)

# Chargement du modèle au démarrage
@app.on_event("startup")
def load_model():
    global model
    try:
        client = MlflowClient(MLFLOW_URI)
        latest_versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
        print("ASCascascascascascascasca1111111111111")

        print(latest_versions)
        if not latest_versions:
            print(f"⚠️ No Production version found for {MODEL_NAME}")
            return
        
        print("ASCascascascascascascasca2222222222éé")
        model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/Production")

        model_info.labels(
            model_name=MODEL_NAME,
            stage=stage
        ).set(1)

        print(f" Model loaded: {MODEL_NAME} v{stage}")

    except Exception as e:
        print(f" Error loading model: {str(e)}")
        model = None

class PredictionInput(BaseModel):
    Pregnancies: float = Field(..., ge=0)
    Glucose: float = Field(..., ge=0)
    BloodPressure: float = Field(..., ge=0)
    SkinThickness: float = Field(..., ge=0)
    Insulin: float = Field(..., ge=0)
    BMI: float = Field(..., ge=0)
    DiabetesPedigreeFunction: float = Field(..., ge=0)
    Age: float = Field(..., ge=0)

class PredictionOutput(BaseModel):
    prediction: float
    duration_ms: float
    model_version: str

@app.get("/")
def root():
    api_requests_total.labels(endpoint="/", method="GET", status="200").inc()
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
    api_requests_total.labels(endpoint="/health", method="GET", status="200").inc()
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
        api_requests_total.labels(endpoint="/predict", method="POST", status="503").inc()
        raise HTTPException(
            status_code=503,
            detail="Le modèle n'est pas chargé. Vérifiez MLflow."
        )

    try:
        df = pd.DataFrame([data.dict()])

        column_order = [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
        ]
        df = df[column_order]

        prediction = model.predict(df)

        duration = time.time() - start_time

        predictions_total.inc()
        prediction_duration.observe(duration)
        api_requests_total.labels(endpoint="/predict", method="POST", status="200").inc()

        return PredictionOutput(
            prediction=float(prediction[0]),
            duration_ms=round(duration * 1000, 2),
            model_version=MODEL_NAME
        )

    except Exception as e:
        errors_total.inc()
        prediction_duration.observe(time.time() - start_time)
        api_requests_total.labels(endpoint="/predict", method="POST", status="500").inc()
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
