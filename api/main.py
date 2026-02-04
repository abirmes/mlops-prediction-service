from fastapi import FastAPI, HTTPException
from pydantic import BaseModel , Field
from prometheus_client import Counter, Histogram, generate_latest
from fastapi.responses import Response
import mlflow.pyfunc
import pandas as pd
from mlflow.tracking import MlflowClient  # ← ADD THIS
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
stage="Production"

# Métriques Prometheus
predictions_total = Counter('predictions_total', 'Nombre total de prédictions')
prediction_duration = Histogram('prediction_duration_seconds', 'Temps de prédiction en secondes')
errors_total = Counter('errors_total', 'Nombre total d\'erreurs')

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
        print(f"✅ Model loaded: {MODEL_NAME}  v{stage}")
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        model = None

class PredictionInput(BaseModel):

        Pregnancies: float = Field(..., description="Nombre de grossesses", ge=0)
        Glucose: float = Field(..., description="Niveau de glucose", ge=0)
        BloodPressure: float = Field(..., description="Pression artérielle (mm Hg)", ge=0)
        SkinThickness: float = Field(..., description="Épaisseur de la peau (mm)", ge=0)
        Insulin: float = Field(..., description="Niveau d'insuline (mu U/ml)", ge=0)
        BMI: float = Field(..., description="Indice de masse corporelle", ge=0)
        DiabetesPedigreeFunction: float = Field(..., description="Fonction de pedigree du diabète", ge=0)
        Age: float = Field(..., description="Âge", ge=0) # You need 8 features for diabetes prediction!

        class Config:
            schema_extra = {
                "example": {
                    "Pregnancies": 6,
                    "Glucose": 148,
                    "BloodPressure": 72,
                    "SkinThickness": 35,
                    "Insulin": 0,
                    "BMI": 33.6,
                    "DiabetesPedigreeFunction": 0.627,
                    "Age": 50,
                }
            }
            

class PredictionOutput(BaseModel):
    prediction: float
    duration_ms: float
    model_version: str

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
    # return model
    # ← REMOVE "return model" FROM HERE!
    df = pd.DataFrame([data.dict()])

    column_order = [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
        ]
    df = df[column_order]
    if model is None:
        errors_total.inc()
        raise HTTPException(
            status_code=503,
            detail="Le modèle n'est pas chargé. Vérifiez MLflow."
        )
    
    try:
        
        prediction = model.predict(df)
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