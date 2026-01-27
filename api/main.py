from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest
from fastapi.responses import Response
import mlflow.pyfunc
import time
import os

# ============================================
# INITIALISATION
# ============================================
app = FastAPI(
    title="MLOps API",
    description="API de prédiction avec monitoring",
    version="1.0.0"
)

# Variables globales
model = None

# ============================================
# MÉTRIQUES PROMETHEUS
# ============================================
predictions_total = Counter(
    'predictions_total',
    'Nombre total de prédictions'
)

prediction_duration = Histogram(
    'prediction_duration_seconds',
    'Temps de prédiction en secondes'
)

errors_total = Counter(
    'errors_total',
    'Nombre total d\'erreurs'
)

# ============================================
# CHARGEMENT DU MODÈLE AU DÉMARRAGE
# ============================================
@app.on_event("startup")
def load_model():
    global model
    
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(mlflow_uri)
    
    model_name = os.getenv("MODEL_NAME", "MonModele")
    model_stage = os.getenv("MODEL_STAGE", "Production")
    
    try:
        print(f"🔄 Chargement du modèle depuis MLflow...")
        print(f"   URI: {mlflow_uri}")
        print(f"   Modèle: {model_name}")
        print(f"   Stage: {model_stage}")
        
        model = mlflow.pyfunc.load_model(f"models:/{model_name}/{model_stage}")
        
        print(f"✅ Modèle {model_name} ({model_stage}) chargé avec succès!")
        
    except Exception as e:
        print(f"❌ Erreur lors du chargement du modèle: {e}")
        print("⚠️  L'API va démarrer mais les prédictions échoueront")
        model = None

# ============================================
# ENDPOINTS
# ============================================

@app.get("/")
def root():
    """Page d'accueil de l'API"""
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
    """Vérification de l'état de l'API"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "mlflow_uri": os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    }

# ============================================
# MODÈLE DE DONNÉES (ADAPTEZ À VOTRE MODÈLE)
# ============================================
class PredictionInput(BaseModel):
    """
    Schéma des données d'entrée pour la prédiction.
    
    ⚠️ IMPORTANT: Adaptez les features selon VOTRE modèle !
    Exemple ci-dessous avec 4 features génériques.
    """
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
    """Schéma de la réponse"""
    prediction: float
    duration_ms: float
    model_version: str

# ============================================
# ENDPOINT DE PRÉDICTION
# ============================================
@app.post("/predict", response_model=PredictionOutput)
def predict(data: PredictionInput):
    """
    Faire une prédiction avec le modèle chargé.
    
    Args:
        data: Données d'entrée au format PredictionInput
        
    Returns:
        PredictionOutput: Prédiction et métadonnées
    """
    start_time = time.time()
    
    # Vérifier que le modèle est chargé
    if model is None:
        errors_total.inc()
        raise HTTPException(
            status_code=503,
            detail="Le modèle n'est pas chargé. Vérifiez MLflow."
        )
    
    try:
        # Préparer les données (adapter selon votre modèle)
        input_data = [[
            data.Pregnancies,
            data.BloodPressure,
            data.SkinThickness,
            data.Insulin,
            data.DiabetesPedigreeFunction,
            data.BMI,
            data.Age
        ]]
        
        # Faire la prédiction
        prediction = model.predict(input_data)
        
        # Calculer la durée
        duration = time.time() - start_time
        
        # Incrémenter les compteurs Prometheus
        predictions_total.inc()
        prediction_duration.observe(duration)
        
        # Retourner le résultat
        return PredictionOutput(
            prediction=float(prediction[0]),
            duration_ms=round(duration * 1000, 2),
            model_version=os.getenv("MODEL_NAME", "MonModele")
        )
        
    except Exception as e:
        errors_total.inc()
        duration = time.time() - start_time
        prediction_duration.observe(duration)
        
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la prédiction: {str(e)}"
        )

# ============================================
# ENDPOINT MÉTRIQUES (PROMETHEUS)
# ============================================
@app.get("/metrics")
def metrics():
    """
    Expose les métriques pour Prometheus.
    
    Métriques disponibles:
    - predictions_total: Nombre total de prédictions
    - prediction_duration_seconds: Temps de prédiction
    - errors_total: Nombre d'erreurs
    """
    return Response(
        content=generate_latest(),
        media_type="text/plain"
    )