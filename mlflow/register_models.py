import os
import pickle
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MODEL_NAME = os.getenv("MODEL_NAME")

if not MLFLOW_TRACKING_URI or not MODEL_NAME:
    raise RuntimeError("Variables d'environnement MLflow manquantes")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("mon_projet_ml")

model_path = os.path.join(os.path.dirname(__file__), "model.pkl")

with open(model_path, "rb") as f:
    model = pickle.load(f)

with mlflow.start_run(run_name="register_model"):

    mlflow.log_metric("accuracy", 0.99)
    mlflow.log_param("algorithm", "LogisticRegression")

    mlflow.sklearn.log_model(
        sk_model=model,
        name="model",
        registered_model_name=MODEL_NAME
    )

client = MlflowClient()

versions = client.search_model_versions(f"name='{MODEL_NAME}'")
latest_version = max(int(v.version) for v in versions)

client.set_registered_model_alias(
    name=MODEL_NAME,
    alias="prod",
    version=latest_version
)

print(f"Modèle {MODEL_NAME} défini comme PROD")
