from mlflow.tracking import MlflowClient

MLFLOW_URI = "http://mlflow:5000"
MODEL_NAME = "diabetes_logistic_regression"
THRESHOLDS = {
    "accuracy": 0.75,
    "f1_score": 0.7,
    "roc_auc": 0.75
}

def validate_model_metrics(model_name: str = MODEL_NAME, thresholds: dict = THRESHOLDS):
    client = MlflowClient(MLFLOW_URI)
    latest_versions = client.get_latest_versions(model_name, stages=["Production"])
    if not latest_versions:
        raise ValueError(f"No Production version found for model '{model_name}'")
    run_id = latest_versions[0].run_id
    metrics = client.get_run(run_id).data.metrics
    passed = True
    for metric_name, threshold in thresholds.items():
        value = metrics.get(metric_name)
        if value is None or value < threshold:
            passed = False
    return metrics, passed


if __name__ == "__main__":
    metrics, passed = validate_model_metrics()
    if not passed:
        exit(1)
    exit(0)
