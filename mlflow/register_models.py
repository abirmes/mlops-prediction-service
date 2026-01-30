import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import f1_score, roc_auc_score
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import os

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
mlflow.set_experiment("diabetes_prediction_experiment")
print(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")

data = pd.read_csv(
    "/app/mlflow/dataset-diabete-68e2810ab0d7e949117525.csv"
).drop("Unnamed: 0", axis=1)

# Imputation
data_copy = data.copy()
for col in data.columns:
    if col != "Pregnancies":
        data_copy[col] = data_copy[col].replace(0, np.nan)
data = pd.DataFrame(KNNImputer(n_neighbors=5).fit_transform(data_copy), columns=data_copy.columns)

# Log transform
for col in ['BloodPressure','Insulin','DiabetesPedigreeFunction']:
    data[col] = np.log1p(data[col])

# Outliers simplifiés (garder logique précédente)
def cap_outliers(df, col, factor_upper=2, factor_lower=1.5):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    upper = Q3 + factor_upper*IQR
    lower = Q1 - factor_lower*IQR
    df[col] = np.where(df[col]>upper, upper, df[col])
    df[col] = np.where(df[col]<lower, lower, df[col])
    return df

for col in ['Glucose','BMI','Age']:
    data = cap_outliers(data, col)

# Feature engineering
data['BMI_Age'] = data['BMI'] * data['Age']
data['Glucose_Insulin'] = data['Glucose'] * data['Insulin']

# **SAVE THE CLEANED DATA HERE**
folder = "/app/cleaned_data"
X = data.drop('Outcome', axis=1)
y = data['Outcome']
os.makedirs(folder, exist_ok=True)
data.to_csv(f"{folder}/data.csv", index=False)
print(f"Cleaned data saved to {folder}/data.csv")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

models = {
    'Logistic Regression': (LogisticRegression(max_iter=1000, random_state=42), {'model__C': [0.01, 0.1, 1], 'model__penalty': ['l2']}),
    'Random Forest': (RandomForestClassifier(random_state=42), {'model__n_estimators': [50, 100], 'model__max_depth': [5, 10]}),
    'Gradient Boosting': (GradientBoostingClassifier(random_state=42), {'model__n_estimators': [50, 100], 'model__learning_rate': [0.01, 0.1]}),
    'SVM': (SVC(probability=True, random_state=42), {'model__C': [0.1, 1], 'model__kernel': ['rbf']}),
    'Decision Tree': (DecisionTreeClassifier(random_state=42), {'model__max_depth': [5, 10], 'model__min_samples_split': [2, 5]}),
    'XGBoost': (XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42), {'model__n_estimators': [50, 100], 'model__learning_rate': [0.01, 0.1]})
}

best_model_info = {'model_name': None, 'roc_auc': 0, 'mlflow_run_id': None}

for name, (model, param_grid) in models.items():
    with mlflow.start_run(run_name=name):
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('sampler', RandomOverSampler(random_state=42)),
            ('model', model)
        ])
        
        grid = GridSearchCV(pipeline, param_grid, cv=StratifiedKFold(n_splits=3), scoring='roc_auc', n_jobs=-1)
        grid.fit(X_train, y_train)
        
        y_pred = grid.predict(X_test)
        y_proba = grid.predict_proba(X_test)[:,1]
        
        test_f1 = f1_score(y_test, y_pred)
        test_roc = roc_auc_score(y_test, y_proba)
        
        mlflow.log_params(grid.best_params_)
        mlflow.log_metric("test_f1", test_f1)
        mlflow.log_metric("test_roc_auc", test_roc)
        mlflow.sklearn.log_model(grid.best_estimator_, f"diabetes_{name.lower().replace(' ','_')}", registered_model_name=f"diabetes_{name.lower().replace(' ','_')}")
        
        print(f"{name}: F1={test_f1:.4f}, ROC-AUC={test_roc:.4f}")
        
        if test_roc > best_model_info['roc_auc']:
            best_model_info = {'model_name': name, 'roc_auc': test_roc, 'mlflow_run_id': mlflow.active_run().info.run_id}

from mlflow.tracking import MlflowClient
client = MlflowClient()

model_registry_name = f"diabetes_{best_model_info['model_name'].lower().replace(' ','_')}"
latest_versions = client.get_latest_versions(name=model_registry_name)

if latest_versions:
    version_number = latest_versions[-1].version
else:
    version_number = 1

client.transition_model_version_stage(
    name=model_registry_name,
    version=version_number,
    stage="Production",
    archive_existing_versions=True
)

print(f"\nBest model: {best_model_info['model_name']} promoted to Production!")