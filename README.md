# Projet MLOps - Déploiement d'un Modèle ML

## 🎯 Objectif

Mettre en place une chaîne MLOps complète pour déployer un modèle de Machine Learning avec :
- ✅ MLflow (versioning & registry)
- ✅ FastAPI (API REST)
- ✅ GitHub Actions (CI/CD)
- ✅ Prometheus & Grafana (monitoring)
- ✅ Docker (conteneurisation)

## 📦 Architecture
```
┌─────────────────────────────────────────────┐
│              MLOPS STACK                    │
├─────────────────────────────────────────────┤
│                                             │
│  MLflow (5000) ← API (8000) → Prometheus    │
│                       ↓                     │
│                   Grafana (3001)            │
│                                             │
└─────────────────────────────────────────────┘
```

## 🚀 Démarrage Rapide

### Prérequis
- Docker & Docker Compose
- Python 3.9+
- Make (optionnel)

### Installation
```bash
# 1. Cloner le projet
git clone <votre-repo>
cd MLOPS-PREDICTION-SERVICE

# 2. Démarrer tous les services
docker-compose up -d
```

### Accès aux services

- 🌐 **API Swagger**: http://localhost:8000/docs
- 📊 **MLflow UI**: http://localhost:5000
- 📈 **Prometheus**: http://localhost:9090
- 📉 **Grafana**: http://localhost:3001 (admin/admin)

## 📝 Utilisation

### 1. Enregistrer le modèle dans MLflow
```bash
# Démarrer MLflow
mlflow ui --host 0.0.0.0 --port 5000

# Dans un autre terminal
python mlflow/register_models.py
```

### 2. Tester l'API
```bash
# Health check
curl http://localhost:8000/health

# Prédiction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "feature1": 5.1,
    "feature2": 3.5,
    "feature3": 1.4,
    "feature4": 0.2
  }'

# Métriques Prometheus
curl http://localhost:8000/metrics
```

### 3. Visualiser dans Grafana

1. Ouvrir http://localhost:3000
2. Login: admin/admin
3. Voir le dashboard "MLOps Monitoring"

## 🧪 Tests
```bash
# Lancer les tests
pytest api/tests/ -v

# Avec coverage
pytest api/tests/ --cov=api --cov-report=html
```

## 🛠️ Commandes Utiles
```bash
# Démarrer
make up

# Arrêter
make down

# Voir les logs
make logs

# Redémarrer
make restart

# Tests
make test
```

## 📊 Métriques Exposées

L'API expose les métriques suivantes pour Prometheus :

- `predictions_total` : Nombre total de prédictions
- `prediction_duration_seconds` : Temps de prédiction
- `errors_total` : Nombre d'erreurs

## 🏗️ Structure du Projet
```
mlops-project/
├── api/                    # Code API FastAPI
├── mlflow/                 # Scripts MLflow
├── monitoring/             # Config Prometheus & Grafana
├── .github/workflows/      # CI/CD
├── docker-compose.yml      # Orchestration
├── Dockerfile              # Image Docker
└── README.md
```

