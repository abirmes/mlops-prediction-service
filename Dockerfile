FROM python:3.9-slim

WORKDIR /app

# Copier et installer dépendances
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code
COPY api/ /app/api/

# Exposer le port
EXPOSE 8000

# Lancer l'API
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]