.PHONY: help up down logs test

help:
	@echo "Commandes disponibles:"
	@echo "  make up      - Démarrer tous les services"
	@echo "  make down    - Arrêter tous les services"
	@echo "  make logs    - Voir les logs"
	@echo "  make test    - Lancer les tests"

up:
	docker-compose up -d
	@echo "✓ Services démarrés!"
	@echo "API:        http://localhost:8000/docs"
	@echo "MLflow:     http://localhost:5000"
	@echo "Prometheus: http://localhost:9090"
	@echo "Grafana:    http://localhost:3000 (admin/admin)"

down:
	docker-compose down

logs:
	docker-compose logs -f

test:
	docker-compose exec api pytest api/tests/ -v