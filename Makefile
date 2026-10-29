.PHONY: help install train test test-features test-data test-api run-backend run-frontend docker-up docker-down

help:
	@echo "TacticRadar Development Commands:"
	@echo "  make install       Install python requirements and frontend dependencies"
	@echo "  make train         Run full ML training and evaluation pipeline"
	@echo "  make train-quick   Run fast training pipeline for sanity testing"
	@echo "  make test          Run all unit tests across features, data, models, and API"
	@echo "  make run-backend   Start FastAPI development server on port 8000"
	@echo "  make run-frontend  Start Vite React development server on port 5173"
	@echo "  make docker-up     Build and start Docker containers"
	@echo "  make docker-down   Stop Docker containers"

install:
	python -m pip install -r requirements.txt
	cd frontend && npm install

train:
	python -m src.train --config configs/default.yaml

train-quick:
	python -m src.train --quick

test:
	python -m pytest tests/ backend/tests/ -v

test-features:
	python -m pytest tests/test_features.py -v

test-data:
	python -m pytest tests/test_data.py -v

test-api:
	python -m pytest backend/tests/test_api.py -v

run-backend:
	uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload

run-frontend:
	cd frontend && npm run dev

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down
