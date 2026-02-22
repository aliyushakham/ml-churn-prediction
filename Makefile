.PHONY: help install train api docker-build docker-run test clean
help:
	@echo "Available commands:"
	@echo "  install      Install dependencies"
	@echo "  train        Run training pipeline"
	@echo "  api          Start FastAPI server"
	@echo "  docker-build Build Docker image"
	@echo "  docker-run   Run Docker container"
	@echo "  test         Run tests"
	@echo "  clean        Clean artifacts"
cat > Makefile << 'EOF'
.PHONY: help install train api docker-build docker-run test clean

help:
	@echo "Available commands:"
	@echo "  install      Install dependencies"
	@echo "  train        Run training pipeline"
	@echo "  api          Start FastAPI server"
	@echo "  docker-build Build Docker image"
	@echo "  docker-run   Run Docker container"
	@echo "  test         Run tests"
	@echo "  clean        Clean artifacts"

install:
	pip install -r requirements.txt

train:
	python -m src.pipeline.training_pipeline

api:
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

docker-build:
	docker build -t churn-prediction .

docker-run:
	docker run -p 8000:8000 churn-prediction

test:
	pytest tests/ -v --cov=src

clean:
	rm -rf artifacts/models/*
	rm -rf artifacts/metrics/*
	rm -rf __pycache__
	rm -rf .pytest_cache
	rm -rf logs/*.log
