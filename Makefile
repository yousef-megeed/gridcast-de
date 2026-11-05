.DEFAULT_GOAL := help
PY := .venv/bin/python
GRIDCAST := .venv/bin/gridcast
export MLFLOW_DISABLE_AGENT_HINT := 1

.PHONY: help setup data tune backtest evaluate analyze bundle reproduce test lint serve docker-build docker-up mlflow

help: ## Show available targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-14s %s\n", $$1, $$2}'

setup: ## Create the virtual environment and install everything
	uv venv --python 3.12
	uv pip install -e ".[train,dev]"

data: ## Download and validate SMARD and Open-Meteo data
	$(GRIDCAST) fetch

tune: ## Hyperparameter search on pre-test data
	$(GRIDCAST) tune ridge
	$(GRIDCAST) tune lightgbm
	$(GRIDCAST) tune attention_net

backtest: ## Walk-forward backtest of all models
	$(GRIDCAST) backtest all

evaluate: ## Score predictions and write results/
	$(GRIDCAST) evaluate

analyze: ## Create figures and the error analysis
	$(GRIDCAST) analyze --part report
	$(GRIDCAST) bundle --member lightgbm --output artifacts/explain
	$(GRIDCAST) analyze --part shap
	$(GRIDCAST) analyze --part attention

bundle: ## Train the final model and package it
	$(GRIDCAST) bundle

reproduce: data tune backtest evaluate bundle analyze ## Run every step from raw data to figures

# Separate processes: torch and LightGBM OpenMP runtimes can deadlock in one macOS process.
test: ## Run the test suite
	$(PY) -m pytest --ignore=tests/test_gbm.py -m "not network" --cov=gridcast --cov-report=term
	$(PY) -m pytest tests/test_gbm.py --cov=gridcast --cov-append --cov-report=term

lint: ## Run formatting, lint and type checks
	.venv/bin/ruff format --check src tests
	.venv/bin/ruff check src tests
	.venv/bin/mypy

serve: ## Run the API on http://127.0.0.1:8000
	$(GRIDCAST) serve

docker-build: ## Build the serving image
	docker compose build api

docker-up: ## Run the serving image
	docker compose up api

mlflow: ## Browse experiments on http://127.0.0.1:5000
	.venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db
