# =============================================================================
# Cricket ML — Makefile
# =============================================================================
# Usage (from project root, venv active):
#
#   make install        Install all dependencies
#   make data           Run EDA + feature engineering notebooks
#   make train          Run full training pipeline (all 4 models)
#   make train-gru      Train GRU score predictor only
#   make serve          Start FastAPI + Streamlit (two terminals auto-managed)
#   make api            Start FastAPI only (port 8000)
#   make dashboard      Start Streamlit only (port 8501)
#   make docker-up      Launch full stack via Docker Compose
#   make docker-down    Stop all Docker services
#   make lint           Run ruff linter
#   make clean          Remove compiled Python files
#   make help           Show this message
#
# Windows: use `nmake` or install `make` via `winget install GnuWin32.Make`
# =============================================================================

PYTHON   := python
PIP      := pip
UVICORN  := uvicorn
STREAMLIT:= streamlit

# Ports
API_PORT  := 8000
DASH_PORT := 8501

.PHONY: help install data train train-rolling train-gru train-fix-sklearn \
        serve api dashboard docker-up docker-down lint clean

# ---------------------------------------------------------------------------
# Default target
# ---------------------------------------------------------------------------
help:
	@echo ""
	@echo "Cricket ML — available targets"
	@echo "────────────────────────────────────────────"
	@echo "  make install          Install all dependencies"
	@echo "  make data             EDA + feature engineering"
	@echo "  make train            Full pipeline: all 4 models"
	@echo "  make train-rolling    Rolling features + retrain match winner (v3)"
	@echo "  make train-gru        Train GRU score predictor"
	@echo "  make train-fix-sklearn  Fix sklearn LabelEncoder version warning"
	@echo "  make api              Start FastAPI on :$(API_PORT)"
	@echo "  make dashboard        Start Streamlit on :$(DASH_PORT)"
	@echo "  make docker-up        Full stack via Docker Compose"
	@echo "  make docker-down      Stop all Docker services"
	@echo "  make lint             Run ruff linter"
	@echo "  make clean            Remove __pycache__ + .pyc files"
	@echo "────────────────────────────────────────────"
	@echo ""

# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------
install:
	$(PIP) install -r requirements.txt
	@echo ""
	@echo "Installing PyTorch 2.5.1 (CPU) — last version stable on Python 3.11.0"
	$(PIP) install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
	@echo ""
	@echo "Install complete. Run 'python -c \"import torch; print(torch.__version__)\"' to verify."

# ---------------------------------------------------------------------------
# Data pipeline
# ---------------------------------------------------------------------------
data:
	$(PYTHON) notebooks/01_eda.py
	$(PYTHON) notebooks/02_feature_engineering.py

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

## Full pipeline: base models → rolling features → fix sklearn warning
train:
	$(PYTHON) notebooks/03_modeling.py
	$(PYTHON) notebooks/04_rolling_features.py
	$(PYTHON) notebooks/06_gru_score_predictor.py
	$(PYTHON) notebooks/05_fix_sklearn_warning.py
	@echo ""
	@echo "All models trained. Restart uvicorn to load updated models."

## Rolling features + match winner v3 retrain only
train-rolling:
	$(PYTHON) notebooks/04_rolling_features.py

## GRU score predictor only
train-gru:
	$(PYTHON) notebooks/06_gru_score_predictor.py

## Fix sklearn LabelEncoder version warning on win_probability.pkl
train-fix-sklearn:
	$(PYTHON) notebooks/05_fix_sklearn_warning.py

# ---------------------------------------------------------------------------
# Serving (local, no Docker)
# ---------------------------------------------------------------------------

## FastAPI on :8000  (blocking)
api:
	$(UVICORN) api.main:app --host 0.0.0.0 --port $(API_PORT)

## Streamlit on :8501  (blocking)
dashboard:
	$(STREAMLIT) run streamlit_app.py

# ---------------------------------------------------------------------------
# Docker
# ---------------------------------------------------------------------------
docker-up:
	docker compose up --build

docker-down:
	docker compose down

# ---------------------------------------------------------------------------
# Code quality
# ---------------------------------------------------------------------------
lint:
	ruff check src/ api/ notebooks/ streamlit_app.py

# ---------------------------------------------------------------------------
# Clean
# ---------------------------------------------------------------------------
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@echo "Cleaned."