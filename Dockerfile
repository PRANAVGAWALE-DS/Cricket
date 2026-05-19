# =============================================================================
# Dockerfile — HF Spaces Docker deployment
# Runs FastAPI (:8000, internal) + Streamlit (:7860, public)
# in a single container. HF Spaces proxies port 7860 publicly.
# Models + parquets are downloaded from HF Hub at container startup.
# =============================================================================

FROM python:3.11-slim

WORKDIR /app

# System deps: libgomp1 for LightGBM/XGBoost, curl for health-check polling
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 curl git \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps (lean serving subset, no training tools)
COPY requirements_tier3.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements_tier3.txt && \
    pip install --no-cache-dir \
        torch==2.5.1 \
        --index-url https://download.pytorch.org/whl/cpu

# Copy source code
COPY src/       src/
COPY api/       api/
COPY streamlit_app.py .

# Pre-create runtime directories (models + data populated at startup)
RUN mkdir -p data/processed data/raw models

# Copy and permission startup script
COPY start.sh .
RUN chmod +x start.sh

# HF Spaces requires port 7860 to be exposed
EXPOSE 7860

ENV PYTHONPATH=/app
# Signal to streamlit_app.py and api/main.py that we're running in a Space
ENV HF_SPACE=1

# HF Spaces runs containers as a non-root user — ensure write permissions
RUN chmod -R 777 /app/data /app/models

CMD ["./start.sh"]