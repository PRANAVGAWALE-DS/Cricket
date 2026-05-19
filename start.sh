#!/bin/bash
# =============================================================================
# start.sh — HF Spaces container startup
# 1. Downloads models + parquets from HF Hub dataset repo
#    (skipped when LOCAL_ARTEFACTS=1 for local Docker testing)
# 2. Starts FastAPI on :8000 (background)
# 3. Waits for API health check to pass
# 4. Starts Streamlit on :7860 (foreground — keeps container alive)
# =============================================================================

set -euo pipefail

ARTEFACTS_REPO="PRANAVGAWALE-DS/cricket-ml-artefacts"
MAX_WAIT=60   # seconds to wait for API health check

echo "========================================"
echo "  Cricket ML — Space startup"
echo "========================================"

# ---------------------------------------------------------------------------
# Step 1 — Download artefacts from HF Hub (skip in local test mode)
# ---------------------------------------------------------------------------
if [ "${LOCAL_ARTEFACTS:-0}" = "1" ]; then
    echo ""
    echo "[1/3] LOCAL_ARTEFACTS=1 — skipping HF Hub download"
    echo "  Using artefacts from mounted /app/models and /app/data/processed"
else
    echo ""
    echo "[1/3] Downloading artefacts from HF Hub: ${ARTEFACTS_REPO}"

    python - <<'PYEOF'
import os, shutil, sys
from pathlib import Path

try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("ERROR: huggingface_hub not installed", file=sys.stderr)
    sys.exit(1)

REPO_ID = "PRANAVGAWALE-DS/cricket-ml-artefacts"

print(f"  Downloading from {REPO_ID}...")
local = snapshot_download(
    repo_id=REPO_ID,
    repo_type="dataset",
    local_dir="/tmp/cricket_artefacts",
)
print(f"  Snapshot at: {local}")

src_models = Path(local) / "models"
if src_models.exists():
    for f in src_models.iterdir():
        dst = Path("/app/models") / f.name
        shutil.copy2(f, dst)
        print(f"  Model:   {f.name} ({f.stat().st_size // 1024} KB)")
else:
    print("  WARNING: no models/ directory found in artefacts repo", file=sys.stderr)

src_proc = Path(local) / "data" / "processed"
if src_proc.exists():
    for f in src_proc.iterdir():
        dst = Path("/app/data/processed") / f.name
        shutil.copy2(f, dst)
        print(f"  Parquet: {f.name} ({f.stat().st_size // 1024} KB)")
else:
    print("  WARNING: no data/processed/ directory found in artefacts repo", file=sys.stderr)

print("  Download complete.")
PYEOF
fi

# ---------------------------------------------------------------------------
# Step 2 — Start FastAPI
# ---------------------------------------------------------------------------
echo ""
echo "[2/3] Starting FastAPI on :8000..."

uvicorn api.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1 \
    --log-level info &
API_PID=$!

echo "  Waiting for API health check (max ${MAX_WAIT}s)..."
for i in $(seq 1 $MAX_WAIT); do
    if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
        echo "  API ready after ${i}s (PID ${API_PID})"
        break
    fi
    if [ $i -eq $MAX_WAIT ]; then
        echo "  WARNING: API health check timed out after ${MAX_WAIT}s"
    fi
    sleep 1
done

# ---------------------------------------------------------------------------
# Step 3 — Start Streamlit (foreground)
# ---------------------------------------------------------------------------
echo ""
echo "[3/3] Starting Streamlit on :7860..."
echo "========================================"

exec streamlit run streamlit_app.py \
    --server.port=7860 \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false