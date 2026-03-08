#!/bin/bash
set -e

# ============================================================
# RunPod Start Script for LTX-2 Video Generation Backend
# ============================================================

# Core paths
export LTX_APP_DATA_DIR="${LTX_APP_DATA_DIR:-/workspace}"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"

# Server config
export LTX_PORT="${LTX_PORT:-8000}"

# Allow RunPod proxy origins; individual dev origins are defaults in app_factory.py
# Add more origins as comma-separated values if needed:
# export LTX_CORS_ORIGINS="https://custom.example.com,http://localhost:5173"

# SageAttention (enabled by default on GPU pods)
export USE_SAGE_ATTENTION="${USE_SAGE_ATTENTION:-1}"

# Create required directories
mkdir -p "${LTX_APP_DATA_DIR}/models"
mkdir -p "${LTX_APP_DATA_DIR}/logs"
mkdir -p "${HF_HOME}"

echo "============================================================"
echo " LTX-2 Video Generation Server — RunPod"
echo "============================================================"
echo " LTX_APP_DATA_DIR : ${LTX_APP_DATA_DIR}"
echo " HF_HOME          : ${HF_HOME}"
echo " LTX_PORT         : ${LTX_PORT}"
echo " USE_SAGE_ATTENTION: ${USE_SAGE_ATTENTION}"
echo " RUNPOD_POD_ID    : ${RUNPOD_POD_ID:-<not set>}"
echo "============================================================"

# Start watchdog in background (auto-stops pod on idle)
if [ -n "${RUNPOD_POD_ID}" ] && [ -n "${RUNPOD_API_KEY}" ]; then
    echo "[start.sh] Starting idle watchdog..."
    python3 /app/watchdog.py &
    WATCHDOG_PID=$!
    echo "[start.sh] Watchdog PID: ${WATCHDOG_PID}"
else
    echo "[start.sh] RUNPOD_POD_ID or RUNPOD_API_KEY not set — watchdog disabled"
fi

# Launch the FastAPI backend
echo "[start.sh] Launching LTX-2 backend on 0.0.0.0:${LTX_PORT}..."
cd /app/backend
exec python3 ltx2_server.py
