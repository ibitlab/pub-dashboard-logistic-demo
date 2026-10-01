#!/bin/bash
# Warehouse Activity Intelligence (RTMPose) — Quick Start

set -e

cd "$(dirname "$0")"

echo "=== Warehouse Activity Intelligence (RTMPose) ==="
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is required"
    exit 1
fi

# Create venv if not exists
if [ ! -d "venv" ]; then
    echo "[1/4] Creating virtual environment..."
    python3 -m venv venv
fi

echo "[2/4] Installing dependencies..."
source venv/bin/activate
pip install -q -r backend/requirements.txt

# Check RTMPose models (auto-download on first use)
echo "[3/4] Checking models..."

REID_MODEL="data/models/osnet_ain_x1_0.onnx"
# The export writes weights to a separate .onnx.data file; both must be present
if [ -f "$REID_MODEL" ] && [ -f "$REID_MODEL.data" ]; then
    echo "       ReID model: found (person re-identification enabled)"
else
    echo "       ReID model: not found (person re-identification disabled)"
    echo "       To enable: run scripts/export_reid_model.py in a separate venv with torch (see README)"
fi

echo "       RTMPose: downloads ~150MB to ~/.cache/rtmlib/ on the first server start"

PORT=${1:-8001}

echo ""
echo "[4/4] Starting server on http://localhost:$PORT"
echo "       API docs: http://localhost:$PORT/docs"
echo "       Pose: RTMPose (COCO 17-point) | ReID: OSNet-AIN | License: Apache 2.0 + MIT"
echo ""

cd backend
uvicorn main:app --host 0.0.0.0 --port "$PORT" --reload
