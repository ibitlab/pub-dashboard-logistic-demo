#!/bin/bash
# Warehouse Activity Intelligence — Quick Start

set -e

cd "$(dirname "$0")"

echo "=== Warehouse Activity Intelligence ==="
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is required"
    exit 1
fi

# Create venv if not exists
if [ ! -d "venv" ]; then
    echo "[1/3] Creating virtual environment..."
    python3 -m venv venv
fi

echo "[2/3] Installing dependencies..."
source venv/bin/activate
pip install -q -r backend/requirements.txt

# Pose models are not in git (backend/models is ignored); fetch any that are missing
MODELS_URL="https://storage.googleapis.com/mediapipe-models/pose_landmarker"
for v in lite full heavy; do
    mkdir -p backend/models
    f="backend/models/pose_landmarker_$v.task"
    if [ ! -f "$f" ]; then
        echo "       Downloading pose_landmarker_$v.task..."
        curl -fsSL -o "$f" "$MODELS_URL/pose_landmarker_$v/float16/latest/pose_landmarker_$v.task"
    fi
done

PORT=${1:-8000}

echo "[3/3] Starting server on http://localhost:$PORT"
echo "       API docs: http://localhost:$PORT/docs"
echo ""

cd backend
uvicorn main:app --host 0.0.0.0 --port "$PORT" --reload
