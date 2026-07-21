#!/bin/bash
# Start the BTC Signal Dashboard backend on VPS
# Usage: ./start.sh [port]

PORT=${1:-8000}
cd "$(dirname "$0")"

echo "Starting BTC Signal Dashboard on port $PORT..."
python3 -m uvicorn api.main:app --host 0.0.0.0 --port "$PORT" --workers 1
