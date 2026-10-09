#!/usr/bin/env bash
# Start the LeanSearch v2 standard-mode FastAPI server.
#
# Paths and tuning are read from ../config.yaml; override any of them by
# exporting the matching environment variable before running this script
# (VECTORDB_DIR, EMBEDDING_MODEL_PATH, RERANKER_MODEL_PATH, NUM_GPUS,
# GPU_MEMORY_UTILIZATION, HOST, PORT).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

# Allow running without `pip install -e .`
export PYTHONPATH="${REPO_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-8000}"

exec python -m uvicorn leansearchv2.server:app --host "$HOST" --port "$PORT"
