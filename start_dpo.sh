#!/usr/bin/env bash
set -euo pipefail

#######################################
# Configuration
#######################################

PYTHON_VERSION="3.12"
UV_PYTHON=(uv run --python "$PYTHON_VERSION" python)

DEFAULT_CONFIG="configs/dpo_hass_qwen3_5_0_8b.yaml"
CONFIG_PATH="${1:-$DEFAULT_CONFIG}"

if [ $# -gt 0 ]; then
  shift
fi
EXTRA_ARGS=("$@")

#######################################
# Environment loading
#######################################

if [ -f ".env" ]; then
  set -a
  . ./.env
  set +a
fi

export MLFLOW_URI="${MLFLOW_TRACKING_URI:-https://mlflow.chezombor.com/ ()}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

#######################################
# Helpers
#######################################

require_uv() {
  command -v uv >/dev/null 2>&1 || {
    echo "Error: uv is required but not installed"
    exit 1
  }
}

is_local_mlflow() {
  case "$1" in
    *localhost*|*127.0.0.1*) return 0 ;;
    *) return 1 ;;
  esac
}

#######################################
# Preflight
#######################################

require_uv

#######################################
# Backend detection (uv‑managed Python)
#######################################

REQUIREMENTS_FILE="requirements.dpo-trl.txt"

if [ ! -f "$REQUIREMENTS_FILE" ]; then
  echo "Error: requirements file not found: $REQUIREMENTS_FILE"
  exit 1
fi

BACKEND="$("${UV_PYTHON[@]}" --with-requirements "$REQUIREMENTS_FILE" - <<'PY' "$CONFIG_PATH"
import sys, yaml

with open(sys.argv[1], "r", encoding="utf-8") as f:
    raw = yaml.safe_load(f) or {}

identity = raw.get("identity") or {}
print(str(identity.get("backend", "trl")).strip().lower())
PY
)"

if [ "$BACKEND" != "trl" ]; then
  echo "Error: unsupported DPO backend: $BACKEND (expected trl)"
  exit 1
fi

#######################################
# Logging
#######################################

RUN_TS="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_LOG_DIR="${DPO_LOG_DIR:-logs/dpo}"
mkdir -p "$RUN_LOG_DIR"

RUN_LOG_FILE="$RUN_LOG_DIR/${RUN_TS}_${BACKEND}.log"
if [ "${RUN_LOG_FILE#/}" = "$RUN_LOG_FILE" ]; then
  RUN_LOG_FILE="$(pwd)/$RUN_LOG_FILE"
fi

export DPO_RUN_TS="$RUN_TS"
export DPO_RUN_BACKEND="$BACKEND"
export DPO_RUN_LOG_FILE="$RUN_LOG_FILE"

exec > >(tee -a "$RUN_LOG_FILE") 2>&1
trap 'echo "DPO run failed. Inspect log: $RUN_LOG_FILE"' ERR

echo "DPO run logs will be persisted to: $RUN_LOG_FILE"

#######################################
# MLflow handling
#######################################

if is_local_mlflow "$MLFLOW_URI"; then
  echo "Using local MLflow (${MLFLOW_URI}), starting docker compose..."
  docker compose up -d
else
  echo "Using remote MLflow (${MLFLOW_URI}), skipping docker compose."
fi

#######################################
# Training
#######################################

echo "Running DPO backend=${BACKEND} with uv (Python ${PYTHON_VERSION})"
echo "Config: ${CONFIG_PATH}"

uv run \
  --python "$PYTHON_VERSION" \
  --with-requirements "$REQUIREMENTS_FILE" \
  python -m src.eliza_trainer.dpo.train \
  --config "$CONFIG_PATH" \
  "${EXTRA_ARGS[@]}"