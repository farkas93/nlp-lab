#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/dpo_hass_qwen3_5_0_8b.yaml}"
if [ $# -gt 0 ]; then
  shift
fi
EXTRA_ARGS=("$@")

if [ -f ".env" ]; then
  set -a
  . ./.env
  set +a
fi

MLFLOW_URI="${MLFLOW_TRACKING_URI:-https://mlflow.chezombor.com/}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

BACKEND="$(uv run --python 3.12 --with-requirements requirements.dpo-trl.txt python - <<'PY' "$CONFIG_PATH"
import sys
import yaml

config_path = sys.argv[1]
with open(config_path, "r", encoding="utf-8") as handle:
    raw = yaml.safe_load(handle) or {}
identity = raw.get("identity") or {}
backend = str(identity.get("backend", "trl")).strip().lower()
print(backend)
PY
)"

if [ "$BACKEND" != "trl" ]; then
  echo "Error: unsupported DPO backend: $BACKEND (expected trl)"
  exit 1
fi

REQUIREMENTS_FILE="requirements.dpo-trl.txt"
if [ ! -f "$REQUIREMENTS_FILE" ]; then
  echo "Error: requirements file not found: $REQUIREMENTS_FILE"
  exit 1
fi

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
echo "DPO run logs will be persisted to: $RUN_LOG_FILE"
trap 'echo "DPO run failed. Inspect log: $RUN_LOG_FILE"' ERR

is_local_mlflow() {
  case "$1" in
    *localhost*|*127.0.0.1*) return 0 ;;
    *) return 1 ;;
  esac
}

if is_local_mlflow "$MLFLOW_URI"; then
  echo "Using local MLflow (${MLFLOW_URI}), starting docker compose..."
  docker compose up -d
else
  echo "Using remote MLflow (${MLFLOW_URI}), skipping docker compose."
fi

echo "Running DPO backend=${BACKEND} with uv (Python 3.12) using config: ${CONFIG_PATH}"

uv run --python 3.12 --with-requirements "$REQUIREMENTS_FILE" python -m src.eliza_trainer.dpo.train --config "$CONFIG_PATH" "${EXTRA_ARGS[@]}"
