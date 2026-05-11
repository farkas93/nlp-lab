#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/sft_general_qwen3_5_0_8b.yaml}"

if [ -f ".env" ]; then
  set -a
  . ./.env
  set +a
fi

MLFLOW_URI="${MLFLOW_TRACKING_URI:-https://mlflow.chezombor.com/}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

BACKEND="$(python - <<'PY' "$CONFIG_PATH"
import sys
import yaml

config_path = sys.argv[1]
with open(config_path, "r", encoding="utf-8") as handle:
    raw = yaml.safe_load(handle) or {}
training = raw.get("training") or {}
backend = str(training.get("backend", "trl")).strip().lower()
print(backend)
PY
)"

if [ "$BACKEND" = "unsloth" ]; then
  REQUIREMENTS_FILE="requirements.sft-unsloth.txt"
else
  REQUIREMENTS_FILE="requirements.sft-trl.txt"
fi

if [ ! -f "$REQUIREMENTS_FILE" ]; then
  if [ -f "requirements.txt" ]; then
    echo "Warning: $REQUIREMENTS_FILE not found, falling back to requirements.txt"
    REQUIREMENTS_FILE="requirements.txt"
  else
    echo "Error: requirements file not found: $REQUIREMENTS_FILE"
    exit 1
  fi
fi

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

echo "Running SFT backend=${BACKEND} with uv (Python 3.12) using config: ${CONFIG_PATH}"
uv run --python 3.12 --with-requirements "$REQUIREMENTS_FILE" python -m src.eliza_trainer.sft.train --config "$CONFIG_PATH"
