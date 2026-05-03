#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/sft_general_qwen3_5_0_8b.yaml}"

if [ -f ".env" ]; then
  set -a
  . ./.env
  set +a
fi

MLFLOW_URI="${MLFLOW_TRACKING_URI:-https://mlflow.chezombor.com/}"

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

echo "Running SFT with uv (Python 3.12) using config: ${CONFIG_PATH}"
uv run --python 3.12 --with-requirements requirements.txt python src/sft_finetune.py --config "$CONFIG_PATH"
