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
identity = raw.get("identity") or {}
backend = str(identity.get("backend", "trl")).strip().lower()
print(backend)
PY
)"

MODEL_NAME="$(python - <<'PY' "$CONFIG_PATH"
import sys
import yaml

config_path = sys.argv[1]
with open(config_path, "r", encoding="utf-8") as handle:
    raw = yaml.safe_load(handle) or {}
model = raw.get("model") or {}
owner = str(model.get("owner", "")).strip()
name = str(model.get("name", "")).strip()
print(f"{owner}/{name}".strip("/"))
PY
)"

case "$BACKEND" in
  trl)
    REQUIREMENTS_FILE="requirements.sft-trl.txt"
    ;;
  unsloth)
    REQUIREMENTS_FILE="requirements.sft-unsloth.txt"
    ;;
  *)
    echo "Error: unsupported training backend: $BACKEND (expected trl or unsloth)"
    exit 1
    ;;
esac

RUN_TS="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_LOG_DIR="${SFT_LOG_DIR:-logs/sft}"
mkdir -p "$RUN_LOG_DIR"
RUN_LOG_FILE="$RUN_LOG_DIR/${RUN_TS}_${BACKEND}.log"
if [ "${RUN_LOG_FILE#/}" = "$RUN_LOG_FILE" ]; then
  RUN_LOG_FILE="$(pwd)/$RUN_LOG_FILE"
fi

export SFT_RUN_TS="$RUN_TS"
export SFT_RUN_BACKEND="$BACKEND"
export SFT_RUN_LOG_FILE="$RUN_LOG_FILE"

exec > >(tee -a "$RUN_LOG_FILE") 2>&1
echo "SFT run logs will be persisted to: $RUN_LOG_FILE"
trap 'echo "SFT run failed. Inspect log: $RUN_LOG_FILE"' ERR

if [ ! -f "$REQUIREMENTS_FILE" ]; then
  echo "Error: requirements file not found: $REQUIREMENTS_FILE"
  exit 1
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

if [ "$BACKEND" = "unsloth" ]; then
  echo "Preflight: validating unsloth import in uv environment..."
  uv run --python 3.12 --with-requirements "$REQUIREMENTS_FILE" python - <<'PY' "$MODEL_NAME"
import sys

from packaging.version import Version
import accelerate
import peft
import torch
import transformers
import trl
import unsloth
import unsloth_zoo
from transformers import AutoConfig

model_name = sys.argv[1]
config = AutoConfig.from_pretrained(model_name)
model_type = str(getattr(config, "model_type", ""))
transformers_version = Version(transformers.__version__)

print(f"unsloth preflight model={model_name}")
print(f"versions torch={torch.__version__} transformers={transformers.__version__} trl={trl.__version__} peft={peft.__version__} accelerate={accelerate.__version__} unsloth={unsloth.__version__} unsloth_zoo={unsloth_zoo.__version__}")
print(f"resolved_model_type={model_type}")

if model_type == "qwen3_5" and transformers_version < Version("5.2.0"):
    raise RuntimeError(
        "Qwen3.5 requires transformers>=5.2.0 for unsloth backend; "
        f"found {transformers.__version__}"
    )

print("unsloth preflight OK")
PY
fi

uv run --python 3.12 --with-requirements "$REQUIREMENTS_FILE" python -m src.eliza_trainer.sft.train --config "$CONFIG_PATH"
