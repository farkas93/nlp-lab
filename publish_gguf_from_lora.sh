#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./publish_gguf_from_lora.sh \
    --adapter-repo zskalo/qwen3.5-0.8b-lora-hass-tools \
    --gguf-repo zskalo/qwen3.5-0.8b-hass-tools-gguf \
    [--train-config configs/sft_hass_qwen3_5_0_8b.yaml] \
    [--tag v1.0.0] \
    [--llama-cpp-dir "$HOME/llama.cpp"] \
    [--quant Q4_K_M] \
    [--public]

Notes:
- Requires HF_HUB_TOKEN in environment.
- Default output is F16 GGUF only. Pass --quant to also upload quantized GGUF.
- Merged safetensors are created in a temporary directory and deleted on exit.
- Uploads training provenance metadata (`nlp_lab_provenance.json`).
- If --train-config is provided, uploads it as `training_config.yaml`.
EOF
}

ADAPTER_REPO=""
GGUF_REPO=""
TAG=""
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$HOME/llama.cpp}"
QUANT_METHOD=""
GGUF_PRIVATE="true"
TRAIN_CONFIG_PATH=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --adapter-repo)
      ADAPTER_REPO="$2"
      shift 2
      ;;
    --gguf-repo)
      GGUF_REPO="$2"
      shift 2
      ;;
    --tag)
      TAG="$2"
      shift 2
      ;;
    --llama-cpp-dir)
      LLAMA_CPP_DIR="$2"
      shift 2
      ;;
    --train-config)
      TRAIN_CONFIG_PATH="$2"
      shift 2
      ;;
    --quant)
      QUANT_METHOD="$2"
      shift 2
      ;;
    --public)
      GGUF_PRIVATE="false"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$ADAPTER_REPO" || -z "$GGUF_REPO" ]]; then
  echo "Both --adapter-repo and --gguf-repo are required."
  usage
  exit 1
fi

if [[ -z "${HF_HUB_TOKEN:-}" ]]; then
  echo "HF_HUB_TOKEN is required in environment."
  exit 1
fi

if [[ -n "$TRAIN_CONFIG_PATH" && ! -f "$TRAIN_CONFIG_PATH" ]]; then
  echo "Train config file does not exist: $TRAIN_CONFIG_PATH"
  exit 1
fi

CONVERT_SCRIPT="$LLAMA_CPP_DIR/convert_hf_to_gguf.py"
if [[ ! -f "$CONVERT_SCRIPT" ]]; then
  echo "convert_hf_to_gguf.py not found at $CONVERT_SCRIPT"
  exit 1
fi

QUANT_BIN=""
if [[ -n "$QUANT_METHOD" ]]; then
  if [[ -x "$LLAMA_CPP_DIR/build/bin/llama-quantize" ]]; then
    QUANT_BIN="$LLAMA_CPP_DIR/build/bin/llama-quantize"
  elif [[ -x "$LLAMA_CPP_DIR/build/bin/quantize" ]]; then
    QUANT_BIN="$LLAMA_CPP_DIR/build/bin/quantize"
  else
    echo "No quantize binary found under $LLAMA_CPP_DIR/build/bin"
    exit 1
  fi
fi

WORKDIR="$(mktemp -d)"
MERGED_DIR="$WORKDIR/merged"
OUT_DIR="$WORKDIR/out"
mkdir -p "$MERGED_DIR" "$OUT_DIR"

NLP_LAB_GIT_COMMIT="unknown"
NLP_LAB_GIT_DESCRIBE="unknown"
NLP_LAB_GIT_BRANCH="unknown"
NLP_LAB_GIT_DIRTY="unknown"
NLP_LAB_GIT_REMOTE="unknown"

if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  NLP_LAB_GIT_COMMIT="$(git rev-parse HEAD 2>/dev/null || true)"
  NLP_LAB_GIT_DESCRIBE="$(git describe --always --tags --dirty 2>/dev/null || true)"
  NLP_LAB_GIT_BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || true)"
  NLP_LAB_GIT_REMOTE="$(git remote get-url origin 2>/dev/null || true)"
  if git diff --quiet >/dev/null 2>&1 && git diff --cached --quiet >/dev/null 2>&1; then
    NLP_LAB_GIT_DIRTY="false"
  else
    NLP_LAB_GIT_DIRTY="true"
  fi
fi

cleanup() {
  rm -rf "$WORKDIR"
}
trap cleanup EXIT

echo "Merging adapter into base model (temporary): $ADAPTER_REPO"
ADAPTER_REPO="$ADAPTER_REPO" MERGED_DIR="$MERGED_DIR" uv run --python 3.12 --with-requirements requirements.sft-trl.txt python - <<'PY'
import os

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

adapter_repo = os.environ["ADAPTER_REPO"]
merged_dir = os.environ["MERGED_DIR"]

cfg = PeftConfig.from_pretrained(adapter_repo, token=os.getenv("HF_HUB_TOKEN"))
base_model = cfg.base_model_name_or_path

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.float16,
    device_map="cpu",
)
model = PeftModel.from_pretrained(model, adapter_repo, token=os.getenv("HF_HUB_TOKEN"))
model = model.merge_and_unload()
model.save_pretrained(merged_dir, safe_serialization=True, max_shard_size="4GB")

try:
    tokenizer = AutoTokenizer.from_pretrained(adapter_repo, token=os.getenv("HF_HUB_TOKEN"))
except Exception:
    tokenizer = AutoTokenizer.from_pretrained(base_model, token=os.getenv("HF_HUB_TOKEN"))
tokenizer.save_pretrained(merged_dir)
PY

F16_GGUF="$OUT_DIR/model-f16.gguf"
echo "Converting merged model to GGUF (F16)"
python "$CONVERT_SCRIPT" "$MERGED_DIR" --outfile "$F16_GGUF" --outtype f16

UPLOAD_FILES=("$F16_GGUF")
if [[ -n "$QUANT_METHOD" ]]; then
  QUANT_GGUF="$OUT_DIR/model-${QUANT_METHOD}.gguf"
  echo "Quantizing GGUF with method=$QUANT_METHOD"
  "$QUANT_BIN" "$F16_GGUF" "$QUANT_GGUF" "$QUANT_METHOD"
  UPLOAD_FILES+=("$QUANT_GGUF")
fi

echo "Uploading GGUF artifacts to $GGUF_REPO"
GGUF_REPO="$GGUF_REPO" GGUF_PRIVATE="$GGUF_PRIVATE" TAG="$TAG" ADAPTER_REPO="$ADAPTER_REPO" UPLOAD_FILES="${UPLOAD_FILES[*]}" TRAIN_CONFIG_PATH="$TRAIN_CONFIG_PATH" NLP_LAB_GIT_COMMIT="$NLP_LAB_GIT_COMMIT" NLP_LAB_GIT_DESCRIBE="$NLP_LAB_GIT_DESCRIBE" NLP_LAB_GIT_BRANCH="$NLP_LAB_GIT_BRANCH" NLP_LAB_GIT_DIRTY="$NLP_LAB_GIT_DIRTY" NLP_LAB_GIT_REMOTE="$NLP_LAB_GIT_REMOTE" LLAMA_CPP_DIR="$LLAMA_CPP_DIR" QUANT_METHOD="$QUANT_METHOD" uv run --python 3.12 --with-requirements requirements.sft-trl.txt python - <<'PY'
import json
import os
import tempfile
from datetime import datetime, timezone

from huggingface_hub import HfApi

repo_id = os.environ["GGUF_REPO"]
private = os.environ["GGUF_PRIVATE"].lower() == "true"
tag = os.environ.get("TAG", "").strip()
adapter_repo = os.environ["ADAPTER_REPO"]
upload_files = [p for p in os.environ["UPLOAD_FILES"].split(" ") if p]
train_config_path = os.environ.get("TRAIN_CONFIG_PATH", "").strip()

provenance = {
    "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "source": {
        "adapter_repo": adapter_repo,
        "gguf_repo": repo_id,
        "tag": tag or None,
        "quant_method": os.environ.get("QUANT_METHOD", "") or None,
    },
    "nlp_lab": {
        "git_commit": os.environ.get("NLP_LAB_GIT_COMMIT", "unknown"),
        "git_describe": os.environ.get("NLP_LAB_GIT_DESCRIBE", "unknown"),
        "git_branch": os.environ.get("NLP_LAB_GIT_BRANCH", "unknown"),
        "git_dirty": os.environ.get("NLP_LAB_GIT_DIRTY", "unknown"),
        "git_remote": os.environ.get("NLP_LAB_GIT_REMOTE", "unknown"),
    },
    "tooling": {
        "llama_cpp_dir": os.environ.get("LLAMA_CPP_DIR", ""),
    },
}

api = HfApi(token=os.getenv("HF_HUB_TOKEN"))
api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)

for path in upload_files:
    api.upload_file(
        path_or_fileobj=path,
        path_in_repo=os.path.basename(path),
        repo_id=repo_id,
        repo_type="model",
    )

if train_config_path:
    api.upload_file(
        path_or_fileobj=train_config_path,
        path_in_repo="training_config.yaml",
        repo_id=repo_id,
        repo_type="model",
    )

with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json", delete=False) as handle:
    json.dump(provenance, handle, ensure_ascii=True, indent=2, sort_keys=True)
    provenance_path = handle.name

api.upload_file(
    path_or_fileobj=provenance_path,
    path_in_repo="nlp_lab_provenance.json",
    repo_id=repo_id,
    repo_type="model",
)

readme = f"""---
library_name: gguf
tags:
- gguf
- qwen
- home-assistant
- tool-calling
---

# {repo_id}

GGUF artifacts generated from LoRA adapter `{adapter_repo}`.

Training provenance:
- `nlp_lab_provenance.json`
- `training_config.yaml` ({"included" if train_config_path else "not provided"})

Files uploaded:
{chr(10).join(f'- `{os.path.basename(path)}`' for path in upload_files)}
"""

with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".md", delete=False) as handle:
    handle.write(readme)
    readme_path = handle.name

api.upload_file(
    path_or_fileobj=readme_path,
    path_in_repo="README.md",
    repo_id=repo_id,
    repo_type="model",
)

if tag:
    try:
        api.create_tag(repo_id=repo_id, tag=tag, repo_type="model")
    except Exception as exc:
        print(f"Warning: failed to create tag {tag}: {exc}")
PY

echo "Done. GGUF repo: https://huggingface.co/$GGUF_REPO"
