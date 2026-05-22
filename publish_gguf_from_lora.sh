#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<'EOF'
Usage:
  ./publish_gguf_from_lora.sh \
    --train-config configs/sft_hass_qwen3_5_0_8b.yaml \
    [--adapter-repo zskalo/qwen3.5-0.8b-lora-hass-tools] \
    [--gguf-repo zskalo/qwen3.5-0.8b-hass-tools-gguf] \
    [--tag v1.0.0] \
    [--llama-cpp-dir ~/llama.cpp] \
    [--convert-script /path/to/convert_hf_to_gguf.py] \
    [--quant Q4_K_M] \
    [--public]

Notes:
- Requires HF_HUB_TOKEN in environment.
- Auto-loads `.env` from current directory when present.
- `--train-config` is required and is used to infer adapter repo when `--adapter-repo` is not provided.
- GGUF repo defaults to adapter repo with `lora` replaced by `gguf` in repo name when `--gguf-repo` is not provided.
- If `--tag` is not provided, attempts to auto-detect the latest tag from the adapter repo.
- Default output is F16 GGUF only. Pass --quant to also upload quantized GGUF.
- Merged safetensors are created in a temporary directory and deleted on exit.
- Automatically generates and uploads Ollama-compatible Modelfile with proper chat template and stop tokens.
- Uploads training provenance metadata (`nlp_lab_provenance.json`).
- Uploads the provided train config as `training_config.yaml`.
- Tokenizer is loaded from adapter repo first (preserving training config), falls back to base model if needed.
EOF
}

ADAPTER_REPO=""
GGUF_REPO=""
TAG=""
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$HOME/llama.cpp}"
CONVERT_SCRIPT=""
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
    --convert-script)
      CONVERT_SCRIPT="$2"
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

if [[ -z "$TRAIN_CONFIG_PATH" ]]; then
  echo "--train-config is required."
  usage
  exit 1
fi

if [[ ! -f "$TRAIN_CONFIG_PATH" ]]; then
  echo "Train config file does not exist: $TRAIN_CONFIG_PATH"
  usage
  exit 1
fi

if [[ -f ".env" ]]; then
  set -a
  . ./.env
  set +a
elif [[ -f "$SCRIPT_DIR/.env" ]]; then
  set -a
  . "$SCRIPT_DIR/.env"
  set +a
fi

if [[ -z "${HF_HUB_TOKEN:-}" ]]; then
  echo "HF_HUB_TOKEN is required in environment."
  exit 1
fi

REQ_FILE="$SCRIPT_DIR/requirements.txt"
if [[ ! -f "$REQ_FILE" ]]; then
  echo "Could not find requirements file: $REQ_FILE"
  exit 1
fi

if [[ -z "$ADAPTER_REPO" ]]; then
  ADAPTER_REPO="$(TRAIN_CONFIG_PATH="$TRAIN_CONFIG_PATH" SCRIPT_DIR="$SCRIPT_DIR" uv run --python 3.12 --with-requirements "$REQ_FILE" python - <<'PY'
import os
import sys

script_dir = os.environ["SCRIPT_DIR"]
config_path = os.environ["TRAIN_CONFIG_PATH"]

if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from src.eliza_trainer.sft.run_config import load_sft_run_config

cfg = load_sft_run_config(config_path)
adapter_repo = str(cfg.hub.adapter_repo_name or "").strip()
if not adapter_repo:
    raise SystemExit(
        "Unable to infer adapter repo from train config. "
        "Set hub.push_to_hub=true with adapter publishing, or pass --adapter-repo."
    )
print(adapter_repo)
PY
)"
fi

if [[ -z "$GGUF_REPO" ]]; then
  GGUF_REPO="$(ADAPTER_REPO="$ADAPTER_REPO" uv run --python 3.12 --with-requirements "$REQ_FILE" python - <<'PY'
import os

adapter_repo = os.environ["ADAPTER_REPO"].strip()
if "/" not in adapter_repo:
    raise SystemExit(f"Invalid adapter repo (expected owner/name): {adapter_repo}")
owner, name = adapter_repo.split("/", 1)

if "lora" in name:
    gguf_name = name.replace("lora", "gguf", 1)
else:
    gguf_name = f"{name}-gguf"
    print(
        f"Warning: adapter repo name has no 'lora' token; defaulting gguf repo suffix: {owner}/{gguf_name}",
        file=os.sys.stderr,
    )

print(f"{owner}/{gguf_name}")
PY
)"
fi

echo "Resolved adapter repo: $ADAPTER_REPO"
echo "Resolved GGUF repo: $GGUF_REPO"

# Auto-detect TAG from adapter repo if not explicitly provided
if [[ -z "$TAG" ]]; then
  echo "No --tag provided, attempting to auto-detect from adapter repo..."
  TAG=$(ADAPTER_REPO="$ADAPTER_REPO" REQ_FILE="$REQ_FILE" uv run --python 3.12 --with-requirements "$REQ_FILE" python - <<'PY'
import os
import sys

try:
    from huggingface_hub import HfApi
    
    adapter_repo = os.environ["ADAPTER_REPO"]
    api = HfApi(token=os.getenv("HF_HUB_TOKEN"))
    
    # List all tags/refs for the adapter repo
    refs = api.list_repo_refs(repo_id=adapter_repo, repo_type="model")
    
    # Get tags (sorted by creation time, most recent first)
    if hasattr(refs, 'tags') and refs.tags:
        # Return the most recent tag name
        latest_tag = refs.tags[0].name if refs.tags else ""
        print(latest_tag)
    else:
        # No tags found
        print("")
except Exception as e:
    print(f"Warning: Could not query adapter tags: {e}", file=sys.stderr)
    print("")
PY
)
  
  if [[ -n "$TAG" ]]; then
    echo "Auto-detected tag from adapter: $TAG"
  else
    echo "No tags found on adapter repo, GGUF will not be tagged"
  fi
fi

if [[ -n "$TAG" ]]; then
  echo "Will apply tag to GGUF repo: $TAG"
fi

if [[ -z "$CONVERT_SCRIPT" ]]; then
  for candidate in \
    "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" \
    "$LLAMA_CPP_DIR/convert-hf-to-gguf.py" \
    "$LLAMA_CPP_DIR/tools/convert_hf_to_gguf.py" \
    "$LLAMA_CPP_DIR/examples/convert_hf_to_gguf.py"; do
    if [[ -f "$candidate" ]]; then
      CONVERT_SCRIPT="$candidate"
      break
    fi
  done
fi

if [[ ! -f "$CONVERT_SCRIPT" ]]; then
  echo "Could not locate GGUF conversion script."
  echo "Checked under --llama-cpp-dir=$LLAMA_CPP_DIR"
  echo "Pass --convert-script explicitly if your layout is custom."
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
ADAPTER_REPO="$ADAPTER_REPO" MERGED_DIR="$MERGED_DIR" uv run --python 3.12 --with-requirements "$REQ_FILE" python - <<'PY'
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

# Try to load tokenizer from adapter first (preserves training-time config)
# Fall back to base model if adapter doesn't have tokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained(adapter_repo, token=os.getenv("HF_HUB_TOKEN"))
    print(f"Loaded tokenizer from adapter: {adapter_repo}")
except Exception as e:
    print(f"Warning: Could not load tokenizer from adapter ({e}), using base model tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(base_model, token=os.getenv("HF_HUB_TOKEN"))
    print(f"Loaded tokenizer from base model: {base_model}")

tokenizer.save_pretrained(merged_dir)
PY

F16_GGUF="$OUT_DIR/model-f16.gguf"
echo "Converting merged model to GGUF (F16)"
uv run --python 3.12 --with-requirements "$REQ_FILE" python "$CONVERT_SCRIPT" "$MERGED_DIR" --outfile "$F16_GGUF" --outtype f16

UPLOAD_FILES=("$F16_GGUF")
if [[ -n "$QUANT_METHOD" ]]; then
  QUANT_GGUF="$OUT_DIR/model-${QUANT_METHOD}.gguf"
  echo "Quantizing GGUF with method=$QUANT_METHOD"
  "$QUANT_BIN" "$F16_GGUF" "$QUANT_GGUF" "$QUANT_METHOD"
  UPLOAD_FILES+=("$QUANT_GGUF")
fi

echo "Generating Ollama Modelfile"
MODELFILE_PATH="$OUT_DIR/Modelfile"
MERGED_DIR="$MERGED_DIR" uv run --python 3.12 --with-requirements "$REQ_FILE" python - > "$MODELFILE_PATH" <<'PY'
import os
from transformers import AutoTokenizer

merged_dir = os.environ["MERGED_DIR"]

# Load tokenizer to extract chat template
try:
    tokenizer = AutoTokenizer.from_pretrained(merged_dir)
    chat_template = getattr(tokenizer, 'chat_template', None)
    eos_token = getattr(tokenizer, 'eos_token', None)
except Exception:
    chat_template = None
    eos_token = None

# Generate Modelfile with Qwen-compatible settings
print("FROM ./model-f16.gguf")
print("")
print("# Model parameters")
print("PARAMETER temperature 0.7")
print("PARAMETER top_p 0.8")
print("PARAMETER top_k 20")
print("PARAMETER repeat_penalty 1.05")
print("PARAMETER num_ctx 4096")
print("")
print("# Stop tokens for Qwen models")
print('PARAMETER stop "<|im_end|>"')
print('PARAMETER stop "<|endoftext|>"')
print('PARAMETER stop "<|im_start|>"')
print("")

# If we have a chat template from the tokenizer, use it
# Otherwise use Qwen's standard template
if chat_template and len(chat_template.strip()) > 0:
    # Try to convert Jinja2 template to Ollama template format
    # This is a simplified conversion - Ollama uses Go templates
    print('# Chat template extracted from model')
    print('# Note: This may need manual adjustment for Ollama compatibility')
    print('# TEMPLATE """')
    print('# ' + chat_template.replace('\n', '\n# '))
    print('# """')
    print('')
    print('# Using Qwen standard template for Ollama compatibility:')

# Qwen standard template (Ollama-compatible)
print('TEMPLATE """{{ if .Messages }}')
print('{{- if or .System .Tools }}<|im_start|>system')
print('{{ .System }}')
print('{{- if .Tools }}')
print('')
print('# Tools')
print('')
print('You are provided with function signatures within <tools></tools> XML tags:')
print('<tools>{{- range .Tools }}')
print('{"type": "function", "function": {{ .Function }}}{{- end }}')
print('</tools>')
print('')
print('For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:')
print('<tool_call>')
print('{"name": <function-name>, "arguments": <args-json-object>}')
print('</tool_call>')
print('{{- end }}<|im_end|>')
print('{{ end }}')
print('{{- range $i, $_ := .Messages }}')
print('{{- $last := eq (len (slice $.Messages $i)) 1 -}}')
print('{{- if eq .Role "user" }}<|im_start|>user')
print('{{ .Content }}<|im_end|>')
print('{{ else if eq .Role "assistant" }}<|im_start|>assistant')
print('{{ if .Content }}{{ .Content }}')
print('{{- else if .ToolCalls }}<tool_call>')
print('{{ range .ToolCalls }}{"name": "{{ .Function.Name }}", "arguments": {{ .Function.Arguments }}}')
print('{{ end }}</tool_call>')
print('{{- end }}{{ if not $last }}<|im_end|>')
print('{{ end }}')
print('{{- else if eq .Role "tool" }}<|im_start|>user')
print('<tool_response>')
print('{{ .Content }}')
print('</tool_response><|im_end|>')
print('{{ end }}')
print('{{- if and (ne .Role "assistant") $last }}<|im_start|>assistant')
print('{{ end }}')
print('{{- end }}')
print('{{- else }}')
print('{{- if .System }}<|im_start|>system')
print('{{ .System }}<|im_end|>')
print('{{ end }}{{ if .Prompt }}<|im_start|>user')
print('{{ .Prompt }}<|im_end|>')
print('{{ end }}<|im_start|>assistant')
print('{{ end }}{{ .Response }}{{ if .Response }}<|im_end|>{{ end }}"""')
print('')
print('# System message')
print('SYSTEM """You are Qwen, created by Alibaba Cloud. You are a helpful assistant."""')
PY

echo "Modelfile generated at: $MODELFILE_PATH"
UPLOAD_FILES+=("$MODELFILE_PATH")

echo "Uploading GGUF artifacts to $GGUF_REPO"
GGUF_REPO="$GGUF_REPO" GGUF_PRIVATE="$GGUF_PRIVATE" TAG="$TAG" ADAPTER_REPO="$ADAPTER_REPO" UPLOAD_FILES="${UPLOAD_FILES[*]}" TRAIN_CONFIG_PATH="$TRAIN_CONFIG_PATH" NLP_LAB_GIT_COMMIT="$NLP_LAB_GIT_COMMIT" NLP_LAB_GIT_DESCRIBE="$NLP_LAB_GIT_DESCRIBE" NLP_LAB_GIT_BRANCH="$NLP_LAB_GIT_BRANCH" NLP_LAB_GIT_DIRTY="$NLP_LAB_GIT_DIRTY" NLP_LAB_GIT_REMOTE="$NLP_LAB_GIT_REMOTE" LLAMA_CPP_DIR="$LLAMA_CPP_DIR" QUANT_METHOD="$QUANT_METHOD" uv run --python 3.12 --with-requirements "$REQ_FILE" python - <<'PY'
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

## Usage with Ollama

A ready-to-use `Modelfile` is included for easy deployment:

```bash
# Download the GGUF and Modelfile
huggingface-cli download {repo_id} model-f16.gguf Modelfile --local-dir ./model

# Create Ollama model
cd model
ollama create my-model -f Modelfile

# Run
ollama run my-model
```

The Modelfile includes:
- Qwen-compatible chat template with tool calling support
- Proper stop tokens (`<|im_end|>`, `<|endoftext|>`)
- Recommended parameters (temperature, top_p, etc.)

## Training provenance

- `nlp_lab_provenance.json` - Training run metadata
- `training_config.yaml` - {"Included" if train_config_path else "Not provided"}
- `Modelfile` - Ollama deployment configuration

## Files

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
