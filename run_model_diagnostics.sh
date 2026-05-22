#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<'EOF'
Usage:
  ./run_model_diagnostics.sh --config <config.yaml> [OPTIONS]

Diagnose SFT training pipeline and model inference issues.

Required:
  --config <path>           Path to SFT YAML config file

Diagnostic Options (at least one required):
  --tokenizer               Analyze tokenizer configuration (pad/eos tokens)
  --training-data           Analyze training data samples for EOS placement
  --gguf <path|auto>        Analyze GGUF file metadata (use 'auto' to discover from config)
  --inference-transformers  Test inference with transformers/peft (requires GPU)
  --inference-ollama <name> Test inference with Ollama model

Additional Options:
  --num-samples <n>         Number of training samples to analyze (default: 5)
  --adapter-repo <repo>     HuggingFace adapter repo for transformers inference
  --ollama-host <url>       Ollama API host (default: http://localhost:11434)
  --output-json <path>      Write JSON output to file
  --all                     Run tokenizer + training-data analysis
  -h, --help                Show this help message

Environment Variables (via .env file):
  HF_HUB_TOKEN              HuggingFace API token
  AWS_ACCESS_KEY_ID         AWS credentials for S3 dataset access
  AWS_SECRET_ACCESS_KEY
  AWS_REGION
  AWS_ENDPOINT_URL

Examples:
  # Analyze tokenizer configuration
  ./run_model_diagnostics.sh --config configs/sft_hass_qwen3_5_0_8b.yaml --tokenizer

  # Analyze training data (5 samples)
  ./run_model_diagnostics.sh --config configs/sft_hass_qwen3_5_0_8b.yaml --training-data

  # Full analysis (tokenizer + training data)
  ./run_model_diagnostics.sh --config configs/sft_hass_qwen3_5_0_8b.yaml --all

  # Analyze GGUF file (explicit path)
  ./run_model_diagnostics.sh --config configs/sft_hass_qwen3_5_0_8b.yaml --gguf ~/models/model.gguf

  # Auto-discover and analyze GGUF from HuggingFace
  ./run_model_diagnostics.sh --config configs/sft_hass_qwen3_5_0_8b.yaml --gguf auto

  # Test inference via Ollama
  ./run_model_diagnostics.sh --config configs/sft_hass_qwen3_5_0_8b.yaml --inference-ollama qwen3_hass

  # Full pipeline with JSON output
  ./run_model_diagnostics.sh --config configs/sft_hass_qwen3_5_0_8b.yaml --all --gguf auto --output-json report.json
EOF
}

# Defaults
CONFIG_PATH=""
RUN_TOKENIZER=false
RUN_TRAINING_DATA=false
RUN_GGUF=false
RUN_INFERENCE_TRANSFORMERS=false
RUN_INFERENCE_OLLAMA=false
GGUF_PATH=""
OLLAMA_MODEL=""
OLLAMA_HOST="http://localhost:11434"
ADAPTER_REPO=""
NUM_SAMPLES=5
OUTPUT_JSON=""

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --tokenizer)
      RUN_TOKENIZER=true
      shift
      ;;
    --training-data)
      RUN_TRAINING_DATA=true
      shift
      ;;
    --gguf)
      RUN_GGUF=true
      GGUF_PATH="$2"
      # Support --gguf auto to trigger auto-discovery
      if [[ "$GGUF_PATH" == "auto" ]]; then
        GGUF_PATH="auto"
      fi
      shift 2
      ;;
    --inference-transformers)
      RUN_INFERENCE_TRANSFORMERS=true
      shift
      ;;
    --inference-ollama)
      RUN_INFERENCE_OLLAMA=true
      OLLAMA_MODEL="$2"
      shift 2
      ;;
    --num-samples)
      NUM_SAMPLES="$2"
      shift 2
      ;;
    --adapter-repo)
      ADAPTER_REPO="$2"
      shift 2
      ;;
    --ollama-host)
      OLLAMA_HOST="$2"
      shift 2
      ;;
    --output-json)
      OUTPUT_JSON="$2"
      shift 2
      ;;
    --all)
      RUN_TOKENIZER=true
      RUN_TRAINING_DATA=true
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

# Validate required arguments
if [[ -z "$CONFIG_PATH" ]]; then
  echo "Error: --config is required"
  usage
  exit 1
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Error: Config file not found: $CONFIG_PATH"
  exit 1
fi

# Check if at least one diagnostic option is specified
if [[ "$RUN_TOKENIZER" == "false" && "$RUN_TRAINING_DATA" == "false" && "$RUN_GGUF" == "false" && "$RUN_INFERENCE_TRANSFORMERS" == "false" && "$RUN_INFERENCE_OLLAMA" == "false" ]]; then
  echo "Error: At least one diagnostic option is required"
  usage
  exit 1
fi

# Load .env if present
if [[ -f "$SCRIPT_DIR/.env" ]]; then
  echo "Loading environment from .env"
  set -a
  . "$SCRIPT_DIR/.env"
  set +a
fi

# Validate environment for specific diagnostics
if [[ "$RUN_TRAINING_DATA" == "true" ]]; then
  if [[ -z "${AWS_ACCESS_KEY_ID:-}" || -z "${AWS_SECRET_ACCESS_KEY:-}" ]]; then
    echo "Error: AWS credentials required for training data analysis"
    echo "Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in .env"
    exit 1
  fi
fi

if [[ "$RUN_GGUF" == "true" && ! -f "$GGUF_PATH" ]]; then
  echo "Error: GGUF file not found: $GGUF_PATH"
  exit 1
fi

REQ_FILE="$SCRIPT_DIR/requirements.txt"

if [[ ! -f "$REQ_FILE" ]]; then
  echo "Error: Requirements file not found: $REQ_FILE"
  exit 1
fi

# Initialize JSON output
JSON_RESULTS="{}"

echo "================================================================================"
echo "MODEL DIAGNOSTICS REPORT"
echo "================================================================================"
echo ""
echo "Config: $CONFIG_PATH"
echo "Date: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
echo ""

# Run tokenizer analysis
if [[ "$RUN_TOKENIZER" == "true" ]]; then
  echo "Running tokenizer analysis..."
  TOKENIZER_JSON=$(CONFIG_PATH="$CONFIG_PATH" uv run --python 3.12 --with-requirements "$REQ_FILE" python - <<'PY'
import os
import sys
sys.path.insert(0, os.environ.get("SCRIPT_DIR", "."))

from tests.test_tokenizer_diagnostics import (
    analyze_tokenizer_config,
    print_tokenizer_report,
    get_tokenizer_diagnostics_json,
)

config_path = os.environ["CONFIG_PATH"]
result = analyze_tokenizer_config(config_path)
print_tokenizer_report(result)
print("__JSON_START__")
print(get_tokenizer_diagnostics_json(result))
print("__JSON_END__")
PY
  )
  
  # Extract JSON from output
  TOKENIZER_JSON_ONLY=$(echo "$TOKENIZER_JSON" | sed -n '/__JSON_START__/,/__JSON_END__/p' | grep -v "__JSON_")
  echo "$TOKENIZER_JSON" | sed '/__JSON_START__/,$d'
  
  if [[ -n "$OUTPUT_JSON" ]]; then
    JSON_RESULTS=$(echo "$JSON_RESULTS" | jq --argjson tok "$TOKENIZER_JSON_ONLY" '. + {tokenizer: $tok}')
  fi
fi

# Run training data analysis
if [[ "$RUN_TRAINING_DATA" == "true" ]]; then
  echo "Running training data analysis..."
  TRAINING_JSON=$(CONFIG_PATH="$CONFIG_PATH" NUM_SAMPLES="$NUM_SAMPLES" uv run --python 3.12 --with-requirements "$REQ_FILE" python - <<'PY'
import os
import sys
sys.path.insert(0, os.environ.get("SCRIPT_DIR", "."))

from tests.test_training_data_diagnostics import (
    analyze_training_samples,
    print_training_data_report,
    get_training_data_diagnostics_json,
)

config_path = os.environ["CONFIG_PATH"]
num_samples = int(os.environ.get("NUM_SAMPLES", "5"))
result = analyze_training_samples(config_path, num_samples)
print_training_data_report(result)
print("__JSON_START__")
print(get_training_data_diagnostics_json(result))
print("__JSON_END__")
PY
  )
  
  TRAINING_JSON_ONLY=$(echo "$TRAINING_JSON" | sed -n '/__JSON_START__/,/__JSON_END__/p' | grep -v "__JSON_")
  echo "$TRAINING_JSON" | sed '/__JSON_START__/,$d'
  
  if [[ -n "$OUTPUT_JSON" ]]; then
    JSON_RESULTS=$(echo "$JSON_RESULTS" | jq --argjson train "$TRAINING_JSON_ONLY" '. + {training_data: $train}')
  fi
fi

# Run GGUF analysis
if [[ "$RUN_GGUF" == "true" ]]; then
  # Auto-discover GGUF path if requested
  if [[ "$GGUF_PATH" == "auto" ]]; then
    echo "Auto-discovering GGUF path from config..."
    DISCOVERED_GGUF=$(CONFIG_PATH="$CONFIG_PATH" uv run --python 3.12 --with-requirements "$REQ_FILE" python - <<'PY'
import os
import sys
import yaml
from pathlib import Path

try:
    from huggingface_hub import HfApi, hf_hub_download
    
    config_path = os.environ["CONFIG_PATH"]
    
    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Get adapter repo from config
    adapter_repo = config.get("adapter_repo")
    if not adapter_repo:
        print("ERROR: No adapter_repo in config", file=sys.stderr)
        sys.exit(1)
    
    # Infer GGUF repo (replace 'lora' with 'gguf')
    if "-lora-" in adapter_repo:
        gguf_repo = adapter_repo.replace("-lora-", "-gguf-")
    elif "lora" in adapter_repo:
        gguf_repo = adapter_repo.replace("lora", "gguf")
    else:
        gguf_repo = adapter_repo + "-gguf"
    
    print(f"Inferred GGUF repo: {gguf_repo}", file=sys.stderr)
    
    # Check if repo exists and list files
    api = HfApi(token=os.getenv("HF_HUB_TOKEN"))
    try:
        files = api.list_repo_files(repo_id=gguf_repo, repo_type="model")
        
        # Look for quantized GGUF first, then F16
        gguf_files = [f for f in files if f.endswith(".gguf")]
        
        if not gguf_files:
            print(f"ERROR: No GGUF files found in {gguf_repo}", file=sys.stderr)
            sys.exit(1)
        
        # Prefer quantized over F16 (smaller, faster download)
        quant_priority = ["Q4_K_M", "Q4_K_S", "Q5_K_M", "Q5_K_S", "Q8_0"]
        selected_file = None
        
        for quant in quant_priority:
            for f in gguf_files:
                if quant.lower() in f.lower():
                    selected_file = f
                    break
            if selected_file:
                break
        
        # Fall back to first GGUF file found
        if not selected_file:
            selected_file = gguf_files[0]
        
        print(f"Selected GGUF file: {selected_file}", file=sys.stderr)
        
        # Download to temp location
        local_path = hf_hub_download(
            repo_id=gguf_repo,
            filename=selected_file,
            repo_type="model",
            token=os.getenv("HF_HUB_TOKEN")
        )
        
        print(local_path)
        
    except Exception as e:
        print(f"ERROR: Could not access GGUF repo {gguf_repo}: {e}", file=sys.stderr)
        sys.exit(1)
        
except Exception as e:
    print(f"ERROR: {e}", file=sys.stderr)
    sys.exit(1)
PY
    )
    
    if [[ $? -ne 0 ]]; then
      echo "Failed to auto-discover GGUF path. Please specify explicitly with --gguf <path>"
      exit 1
    fi
    
    GGUF_PATH="$DISCOVERED_GGUF"
    echo "Auto-discovered GGUF: $GGUF_PATH"
  fi
  
  echo "Running GGUF analysis..."
  GGUF_JSON=$(GGUF_PATH="$GGUF_PATH" uv run --python 3.12 --with-requirements "$REQ_FILE" --with gguf python - <<'PY'
import os
import sys
sys.path.insert(0, os.environ.get("SCRIPT_DIR", "."))

from tests.test_gguf_analysis import (
    analyze_gguf_metadata,
    print_gguf_report,
    get_gguf_diagnostics_json,
)

gguf_path = os.environ["GGUF_PATH"]
result = analyze_gguf_metadata(gguf_path)
print_gguf_report(result)
print("__JSON_START__")
print(get_gguf_diagnostics_json(result))
print("__JSON_END__")
PY
  )
  
  GGUF_JSON_ONLY=$(echo "$GGUF_JSON" | sed -n '/__JSON_START__/,/__JSON_END__/p' | grep -v "__JSON_")
  echo "$GGUF_JSON" | sed '/__JSON_START__/,$d'
  
  if [[ -n "$OUTPUT_JSON" ]]; then
    JSON_RESULTS=$(echo "$JSON_RESULTS" | jq --argjson gguf "$GGUF_JSON_ONLY" '. + {gguf: $gguf}')
  fi
fi

# Run transformers inference test
if [[ "$RUN_INFERENCE_TRANSFORMERS" == "true" ]]; then
  echo "Running transformers inference test..."
  INFERENCE_JSON=$(CONFIG_PATH="$CONFIG_PATH" ADAPTER_REPO="${ADAPTER_REPO:-}" uv run --python 3.12 --with-requirements "$REQ_FILE" python - <<'PY'
import os
import sys
sys.path.insert(0, os.environ.get("SCRIPT_DIR", "."))

from tests.test_inference_testing import (
    test_transformers_inference,
    print_inference_report,
    get_inference_diagnostics_json,
)

config_path = os.environ["CONFIG_PATH"]
adapter_repo = os.environ.get("ADAPTER_REPO") or None
result = test_transformers_inference(config_path, adapter_repo=adapter_repo)
print_inference_report(result)
print("__JSON_START__")
print(get_inference_diagnostics_json(result))
print("__JSON_END__")
PY
  )
  
  INFERENCE_JSON_ONLY=$(echo "$INFERENCE_JSON" | sed -n '/__JSON_START__/,/__JSON_END__/p' | grep -v "__JSON_")
  echo "$INFERENCE_JSON" | sed '/__JSON_START__/,$d'
  
  if [[ -n "$OUTPUT_JSON" ]]; then
    JSON_RESULTS=$(echo "$JSON_RESULTS" | jq --argjson inf "$INFERENCE_JSON_ONLY" '. + {inference_transformers: $inf}')
  fi
fi

# Run Ollama inference test
if [[ "$RUN_INFERENCE_OLLAMA" == "true" ]]; then
  echo "Running Ollama inference test..."
  OLLAMA_JSON=$(OLLAMA_MODEL="$OLLAMA_MODEL" OLLAMA_HOST="$OLLAMA_HOST" uv run --python 3.12 --with-requirements "$REQ_FILE" python - <<'PY'
import os
import sys
sys.path.insert(0, os.environ.get("SCRIPT_DIR", "."))

from tests.test_inference_testing import (
    test_ollama_inference,
    print_inference_report,
    get_inference_diagnostics_json,
)

model_name = os.environ["OLLAMA_MODEL"]
ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
result = test_ollama_inference(model_name, ollama_host=ollama_host)
print_inference_report(result)
print("__JSON_START__")
print(get_inference_diagnostics_json(result))
print("__JSON_END__")
PY
  )
  
  OLLAMA_JSON_ONLY=$(echo "$OLLAMA_JSON" | sed -n '/__JSON_START__/,/__JSON_END__/p' | grep -v "__JSON_")
  echo "$OLLAMA_JSON" | sed '/__JSON_START__/,$d'
  
  if [[ -n "$OUTPUT_JSON" ]]; then
    JSON_RESULTS=$(echo "$JSON_RESULTS" | jq --argjson oll "$OLLAMA_JSON_ONLY" '. + {inference_ollama: $oll}')
  fi
fi

# Write JSON output if requested
if [[ -n "$OUTPUT_JSON" ]]; then
  echo "$JSON_RESULTS" | jq '.' > "$OUTPUT_JSON"
  echo ""
  echo "JSON output written to: $OUTPUT_JSON"
fi

echo ""
echo "================================================================================"
echo "DIAGNOSTICS COMPLETE"
echo "================================================================================"
