# nlp-lab

`nlp-lab` trains post-training models from manifest-driven datasets.

## Documentation

- `docs/README.md`
- `docs/architecture.md`
- `docs/config-schema-v2.md`
- `docs/data-contracts/sft.md`
- `docs/data-contracts/dpo.md`
- `docs/migration-v1-to-v2.md`
- `docs/runs/sft.md`
- `docs/runs/dpo.md`
- `docs/experiments/mlflow.md`
- `docs/runs/benchmarking.md`
- `docs/troubleshooting.md`

## Repository areas

- Production SFT path: `src/eliza_trainer/sft/`
- Production DPO path: `src/eliza_trainer/dpo/`
- Common runtime helpers: `src/eliza_trainer/common/`
- Legacy post-training experiments: `experiments/legacy_post_training/`

## Environment

Copy the env template and set values:

```bash
cp .env.template .env
```

Important variables:

- `HF_HUB_TOKEN` for model download/upload
- `MLFLOW_TRACKING_URI` (default: `https://mlflow.chezombor.com/`)
- `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_ENDPOINT_URL`, `AWS_REGION` for `s3://` dataset manifests

## SFT config

SFT runs are YAML-driven. Default config:

- `configs/sft_general_qwen3_5_0_8b.yaml`

Main fields:

- `config_schema_version` -> must be `2`
- `identity.experiment_tag` + `identity.backend` -> run naming and backend selection
- `data.bucket` + `data.dataset_id` + `data.dataset_version` -> manifest URI is auto-derived
- `data.cache_mode` -> `reuse` (default) or `refresh` to force dataset redownload
- `data.expected_manifest_sha256` -> optional guard to fail on unexpected manifest content
- `model.owner` + `model.name` -> base model HF repo is auto-derived as `{owner}/{name}`
- `training.*` -> experiment and training hyperparameters
- `hub.owner` + publish flags -> adapter/full target repos are auto-derived with LoRA suffix policy
- manifest governance metadata (`governance.policy_*`) -> policy class logging at train start and policy gates

Policy enforcement helpers live in `src/eliza_trainer/common/data_policy.py` and are reusable across SFT/DPO/GRPO manifest-based runners.

See full runbook: `docs/runs/sft.md`.

## Run SFT

Run training via `uv` (Python 3.12):

```bash
./start_sft.sh
```

`start_sft.sh` only starts local MLflow via docker-compose when `MLFLOW_TRACKING_URI` points to `localhost` or `127.0.0.1`.
It reads `identity.backend` from the YAML config and selects backend-specific dependency overlays:

- `trl` -> `requirements.sft-trl.txt`
- `unsloth` -> `requirements.sft-unsloth.txt`

For `unsloth`, it runs a preflight import check before starting training.

Run with custom config path:

```bash
./start_sft.sh configs/sft_general_qwen3_5_0_8b.yaml
```

If dataset policy classes include `P3` or `P4`, training requires explicit confirmation.
Use `--assume-yes` for non-interactive runs:

```bash
./start_sft.sh configs/sft_general_qwen3_5_0_8b.yaml --assume-yes
```

Safety guard: when dataset policy class `P4` is present, training refuses to continue if `hub.push_to_hub=true`.

For Ollama-first deployment, publish LoRA adapters from training, then export GGUF with:

```bash
./publish_gguf_from_lora.sh --help
```

The GGUF publish script can also upload the exact training config and `nlp-lab` git provenance to the model repo.

Direct Python command:

```bash
uv run --python 3.12 --with-requirements requirements.sft-trl.txt python -m src.eliza_trainer.sft.train --config configs/sft_general_qwen3_5_0_8b.yaml
```

## Run DPO

Run DPO training via `uv` (Python 3.12):

```bash
./start_dpo.sh configs/dpo_hass_qwen3_5_0_8b.yaml
```

`start_dpo.sh` reads `identity.backend` from the YAML config and currently supports:

- `trl` -> `requirements.dpo-trl.txt`

Run with config dry-run only (no training):

```bash
./start_dpo.sh configs/dpo_hass_qwen3_5_0_8b.yaml --dry-run-config
```

DPO policy gates mirror SFT behavior:

- `P3/P4` policy classes require explicit confirmation unless `--assume-yes` is set
- `P4` blocks runs when `hub.push_to_hub=true`

Full runbook: `docs/runs/dpo.md`.

## Dataset contract expectations

The loader expects a manifest JSON with `splits` entries containing `split` and `key`, with Parquet files for `train` and `eval` rows.

Each dataset row should include at least:

- `messages` (list of role/content context messages)
- `target_text` (assistant target string)

Tokenization applies model chat templates at runtime, and training uses assistant-only loss masking.

Full contract details: `docs/data-contracts/sft.md`.

## Diagnostic Tools

### Diagnose SFT Pipeline

Analyze tokenizer configuration and training data for issues that can cause inference problems (e.g., infinite loops after EOS tokens):

```bash
# Basic analysis
./diagnose_sft_pipeline.py --config configs/sft_hass_qwen3_5_0_8b.yaml

# Analyze more samples with verbose output
./diagnose_sft_pipeline.py --config configs/sft_hass_qwen3_5_0_8b.yaml --num-samples 10 --verbose

# Include GGUF metadata analysis
./diagnose_sft_pipeline.py --config configs/sft_hass_qwen3_5_0_8b.yaml --gguf-path ~/models/model.gguf
```

The diagnostic script checks:
- Tokenizer special token configuration (EOS, PAD, BOS)
- Detects `pad_token == eos_token` bug with `full_conversation` loss mode
- Analyzes training samples for proper EOS token placement
- Optional GGUF metadata extraction and comparison

### Test Model Inference

Test trained models (LoRA adapters or GGUF via Ollama) for EOS stopping behavior and infinite loop detection:

```bash
# Test LoRA adapter (inferred from config)
./test_model_inference.py --config configs/sft_hass_qwen3_5_0_8b.yaml

# Test specific adapter repo
./test_model_inference.py --config configs/sft_hass_qwen3_5_0_8b.yaml --adapter-repo zskalo/qwen3.5-0.8b-lora-hass-tools

# Test GGUF via Ollama
./test_model_inference.py --ollama-model qwen3_hass

# Custom prompts
./test_model_inference.py --config configs/sft_hass_qwen3_5_0_8b.yaml --prompt "Turn on the lights"
```

The inference test script:
- Runs multiple test prompts through the model
- Detects if model stops at EOS or continues generating
- Identifies repetitive patterns indicating infinite loops
- Works with both transformers/peft (GPU) and Ollama (CPU/GPU)
