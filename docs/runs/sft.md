# SFT Runbook

## Prerequisites

1. `materialize_sft_release` has produced a manifest and Parquet splits.
2. `.env` exists and contains model, tracking, and storage credentials.
3. YAML run config points to the correct dataset manifest URI.
4. `uv` is installed and Python 3.12 is available.

## Minimal run

```bash
./start_sft.sh configs/sft_general_qwen3_5_0_8b.yaml
```

`start_sft.sh` uses `uv run --python 3.12 --with-requirements ...` and only starts local MLflow via docker-compose when `MLFLOW_TRACKING_URI` points to `localhost` or `127.0.0.1`.
`identity.backend` controls trainer backend selection (`trl` or `unsloth`) and dependency overlays are selected explicitly:

- `trl` -> `requirements.sft-trl.txt`
- `unsloth` -> `requirements.sft-unsloth.txt`

For `unsloth`, `start_sft.sh` also runs a preflight import check before launching training.
The unsloth requirements overlay is intentionally version-pinned and independent from the vLLM/TRL latest stack to avoid known `transformers` 5.x compatibility breakages.
Each run also persists host logs under `logs/sft/<timestamp>_<backend>.log` (or `SFT_LOG_DIR` if set), and exports the log path to the trainer process.

Equivalent direct command:

```bash
uv run --python 3.12 --with-requirements requirements.sft-trl.txt python -m src.eliza_trainer.sft.train --config configs/sft_general_qwen3_5_0_8b.yaml
```

Resolve and validate config only (no training):

```bash
uv run --python 3.12 --with-requirements requirements.sft-trl.txt python -m src.eliza_trainer.sft.train --config configs/sft_general_qwen3_5_0_8b.yaml --dry-run-config
```

Bypass interactive policy confirmations (for non-interactive CI/jobs):

```bash
uv run --python 3.12 --with-requirements requirements.sft-trl.txt python -m src.eliza_trainer.sft.train --config configs/sft_general_qwen3_5_0_8b.yaml --assume-yes
```

## Config fields

- `config_schema_version`: must be `2`
- `identity.experiment_tag`: experiment label used in derived run names
- `identity.backend`: `trl` or `unsloth`
- `data.bucket`, `data.dataset_id`, `data.dataset_version`: manifest URI is derived as `s3://<bucket>/dataset_id=<id>/dataset_version=<version>/manifest.json`
- `data.train_split`, `data.eval_split`: split names in manifest
- `data.cache_mode`: `reuse` (default) or `refresh` (force redownload)
- `data.expected_manifest_sha256`: optional reproducibility guard
- `data.max_train_samples`, `data.max_eval_samples`: optional smoke-run caps
- `model.owner`, `model.name`: base model HF repo components
- `model.max_seq_len`: max tokenized sequence length
- `model.load_in_4bit`: set `true` for lower VRAM footprint on constrained GPUs
- `model.lora_*`: LoRA adapter config used when `load_in_4bit=true`
- `training.output_root`: parent directory for derived run output paths
- `training.*`: learning rate, epochs, batch sizes, scheduler, etc.
- `training.gradient_checkpointing`: enable to reduce activation memory
- `tracking.mlflow_tracking_uri`: optional override for tracking backend
- `hub.push_to_hub`: enable model publishing
- `hub.owner`: HF namespace owner for derived target repos
- `hub.publish_adapter`, `hub.publish_full_model`: explicit publish targets
- `hub.adapter_tag_strategy`, `hub.full_model_tag_strategy`: `run_name` / `none` / `custom`
- `hub.allow_existing_tags`: ignore tag `409 Conflict` and continue
- dataset manifest governance metadata (`governance.policy_classes_present`, `governance.policy_class_counts`, `governance.max_policy_class`) is logged at train start

Policy gates:

- If manifest includes `P3` or `P4`, training requires explicit confirmation unless `--assume-yes` is provided.
- Non-interactive runs fail closed when confirmation is required and `--assume-yes` is absent.
- If manifest includes `P4` and `hub.push_to_hub=true`, training fails before training starts.

Example hub config for adapter + merged publish:

```yaml
hub:
  push_to_hub: true
  owner: your-org
  publish_adapter: true
  publish_full_model: true
  adapter_tag_strategy: run_name
  full_model_tag_strategy: run_name
```

## Outputs

- Intermediate checkpoints under `training.output_dir`.
- Final checkpoint and tokenizer under `training.output_dir/final`.
- MLflow run with dataset lineage parameters and training metrics.
- MLflow tokenization-fit metrics (`train_context_fit_pct`, `eval_context_fit_pct`, truncation/drop counts).
- MLflow run diagnostics artifacts under `run_diagnostics/` (`effective_config.yaml`, host run log, and `crash_report.json` when a run fails).

## Ollama-first publish flow (GGUF)

If your deployment target is Ollama, keep only adapter publishing in SFT config and produce GGUF as a separate post-training step.

Recommended `hub` config:

```yaml
hub:
  push_to_hub: true
  owner: your-org
  publish_adapter: true
  publish_full_model: false
  adapter_tag_strategy: run_name
```

Then generate and publish GGUF from the adapter:

```bash
HF_HUB_TOKEN=... ./publish_gguf_from_lora.sh \
  --train-config configs/sft_hass_qwen3_5_0_8b.yaml \
  --tag v1.0.0
```

Notes:

- The script infers adapter repo from `--train-config` and infers GGUF repo by replacing `lora` with `gguf` in the repo name.
- You can still override with `--adapter-repo` and/or `--gguf-repo` when needed.
- Default output is F16 GGUF only (no quantization).
- Add `--quant Q4_K_M` when you want an additional quantized file.
- The script creates merged safetensors in a temporary directory and deletes them at exit.
- The script uploads `training_config.yaml` and `nlp_lab_provenance.json` so the GGUF repo records run config and `nlp-lab` git version used.

## Run hygiene

- Keep one dataset version per run for reproducibility.
- Do not mutate manifest content after training starts.
- Keep model/template config fixed for benchmark comparisons.

## Native Tool-Call Dataset Notes

- Tool-calling datasets can include structured fields (`tool_calls`, `tool_name`) in `messages` and `target_message`.
- Tokenization keeps these fields when present and still supports legacy `target_text` rows.
- For first runs on new synthetic cuts, start with a smoke config (`max_train_samples` / `max_eval_samples`) and verify:
  - non-zero `train_rows_effective` and `eval_rows_effective`
  - no large-scale row drops from template errors
  - expected behavior on tool-call prompts after training
