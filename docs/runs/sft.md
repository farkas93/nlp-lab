# SFT Runbook

## Prerequisites

1. `build_general_sft_dataset` has produced a manifest and Parquet splits.
2. `.env` exists and contains model, tracking, and storage credentials.
3. YAML run config points to the correct dataset manifest URI.
4. `uv` is installed and Python 3.12 is available.

## Minimal run

```bash
./start_sft.sh configs/sft_general_qwen3_5_0_8b.yaml
```

`start_sft.sh` uses `uv run --python 3.12 --with-requirements ...` and only starts local MLflow via docker-compose when `MLFLOW_TRACKING_URI` points to `localhost` or `127.0.0.1`.
`training.backend` controls trainer backend selection (`trl` or `unsloth`) and dependency overlays are selected explicitly:

- `trl` -> `requirements.sft-trl.txt`
- `unsloth` -> `requirements.sft-unsloth.txt`

For `unsloth`, `start_sft.sh` also runs a preflight import check before launching training.
The unsloth requirements overlay is intentionally version-pinned and independent from the vLLM/TRL latest stack to avoid known `transformers` 5.x compatibility breakages.
Each run also persists host logs under `logs/sft/<timestamp>_<backend>.log` (or `SFT_LOG_DIR` if set), and exports the log path to the trainer process.

Equivalent direct command:

```bash
uv run --python 3.12 --with-requirements requirements.sft-trl.txt python -m src.eliza_trainer.sft.train --config configs/sft_general_qwen3_5_0_8b.yaml
```

## Config fields

- `data.dataset_manifest_uri`: manifest path (`s3://...` or local path)
- `data.train_split`, `data.eval_split`: split names in manifest
- `data.cache_mode`: `reuse` (default) or `refresh` (force redownload)
- `data.expected_manifest_sha256`: optional reproducibility guard
- `data.max_train_samples`, `data.max_eval_samples`: optional smoke-run caps
- `model.model_name`: base model HF repo, example `Qwen/Qwen3.5-0.8B`
- `model.max_seq_len`: max tokenized sequence length
- `model.load_in_4bit`: set `true` for lower VRAM footprint on constrained GPUs
- `model.lora_*`: LoRA adapter config used when `load_in_4bit=true`
- `training.*`: output directory, learning rate, epochs, batch sizes
- `training.backend`: `trl` or `unsloth`
- `training.gradient_checkpointing`: enable to reduce activation memory
- `tracking.mlflow_tracking_uri`: optional override for tracking backend
- `hub.push_to_hub`, `hub.repo_name`: optional publishing
- `hub.full_model`: merge adapter into a full model export after training
- `hub.full_model_repo_name`: optional HF repo for merged model (defaults to `<repo_name>-merged`)
- `hub.adapter_tag`: optional HF git tag to create on adapter repo after publish
- `hub.full_model_tag`: optional HF git tag to create on merged repo after publish

Example hub config for adapter + merged publish:

```yaml
hub:
  push_to_hub: true
  repo_name: your-org/qwen35-08b-hass-tools-adapter
  full_model: true
  full_model_repo_name: your-org/qwen35-08b-hass-tools-merged
  adapter_tag: v1.0.0
  full_model_tag: v1.0.0
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
  repo_name: your-org/qwen35-08b-hass-tools-lora
  adapter_tag: v1.0.0
  full_model: false
```

Then generate and publish GGUF from the adapter:

```bash
HF_HUB_TOKEN=... ./publish_gguf_from_lora.sh \
  --adapter-repo your-org/qwen35-08b-hass-tools-lora \
  --gguf-repo your-org/qwen35-08b-hass-tools-gguf \
  --train-config configs/sft_hass_qwen3_5_0_8b.yaml \
  --tag v1.0.0
```

Notes:

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
