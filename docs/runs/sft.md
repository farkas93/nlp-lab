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

`start_sft.sh` uses `uv run --python 3.12 --with-requirements requirements.txt ...` and only starts local MLflow via docker-compose when `MLFLOW_TRACKING_URI` points to `localhost` or `127.0.0.1`.

Equivalent direct command:

```bash
uv run --python 3.12 python src/sft_finetune.py --config configs/sft_general_qwen3_5_0_8b.yaml
```

## Config fields

- `data.dataset_manifest_uri`: manifest path (`s3://...` or local path)
- `data.train_split`, `data.eval_split`: split names in manifest
- `model.model_name`: base model HF repo, example `Qwen/Qwen3.5-0.8B`
- `model.max_seq_len`: max tokenized sequence length
- `model.load_in_4bit`: set `true` for lower VRAM footprint on constrained GPUs
- `model.lora_*`: LoRA adapter config used when `load_in_4bit=true`
- `training.*`: output directory, learning rate, epochs, batch sizes
- `training.gradient_checkpointing`: enable to reduce activation memory
- `tracking.mlflow_tracking_uri`: optional override for tracking backend
- `hub.push_to_hub`, `hub.repo_name`: optional publishing

## Outputs

- Intermediate checkpoints under `training.output_dir`.
- Final checkpoint and tokenizer under `training.output_dir/final`.
- MLflow run with dataset lineage parameters and training metrics.

## Run hygiene

- Keep one dataset version per run for reproducibility.
- Do not mutate manifest content after training starts.
- Keep model/template config fixed for benchmark comparisons.
