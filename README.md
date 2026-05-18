# nlp-lab

`nlp-lab` trains post-training models from manifest-driven datasets.

## Documentation

- `docs/README.md`
- `docs/architecture.md`
- `docs/config-schema-v2.md`
- `docs/data-contracts/sft.md`
- `docs/migration-v1-to-v2.md`
- `docs/runs/sft.md`
- `docs/experiments/mlflow.md`
- `docs/runs/benchmarking.md`
- `docs/troubleshooting.md`

## Repository areas

- Production SFT path: `src/eliza_trainer/sft/`
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

For Ollama-first deployment, publish LoRA adapters from training, then export GGUF with:

```bash
./publish_gguf_from_lora.sh --help
```

The GGUF publish script can also upload the exact training config and `nlp-lab` git provenance to the model repo.

Direct Python command:

```bash
uv run --python 3.12 --with-requirements requirements.sft-trl.txt python -m src.eliza_trainer.sft.train --config configs/sft_general_qwen3_5_0_8b.yaml
```

## Dataset contract expectations

The loader expects a manifest JSON with `splits` entries containing `split` and `key`, with Parquet files for `train` and `eval` rows.

Each dataset row should include at least:

- `messages` (list of role/content context messages)
- `target_text` (assistant target string)

Tokenization applies model chat templates at runtime, and training uses assistant-only loss masking.

Full contract details: `docs/data-contracts/sft.md`.
