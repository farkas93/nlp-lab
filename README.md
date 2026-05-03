# nlp-lab

`nlp-lab` trains post-training models from manifest-driven datasets.

## Documentation

- `docs/README.md`
- `docs/architecture.md`
- `docs/data-contracts/sft.md`
- `docs/runs/sft.md`
- `docs/experiments/mlflow.md`
- `docs/runs/benchmarking.md`
- `docs/troubleshooting.md`

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

- `data.dataset_manifest_uri` -> manifest created by `build_general_sft_dataset`
- `model.model_name` -> base model HF repo
- `training.*` -> experiment and training hyperparameters

See full runbook: `docs/runs/sft.md`.

## Run SFT

Run training via `uv` (Python 3.12):

```bash
./start_sft.sh
```

`start_sft.sh` only starts local MLflow via docker-compose when `MLFLOW_TRACKING_URI` points to `localhost` or `127.0.0.1`.

Run with custom config path:

```bash
./start_sft.sh configs/sft_general_qwen3_5_0_8b.yaml
```

Direct Python command:

```bash
uv run --python 3.12 python src/sft_finetune.py --config configs/sft_general_qwen3_5_0_8b.yaml
```

## Dataset contract expectations

The loader expects a manifest JSON with `splits` entries containing `split` and `key`, with Parquet files for `train` and `eval` rows.

Each dataset row should include at least:

- `messages` (list of role/content context messages)
- `target_text` (assistant target string)

Tokenization applies model chat templates at runtime, and training uses assistant-only loss masking.

Full contract details: `docs/data-contracts/sft.md`.
