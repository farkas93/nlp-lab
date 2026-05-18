# SFT Config Migration: v1 -> v2

Schema v1 keys are not accepted anymore. Use `config_schema_version: 2`.

## Field mapping

- `training.backend` -> `identity.backend`
- `model.model_name` -> `model.owner` + `model.name`
- `data.dataset_manifest_uri` -> `data.bucket` + `data.dataset_id` + `data.dataset_version`
- `training.experiment_name` -> derived automatically
- `training.run_name` -> derived automatically
- `training.output_dir` -> derived automatically from `training.output_root`
- `hub.repo_name` -> derived automatically (or `hub.repo_adapter` override)
- `hub.full_model` -> `hub.publish_full_model`

## Minimal v2 skeleton

```yaml
config_schema_version: 2

identity:
  experiment_tag: hass_sft
  backend: trl

data:
  bucket: sft-datasets
  dataset_id: general
  dataset_version: v1.0.1

model:
  owner: Qwen
  name: Qwen3.5-0.8B

training:
  output_root: ./outputs

hub:
  push_to_hub: true
  owner: zskalo
  publish_adapter: true
  publish_full_model: false
  adapter_tag_strategy: run_name

tracking:
  mlflow_tracking_uri: https://mlflow.chezombor.com/

runtime:
  hf_model_cache_dir: ./hf_models
```

## Fast validation

```bash
./start_sft.sh configs/sft_hass_qwen3_5_0_8b.yaml
```

or validate without training:

```bash
uv run --python 3.12 --with-requirements requirements.sft-trl.txt \
  python -m src.eliza_trainer.sft.train --config configs/sft_hass_qwen3_5_0_8b.yaml --dry-run-config
```
