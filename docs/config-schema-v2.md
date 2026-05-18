# SFT Config Schema v2

`nlp-lab` SFT configs use a strict schema with `config_schema_version: 2`.

## Required top-level sections

- `config_schema_version`
- `identity`
- `data`
- `model`
- `training`
- `hub`
- `tracking`
- `runtime`

## Core fields

- `identity.experiment_tag`: slug used in derived naming
- `identity.backend`: `trl` or `unsloth`
- `data.bucket`, `data.dataset_id`, `data.dataset_version`: derive manifest URI
- `model.owner`, `model.name`: derive model FQN as `{owner}/{name}`
- `hub.owner`: target HF owner namespace when pushing
- `hub.publish_adapter`, `hub.publish_full_model`: explicit publish intents

## Derived values

- `model.model_name` -> `{model.owner}/{model.name}`
- `data.manifest_uri` -> `s3://{bucket}/dataset_id={dataset_id}/dataset_version={dataset_version}/manifest.json`
- `training.experiment_name` -> `{slug(model.name)}_{identity.experiment_tag}`
- `training.run_name` -> `{experiment_tag}-{backend}-{utc_ts}` (+ optional `-run_label`)
- `training.output_dir` -> `{training.output_root}/{slug(model.name)}/{experiment_tag}/{backend}/{utc_ts}`

## Hub naming policy

- If `hub.owner != model.owner`, base stem is `{model.name}-{experiment_tag}-{backend}`.
- If `hub.owner == model.owner`, base stem is `{model.name}` (same-owner lineage mode).
- Adapter target repo always ends with `-lora`.
- Full-model target repo never ends with `-lora`.

## Tag policy

- `hub.adapter_tag_strategy` and `hub.full_model_tag_strategy`:
  - `run_name` -> use resolved run name as tag
  - `none` -> skip tag creation
  - `custom` -> use explicit tag field
- `hub.allow_existing_tags: true` makes existing HF tags (`409`) non-fatal.

## Dry-run validation

Validate/resolve config only (no training):

```bash
uv run --python 3.12 --with-requirements requirements.sft-trl.txt \
  python -m src.eliza_trainer.sft.train --config configs/sft_hass_qwen3_5_0_8b.yaml --dry-run-config
```

This prints the resolved run plan and exits.
