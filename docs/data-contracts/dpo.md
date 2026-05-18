# DPO Data Contract

This contract defines what `nlp-lab` expects from `materialize_dpo_release` outputs.

## Storage layout

- `dataset_id=<id>/dataset_version=<version>/split=train/part-*.parquet`
- `dataset_id=<id>/dataset_version=<version>/split=eval/part-*.parquet`
- `dataset_id=<id>/dataset_version=<version>/manifest.json`

## Manifest requirements

Manifest must be a JSON object with:

- `dataset_id`
- `dataset_version`
- `splits` list where each item contains:
  - `split` (for example `train`, `eval`)
  - `key` (object key or URI to Parquet file)

Recommended manifest fields for lineage/governance:

- `construction.split_seed`
- `construction.eval_percent`
- `stats.train_rows`, `stats.eval_rows`, `stats.total_rows`
- `governance.policy_classes_present`
- `governance.policy_class_counts`
- `governance.max_policy_class`

## Row-level requirements

Each DPO row must include:

- `prompt`
- `chosen`
- `rejected`

Optional but recommended metadata:

- `example_id`
- `session_id`
- `source_family`
- `source_dataset`
- `source_variant`
- `tool_contract`
- `scenario_type`
- `policy_class`
