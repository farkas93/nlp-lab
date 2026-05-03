# SFT Data Contract

This contract defines what `nlp-lab` expects from `build_general_sft_dataset` outputs.

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

Recommended manifest fields for lineage:

- `construction.target_mode`
- `construction.token_budget`
- `construction.context_overhead_tokens`
- `construction.split_seed`
- `construction.eval_percent`
- `stats.train_rows`, `stats.eval_rows`, `stats.total_rows`

## Row-level requirements

Each SFT row must include:

- `messages`: ordered context messages
- `target_text`: assistant target string

Expected `messages` item shape:

- `role`: `system|user|assistant|tool`
- `content`: text content

Optional but recommended metadata:

- `session_id`
- `source_export_ts`
- `cleaned_run_ts`
- `session_language`
- `target_mode`

## Training semantics

- Context messages are prompt context.
- `target_text` is the supervised assistant output.
- Loss is computed only on assistant target tokens.
- Context tokens are masked with `-100` labels.

## Target modes

- `single_turn`: each assistant turn becomes one target example.
- `merge_consecutive_assistant`: contiguous assistant turns are merged with separator `\n\n`.

`nlp-lab` supports both modes and reads mode metadata from manifest/rows for lineage.
