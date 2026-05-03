# MLflow Tracking Conventions

`nlp-lab` uses MLflow as the source of truth for experiment lineage.

Default tracking target is `https://mlflow.chezombor.com/`.

## Required run parameters

Every SFT run must log:

- `dataset_manifest_uri`
- `dataset_manifest_sha256`
- `dataset_id`
- `dataset_version`
- `dataset_target_mode`
- `dataset_token_budget`
- `dataset_context_overhead_tokens`
- `dataset_split_seed`
- `dataset_eval_percent`
- `model_name`
- `max_seq_len`
- `assistant_only_loss=true`

## Required metrics

- `train_rows`
- `eval_rows`
- `total_rows`
- training/eval loss from trainer logs

## Suggested naming conventions

- Experiment: `<family>_<stage>` (example `qwen3_5_0_8b_sft`)
- Run name: `<dataset_id>_<dataset_version>_<target_mode>`

## Artifact expectations

- YAML config used for the run
- training logs
- optional generated samples for qualitative checks

## Reproducibility checklist

- Dataset manifest hash logged
- Split seed logged
- Model/tokenizer repo string logged
- Output checkpoint path retained
