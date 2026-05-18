# DPO Runbook

## Prerequisites

1. `materialize_dpo_release` has produced a manifest and Parquet splits.
2. `.env` exists and contains model, tracking, and storage credentials.
3. YAML run config points to the correct DPO manifest via `data.bucket`, `data.dataset_id`, `data.dataset_version`.
4. `uv` is installed and Python 3.12 is available.

## Minimal run

```bash
./start_dpo.sh configs/dpo_hass_qwen3_5_0_8b.yaml
```

Resolve and validate config only (no training):

```bash
./start_dpo.sh configs/dpo_hass_qwen3_5_0_8b.yaml --dry-run-config
```

## Config fields

- `config_schema_version`: must be `2`
- `identity.experiment_tag`: experiment label used in derived run names
- `identity.backend`: currently `trl`
- `data.bucket`, `data.dataset_id`, `data.dataset_version`: manifest URI is derived as `s3://<bucket>/dataset_id=<id>/dataset_version=<version>/manifest.json`
- `data.train_split`, `data.eval_split`: split names in manifest
- `data.cache_mode`: `reuse` (default) or `refresh`
- `data.expected_manifest_sha256`: optional reproducibility guard
- `model.owner`, `model.name`: base model HF repo components
- `model.load_in_4bit`: optional QLoRA-style training
- `training.*`: learning rate, epochs, batch sizes, scheduler, etc.
- `dpo.beta`: DPO beta hyperparameter
- `dpo.max_prompt_length`: max prompt token length in trainer
- `dpo.max_length`: max total sequence length in trainer
- `hub.push_to_hub`: enable model publishing
- `hub.owner`: HF namespace owner for derived target repos
- `hub.publish_adapter`, `hub.publish_full_model`: explicit publish targets

## Outputs

- Intermediate checkpoints under `training.output_dir`.
- Final checkpoint and tokenizer under `training.output_dir/final`.
- MLflow run with dataset lineage, governance policy metadata, and training metrics.
- MLflow run diagnostics artifacts under `run_diagnostics/` (`effective_config.yaml`, host run log, and `crash_report.json` on failures).

## Policy gates

- If dataset governance includes `P3` or `P4`, training requires explicit confirmation.
- Use `--assume-yes` for non-interactive intentional runs.
- If governance includes `P4`, training refuses to continue when `hub.push_to_hub=true`.

## DPO eval metrics (recommended)

- Track pairwise preference quality with standard DPO validation loss and reward margin trend.
- Add tool-call decision accuracy on eval examples where both `chosen` and `rejected` represent callable actions.
- Add argument precision for tool rows by comparing predicted argument keys/values against expected fields.
- Report both strict-match and relaxed-match slices (strict JSON equality, relaxed key overlap + normalized scalar values).
- Segment metrics by `scenario_type` and `policy_class` to avoid hiding regressions in high-risk subsets.
