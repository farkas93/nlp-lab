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

System telemetry parameters (default-on):

- `system_host_name`
- `system_host_name_mode` (`raw` by default; `anonymized` when opted out)
- `system_host_os`
- `system_host_release`
- `system_host_arch`
- `system_python_version`
- `system_cpu_logical_cores`
- `system_ram_total_gb`
- `system_gpu_count`

## Required metrics

- `train_rows`
- `eval_rows`
- `total_rows`
- training/eval loss from trainer logs

System telemetry metrics (default-on):

- `cpu_util_pct_avg`, `cpu_util_pct_peak`
- `ram_util_pct_avg`, `ram_util_pct_peak`
- per-GPU metrics when available (for example `gpu_0_util_pct_avg`, `gpu_0_mem_util_pct_peak`, `gpu_0_mem_used_gb_peak`)
- `run_wall_time_sec`

Opt-out controls:

- `NLP_LAB_LOG_SYSTEM_TELEMETRY=0`
- `NLP_LAB_LOG_RAW_HOSTNAME=0`
- `NLP_LAB_SYSTEM_TELEMETRY_INTERVAL_SEC=<seconds>`

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
