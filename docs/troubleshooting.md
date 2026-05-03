# Troubleshooting

## Manifest cannot be loaded

- Verify `data.dataset_manifest_uri` points to an existing object/path.
- If using S3, check `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_ENDPOINT_URL`, `AWS_REGION`.

## No train/eval files found in manifest

- Confirm manifest `splits` contains entries for configured `train_split` and `eval_split`.
- Confirm each split item has a valid `key`.

## Chat template/tokenization failures

- Verify model tokenizer supports chat templates.
- Inspect a sample row: `messages` must be a list of role/content dicts.

## OOM during training

- Lower `model.max_seq_len`.
- Lower `training.train_batch_size`.
- Increase `gradient_accumulation_steps` to preserve effective batch size.

## MLflow run not visible

- Check `MLFLOW_TRACKING_URI` in `.env` or YAML `tracking.mlflow_tracking_uri`.
- Ensure `docker compose up -d` has started MLflow service.

## Dependency resolution issues on Python 3.14

- Some optional packages (notably around vLLM/torch combinations) may not resolve for all ABIs.
- For SFT unit tests, use a minimal dependency subset when needed.
