# Architecture

`nlp-lab` is the training and experimentation layer that consumes curated datasets and produces model checkpoints with full lineage in MLflow.

## End-to-end flow

1. `eliza-data-pipelines` exports and curates sessions (`raw` -> `review` -> `cleaned`).
2. `eliza-data-pipelines` materializes SFT datasets into `sft-datasets` as Parquet + manifest.
3. `nlp-lab` loads the manifest and Parquet splits.
4. `nlp-lab` applies chat template rendering at runtime for the selected model tokenizer.
5. `nlp-lab` trains with assistant-only loss masking.
6. `nlp-lab` logs run metadata and artifacts to MLflow.

## Design principles

- Canonical SFT data is model-agnostic.
- Chat template rendering is model-specific and happens at train time.
- Dataset lineage is immutable per dataset version.
- Experiment metadata must always include dataset manifest and hash.

## Repository components

- `configs/`: YAML run configs.
- `src/sft_run_config.py`: YAML config parser and env integration.
- `src/sft_dataset_loader.py`: manifest + Parquet loader and tokenization bridge.
- `src/training/loss_masking.py`: assistant-only loss collator.
- `src/sft_finetune.py`: end-to-end SFT entrypoint.

## Outputs

- Model checkpoints under configured `training.output_dir`.
- Final model under `<output_dir>/final`.
- MLflow run with dataset lineage, model/training params, and metrics.
