# SHARED_CACHE

## Coordination
- Agent name: `bravo`
- Scope owner (current): manifest-driven DPO runtime foundation in `nlp-lab`
- Started: 2026-05-18 UTC
- Last pushed commit (this scope): `80122f7`
- Concurrency note: keep SFT runtime untouched while adding DPO path in parallel.

## Bravo Active/Reserved Files
- `src/eliza_trainer/dpo/*`
- `src/dpo_dataset_loader.py`
- `start_dpo.sh`
- `configs/dpo_hass_qwen3_5_0_8b.yaml`
- `requirements.dpo-trl.txt`
- `docs/data-contracts/dpo.md`
- `docs/runs/dpo.md`
- `docs/README.md`
- `README.md`
- `src/eliza_trainer/__init__.py`

## Planned This Batch
1. Add DPO config loader (schema-v2 style, derived naming, manifest URI derivation).
2. Add DPO dataset loader for manifest + parquet splits (`prompt/chosen/rejected`).
3. Add TRL DPO backend and `python -m src.eliza_trainer.dpo.train` entrypoint.
4. Replace archived `start_dpo.sh` with active launcher.
5. Add DPO docs and baseline config.

## Implemented In This Batch
- Added DPO runtime package under `src/eliza_trainer/dpo/`:
  - `run_config.py`
  - `dataset_loader.py`
  - `backends/trl_backend.py`
  - `train.py`
- Added shared manifest loader `src/dpo_dataset_loader.py` for DPO parquet manifests.
- Replaced archived `start_dpo.sh` with active launcher and dry-run support.
- Added `requirements.dpo-trl.txt` and baseline config `configs/dpo_hass_qwen3_5_0_8b.yaml`.
- Added docs:
  - `docs/data-contracts/dpo.md`
  - `docs/runs/dpo.md`
  - updated `docs/README.md`, `docs/architecture.md`, and root `README.md`.

## Validation
- `uv run --python 3.12 python -m compileall src/eliza_trainer/dpo src/dpo_dataset_loader.py`
- `uv run --python 3.12 --with-requirements requirements.dpo-trl.txt python -m src.eliza_trainer.dpo.train --config configs/dpo_hass_qwen3_5_0_8b.yaml --dry-run-config`
- `./start_dpo.sh configs/dpo_hass_qwen3_5_0_8b.yaml --dry-run-config`

## Alpha Alignment Check
- Verified alignment with alpha via `eliza-data-pipelines/SHARED_CACHE.md` before starting this batch.
- No overlap with alpha-owned paths (`pipeline_core/external_normalization/*`, normalization ops/jobs, related run-configs).

## Guardrails
- Keep SFT code paths and existing behavior unchanged.
- Reuse policy gates from `src/eliza_trainer/common/data_policy.py`.
- No compatibility alias to legacy archived DPO flow.

## Next Action Queue (post-push)
- [x] Add DPO troubleshooting section in `docs/troubleshooting.md`.
- [x] Add small synthetic manifest fixture tests for `src/dpo_dataset_loader.py` and `dpo/run_config.py`.
- [x] Add DPO metric/eval notes (tool-call decision, arg precision) in docs.

## Upcoming Work Window (Bravo)
- Keep active ownership on `src/eliza_trainer/dpo/*` and DPO docs during hardening.
- First hardening patch:
  - loader/config unit tests
  - troubleshooting docs
  - minor trainer compatibility guardrails if TRL API drift appears
