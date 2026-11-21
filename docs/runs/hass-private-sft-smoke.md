# First private HASS SFT smoke

Run this from Jupyter with access to the deployed Dagster jobs, private MinIO
buckets and DGX control token. Use the current `feature/training-boundaries`
checkout and its editable `packages/training-client` in the notebook kernel:
the installed image client must support `restricted_manifest_ack_sha256` before
submission. Restart the kernel after updating the package and use `runtime_source`
below to send the same worker and runtime profile to the DGX.

For an existing image, install from the updated checkout in the active notebook
kernel and then restart it:

```ipython
%pip install -e /home/jovyan/work/nlp-lab/packages/training-client
```

## Freeze and configure

Use the HASS export notebook to review all episodes, save explicit grouped
train/eval/excluded decisions and resolve the exact frozen export. Keep the
manifest URI and stored-byte SHA-256. Copy
`configs/sft_hass_private_smoke.template.yaml` to `configs/private-hass-smoke.yaml`
and set `data.bucket`, `data.dataset_id`, `data.dataset_version` and
`data.expected_manifest_sha256` from the export receipt. The template pins the
public Qwen3.5-0.8B model and tokenizer to the same commit, uses BF16 LoRA,
target-only loss, 12 optimizer steps and local MLflow tracking inside `artifacts/`.

With the SFT dependencies and access to the dataset, the following loads the
real splits, checks their bytes, renders and tokenizes all rows, reports maximum
context length, and exits before loading model weights. Run interactively to
review the P4 acknowledgement. The Slurm training command repeats these checks
in the allocation even if this optional separate preflight has passed.

```bash
python -m src.eliza_trainer.sft.train --config configs/private-hass-smoke.yaml --preflight-only
```

## Submit and monitor

Set `export_receipt` to the verified export result and `artifact_bucket` to the
configured private writable bucket. Use the current `eliza_training` adapter
with the notebook's MinIO secret mounts. The signed bundle URL must outlive
queue wait and bootstrap. Confirm the DGX is available before switching modes.

```python
from uuid import uuid4
from eliza_training import ElizaTraining

repo = "/home/jovyan/work/nlp-lab"
pin = {"uri": export_receipt["manifest_uri"], "sha256": export_receipt["manifest_sha256"]}
training = ElizaTraining(bucket=artifact_bucket)
run_id = str(uuid4())
print(run_id)  # Retain before submitting; reconcile this ID if the response is lost.
# Set this to the exact manifest digest only after explicitly accepting this P4
# dataset for the private experiment; do not use a notebook-wide auto-confirm flag.
acknowledged_digest = input("Enter the reviewed P4 manifest SHA-256: ").strip()
if acknowledged_digest != pin["sha256"]:
    raise ValueError("Acknowledgement does not match the frozen manifest")
training.environment("training")
run = training.submit(
    repo=repo, config="configs/private-hass-smoke.yaml", run_id=run_id,
    dataset_manifest=pin, restricted_manifest_ack_sha256=acknowledged_digest,
    runtime_source=f"{repo}/packages/training-client", runtime_profile="trl",
    cpus=8, memory_mb=32768, time_minutes=90,
)
training.watch(run_id, on_update=print)
```

For an ambiguous submit, use `training.recover_submission(run_id)` rather than
creating a replacement. Check the terminal worker `result.json` in the private run
prefix and `train.log`, `environment.json`, `dataset-manifest.json`, and
`artifact-index.json`. A scheduler COMPLETED state alone is not acceptance.

## Acceptance evidence

1. Check the worker outcome is `succeeded`, both preflight splits are nonempty
   with no lost/truncated rows, and the final Trainer global step is 12.
2. Download the `final/` adapter and tokenizer files using the private storage
   adapter and verify their hashes against the artifact index. Check that
   `adapter_config.json` records the configured base model and revision.
3. In a compatible environment, load the exact pinned
   `Qwen3_5ForConditionalGeneration` base, then attach the local adapter with
   `PeftModel.from_pretrained(base, local_final_path)`. Use deterministic decoding
   on fixed held-out call and post-tool speech examples, comparing with the same
   base with its adapter disabled. Keep inputs and results private and never
   execute the generated calls during this test. The small smoke demonstrates
   the round trip; it is not evidence of generalization.
4. After the GPU jobs finish, call `training.environment("inference")` and
   verify the saved inference services are healthy.
