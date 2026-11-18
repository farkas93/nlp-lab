# nlp-lab training client and execution package

This is the canonical owner of run bundles, Slurm submission/reconciliation,
host-control client transport, execution workers, GPU probes and training runtime
profiles. The base package needs only boto3/PyYAML and never imports PyTorch,
Transformers or Spark during client import.

```bash
python -m pip install ./packages/training-client
```

```python
from nlp_training import TrainingClient
# Inject your infrastructure's storage adapter (read_json/read_bytes/create_bytes).
training = TrainingClient(storage=my_storage, bucket="private-artifacts")
training.environment("training")
run = training.submit(repo="/workspace/nlp-lab", config="configs/private.yaml",
                      dataset_manifest={"uri": "s3://bucket/frozen/manifest.json", "sha256": "..."})
training.watch(run["run_id"])
```

The worker is `python -m nlp_training.worker`; the infrastructure probe is
`python -m nlp_training.smoke`. Base installation is CPU-safe. Jobs carry their
bootstrap/runtime source and prepare GPU dependencies inside their Slurm
allocation with `nlp_training.provision`, using the caller-selected profile in
`nlp_training/profiles/runtime.json`. Root `requirements.sft-trl.txt` and
`requirements.dpo-trl.txt` consume the same canonical TRL requirements. The ARM64
CUDA overlay adds its pinned wheel to that common profile; inference repositories
do not duplicate these dependency versions.

Jupyter packages the base distribution and supplies its generic MinIO adapter.
The host only advertises scheduler capabilities and controls GPU handover; it
does not install or pin this package. Submitted bundle version 2 records the
runtime source/profile and uses signed downloads or an inline probe payload.
See `docs/runs/jupyter-slurm-training.md` for job credentials and recovery.

Run tests with `workspace-dev bootstrap` and `workspace-dev verify` in this
subproject. They require no GPU frameworks. Model fine-tuning dependencies and
actual GPU compatibility are validated separately in the selected runtime.
