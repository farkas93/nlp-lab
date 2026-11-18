# Jupyter to standalone DGX training

Ownership:

- `nlp-lab/packages/training-client`: training client, worker, bundle protocol,
  runtime profiles and GPU probe;
- `eliza-inference`: host installation, Slurm configuration, environment exclusion
  and authenticated lifecycle/token endpoints;
- `eliza-spark`: Jupyter image, generic storage adapter, secrets and notebooks;
- `eliza-data-pipelines`: frozen dataset selection/export.

Install host components through `eliza-inference`'s `./scripts/setup control`.
That installer consumes a pinned snapshot of the training subproject and delegates
runtime preparation to `python -m nlp_training.provision`. It keeps GPU libraries
separate from control. Jupyter installs only the base package, so no CUDA packages
are needed in the notebook image.

In Jupyter, the convenience `eliza_training.ElizaTraining` is a thin adapter that
injects `ElizaStorage` into the canonical `nlp_training.TrainingClient`.

```python
from eliza_training import ElizaTraining
training = ElizaTraining(bucket="<private-artifact-bucket>")
training.environment("training")
run = training.submit(repo="/home/jovyan/work/nlp-lab", config="configs/private.yaml",
    dataset_manifest={"uri": "s3://<bucket>/<manifest>", "sha256": "<exact digest>"})
training.watch(run["run_id"], on_update=print)
```

Freeze exact source/config bytes before submission. Private configs require
`hub.push_to_hub: false` and outputs inside `artifacts/`. The worker validates the
bundle and frozen dataset manifest before running the existing SFT/DPO module.
After restarting a notebook, reconnect using the saved run ID. Do not automatically
retry an ambiguous submission; reconcile it with `recover_submission`.

The scheduler owns placement/status/cancellation. The host controller owns mode
switches and inference restoration; notebook kernel exit never switches modes or
kills jobs. Return to inference explicitly after the queue has drained.
