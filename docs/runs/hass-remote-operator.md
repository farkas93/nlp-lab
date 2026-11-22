# Training workspace: installation, runs and live feedback

Jupyter is a remote operator. Dagster prepares data; Slurm tokenizes, trains and
evaluates. Notebook kernels do not import Torch or download model weights.

## Install once in Jupyter

Update the `nlp-lab` checkout, then run this in JupyterLab Terminal (adjust the
checkout path if necessary):

```bash
python /home/jovyan/eliza/nlp-lab/scripts/setup_training_workspace.py
```

This creates `$HOME/eliza/training/training.env`, both operator notebooks and
`runs/`. Existing configuration and edited notebooks are preserved. A changed
template is installed as `*.updated-<hash>.ipynb` rather than overwriting your
notebook. Open the newly installed version when updating code.

The MLflow REST write/read probe was exercised from the coding workspace at
`https://mlflow.chezombor.com/`, experiment `eliza_pipeline_connectivity`, run
`88ee7c32876c4f7ebd67c7cc27d696ce`. This proves that endpoint accepted and returned
a metric, not that the DGX network path works. Remote preflight repeats it there.

Update the lightweight client in the kernel if the image has an older package:

```python
%pip install --target /home/jovyan/eliza/training/.client --upgrade /home/jovyan/eliza/nlp-lab/packages/training-client
```

Restart the kernel afterwards. The notebooks select this overlay before imports.
Edit the visible `training.env` through the file browser. It is loaded explicitly;
no manual `load_dotenv` calls are needed. Precedence is notebook overrides → this
file → deployed environment. The setup reports sources for nonsecret settings.
MLflow credentials, if needed, are file references; credential values are never
written in notebook plans or bundles. Existing MinIO secret mounts remain usable.

## What appears under runs/

```text
runs/<dataset-label>/dataset-progress.json  # exact Dagster run IDs
runs/<dataset-label>/dataset-preview.json   # coverage, split counts, examples
runs/<dataset-label>/release.json           # frozen URI and checksum for all models
runs/<training-label>/plan.yaml             # requested model/data/resources
runs/<training-label>/<stage>/run.json      # UUID and immutable binding
runs/<training-label>/<stage>/resolved-config.yaml
runs/<training-label>/<stage>/submission.json
runs/<training-label>/<stage>/status.json
runs/<training-label>/<stage>/live-log.txt
runs/<training-label>/<stage>/result.json
runs/<training-label>/<stage>/*-report.json  # verified returned reports (names vary)
```

Dataset folders appear when setup runs. Stage records appear when a job is
prepared, not when a notebook file is merely opened. Existing job 8 artifacts
remain in `slurm-artifacts/training/runs/94dbf110-1d23-43f6-88cb-7c5f4304b293/`;
they are not silently moved into the new workspace.

## Dataset → three model runs

Deploy the pipeline image containing `prepare_hass_household_cut` and
`export_hass_household_cut` (both code server and run workers). Open
`hass-dataset.ipynb`, set capture path and household UUID, then run its sections.
The requested default is 1,200 episodes; the recipe reports actual supported
counts rather than padding to that quota. Freeze the previewed index and retain
`release.json`. The cut remains private, unreviewed, simulator-generated and
single-household; it is not a canonically moderated release.

Open `hass-sft.ipynb` three times, use separate RUN_LABEL values and one shared
DATASET_LABEL. Presets:

| Preset | Checkpoint | Native tokenizer status | DGX training status |
| --- | --- | --- | --- |
| `sft_hass_household_qwen08.yaml` | Qwen3.5-0.8B | pinned call/reply tests pass | 12-step pilot passed; larger cut not yet run |
| `sft_hass_household_gemma4.yaml` | Gemma 4 E4B-it | pinned native nested-call/reply tests pass | longest-row qualification and SFT pending |
| `sft_hass_household_qwen9.yaml` | Qwen3.5-9B | pinned call/reply tests pass | longest-row qualification and SFT pending |

Presets request 100 steps, BF16 LoRA, batch one and accumulation four. Runtime and
memory/time overrides are explicit in the setup cell. Context defaults to 128K
admission, with dynamic row lengths and no truncation; native and actual GPU fit
are checked separately. These are starting experiments, not verified optimal recipes.

## MLflow is checked, not assumed

The remote preflight creates a labelled `remote-connectivity-check` run on the
configured server, logs one metric and reads it back. Authentication, route or
write failures stop preflight with the failing operation. The returned report
contains its server URL, experiment ID and run URL. Training uses the same URI
and forwarded credentials; its verified `training-summary.json` carries the real
training run ID/link. No SQLite fallback occurs in this notebook. Job 8 used the
old local SQLite configuration and will not retroactively appear on the server.

## Streaming and failure

The worker publishes a bounded 32 KiB live log every three seconds, including
while output is quiet. Notebook polls save and display it with stage and step
progress in one updating view. Full logs remain in object storage; the live log
is a tail, not the complete archive. Bootstrap/dependency setup reports a stage
but may have no trainer log yet. Slurm RUNNING and a nonterminal exit code are
not success; worker result and archive verification decide completion.

Re-run a monitor to reconnect. A lost submit response requires recovery of the
original UUID; never delete intent markers and resubmit. Use new labels only
for deliberately new attempts/plans.

## Job 8 reload

Use SOURCE_RUN_ID `94dbf110-1d23-43f6-88cb-7c5f4304b293`, the original frozen
manifest pins and `sft_hass_private_smoke.template.yaml`. Preflight verifies the
server and context; skip training, then run remote reload/evaluation. It does
not retrain or deploy the adapter. Restore inference explicitly when jobs end.
Live access to MinIO, DGX and MLflow is required; source tests cannot certify it.
