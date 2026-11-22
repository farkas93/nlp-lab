# HASS training from a remote control notebook

Keep notebooks and run journals in `$HOME/eliza/training`. Copy
`configs/training-workspace.env.example` to `.env` in that directory. Set your
checkout path, DGX control endpoint and token-file path, private artifact bucket,
and remote MLflow URI. Use mounted credential files, never credential values in
the notebook. Nonempty settings resolve as notebook override → `.env` → pod env.
The helper does not export the dotenv file. Only the storage credentials and
explicit MLflow authentication settings needed by the worker are passed to jobs.

Use the updated lightweight client. If the Jupyter image has an older client,
run this in a Jupyter cell (adjust paths to match your checkout), then restart
the kernel before opening the new notebook:

```python
%pip install --target /home/jovyan/eliza/training/.client --upgrade /home/jovyan/eliza/nlp-lab/packages/training-client
```

Open `notebooks/hass-sft.ipynb`. Its setup cell holds the run label, model preset,
frozen manifest URI and SHA-256, and optional resource overrides. Use the dataset
sections of the older end-to-end notebook to prepare or reconnect to an export.
Each launch has its own UUID and journal under `runs/<label>/`; use a new label
for a changed plan. For an uncertain submission, recover the original UUID
before creating another attempt.

The context check runs in a CPU-only DGX Slurm allocation and loads the tokenizer and
config only. Training then uses its measured budget and probes the longest row.
The monitor shows allocation state and worker progress; while bootstrap is
running there may not yet be a progress receipt. A successful worker receipt is
required to read the checksum-verified run reports. MLflow receives metrics and
lineage; private logs, diagnostics and adapter files stay in the Slurm archive.

## Evaluate job 8

Set `SOURCE_RUN_ID` to `94dbf110-1d23-43f6-88cb-7c5f4304b293`, use the Qwen 0.8B
smoke preset, and enter the same manifest URI and byte SHA-256 used for that run.
Run Setup, skip the context and train cells, and run **Reload and compare on DGX**.
It verifies the source run and adapter hashes, reloads the adapter in Slurm, and
returns two private held-out comparisons. Restore inference after the job exits.
Restoration does not deploy the adapter into `eliza-small`.

This evaluator currently qualifies the Qwen3.5 smoke path. The larger synthetic
cut still needs a versioned generation recipe, sharded export and verified
loading. Gemma E4B-it additionally needs its own tool-template and model backend
qualification before the three-model comparison can run.
