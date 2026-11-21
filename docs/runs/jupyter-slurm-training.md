# Job-owned runtime on an advertised Slurm scheduler

`eliza-inference` installs/advertises Slurm and controls inference/Slurm GPU
handover. `nlp-lab` owns every training decision and the submitted batch job.
Jupyter packaging/storage integration is supplied by `eliza-spark`.

The host needs no `nlp-lab` package or model-training environment installed.
The job carries a stdlib bootstrap plus checksum-pinned runtime source/profile
and prepares its environments under the non-root Slurm job account. No sudo,
host Git checkout or inference-owned runtime configuration is used by the job.

```python
from eliza_training import ElizaTraining
training = ElizaTraining(bucket="<private-artifacts>")
training.environment("training")
print(training.discover())
run = training.submit(
    repo="/home/jovyan/work/nlp-lab",
    config="configs/private.yaml",
    dataset_manifest={"uri": "s3://<bucket>/<frozen-manifest>", "sha256": "<exact digest>"},
    runtime_profile="trl",
    # Supply this when editing runtime code/dependencies in the working checkout:
    runtime_source="/home/jovyan/work/nlp-lab/packages/training-client",
)
training.watch(run["run_id"], on_update=print)
```

The client discovers the REST URL/API versions, partition resources and writable
job workspace through `GET /slurm`. An explicit endpoint/workspace override is
also possible. Scheduler credentials come from `POST /slurm/token`.

Runtime defaults come from the installed training client, never from the host.
An explicit `runtime_source` captures the selected working package's code,
pyproject and dependency profiles (including edits). The submitted manifest's
file checksums bind those exact bytes. Use a compatible client/worker protocol;
this self-contained execution format is bundle version 2.

Source/config/runtime bundles are immutable and contain no job credentials. The
generic storage adapter supplies a signed download URL with a caller-selected
lifetime (default one day, S3 maximum seven days). Ensure it outlives queue wait
and bootstrap; an expired URL fails the job rather than switching data silently.

Storage access and other experiment variables are submitted as Slurm job
environment values, not archived in run manifests or submission intents. The
Jupyter adapter resolves its mounted MinIO access; explicit `job_environment`
values can add/override caller-owned settings such as MLflow/HF credentials.
Outside Jupyter, inject a storage adapter implementing `read_json`, `read_bytes`,
`create_bytes` and `presign_get`, plus a job environment/provider. A job-owned
`runtime_env` file is another explicit caller option; it is never provisioned
or selected by inference.

Private configs require `hub.push_to_hub: false` and outputs inside `artifacts/`.
Their `data.bucket`, `data.dataset_id`, `data.dataset_version` and
`data.expected_manifest_sha256` must resolve to the same manifest URI and stored-byte
digest as the submission pin. For P3/P4, pass `restricted_manifest_ack_sha256` only
after explicit operator acknowledgement of that digest. See the
[private HASS smoke runbook](hass-private-sft-smoke.md) for the pinned example.
The worker validates the source bundle and dataset-manifest digest before invoking
the selected SFT/DPO module. Runtime preparation counts against allocated job
resources and time. Each run has isolated runtime directories.

## GPU acceptance probe

From an environment with the base training client and authenticated control access:

```bash
python -m nlp_training.smoke
```

The probe sends its runtime inline, prepares it in the allocation, performs
CUDA forward/backward, saves an artifact and briefly holds the allocation so a
busy mode switch can be tested. It requires no HASS dataset or MinIO access,
but runtime package downloads require network access. Its logs/artifact paths
are printed with the job ID.

## Recovery

Save the run UUID before submission. A lost response can follow acceptance;
use `recover_submission` instead of submitting again. Kernel/watch timeout never
cancels jobs. Cancellation and return to inference are explicit. Missing worker
outcomes are not success; consult durable scheduler completion records.

Model/template fidelity, frozen dataset selection and saved-model evaluation
remain separate requirements for a useful HASS fine-tuning experiment.

## Pinned NVIDIA ARM64 metadata repair

The selected PyTorch 2.9.1 CUDA 13 build requires cuSPARSELt 0.8.0 exactly.
NVIDIA's ARM64 wheel filename is correct, but its internal tag is the invalid
`py3-none-manylinux2014_sbsa` (upstream: astral-sh/uv issue 17711 and
pytorch/pytorch issue 202632). Resolving an ARM64 filename alone does not catch
this installed-wheel defect.

The TRL profile runs `nlp_training.wheel_metadata` before the complete `uv pip
check`. It is restricted to Linux aarch64, version 0.8.0 and the recorded URL/
SHA-256 of the pinned NVIDIA wheel. Every vendor file is checked against the
vendor RECORD, and the library must be an ARM64 ELF binary. Only the known WHEEL
tag and its RECORD row are normalized; binary payloads and dependency versions
are unchanged. Unknown metadata, a different artifact or changed files fail.

The repair is repeatable and recovers an interrupted WHEEL/RECORD update. It
records `nlp-training-metadata-repair.json` in the job's training environment;
dataset workers include that receipt in archived `environment.json`. Full
dependency/platform checks and CUDA tests remain required after normalization.
Remove this profile hook when changing to a verified upstream fixed pairing.

UV may write the correct direct URL with an empty `archive_info`. In that case,
the repair first validates all installed vendor files and ARM64 ELF architecture,
then independently streams/hashes the pinned NVIDIA reference archive before
editing metadata. The reference is approximately 211 MiB; verification requires
network access and retains no downloaded archive on disk. A recorded hash mismatch
is always rejected. The repair receipt distinguishes installer-recorded hashes
from independent reference downloads. Neither case skips the full dependency check.
