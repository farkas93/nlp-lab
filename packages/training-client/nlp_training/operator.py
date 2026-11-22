"""Configuration and read-only reporting for the remote notebook operator."""
import hashlib
import json
import os
import uuid
from pathlib import Path
from urllib.parse import urlsplit


SETTINGS = ("NLP_LAB_REPO", "INFERENCE_CONTROL_URL", "CONTROL_DAEMON_TOKEN_FILE",
            "SLURM_REST_URL", "SLURM_ARTIFACT_BUCKET", "MLFLOW_TRACKING_URI",
            "DAGSTER_GRAPHQL_URL", "STORAGE_URL")
MLFLOW_AUTH = ("MLFLOW_TRACKING_TOKEN", "MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD")


def workspace_settings(root, overrides=None):
    from dotenv import dotenv_values

    keys = (*SETTINGS, *(name + "_FILE" for name in MLFLOW_AUTH))
    path = Path(root).expanduser() / ".env"
    values = dotenv_values(path, interpolate=False) if path.is_file() else {}
    if set(values) - set(keys):
        raise ValueError("Unknown workspace .env keys: " + ", ".join(sorted(set(values) - set(keys))))
    overrides = overrides or {}
    if set(overrides) - set(keys):
        raise ValueError("Unknown workspace overrides")
    resolved, origins = {}, {}
    for key in keys:
        for source, options in (("notebook", overrides), ("workspace .env", values), ("deployment", os.environ)):
            if options.get(key):
                resolved[key] = options[key]
                origins[key] = source
                break
    for key in ("NLP_LAB_REPO", "INFERENCE_CONTROL_URL", "SLURM_ARTIFACT_BUCKET", "MLFLOW_TRACKING_URI"):
        if key not in resolved:
            raise ValueError(f"Set {key} in {path}")
    for key in ("INFERENCE_CONTROL_URL", "SLURM_REST_URL", "MLFLOW_TRACKING_URI", "DAGSTER_GRAPHQL_URL", "STORAGE_URL"):
        if key in resolved:
            url = urlsplit(resolved[key])
            if url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password or url.query or url.fragment:
                raise ValueError(f"Use a plain HTTP(S) endpoint for {key}")
    return resolved, origins


def mlflow_environment(settings):
    result = {}
    for key in MLFLOW_AUTH:
        filename = settings.get(key + "_FILE")
        if filename:
            result[key] = Path(filename).expanduser().read_text().strip()
        elif os.environ.get(key):
            result[key] = os.environ[key]
    return result


def format_status(status):
    job = status.get("job_id", "?")
    if status.get("source") == "trainer-process":
        return f"Job {job} · worker {status['outcome']} · {status.get('artifact_count', 0)} archived files"
    states = ", ".join(status.get("states", [])) or "unknown"
    note = "worker receipt not available" if status.get("terminal") else "allocation status"
    progress = status.get("progress", {})
    if progress.get("stage"):
        note = progress["stage"].replace("_", " ")
        if "step" in progress:
            note += f" {progress['step']}/{progress['total']}"
    return f"Job {job} · Slurm {states} · {note}"


def prepare_run(repo, run_dir, *, kind, config, pin, bucket, resources):
    """Reuse exact run intent on reconnect; change labels for a new intent."""
    import yaml

    run_dir = Path(run_dir)
    run_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    journal = run_dir / "run.json"
    config_text = yaml.safe_dump(config, sort_keys=True)
    binding = dict(kind=kind, dataset_manifest=pin, bucket=bucket, resources=resources,
                   config_sha256=hashlib.sha256(config_text.encode()).hexdigest())
    if not journal.exists():
        with journal.open("x") as out:
            json.dump({**binding, "run_id": str(uuid.uuid4())}, out)
    existing = json.loads(journal.read_text())
    if any(existing.get(key) != value for key, value in binding.items()):
        raise ValueError("Run label already binds a different plan; choose a new label")
    run_id = str(uuid.UUID(existing["run_id"]))
    relative = f"configs/private-hass-{run_id}.yaml"
    path = Path(repo) / relative
    if not path.exists():
        with path.open("x") as out:
            out.write(config_text)
    elif path.read_text() != config_text:
        raise ValueError("Prepared run config has changed")
    return run_id, relative


def submit_prepared(training, repo, run_dir, *, kind, config, pin, resources, settings):
    run_id, relative = prepare_run(repo, run_dir, kind=kind, config=config, pin=pin,
                                   bucket=training.bucket, resources=resources)
    print(f"{kind} run UUID: {run_id}", flush=True)
    marker = Path(run_dir) / "submit-attempt.json"
    if marker.exists():
        print("Existing submission attempt; monitor or reconcile this UUID, do not resubmit.")
        return run_id
    if input(f"Type SUBMIT to run {kind} on the private dataset {pin['sha256']}: ").strip() != "SUBMIT":
        raise RuntimeError("Submission not requested")
    env = {**mlflow_environment(settings), "ELIZA_MLFLOW_METRICS_ONLY": "1"} if kind == "sft" else {}
    training.environment("training")
    with marker.open("x") as handle:
        json.dump({"run_id": run_id}, handle)
    training.submit(repo=repo, config=relative, run_id=run_id, kind=kind, dataset_manifest=pin,
                    restricted_manifest_ack_sha256=pin["sha256"], runtime_source=Path(repo) / "packages/training-client",
                    runtime_profile="context" if kind == "hass-context" else "trl",
                    job_environment=env, **resources)
    return run_id


def read_report(training, run_id, name):
    if name not in {"context-preflight.json", "held-out-comparison.json", "training-summary.json"}:
        raise ValueError("Unknown run report")
    root = f"{training.prefix}/{run_id}"
    store = training.storage
    result = store.read_json(bucket=training.bucket, key=f"{root}/result.json", max_bytes=1024 * 1024)
    if result.get("run_id") != run_id or result.get("outcome") != "succeeded":
        raise RuntimeError(format_status(result))
    index = store.read_json(bucket=training.bucket, key=f"{root}/artifact-index.json", max_bytes=1024 * 1024)
    matches = [entry for entry in index if entry["path"] == name]
    if len(matches) != 1:
        raise ValueError(f"Missing or ambiguous archived report: {name}")
    payload = store.read_bytes(bucket=training.bucket, key=f"{root}/artifacts/{name}", max_bytes=1024 * 1024)
    if len(payload) != matches[0]["size"] or hashlib.sha256(payload).hexdigest() != matches[0]["sha256"]:
        raise ValueError(f"Archived report checksum mismatch: {name}")
    return json.loads(payload)
