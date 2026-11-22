"""Configuration and read-only reporting for the remote notebook operator."""
import hashlib
import json
import os
import uuid
from pathlib import Path
from urllib.parse import urlsplit

from .logtail import compact


SETTINGS = ("NLP_LAB_REPO", "INFERENCE_CONTROL_URL", "CONTROL_DAEMON_TOKEN_FILE",
            "SLURM_REST_URL", "SLURM_ARTIFACT_BUCKET", "MLFLOW_TRACKING_URI",
            "DAGSTER_GRAPHQL_URL", "STORAGE_URL", "STORAGE_ACCOUNT_FILE", "STORAGE_KEY_FILE", "AWS_REGION", "HF_TOKEN_FILE")
MLFLOW_AUTH = ("MLFLOW_TRACKING_TOKEN", "MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD")


def workspace_settings(root, overrides=None, *, filename="training.env"):
    from dotenv import dotenv_values

    keys = (*SETTINGS, *(name + "_FILE" for name in MLFLOW_AUTH))
    path = Path(root).expanduser() / filename
    values = dotenv_values(path, interpolate=False) if path.is_file() else {}
    if not path.is_file():
        raise ValueError(f"Missing visible workspace configuration: {path}; run scripts/setup_training_workspace.py")
    if set(values) - set(keys):
        raise ValueError(f"Unknown keys in {path}: " + ", ".join(sorted(set(values) - set(keys))))
    overrides = overrides or {}
    if set(overrides) - set(keys):
        raise ValueError("Unknown workspace overrides")
    resolved, origins = {}, {}
    for key in keys:
        for source, options in (("notebook", overrides), (str(path), values), ("deployment", os.environ)):
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


def connect_workspace(root, overrides=None):
    from eliza_storage import ElizaStorage
    from eliza_training import ElizaTraining
    from eliza_dagster import ElizaDagster

    settings, origins = workspace_settings(root, overrides)
    values = {name: Path(settings[name + "_FILE"]).expanduser().read_text().strip()
              for name in ("STORAGE_ACCOUNT", "STORAGE_KEY") if settings.get(name + "_FILE")}
    storage = ElizaStorage(endpoint_url=settings.get("STORAGE_URL"), access_key=values.get("STORAGE_ACCOUNT"),
                           secret_key=values.get("STORAGE_KEY"), region_name=settings.get("AWS_REGION"))
    token_file = settings.get("CONTROL_DAEMON_TOKEN_FILE")
    token = Path(token_file).expanduser().read_text().strip() if token_file else None
    training = ElizaTraining(bucket=settings["SLURM_ARTIFACT_BUCKET"], storage=storage,
                             control_url=settings["INFERENCE_CONTROL_URL"], endpoint_url=settings.get("SLURM_REST_URL"),
                             control_token=token)
    dagster = ElizaDagster(graphql_url=settings.get("DAGSTER_GRAPHQL_URL"))
    print("Configuration file:", Path(root) / "training.env")
    print("Run records:", Path(root) / "runs")
    print("Resolved settings:", {key: {"value": value, "source": origins[key]} for key, value in settings.items() if not key.endswith("_FILE")})
    print("DGX mode:", training.environment().get("phase"))
    return settings, storage, training, dagster


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
    local_config = run_dir / "resolved-config.yaml"
    if local_config.exists() and local_config.read_text() != config_text:
        raise ValueError("Local resolved config differs from the prepared plan")
    local_config.write_text(config_text)
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
    env = {**mlflow_environment(settings), "ELIZA_MLFLOW_METRICS_ONLY": "1"} if kind in {"sft", "hass-context"} else {}
    if settings.get("HF_TOKEN_FILE"):
        env["HF_TOKEN"] = Path(settings["HF_TOKEN_FILE"]).expanduser().read_text().strip()
    training.environment("training")
    with marker.open("x") as handle:
        json.dump({"run_id": run_id}, handle)
    receipt = training.submit(repo=repo, config=relative, run_id=run_id, kind=kind, dataset_manifest=pin,
                    restricted_manifest_ack_sha256=pin["sha256"], runtime_source=Path(repo) / "packages/training-client",
                    runtime_profile=("context" if kind == "hass-context" else
                                     "unsloth-gemma4" if config["identity"]["backend"] == "unsloth" else "trl"),
                    job_environment=env, **resources)
    (Path(run_dir) / "submission.json").write_text(json.dumps(receipt, indent=2))
    return run_id


def watch_run(training, run_id, run_dir, *, poll_seconds=5, timeout_seconds=7200, render=None):
    """Persist state and display a bounded live log without importing model packages."""
    import time
    from botocore.exceptions import ClientError

    folder = Path(run_dir)
    folder.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout_seconds
    previous = None
    training.discover()
    while True:
        state = training.status(run_id)
        tail = ""
        try:
            tail = training.storage.read_bytes(bucket=training.bucket,
                key=f"{training.prefix}/{run_id}/live-log.txt", max_bytes=65536).decode("utf-8", errors="replace")
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") not in {"NoSuchKey", "404"}:
                raise
        tail = compact(tail)
        view = (state, tail)
        if view != previous:
            (folder / "status.json").write_text(json.dumps(state, indent=2))
            (folder / "live-log.txt").write_text(tail + ("\n" if tail else ""))
            if render:
                render(state, tail)
            else:
                print(format_status(state))
            previous = view
        if state.get("terminal"):
            (folder / "result.json").write_text(json.dumps(state, indent=2))
            if state.get("outcome") != "succeeded":
                raise RuntimeError(f"{format_status(state)}; see {folder}/live-log.txt and the remote train.log")
            return state
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Monitoring stopped; remote run {run_id} continues. Re-run this monitor to reconnect.")
        time.sleep(poll_seconds)


def read_report(training, run_id, name, *, run_dir=None):
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
    report = json.loads(payload)
    if run_dir is not None:
        Path(run_dir).mkdir(parents=True, exist_ok=True)
        (Path(run_dir) / name).write_bytes(payload)
    return report
