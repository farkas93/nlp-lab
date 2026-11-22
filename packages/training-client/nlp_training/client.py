"""Notebook client for on-demand Slurm and immutable nlp-lab run bundles."""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import re
import shlex
import subprocess
import tarfile
import time
import uuid
import base64
import importlib.metadata
from pathlib import Path, PurePosixPath
from urllib.request import Request, urlopen

from .credentials import credential as _credential
from .control import HostControl, request_json
from .dataset_pin import validate_dataset_pin, validate_acknowledgement
from .provision import profile

MAX_BUNDLE_BYTES = 64 * 1024 * 1024
TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY", "BOOT_FAIL", "DEADLINE", "PREEMPTED"}


class SubmissionUncertain(RuntimeError):
    """Do not resubmit automatically: the controller may have accepted the job."""


def _json(data):
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _relative(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or ".." in path.parts or "\\" in value:
        raise ValueError("Run paths must be relative without traversal")
    return path.as_posix()


def _run_id(value: str) -> str:
    return str(uuid.UUID(value))


def runtime_files(runtime_profile, runtime_source=None):
    root = Path(runtime_source) / "nlp_training" if runtime_source else Path(__file__).parent
    definitions = json.loads((root / "profiles/runtime.json").read_text())
    if runtime_profile not in definitions:
        raise ValueError("Unknown submitted runtime profile")
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts and path.suffix in {".py", ".json", ".txt"}:
            if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
                raise ValueError("Runtime package links are not supported")
            files["runtime-package/nlp_training/" + path.relative_to(root).as_posix()] = path.read_bytes()
    if runtime_source:
        files["runtime-package/pyproject.toml"] = (Path(runtime_source) / "pyproject.toml").read_bytes()
        return files
    distribution = importlib.metadata.distribution("nlp-lab-training")
    requires = [item.split(";", 1)[0].strip() for item in distribution.requires or []
                if ";" not in item or "provisioning" in item.split(";", 1)[1]]
    metadata = ('[build-system]\nrequires=["setuptools>=68"]\nbuild-backend="setuptools.build_meta"\n'
                '[project]\nname="nlp-lab-training"\nversion=' + json.dumps(distribution.version)
                + '\nrequires-python=">=3.10"\ndependencies=' + json.dumps(requires)
                + '\n[project.optional-dependencies]\nprovisioning=[]\n'
                + '\n[tool.setuptools.packages.find]\ninclude=["nlp_training*"]\n'
                + '[tool.setuptools.package-data]\nnlp_training=["profiles/*.txt","profiles/*.json"]\n')
    files["runtime-package/pyproject.toml"] = metadata.encode()
    return files


def archive_files(files, manifest):
    manifest["files"] = {name: hashlib.sha256(data).hexdigest() for name, data in sorted(files.items())}
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w") as archive:
        for name, data in sorted({**files, "run-manifest.json": _json(manifest)}.items()):
            info = tarfile.TarInfo(name)
            info.size, info.mode = len(data), 0o644
            archive.addfile(info, io.BytesIO(data))
    bundle = gzip.compress(raw.getvalue(), mtime=0)
    if sum(map(len, files.values())) > MAX_BUNDLE_BYTES or len(bundle) > MAX_BUNDLE_BYTES:
        raise ValueError("Run bundle exceeds size limit")
    return bundle, manifest


def batch_script(args, runtime_env=None):
    # This bootstrap requires only the host's OS Python; all runtime code is submitted.
    source = (Path(__file__).parent / "bootstrap.py").read_text()
    prefix = "#!/bin/bash\nset -euo pipefail\n"
    if runtime_env:
        prefix += "set -a\nsource " + shlex.quote(runtime_env) + "\nset +a\n"
    return prefix + "/usr/bin/python3 - " + shlex.join(args) + " <<'NLP_JOB_BOOTSTRAP'\n" + source + "\nNLP_JOB_BOOTSTRAP\n"


def freeze_bundle(repo: str | Path, config: str, *, run_id: str, dataset_manifest: dict,
                  include: tuple[str, ...] = ("src",), kind: str = "sft", runtime_profile="trl", runtime_source=None,
                  restricted_manifest_ack_sha256=None) -> tuple[bytes, dict]:
    """Capture current files, including edits, without packaging the whole workspace."""
    import yaml

    root = Path(repo).resolve(strict=True)
    config = _relative(config)
    if kind not in {"sft", "dpo", "hass-context", "hass-eval"}:
        raise ValueError("Unsupported training job kind")
    files = {}
    size = 0
    for name in (*include, config):
        candidate = root / _relative(name)
        if candidate.is_symlink():
            raise ValueError(f"Symlinks are not bundle inputs: {name}")
        if not candidate.exists():
            raise ValueError(f"Missing bundle input: {name}")
        for path in sorted(candidate.rglob("*")) if candidate.is_dir() else (candidate,):
            relative = path.relative_to(root).as_posix()
            parts = path.relative_to(root).parts
            if any(p.startswith(".") or p in {"__pycache__", "logs", "outputs", "hf_models", "node_modules"} for p in parts):
                continue
            if path.is_symlink() or not path.resolve().is_relative_to(root):
                raise ValueError(f"Symlinks are not bundle inputs: {relative}")
            if path.is_file():
                if path.stat().st_size + size > MAX_BUNDLE_BYTES:
                    raise ValueError("Run bundle exceeds the configured size limit")
                contents = path.read_bytes()
                if relative not in files:
                    size += len(contents)
                files[relative] = contents
    if config not in files:
        raise ValueError("Configuration was excluded from the bundle")
    parsed = yaml.safe_load(files[config])
    if not isinstance(parsed, dict) or parsed.get("config_schema_version") != 2:
        raise ValueError("Expected nlp-lab config schema 2")
    if parsed.get("hub", {}).get("push_to_hub") is not False:
        raise ValueError("Private runs require hub.push_to_hub: false")
    output = _relative(parsed.get("training", {}).get("output_root", ""))
    if PurePosixPath(output).parts[0] != "artifacts":
        raise ValueError("training.output_root must be inside artifacts/ for durable upload")
    _, digest = validate_dataset_pin(parsed, dataset_manifest)
    validate_acknowledgement(restricted_manifest_ack_sha256, digest)
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True)
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, check=True)
    if any(name.startswith("runtime-package/") for name in files):
        raise ValueError("runtime-package is reserved for the submitted execution runtime")
    files.update(runtime_files(runtime_profile, runtime_source))
    manifest = {
        "version": 2, "run_id": _run_id(run_id), "kind": kind, "config": config,
        "runtime": {"profile": runtime_profile, "package_version": importlib.metadata.version("nlp-lab-training")},
        "git_base": git.stdout.strip(), "git_status": dirty.stdout.splitlines(),
        "dataset_manifest": dataset_manifest,
    }
    if restricted_manifest_ack_sha256 is not None:
        manifest["restricted_manifest_ack_sha256"] = restricted_manifest_ack_sha256
    return archive_files(files, manifest)


class TrainingClient:
    """Slurm v0.0.45 adapter; it does not execute or supervise a GPU process."""
    def __init__(self, *, storage=None, bucket: str = "", prefix="training/runs",
                 control_url=None, endpoint_url=None, user=None, token=None,
                 control_token=None, run_root=None, runtime_env=None, request=None,
                 job_environment_provider=None):
        self.storage = storage
        self.bucket = bucket
        self.prefix = _relative(prefix).rstrip("/")
        self.control_url = (control_url or os.getenv("INFERENCE_CONTROL_URL", "")).rstrip("/")
        self.endpoint = (endpoint_url or os.getenv("SLURM_REST_URL", "")).rstrip("/")
        self.user, self.token, self.control_token = user, token, control_token
        self._runtime_token, self._runtime_user, self._token_expires = None, None, 0
        self.run_root, self.runtime_env = run_root, runtime_env
        self.job_environment_provider = job_environment_provider
        for path in [path for path in (run_root, runtime_env) if path]:
            if not PurePosixPath(path).is_absolute() or "\n" in path:
                raise ValueError("DGX execution paths must be absolute")
        self.request = request or self._http
        self.control = HostControl(self.control_url, token=control_token, request=self.request)

    def discover(self):
        advertised = self.control.slurm()
        if "v0.0.45" not in advertised["api_versions"]:
            raise ValueError("Scheduler does not advertise the supported API")
        self.endpoint = self.endpoint or advertised["rest_url"].rstrip("/")
        self.run_root = self.run_root or advertised["workspace"]
        return advertised

    @staticmethod
    def _http(url, method, headers, payload):
        return request_json(url, method, headers=headers, payload=payload,
                            timeout=1200 if "/environment/" in url else 60)

    def environment(self, mode=None):
        return self.control.environment(mode)

    def _slurm(self, path, method="GET", payload=None):
        user = _credential(self.user, "SLURM_USER_NAME")
        token = _credential(self.token, "SLURM_USER_TOKEN")
        if not token:
            if not self._runtime_token or time.time() >= self._token_expires - 60:
                self.refresh_token()
            token = self._runtime_token
            if user and user != self._runtime_user:
                raise ValueError("Configured Slurm user differs from delegated training identity")
            user = self._runtime_user
        if not self.endpoint or not user or not token:
            raise ValueError("Configure SLURM_REST_URL, SLURM_USER_NAME and SLURM_USER_TOKEN (or *_FILE)")
        return self.request(self.endpoint + path, method,
                            {"X-SLURM-USER-NAME": user, "X-SLURM-USER-TOKEN": token}, payload)

    def refresh_token(self):
        """Obtain the configured training identity; keep the JWT out of notebook outputs."""
        result = self.control.training_token()
        self._runtime_token, self._runtime_user = result["token"], result["user_name"]
        self._token_expires = result["expires_at"]
        return {"user_name": self._runtime_user, "expires_at": self._token_expires}

    def _key(self, run_id, name):
        return f"{self.prefix}/{_run_id(run_id)}/{name}"

    def _read(self, run_id, name):
        return self.storage.read_json(bucket=self.bucket, key=self._key(run_id, name), max_bytes=1024 * 1024)

    def _create(self, run_id, name, data):
        return self.storage.create_bytes(bucket=self.bucket, key=self._key(run_id, name), data=_json(data))

    def submit(self, *, repo, config, dataset_manifest, run_id=None, kind="sft", include=("src",),
               cpus=4, memory_mb=16384, time_minutes=60, runtime_profile="trl", job_environment=None,
               bundle_url_lifetime=86400, runtime_source=None, restricted_manifest_ack_sha256=None):
        if self.storage is None or not self.bucket:
            raise ValueError("Supply a storage adapter and private artifact bucket for dataset jobs")
        run_id = _run_id(run_id or str(uuid.uuid4()))
        for number in (cpus, memory_mb, time_minutes):
            if type(number) is not int or number < 1:
                raise ValueError("Resource requests must be positive integers")
        if self.environment().get("phase") != "training":
            raise RuntimeError("Switch the DGX to training mode before submission")
        advertised = self.discover()
        if not advertised["ready"]:
            raise RuntimeError("Advertised Slurm is not ready")
        extra_env = dict(self.job_environment_provider() if self.job_environment_provider else {})
        extra_env.update(job_environment or {})
        for name, value in extra_env.items():
            if not re.fullmatch(r"[A-Z_][A-Z0-9_]*", name) or not isinstance(value, str) or "\x00" in value:
                raise ValueError("Invalid job environment")
        schema = self._slurm("/openapi/v3")
        if "/slurm/v0.0.45/job/submit" not in schema.get("paths", {}):
            raise ValueError("This client requires the Slurm v0.0.45 API; check deployed /openapi/v3")
        bundle, manifest = freeze_bundle(repo, config, run_id=run_id, dataset_manifest=dataset_manifest,
                                         include=include, kind=kind, runtime_profile=runtime_profile, runtime_source=runtime_source,
                                         restricted_manifest_ack_sha256=restricted_manifest_ack_sha256)
        digest = hashlib.sha256(bundle).hexdigest()
        bundle_key = self._key(run_id, "bundle.tar.gz")
        self.storage.create_bytes(bucket=self.bucket, key=bundle_key, data=bundle,
                                  content_type="application/gzip", max_bytes=MAX_BUNDLE_BYTES)
        self._create(run_id, "manifest.json", {**manifest, "bundle_sha256": digest})
        log_root = self.run_root + "/logs"
        url = self.storage.presign_get(bucket=self.bucket, key=bundle_key, expires_seconds=bundle_url_lifetime)
        script = batch_script(["--bundle-url", url, "--bucket", self.bucket, "--key", bundle_key,
                               "--sha256", digest, "--run-id", run_id, "--run-root", self.run_root], self.runtime_env)
        job = {"name": "eliza-" + run_id, "partition": "training", "nodes": "1", "tasks": 1,
               "cpus_per_task": cpus, "requeue": False,
               "memory_per_node": {"set": True, "number": memory_mb, "infinite": False},
               "time_limit": {"set": True, "number": time_minutes, "infinite": False},
               "current_working_directory": self.run_root,
               "standard_output": f"{log_root}/{run_id}-%j.out",
               "standard_error": f"{log_root}/{run_id}-%j.err",
               "environment": ["PATH=/usr/local/bin:/usr/bin:/bin", "PYTHONUNBUFFERED=1"]}
        if kind != "hass-context":
            job["tres_per_node"] = "gres/gpu:1"
        intent = {"run_id": run_id, "bundle_sha256": digest, "job": dict(job),
                  "job_environment_keys": sorted(extra_env),
                  "script_sha256": hashlib.sha256(script.encode()).hexdigest()}
        if self._create(run_id, "submission-intent.json", intent) != "created":
            raise SubmissionUncertain(f"Run {run_id} already has an intent; use recover_submission(), not resubmit")
        try:
            job["environment"] = job["environment"] + [name + "=" + value for name, value in extra_env.items()]
            result = self._slurm("/slurm/v0.0.45/job/submit", "POST", {"script": script, "job": job})
            job_id = result["job_id"]
            if type(job_id) is not int or job_id < 1:
                raise ValueError("Slurm did not return a valid job ID")
            receipt = {"run_id": run_id, "job_id": job_id, "bundle_sha256": digest}
            self._create(run_id, "submission.json", receipt)
            return receipt
        except Exception as exc:
            raise SubmissionUncertain(f"Submission outcome for {run_id} needs reconciliation; do not retry submit") from exc

    def recover_submission(self, run_id):
        intent = self._read(run_id, "submission-intent.json")
        jobs = self._slurm("/slurm/v0.0.45/jobs/").get("jobs", [])
        matches = [j for j in jobs if j.get("name") == "eliza-" + _run_id(run_id)]
        if len(matches) != 1:
            raise SubmissionUncertain("Cannot identify exactly one job; inspect persisted Slurm completion records before resubmitting")
        receipt = {"run_id": _run_id(run_id), "job_id": matches[0]["job_id"], "bundle_sha256": intent["bundle_sha256"]}
        self._create(run_id, "submission.json", receipt)
        return receipt

    def status(self, run_id):
        # Worker outcomes are useful even with Slurm stopped; they are not scheduler receipts.
        from botocore.exceptions import ClientError
        for name in ("result.json", "scheduler-result.json"):
            try:
                return self._read(run_id, name)
            except ClientError as exc:
                if exc.response.get("Error", {}).get("Code") not in {"NoSuchKey", "404"}:
                    raise
        receipt = self._read(run_id, "submission.json")
        result = self._slurm(f"/slurm/v0.0.45/job/{receipt['job_id']}")
        jobs = result.get("jobs", [])
        if len(jobs) != 1 or jobs[0].get("name") != "eliza-" + _run_id(run_id):
            raise RuntimeError("Job is missing or identity differs; missing does not mean completed")
        job = jobs[0]
        states = job.get("job_state", [])
        states = [states] if isinstance(states, str) else states
        snapshot = {"run_id": _run_id(run_id), "job_id": receipt["job_id"],
                    "source": "slurm", "states": states, "terminal": bool(TERMINAL.intersection(states)) and "COMPLETING" not in states,
                    "exit_code": job.get("exit_code")}
        if snapshot["terminal"]:
            self._create(run_id, "scheduler-result.json", snapshot)
        else:
            try:
                progress = self._read(run_id, "progress.json")
                if progress.get("run_id") == run_id:
                    snapshot["progress"] = progress
            except ClientError as exc:
                if exc.response.get("Error", {}).get("Code") not in {"NoSuchKey", "404"}:
                    raise
            snapshot.pop("exit_code")
        return snapshot

    def watch(self, run_id, *, poll_seconds=5, timeout_seconds=3600, on_update=None):
        if poll_seconds <= 0 or timeout_seconds <= 0:
            raise ValueError("Polling interval and timeout must be positive")
        deadline = time.monotonic() + timeout_seconds
        previous = None
        while True:
            result = self.status(run_id)
            if on_update and result != previous:
                on_update(result)
            previous = result
            if result.get("terminal"):
                return result
            if time.monotonic() >= deadline:
                raise TimeoutError("Watch timed out; the Slurm job continues independently")
            time.sleep(poll_seconds)

    def cancel(self, run_id):
        receipt = self._read(run_id, "submission.json")
        result = self._slurm(f"/slurm/v0.0.45/job/{receipt['job_id']}")
        jobs = result.get("jobs", [])
        if len(jobs) != 1 or jobs[0].get("name") != "eliza-" + _run_id(run_id):
            raise RuntimeError("Job identity differs or has expired; cancellation blocked")
        return self._slurm(f"/slurm/v0.0.45/job/{receipt['job_id']}", "DELETE")

    def logs(self, run_id, *, max_bytes=1024 * 1024):
        return self.storage.read_bytes(bucket=self.bucket, key=self._key(run_id, "train.log"), max_bytes=max_bytes).decode("utf-8", errors="replace")
