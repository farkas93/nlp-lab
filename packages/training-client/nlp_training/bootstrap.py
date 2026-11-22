"""Stdlib-only job entrypoint. This source is carried in the Slurm batch script."""
import argparse
import base64
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path, PurePosixPath
from urllib.request import urlopen

LIMIT = 64 * 1024 * 1024


def verified_files(data, digest, run_id):
    if len(data) > LIMIT or hashlib.sha256(data).hexdigest() != digest:
        raise ValueError("Job bundle checksum/size mismatch")
    files, total = {}, 0
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        for member in archive:
            path = PurePosixPath(member.name)
            if not member.isfile() or path.is_absolute() or ".." in path.parts or path.as_posix() != member.name or "\\" in member.name or member.name in files:
                raise ValueError("Unsafe or duplicate job bundle member")
            total += member.size
            if total > LIMIT:
                raise ValueError("Expanded job bundle exceeds limit")
            with archive.extractfile(member) as handle:
                files[member.name] = handle.read()
    manifest = json.loads(files.pop("run-manifest.json"))
    if manifest.get("version") != 2 or manifest.get("run_id") != run_id:
        raise ValueError("Job bundle identity/protocol mismatch")
    if set(files) != set(manifest["files"]):
        raise ValueError("Job bundle membership mismatch")
    for name, content in files.items():
        if hashlib.sha256(content).hexdigest() != manifest["files"][name]:
            raise ValueError("Job file checksum mismatch: " + name)
    return files, manifest


def execute(data, digest, run_id, run_root, *, bucket=None, key=None):
    files, manifest = verified_files(data, digest, run_id)
    root = Path(run_root) / ("runtime-" + run_id)
    root.mkdir(mode=0o700, parents=True, exist_ok=False)
    package = root / "package"
    for name, content in files.items():
        if name.startswith("runtime-package/"):
            path = package / name.removeprefix("runtime-package/")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
    if not (package / "pyproject.toml").is_file():
        raise ValueError("Submitted runtime package is missing")
    bootstrap = root / "bootstrap"
    subprocess.run([sys.executable, "-m", "venv", str(bootstrap)], check=True)
    python = str(bootstrap / "bin/python")
    subprocess.run([python, "-m", "pip", "install", str(package) + "[provisioning]"], check=True)
    if bucket and key and manifest["kind"] != "probe":
        subprocess.run([python, "-c", (
            "import json, sys; from nlp_training.worker import client; "
            "client().put_object(Bucket=sys.argv[1], Key=sys.argv[2], "
            "Body=json.dumps(dict(run_id=sys.argv[3], stage='preparing_runtime')).encode(), "
            "ContentType='application/json')"
        ), bucket, key.removesuffix("/bundle.tar.gz") + "/progress.json", run_id], check=False)
    subprocess.run([python, "-m", "nlp_training.provision", "--profile", manifest["runtime"]["profile"],
                    "--root", str(root), "--source", str(package)], check=True)
    env = {**os.environ, "ELIZA_TRAINING_PYTHON": str(root / "venv/bin/python")}
    if manifest["kind"] == "probe":
        script = files["probe.py"].decode()
        return subprocess.run([env["ELIZA_TRAINING_PYTHON"], "-c", script], cwd=root, env=env).returncode
    if not bucket or not key:
        raise ValueError("Dataset job requires its bundle object reference")
    return subprocess.run([python, "-m", "nlp_training.worker", "--bucket", bucket, "--key", key,
                           "--sha256", digest, "--run-id", run_id, "--run-root", run_root], env=env).returncode


def main():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--bundle-url")
    source.add_argument("--inline-bundle")
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--bucket")
    parser.add_argument("--key")
    args = parser.parse_args()
    if args.inline_bundle:
        data = base64.b64decode(args.inline_bundle, validate=True)
    else:
        with urlopen(args.bundle_url, timeout=120) as response:
            data = response.read(LIMIT + 1)
    return execute(data, args.sha256, args.run_id, args.run_root, bucket=args.bucket, key=args.key)


if __name__ == "__main__":
    raise SystemExit(main())
