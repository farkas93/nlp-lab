#!/usr/bin/env python3
"""Canonical foreground training worker, independent of the host controller."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import signal
import subprocess
import tarfile
import time
import uuid
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

MAX_BYTES = 64 * 1024 * 1024


def credential(name):
    return os.environ.get(name) or Path(os.environ[name + "_FILE"]).read_text().strip()


def client():
    import boto3
    from botocore.config import Config
    return boto3.client("s3", endpoint_url=os.environ["STORAGE_URL"],
                        aws_access_key_id=credential("STORAGE_ACCOUNT"),
                        aws_secret_access_key=credential("STORAGE_KEY"),
                        region_name=os.environ.get("AWS_REGION", "us-east-1"),
                        config=Config(signature_version="s3v4", s3={"addressing_style": "path"}))


def download(storage, bucket, key):
    response = storage.get_object(Bucket=bucket, Key=key)
    body = response["Body"]
    try:
        if response.get("ContentLength", 0) > MAX_BYTES:
            raise ValueError("Input object exceeds size limit")
        data = body.read(MAX_BYTES + 1)
        if len(data) > MAX_BYTES:
            raise ValueError("Input object exceeds size limit")
        return data
    finally:
        body.close()


def extract_verified(bundle: bytes, digest: str, destination: Path, run_id: str):
    if hashlib.sha256(bundle).hexdigest() != digest:
        raise ValueError("Run bundle checksum mismatch")
    contents = {}
    total = 0
    with tarfile.open(fileobj=io.BytesIO(bundle), mode="r:gz") as archive:
        for entry in archive:
            path = PurePosixPath(entry.name)
            if not entry.isfile() or path.is_absolute() or ".." in path.parts or not path.parts or "\\" in entry.name or path.as_posix() != entry.name:
                raise ValueError("Run bundle contains an unsafe path or non-regular file")
            if entry.name in contents or entry.size < 0:
                raise ValueError("Duplicate or invalid archive member")
            total += entry.size
            if total > MAX_BYTES:
                raise ValueError("Expanded run bundle exceeds size limit")
            with archive.extractfile(entry) as handle:
                contents[entry.name] = handle.read()
    manifest = json.loads(contents.pop("run-manifest.json"))
    if manifest.get("version") != 2 or manifest.get("run_id") != run_id or manifest.get("kind") not in {"sft", "dpo"}:
        raise ValueError("Run manifest identity/version mismatch")
    if set(contents) != set(manifest["files"]):
        raise ValueError("Run manifest file membership mismatch")
    for name, data in contents.items():
        if hashlib.sha256(data).hexdigest() != manifest["files"][name]:
            raise ValueError("File checksum mismatch: " + name)
    if manifest.get("config") not in contents:
        raise ValueError("Configuration is missing from bundle")
    destination.mkdir(parents=True, exist_ok=False)
    for name, data in contents.items():
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    (destination.parent / "run-manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def execute(args, storage):
    run_id = str(uuid.UUID(args.run_id))
    if not args.key.endswith("/" + run_id + "/bundle.tar.gz"):
        raise ValueError("Bundle key does not match run identity")
    prefix = args.key.removesuffix("/bundle.tar.gz")
    run_dir = Path(args.run_root).resolve() / run_id
    # Never reuse outputs from an earlier attempt of this run.
    run_dir.mkdir(parents=True, exist_ok=False)
    workspace = run_dir / "workspace"
    child = None
    interrupted = None

    def stop(signum, frame):
        nonlocal interrupted
        interrupted = signum
        if child is not None:
            child.send_signal(signum)

    previous = {sig: signal.signal(sig, stop) for sig in (signal.SIGTERM, signal.SIGINT)}
    result = {"run_id": run_id, "job_id": os.environ.get("SLURM_JOB_ID"),
              "source": "trainer-process", "terminal": True,
              "artifact_prefix": f"s3://{args.bucket}/{prefix}/artifacts/"}
    return_code = 1
    log_path = run_dir / "train.log"
    try:
        with log_path.open("wb") as log:
            manifest = extract_verified(download(storage, args.bucket, args.key), args.sha256, workspace, run_id)
            source = manifest["dataset_manifest"]
            uri = urlsplit(source["uri"])
            if uri.scheme != "s3" or not uri.netloc or not uri.path.strip("/"):
                raise ValueError("Frozen dataset manifest must have an s3:// URI")
            evidence = download(storage, uri.netloc, uri.path.lstrip("/"))
            if hashlib.sha256(evidence).hexdigest() != source["sha256"]:
                raise ValueError("Frozen dataset manifest checksum mismatch")
            (run_dir / "dataset-manifest.json").write_bytes(evidence)
            if interrupted:
                raise InterruptedError("Cancelled before training startup")
            training_python = os.environ["ELIZA_TRAINING_PYTHON"]
            environment = subprocess.check_output([
                training_python, "-c",
                "import json,sys,pathlib,importlib.metadata as m; "
                "repair=pathlib.Path(sys.prefix)/'nlp-training-metadata-repair.json'; "
                "print(json.dumps({'python':sys.version,'packages':sorted((d.metadata['Name'],d.version) for d in m.distributions()),"
                "'metadata_repair':json.loads(repair.read_text()) if repair.exists() else None}))",
            ], cwd=workspace, text=True)
            environment = json.loads(environment)
            environment.update(training_python=training_python,
                               runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
            (run_dir / "environment.json").write_text(json.dumps(environment, indent=2))
            command = [training_python, "-m", f"src.eliza_trainer.{manifest['kind']}.train", "--config", manifest["config"]]
            child = subprocess.Popen(command, cwd=workspace, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            last_upload = time.monotonic()
            while True:
                chunk = child.stdout.read1(64 * 1024)
                if not chunk:
                    break
                log.write(chunk)
                log.flush()
                os.write(1, chunk)
                if time.monotonic() - last_upload >= 30:
                    storage.upload_file(str(log_path), args.bucket, prefix + "/train.log")
                    last_upload = time.monotonic()
            return_code = child.wait()
            result["return_code"] = return_code
    except Exception as exc:
        result["error"] = str(exc)
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
    finally:
        result.setdefault("return_code", return_code)
        result["signal"] = interrupted
        result["outcome"] = "cancelled" if interrupted else "succeeded" if return_code == 0 else "failed"
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        # Keep local outcomes even when object storage is unavailable.
        (run_dir / "result.json").write_text(json.dumps(result, indent=2))
        try:
            if log_path.exists():
                storage.upload_file(str(log_path), args.bucket, prefix + "/train.log")
            for name in ("run-manifest.json", "dataset-manifest.json", "environment.json"):
                if (run_dir / name).exists():
                    storage.upload_file(str(run_dir / name), args.bucket, prefix + "/" + name)
            artifacts = workspace / "artifacts"
            if artifacts.is_symlink() or not artifacts.resolve().is_relative_to(workspace.resolve()):
                raise ValueError("Artifact root must be inside the run workspace")
            count = 0
            index = []
            for path in sorted(artifacts.rglob("*")) if artifacts.exists() else ():
                if path.is_symlink() or not path.resolve().is_relative_to(artifacts.resolve()):
                    raise ValueError("Artifact symlinks are not supported")
                if path.is_file():
                    storage.upload_file(str(path), args.bucket, prefix + "/artifacts/" + path.relative_to(artifacts).as_posix())
                    digest = hashlib.sha256()
                    with path.open("rb") as handle:
                        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                            digest.update(chunk)
                    index.append({"path": path.relative_to(artifacts).as_posix(),
                                  "size": path.stat().st_size, "sha256": digest.hexdigest()})
                    count += 1
            storage.put_object(Bucket=args.bucket, Key=prefix + "/artifact-index.json",
                               Body=json.dumps(index).encode(), ContentType="application/json", IfNoneMatch="*")
            result["artifact_count"] = count
        except Exception as exc:
            result.update(outcome="archive_failed", archive_error=str(exc))
            return_code = return_code or 1
        (run_dir / "result.json").write_text(json.dumps(result, indent=2))
        storage.put_object(Bucket=args.bucket, Key=prefix + "/result.json",
                           Body=json.dumps(result).encode(), ContentType="application/json", IfNoneMatch="*")
    return return_code if return_code > 0 else 128 + (interrupted or -return_code) if return_code < 0 or interrupted else 0


def main():
    parser = argparse.ArgumentParser()
    for name in ("bucket", "key", "sha256", "run-id", "run-root"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    return execute(args, client())


if __name__ == "__main__":
    raise SystemExit(main())
