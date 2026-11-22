#!/usr/bin/env python3
"""Canonical foreground training worker, independent of the host controller."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import signal
import selectors
import subprocess
import tarfile
import time
import uuid
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

from .dataset_pin import validate_dataset_pin, validate_acknowledgement
from .logtail import compact

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
    if manifest.get("version") != 2 or manifest.get("run_id") != run_id or manifest.get("kind") not in {"sft", "dpo", "hass-context", "hass-eval"}:
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

    def progress(stage, **fields):
        try:
            storage.put_object(Bucket=args.bucket, Key=prefix + "/progress.json",
                               Body=json.dumps(dict(run_id=run_id, stage=stage, **fields)).encode(),
                               ContentType="application/json")
        except Exception:
            pass  # progress is advisory; terminal archival remains authoritative

    try:
        progress("verifying_inputs")
        with log_path.open("wb") as log:
            manifest = extract_verified(download(storage, args.bucket, args.key), args.sha256, workspace, run_id)
            import yaml
            config = yaml.safe_load((workspace / manifest["config"]).read_bytes())
            source, digest = validate_dataset_pin(config, manifest["dataset_manifest"])
            ack = manifest.get("restricted_manifest_ack_sha256")
            validate_acknowledgement(ack, digest)
            if config.get("hub", {}).get("push_to_hub") is not False:
                raise ValueError("Private worker requires hub.push_to_hub: false")
            uri = urlsplit(source)
            evidence = download(storage, uri.netloc, uri.path.lstrip("/"))
            if hashlib.sha256(evidence).hexdigest() != digest:
                raise ValueError("Frozen dataset manifest checksum mismatch")
            (run_dir / "dataset-manifest.json").write_bytes(evidence)
            if json.loads(evidence).get("kind") in {"hass-experimental-sft-v1", "hass-household-sft-v1"} and ack is None:
                raise ValueError("Private HASS SFT requires explicit acknowledgement of its manifest SHA-256")
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
            kind = manifest["kind"]
            if kind in {"hass-context", "hass-eval"}:
                if json.loads(evidence).get("kind") not in {"hass-experimental-sft-v1", "hass-household-sft-v1"}:
                    raise ValueError("Expected a private HASS export")
                command = [training_python, "-m", "src.eliza_trainer.sft.remote_checks", "--kind", kind,
                           "--config", manifest["config"], "--bucket", args.bucket, "--prefix", prefix]
            else:
                command = [training_python, "-m", f"src.eliza_trainer.{kind}.train", "--config", manifest["config"]]
            if ack is not None and kind in {"sft", "dpo"}:
                command.append("--assume-yes")
            if config.get("identity", {}).get("backend") == "unsloth" and kind != "hass-context":
                if manifest["runtime"]["profile"] != "unsloth-gemma4":
                    raise ValueError("Unsloth jobs require their isolated unsloth-gemma4 runtime")
                command = [training_python, "-m", "nlp_training.unsloth_entry", *command[2:]]
            child_env = os.environ.copy()
            if child_env.get("ELIZA_MLFLOW_METRICS_ONLY") == "1":
                child_env.pop("HF_MLFLOW_LOG_ARTIFACTS", None)
            child = subprocess.Popen(command, cwd=workspace, env=child_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            progress({"hass-context": "measuring_context", "hass-eval": "evaluating_adapter"}.get(kind, "trainer_starting"))
            last_upload = 0
            pending = b""
            tail = b""

            def publish_tail():
                body = compact(tail.decode("utf-8", errors="replace"), max_lines=60).encode()
                storage.put_object(Bucket=args.bucket, Key=prefix + "/live-log.txt", Body=body,
                                   ContentType="text/plain; charset=utf-8")

            with selectors.DefaultSelector() as selector:
                selector.register(child.stdout, selectors.EVENT_READ)
                while selector.get_map():
                    for key_event, _ in selector.select(timeout=1):
                        chunk = os.read(key_event.fileobj.fileno(), 64 * 1024)
                        if not chunk:
                            selector.unregister(key_event.fileobj)
                            continue
                        log.write(chunk)
                        log.flush()
                        os.write(1, chunk)
                        tail = (tail + chunk)[-262144:]
                        pending = (pending + chunk)[-131072:]
                        while b"\n" in pending:
                            line, pending = pending.split(b"\n", 1)
                            if line.startswith(b"ELIZA_PROGRESS "):
                                try:
                                    event = json.loads(line.removeprefix(b"ELIZA_PROGRESS "))
                                    progress(event.pop("stage"), **event)
                                except (ValueError, KeyError, TypeError):
                                    pass
                    if time.monotonic() - last_upload >= 3:
                        try:
                            publish_tail()
                        except Exception:
                            pass
                        last_upload = time.monotonic()
            try:
                publish_tail()
            except Exception:
                pass
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
            progress("archiving_outputs")
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
