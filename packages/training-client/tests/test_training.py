import hashlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tarfile
import uuid
from pathlib import Path
from unittest.mock import patch

import pytest
from botocore.exceptions import ClientError

from nlp_training import TrainingClient as ElizaTraining, SubmissionUncertain, freeze_bundle
from nlp_training import worker as runner

RUN = "00000000-0000-4000-8000-000000000001"
DATASET = {"uri": "s3://sft-datasets/pinned/manifest.json", "sha256": hashlib.sha256(b"{}").hexdigest()}


class Storage:
    def __init__(self):
        self.objects = {}

    def create_bytes(self, *, bucket, key, data, **kwargs):
        if key in self.objects:
            if self.objects[key] != data:
                raise ValueError("Immutable conflict")
            return "replayed"
        self.objects[key] = data
        return "created"

    def read_json(self, *, bucket, key, **kwargs):
        if key not in self.objects:
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        return json.loads(self.objects[key])

    def get_object(self, *, Bucket, Key):
        data = b"{}" if Key == "pinned/manifest.json" else self.objects[Key]
        return {"Body": io.BytesIO(data), "ContentLength": len(data)}

    def put_object(self, *, Bucket, Key, Body, **kwargs):
        self.objects[Key] = Body

    def upload_file(self, filename, bucket, key):
        self.objects[key] = Path(filename).read_bytes()


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "nlp-lab"
    (root / "src/eliza_trainer/sft").mkdir(parents=True)
    (root / "configs").mkdir()
    (root / "src/model.py").write_text("value = 'original'\n")
    (root / "configs/private.yaml").write_text("config_schema_version: 2\nhub:\n  push_to_hub: false\ntraining:\n  output_root: artifacts/checkpoints\n")
    # Use a real Git base so bundle provenance and edited contents are exercised.
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "fixture"], cwd=root, check=True)
    return root


def transport(fail_submit=False):
    calls = []
    def request(url, method, headers, payload):
        calls.append((url, method, payload))
        if url.endswith("/environment"):
            return {"phase": "training"}
        if url.endswith("/openapi/v3"):
            return {"paths": {"/slurm/v0.0.45/job/submit": {}}}
        if url.endswith("/job/submit"):
            if fail_submit:
                raise TimeoutError("response lost after acceptance")
            return {"job_id": 42}
        if url.endswith("/jobs/"):
            return {"jobs": [{"job_id": 42, "name": "eliza-" + RUN}]}
        return {"jobs": [{"job_id": 42, "name": "eliza-" + RUN, "job_state": ["COMPLETED"], "exit_code": {"return_code": {"number": 0}}}]}
    return request, calls


def client(storage, request):
    return ElizaTraining(storage=storage, bucket="private-runs", control_url="http://control", endpoint_url="http://slurm", user="trainer", token="test-token", request=request)


def test_freeze_preserves_edits_and_excludes_credentials(repo, tmp_path):
    (repo / "src/model.py").write_text("value = 'edited'\n")
    (repo / "src/.env").write_text("SECRET=hidden")
    (repo / ".env").write_text("SECRET=hidden")
    bundle, manifest = freeze_bundle(repo, "configs/private.yaml", run_id=RUN, dataset_manifest=DATASET)
    assert "src/.env" not in manifest["files"]
    assert manifest["git_status"]
    destination = tmp_path / "extracted"
    assert runner.extract_verified(bundle, hashlib.sha256(bundle).hexdigest(), destination, RUN) == manifest
    assert "edited" in (destination / "src/model.py").read_text()
    (repo / "src/model.py").write_text("value = 'later'\n")
    assert "edited" in (destination / "src/model.py").read_text()


def test_bundle_rejects_publication_and_output_escape(repo):
    config = repo / "configs/private.yaml"
    config.write_text(config.read_text().replace("false", "true"))
    with pytest.raises(ValueError, match="Private runs"):
        freeze_bundle(repo, "configs/private.yaml", run_id=RUN, dataset_manifest=DATASET)
    config.write_text("config_schema_version: 2\nhub: {push_to_hub: false}\ntraining: {output_root: ../outputs}\n")
    with pytest.raises(ValueError, match="traversal"):
        freeze_bundle(repo, "configs/private.yaml", run_id=RUN, dataset_manifest=DATASET)


def test_bundle_rejects_links_and_corrupt_or_unsafe_archives(repo, tmp_path):
    (repo / "src/link").symlink_to(repo / "src/model.py")
    with pytest.raises(ValueError, match="Symlinks"):
        freeze_bundle(repo, "configs/private.yaml", run_id=RUN, dataset_manifest=DATASET)
    (repo / "src/link").unlink()
    bundle, _ = freeze_bundle(repo, "configs/private.yaml", run_id=RUN, dataset_manifest=DATASET)
    with pytest.raises(ValueError, match="checksum"):
        runner.extract_verified(bundle, "0" * 64, tmp_path / "out", RUN)
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w:gz") as archive:
        entry = tarfile.TarInfo("../escaped")
        entry.size = 1
        archive.addfile(entry, io.BytesIO(b"x"))
    data = raw.getvalue()
    with pytest.raises(ValueError, match="unsafe"):
        runner.extract_verified(data, hashlib.sha256(data).hexdigest(), tmp_path / "out", RUN)


def test_submit_persists_before_post_and_never_retries_an_existing_intent(repo):
    storage = Storage()
    request, calls = transport()
    training = client(storage, request)
    run = training.submit(repo=repo, config="configs/private.yaml", run_id=RUN, dataset_manifest=DATASET)
    assert run["job_id"] == 42
    assert json.loads(storage.objects[f"training/runs/{RUN}/submission-intent.json"])["job"]["tres_per_node"] == "gres/gpu:1"
    with pytest.raises(SubmissionUncertain):
        training.submit(repo=repo, config="configs/private.yaml", run_id=RUN, dataset_manifest=DATASET)
    assert len([c for c in calls if c[0].endswith("/job/submit")]) == 1


def test_ambiguous_submission_recovers_unique_job_without_resubmitting(repo):
    storage = Storage()
    request, calls = transport(fail_submit=True)
    training = client(storage, request)
    with pytest.raises(SubmissionUncertain):
        training.submit(repo=repo, config="configs/private.yaml", run_id=RUN, dataset_manifest=DATASET)
    assert training.recover_submission(RUN)["job_id"] == 42
    assert len([c for c in calls if c[0].endswith("/job/submit")]) == 1


def test_terminal_status_is_readable_after_scheduler_stops(repo):
    storage = Storage()
    request, _ = transport()
    training = client(storage, request)
    training.submit(repo=repo, config="configs/private.yaml", run_id=RUN, dataset_manifest=DATASET)
    result = training.status(RUN)
    assert result["source"] == "slurm" and result["terminal"]
    training.request = lambda *args: pytest.fail("Scheduler should not be contacted for an archived result")
    assert training.status(RUN) == result


def test_watch_timeout_does_not_cancel():
    training = client(Storage(), lambda *args: {})
    training.status = lambda _: {"terminal": False}
    with patch("nlp_training.client.time.monotonic", side_effect=[0, 2]):
        with pytest.raises(TimeoutError, match="continues independently"):
            training.watch(RUN, timeout_seconds=1)


def test_cancel_refuses_reused_or_wrong_job_identity(repo):
    storage = Storage()
    request, _ = transport()
    training = client(storage, request)
    training.submit(repo=repo, config="configs/private.yaml", run_id=RUN, dataset_manifest=DATASET)
    training.request = lambda *args: {"jobs": [{"job_id": 42, "name": "someone-elses-run"}]}
    with pytest.raises(RuntimeError, match="identity"):
        training.cancel(RUN)


def test_scoped_token_is_obtained_cached_and_refreshed_without_notebook_output():
    issued = []
    def request(url, method, headers, payload):
        if url.endswith("/training/token"):
            issued.append(url)
            return {"user_name": "training-account", "token": "delegated-jwt", "expires_at": 1000}
        assert headers["X-SLURM-USER-NAME"] == "training-account"
        assert headers["X-SLURM-USER-TOKEN"] == "delegated-jwt"
        return {}
    training = ElizaTraining(bucket="private-runs", control_url="http://control", endpoint_url="http://slurm", control_token="test-control", request=request)
    with patch.dict(os.environ, {}, clear=True), patch("nlp_training.client.time.time", return_value=100):
        training._slurm("/slurm/v0.0.45/ping/")
        training._slurm("/slurm/v0.0.45/ping/")
        assert len(issued) == 1
        assert "token" not in training.refresh_token()
    with patch.dict(os.environ, {}, clear=True), patch("nlp_training.client.time.time", return_value=950):
        training._slurm("/slurm/v0.0.45/ping/")
    assert len(issued) == 3


def test_real_foreground_worker_archives_private_training_outputs(repo, tmp_path):
    source = repo / "src/eliza_trainer/sft/train.py"
    source.write_text("from pathlib import Path\nprint('training fixture')\nPath('artifacts/checkpoints').mkdir(parents=True)\nPath('artifacts/checkpoints/adapter.bin').write_bytes(b'adapter')\n")
    storage = Storage()
    bundle, _ = freeze_bundle(repo, "configs/private.yaml", run_id=RUN, dataset_manifest=DATASET)
    key = f"training/runs/{RUN}/bundle.tar.gz"
    storage.objects[key] = bundle
    args = type("Args", (), {"run_id": RUN, "run_root": str(tmp_path / "runs"), "bucket": "private-runs", "key": key, "sha256": hashlib.sha256(bundle).hexdigest()})()
    with patch.dict(os.environ, {"ELIZA_TRAINING_PYTHON": sys.executable, "SLURM_JOB_ID": "42"}):
        assert runner.execute(args, storage) == 0
    result = json.loads(storage.objects[f"training/runs/{RUN}/result.json"])
    assert result["outcome"] == "succeeded" and result["artifact_count"] == 1
    assert storage.objects[f"training/runs/{RUN}/artifacts/checkpoints/adapter.bin"] == b"adapter"
    index = json.loads(storage.objects[f"training/runs/{RUN}/artifact-index.json"])
    assert index[0]["sha256"] == hashlib.sha256(b"adapter").hexdigest()
    assert "packages" in json.loads(storage.objects[f"training/runs/{RUN}/environment.json"])
    with pytest.raises(FileExistsError):
        runner.execute(args, storage)


def test_worker_verifies_dataset_manifest_before_launch(repo, tmp_path):
    storage = Storage()
    bundle, _ = freeze_bundle(repo, "configs/private.yaml", run_id=RUN,
                              dataset_manifest={**DATASET, "sha256": "0" * 64})
    key = f"training/runs/{RUN}/bundle.tar.gz"
    storage.objects[key] = bundle
    args = type("Args", (), {"run_id": RUN, "run_root": str(tmp_path / "runs"), "bucket": "private-runs", "key": key, "sha256": hashlib.sha256(bundle).hexdigest()})()
    assert runner.execute(args, storage) == 1
    result = json.loads(storage.objects[f"training/runs/{RUN}/result.json"])
    assert result["outcome"] == "failed" and "manifest checksum" in result["error"]
