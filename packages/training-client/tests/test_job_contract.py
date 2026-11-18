import json

from test_training import Storage, transport, client, DATASET, RUN, repo
from nlp_training.client import runtime_files, freeze_bundle
from nlp_training import bootstrap
import hashlib


def test_credentials_are_submitted_to_slurm_but_not_archived(repo):
    storage = Storage()
    request, calls = transport()
    training = client(storage, request)
    training.submit(repo=repo, config="configs/private.yaml", dataset_manifest=DATASET, run_id=RUN,
                    job_environment={"STORAGE_KEY": "dummy"}, runtime_profile="bootstrap")
    submitted = next(payload for url, method, payload in calls if url.endswith("/job/submit"))
    assert "STORAGE_KEY=dummy" in submitted["job"]["environment"]
    intent = json.loads(storage.objects[f"training/runs/{RUN}/submission-intent.json"])
    assert "dummy" not in json.dumps(intent)
    assert intent["job_environment_keys"] == ["STORAGE_KEY"]
    assert "/opt/eliza-training" not in submitted["script"]


def test_caller_can_submit_an_edited_runtime_profile(repo, tmp_path):
    source = tmp_path / "runtime-source"
    for name, data in runtime_files("bootstrap").items():
        path = source / name.removeprefix("runtime-package/")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    profile = source / "nlp_training/profiles/runtime.json"
    profile.write_text(json.dumps({"caller-profile": {"python": "3.12", "requirements": None, "minimum_driver_major": 0}}))
    data, manifest = freeze_bundle(repo, "configs/private.yaml", run_id=RUN, dataset_manifest=DATASET,
                                   runtime_profile="caller-profile", runtime_source=source)
    files, _ = bootstrap.verified_files(data, hashlib.sha256(data).hexdigest(), RUN)
    assert manifest["runtime"]["profile"] == "caller-profile"
    assert files["runtime-package/nlp_training/profiles/runtime.json"] == profile.read_bytes()
