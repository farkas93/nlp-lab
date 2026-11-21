import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

from nlp_training import provision
from nlp_training.smoke import job
from nlp_training import bootstrap
import base64
import shlex


def test_base_import_does_not_import_frameworks():
    result = subprocess.run([sys.executable, "-c", "import sys,nlp_training; assert not {'torch','transformers','pyspark'} & set(sys.modules); print('cpu-safe')"], capture_output=True, text=True, check=True)
    assert "cpu-safe" in result.stdout


def test_profile_delegation_uses_the_canonical_runtime_requirements(tmp_path):
    root = tmp_path / "runtime"
    (root / "venv/bin").mkdir(parents=True)
    (root / "venv/bin/python").touch()
    calls = []
    with patch.object(provision, "run", side_effect=lambda args, **kwargs: calls.append([str(arg) for arg in args])):
        report = provision.provision("trl", root, tmp_path / "source")
    installs = [call for call in calls if call[1:3] == ["pip", "install"]]
    assert len(installs) == 1
    assert str(tmp_path / "source") in installs[0]
    requirements = installs[0][-1]
    assert requirements.endswith("nlp_training/profiles/sft-trl.arm64-cu130.txt")
    assert report["definition"]["minimum_driver_major"] == 580
    assert any("pip" in call and "check" in call for call in calls)
    repair = next(i for i, call in enumerate(calls) if "nlp_training.wheel_metadata" in call)
    check = next(i for i, call in enumerate(calls) if "pip" in call and "check" in call)
    assert repair < check
    assert not any("cuda.is_available" in " ".join(call) for call in calls)


def test_probe_is_gpu_allocated_and_python_body_compiles():
    payload = job("00000000-0000-4000-8000-000000000001", 30)
    assert payload["job"]["tres_per_node"] == "gres/gpu:1"
    body = payload["script"].split("<<'NLP_JOB_BOOTSTRAP'\n", 1)[1].rsplit("\nNLP_JOB_BOOTSTRAP", 1)[0]
    compile(body, "probe", "exec")
    line = next(line for line in payload["script"].splitlines() if line.startswith("/usr/bin/python3"))
    args = shlex.split(line.split(" <<", 1)[0])[2:]
    values = dict(zip(args[::2], args[1::2]))
    data = base64.b64decode(values["--inline-bundle"])
    files, manifest = bootstrap.verified_files(data, values["--sha256"], values["--run-id"])
    assert b"loss.backward()" in files["probe.py"]
    assert "runtime-package/nlp_training/provision.py" in files
    assert manifest["runtime"]["profile"] == "trl"


def test_job_bootstrap_prepares_only_the_submitted_runtime(tmp_path):
    payload = job("00000000-0000-4000-8000-000000000001", 0)
    line = next(line for line in payload["script"].splitlines() if line.startswith("/usr/bin/python3"))
    args = shlex.split(line.split(" <<", 1)[0])[2:]
    values = dict(zip(args[::2], args[1::2]))
    calls = []
    with patch.object(bootstrap.subprocess, "run", side_effect=lambda args, **kwargs: calls.append([str(arg) for arg in args]) or type("Result", (), {"returncode": 0})()):
        assert bootstrap.execute(base64.b64decode(values["--inline-bundle"]), values["--sha256"], values["--run-id"], str(tmp_path)) == 0
    assert any("nlp_training.provision" in call for call in calls)
    assert not any("sudo" in call or "git" in call or "/opt/eliza-training" in " ".join(call) for call in calls)


def test_root_training_manifests_reference_one_owned_profile():
    repo = Path(__file__).resolve().parents[3]
    assert "nlp_training/profiles/sft-trl.txt" in (repo / "requirements.sft-trl.txt").read_text()
    assert "-r requirements.sft-trl.txt" in (repo / "requirements.dpo-trl.txt").read_text()
