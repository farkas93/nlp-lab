import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

from nlp_training import provision
from nlp_training.smoke import job


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
    requirements = next(call[-1] for call in calls if "-r" in call)
    assert requirements.endswith("nlp_training/profiles/sft-trl.arm64-cu130.txt")
    assert report["definition"]["minimum_driver_major"] == 580
    assert any("pip" in call and "check" in call for call in calls)
    assert not any("cuda.is_available" in " ".join(call) for call in calls)


def test_probe_is_gpu_allocated_and_python_body_compiles():
    payload = job("00000000-0000-4000-8000-000000000001", 30)
    assert payload["job"]["tres_per_node"] == "gres/gpu:1"
    body = payload["script"].split("<<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
    compile(body, "probe", "exec")
    assert "loss.backward()" in body and "torch.save" in body


def test_root_training_manifests_reference_one_owned_profile():
    repo = Path(__file__).resolve().parents[3]
    assert "nlp_training/profiles/sft-trl.txt" in (repo / "requirements.sft-trl.txt").read_text()
    assert "-r requirements.sft-trl.txt" in (repo / "requirements.dpo-trl.txt").read_text()
