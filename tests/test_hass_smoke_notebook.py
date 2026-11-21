import hashlib
import json
import re
import shutil
import uuid
from pathlib import Path, PurePosixPath
from unittest.mock import Mock

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_notebook_binds_config_and_never_repeats_submission(tmp_path):
    notebook = json.loads((ROOT / "notebooks/hass_private_sft_end_to_end.ipynb").read_text())
    code = {cell["id"]: compile("".join(cell["source"]), cell["id"], "exec")
            for cell in notebook["cells"] if cell["cell_type"] == "code"}
    repo = tmp_path / "repo"
    (repo / "configs").mkdir(parents=True)
    shutil.copy(ROOT / "configs/sft_hass_private_smoke.template.yaml", repo / "configs")
    private_dir = tmp_path / "private"
    private_dir.mkdir()
    digest = "a" * 64
    training = Mock(prefix="training/runs")
    storage = Mock()
    namespace = dict(REPO=repo, PRIVATE_DIR=private_dir, ARTIFACT_BUCKET="private-runs",
                     receipt=dict(manifest_uri="s3://sft-datasets/test/manifest.json", manifest_sha256=digest,
                                  bucket="sft-datasets", dataset_id="hass", dataset_version="experimental-test"),
                     yaml=yaml, json=json, uuid=uuid, storage=storage, ElizaTraining=Mock(return_value=training),
                     input=Mock(return_value="wrong"), LIMIT=32 * 1024 * 1024, PurePosixPath=PurePosixPath,
                     re=re, sha=lambda data: hashlib.sha256(data).hexdigest())
    exec(code["prepare"], namespace)
    run_id = namespace["run_id"]
    assert yaml.safe_load(namespace["config_path"].read_text())["data"]["expected_manifest_sha256"] == digest
    with pytest.raises(ValueError, match="Acknowledgement"):
        exec(code["submit"], namespace)
    training.environment.assert_not_called()
    namespace["input"].return_value = digest
    training.submit.side_effect = TimeoutError("simulated lost response")
    with pytest.raises(TimeoutError):
        exec(code["submit"], namespace)
    exec(code["prepare"], namespace)
    with pytest.raises(RuntimeError, match="already attempted"):
        exec(code["submit"], namespace)
    assert training.submit.call_count == 1
    assert training.submit.call_args.kwargs["run_id"] == namespace["run_id"] == run_id
    assert training.submit.call_args.kwargs["restricted_manifest_ack_sha256"] == digest

    model = namespace["cfg"]["model"]
    artifacts = {"adapter_config.json": json.dumps({"base_model_name_or_path": f"{model['owner']}/{model['name']}",
                                                     "revision": model["revision"]}).encode(),
                 "adapter_model.safetensors": b"adapter", "tokenizer.json": b"tokenizer"}
    index = [{"path": f"checkpoints/run/final/{name}", "size": len(data), "sha256": namespace["sha"](data)}
             for name, data in artifacts.items()]
    storage.read_json.side_effect = lambda *, bucket, key, max_bytes: (
        index if key.endswith("artifact-index.json") else {"run_id": run_id, "outcome": "succeeded"})
    storage.read_bytes.side_effect = lambda *, bucket, key, max_bytes: artifacts[PurePosixPath(key).name]
    training.logs.return_value = "Training finished at optimizer global_step=12\n"
    exec(code["artifacts"], namespace)
    assert (private_dir / "final/adapter_model.safetensors").read_bytes() == b"adapter"
    artifacts["adapter_model.safetensors"] = b"changed"
    with pytest.raises(ValueError, match="Artifact checksum mismatch"):
        exec(code["artifacts"], namespace)
