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


def notebook_cells():
    notebook = json.loads((ROOT / "notebooks/hass_private_sft_end_to_end.ipynb").read_text())
    return {cell["id"]: compile("".join(cell["source"]), cell["id"], "exec")
            for cell in notebook["cells"] if cell["cell_type"] == "code"}


def test_notebook_binds_config_and_never_repeats_submission(tmp_path):
    code = notebook_cells()
    repo = tmp_path / "repo"
    (repo / "configs").mkdir(parents=True)
    shutil.copy(ROOT / "configs/sft_hass_private_smoke.template.yaml", repo / "configs")
    private_dir = tmp_path / "private"
    private_dir.mkdir()
    digest = "a" * 64
    training = Mock(prefix="training/runs")
    storage = Mock()
    namespace = dict(REPO=repo, PRIVATE_DIR=private_dir, SLURM_ARTIFACT_BUCKET="private-runs",
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


def test_capture_handoff_computes_candidate_pin_and_resumes(tmp_path):
    code = notebook_cells()
    pilot = Mock(source_key="hass/raw/snapshots/home/capture.json")
    pilot.launch_ingestion.return_value = "ingest-1"
    pilot.launch_generation.return_value = "generate-1"
    pilot.preview.return_value = {"key": "hass/cleaned/candidates/fixture/candidates.json"}
    factory = Mock(return_value=pilot)
    progress = {}
    storage = Mock()
    storage.read_bytes.return_value = b'{"sha256":"semantic digest"}'
    namespace = dict(HOUSEHOLD_UUID=str(uuid.uuid4()), CAPTURE_FILE="capture.json", SOURCE_BUCKET="synthetic",
                     storage=storage, HassPilotWorkflow=factory, remember=lambda **values: progress.update(values),
                     progress=progress, LIMIT=1024, sha=lambda data: hashlib.sha256(data).hexdigest(),
                     PRIVATE_DIR=tmp_path, json=json)
    for stage in ("ingest", "generate", "candidate-pin"):
        exec(code[stage], namespace)
    pilot.archive_file.assert_called_once_with("capture.json")
    assert progress["candidates_sha256"] == hashlib.sha256(storage.read_bytes.return_value).hexdigest()
    namespace["CAPTURE_FILE"] = ""
    for stage in ("ingest", "generate", "candidate-pin"):
        exec(code[stage], namespace)
    pilot.use_archived_source.assert_called_once_with(progress["source_key"])
    assert pilot.ingest_run_id == "ingest-1" and pilot.generate_run_id == "generate-1"

    steps = [{"role": "assistant", "tool_calls": [{}]}, {"role": "tool"}, {"role": "assistant", "content": "Done."}]
    namespace["episodes"] = {i: {"episode_id": str(i), "base_task_id": str(i), "household_group_id": "home", "steps": steps}
                             for i in (1, 2)}
    namespace["input"] = Mock(side_effect=["1", "2", "", "operator", "smoke"])
    exec(code["decisions"], namespace)
    assert json.loads((tmp_path / "decisions.json").read_text())["assignments"] == {"1": "train", "2": "eval"}
    namespace["input"] = Mock(side_effect=AssertionError("must not prompt on resume"))
    exec(code["decisions"], namespace)
