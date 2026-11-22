import hashlib
import io
import json

import pytest

from nlp_training.operator import prepare_run, workspace_settings


def test_workspace_precedence_and_attempt_binding(tmp_path, monkeypatch):
    (tmp_path / '.env').write_text('INFERENCE_CONTROL_URL=http://workspace:8030\n'
                                  'NLP_LAB_REPO=/repo\nSLURM_ARTIFACT_BUCKET=private-runs\n'
                                  'MLFLOW_TRACKING_URI=https://tracking.invalid\n')
    monkeypatch.setenv('INFERENCE_CONTROL_URL', 'http://deployment:8030')
    settings, origins = workspace_settings(tmp_path, {'MLFLOW_TRACKING_URI': 'https://override.invalid'})
    assert settings['INFERENCE_CONTROL_URL'] == 'http://workspace:8030'
    assert origins['MLFLOW_TRACKING_URI'] == 'notebook'
    (tmp_path / 'configs').mkdir()
    args = dict(kind='hass-eval', config={'evaluation': {}}, pin={'sha256': 'a' * 64},
                bucket='private-runs', resources={'cpus': 8})
    first = prepare_run(tmp_path, tmp_path / 'runs/example', **args)
    assert prepare_run(tmp_path, tmp_path / 'runs/example', **args) == first
    with pytest.raises(ValueError, match='different plan'):
        prepare_run(tmp_path, tmp_path / 'runs/example', **{**args, 'resources': {'cpus': 4}})


def test_remote_reload_checks_archive_integrity(tmp_path):
    pytest.importorskip('src.eliza_trainer.sft.remote_checks')
    from src.eliza_trainer.sft.remote_checks import fetch_adapter
    source = '00000000-0000-4000-8000-000000000001'
    root = f'training/runs/{source}'
    pin = {'uri': 's3://datasets/example', 'sha256': 'a' * 64}
    files = {'adapter_config.json': b'{"revision":"rev"}', 'adapter_model.safetensors': b'weights',
             'tokenizer.json': b'{}', 'tokenizer_config.json': b'{}'}
    index = [{'path': f'checkpoints/final/{name}', 'size': len(payload),
              'sha256': hashlib.sha256(payload).hexdigest()} for name, payload in files.items()]
    objects = {root + '/result.json': json.dumps({'run_id': source, 'outcome': 'succeeded'}).encode(),
               root + '/run-manifest.json': json.dumps({'run_id': source, 'kind': 'sft', 'dataset_manifest': pin}).encode(),
               root + '/artifact-index.json': json.dumps(index).encode(),
               **{f'{root}/artifacts/checkpoints/final/{name}': payload for name, payload in files.items()}}

    class Storage:
        def get_object(self, *, Bucket, Key):
            assert Bucket == 'private-runs'
            return {'Body': io.BytesIO(objects[Key])}

    def fetch(destination, dataset=pin):
        return fetch_adapter(Storage(), 'private-runs', 'training/runs/evaluator', source, dataset, destination)

    assert fetch(tmp_path / 'adapter') == {'revision': 'rev'}
    with pytest.raises(ValueError, match='exact dataset'):
        fetch(tmp_path / 'wrong-dataset', {'uri': 'other'})
    objects[root + '/artifacts/checkpoints/final/adapter_model.safetensors'] = b'corrupt'
    with pytest.raises(ValueError, match='checksum mismatch'):
        fetch(tmp_path / 'corrupt-adapter')
