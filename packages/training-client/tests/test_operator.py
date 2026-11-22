import hashlib
import io
import json

import pytest

from nlp_training.operator import prepare_run, workspace_settings


def test_workspace_precedence_and_attempt_binding(tmp_path, monkeypatch):
    (tmp_path / 'training.env').write_text('INFERENCE_CONTROL_URL=http://workspace:8030\n'
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
    assert (tmp_path / 'runs/example/resolved-config.yaml').is_file()


def test_workspace_install_preserves_edits_and_supplies_both_notebooks(tmp_path):
    from pathlib import Path
    from scripts.setup_training_workspace import install
    root = Path(__file__).resolve().parents[3]
    target = tmp_path / 'training'
    install(root, target)
    assert (target / 'runs').is_dir()
    assert (target / 'hass-dataset.ipynb').is_file()
    (target / 'training.env').write_text('USER_SETTINGS=yes')
    (target / 'hass-sft.ipynb').write_text('user work')
    result = install(root, target)
    assert (target / 'training.env').read_text() == 'USER_SETTINGS=yes'
    assert (target / 'hass-sft.ipynb').read_text() == 'user work'
    assert 'updated-' in result['hass-sft.ipynb']


def test_tracking_probe_creates_run_and_verifies_metric(monkeypatch):
    from nlp_training.tracking import verify_tracking
    from urllib.error import HTTPError
    calls = []

    class Response(io.BytesIO):
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.close()

    def request(req, timeout):
        route = req.full_url.split('/api/2.0/mlflow/')[1].split('?')[0]
        calls.append((route, json.loads(req.data) if req.data else None))
        if route == 'experiments/get-by-name':
            raise HTTPError(req.full_url, 404, 'missing', {}, io.BytesIO(b'{"error_code":"RESOURCE_DOES_NOT_EXIST"}'))
        result = {'experiments/create': {'experiment_id': '7'},
                  'runs/create': {'run': {'info': {'run_id': 'test-run'}}},
                  'runs/get': {'run': {'data': {'metrics': [{'key': 'eliza_connectivity', 'value': 1}]}}}}.get(route, {})
        return Response(json.dumps(result).encode())

    monkeypatch.setattr('nlp_training.tracking.urlopen', request)
    result = verify_tracking('https://tracking.invalid', 'test-experiment')
    assert result['verified'] and result['experiment_id'] == '7'
    assert [route for route, _ in calls] == ['experiments/get-by-name', 'experiments/create', 'runs/create',
                                           'runs/log-metric', 'runs/update', 'runs/get']


def test_live_monitor_saves_status_log_and_terminal_receipt(tmp_path):
    from unittest.mock import Mock, patch
    from nlp_training.operator import watch_run
    training = Mock(prefix='training/runs', bucket='private')
    training.storage.read_bytes.return_value = b'loss=0.5 step=1\n'
    training.status.side_effect = [dict(run_id='test', job_id=8, states=['RUNNING'], terminal=False),
                                   dict(run_id='test', job_id=8, source='trainer-process', outcome='succeeded', terminal=True)]
    render = Mock()
    with patch('time.sleep'):
        assert watch_run(training, 'test', tmp_path, render=render)['outcome'] == 'succeeded'
    assert json.loads((tmp_path / 'result.json').read_text())['outcome'] == 'succeeded'
    assert (tmp_path / 'live-log.txt').read_text() == 'loss=0.5 step=1\n'
    assert render.call_count == 2


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
