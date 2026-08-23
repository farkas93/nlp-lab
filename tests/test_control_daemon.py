from __future__ import annotations

import json
import urllib.request
from unittest.mock import MagicMock, patch

from eliza_trainer.orchestration import control_daemon


def test_request_uses_optional_token_from_environment() -> None:
    response = MagicMock()
    response.read.return_value = json.dumps({"status": "stopped"}).encode()
    response.__enter__.return_value = response

    with (
        patch.dict("os.environ", {"CONTROL_DAEMON_TOKEN": "secret"}),
        patch("urllib.request.urlopen", return_value=response) as urlopen,
    ):
        control_daemon.stop_stack("http://10.42.47.8:8030")

    request = urlopen.call_args.args[0]
    assert isinstance(request, urllib.request.Request)
    assert request.get_header("Authorization") == "Bearer secret"
    assert urlopen.call_args.kwargs["timeout"] == 180


def test_start_stack_uses_large_model_timeout() -> None:
    response = MagicMock()
    response.read.return_value = json.dumps({"status": "started"}).encode()
    response.__enter__.return_value = response

    with patch("urllib.request.urlopen", return_value=response) as urlopen:
        control_daemon.start_stack("http://10.42.47.8:8030", token="explicit")

    request = urlopen.call_args.args[0]
    assert request.get_header("Authorization") == "Bearer explicit"
    assert urlopen.call_args.kwargs["timeout"] == 1200
