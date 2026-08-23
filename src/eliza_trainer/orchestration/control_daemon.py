from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any


class ControlDaemonError(RuntimeError):
    pass


def _request_json(
    url: str,
    method: str = "GET",
    timeout: int = 30,
    token: str | None = None,
) -> dict[str, Any]:
    headers: dict[str, str] = {}
    control_token = token if token is not None else os.environ.get("CONTROL_DAEMON_TOKEN", "")
    if control_token:
        headers["Authorization"] = f"Bearer {control_token}"
    request = urllib.request.Request(url=url, method=method, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace").strip()
        detail = body or f"HTTP {exc.code}"
        raise ControlDaemonError(f"Control daemon request failed: {method} {url}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise ControlDaemonError(f"Control daemon unreachable: {method} {url}: {exc}") from exc


def get_status(base_url: str, timeout: int = 30, token: str | None = None) -> dict[str, Any]:
    return _request_json(f"{base_url.rstrip('/')}/status", method="GET", timeout=timeout, token=token)


def stop_stack(base_url: str, timeout: int = 180, token: str | None = None) -> dict[str, Any]:
    return _request_json(f"{base_url.rstrip('/')}/stack/stop", method="POST", timeout=timeout, token=token)


def start_stack(base_url: str, timeout: int = 1200, token: str | None = None) -> dict[str, Any]:
    return _request_json(f"{base_url.rstrip('/')}/stack/start", method="POST", timeout=timeout, token=token)


def wait_for_stack_state(
    base_url: str,
    expected_substring: str,
    timeout_sec: int = 1200,
    poll_interval_sec: int = 3,
    token: str | None = None,
) -> dict[str, Any]:
    deadline = time.time() + timeout_sec
    expected = expected_substring.lower()
    last_status: dict[str, Any] = {}

    while time.time() < deadline:
        last_status = get_status(base_url, token=token)
        output = str(last_status.get("output", "")).lower()
        if expected in output:
            return last_status
        time.sleep(poll_interval_sec)

    raise ControlDaemonError(
        f"Timed out waiting for stack state containing '{expected_substring}'. Last status: {last_status}"
    )
