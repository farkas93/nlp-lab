from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any


class ControlDaemonError(RuntimeError):
    pass


def _request_json(url: str, method: str = "GET", timeout: int = 30) -> dict[str, Any]:
    request = urllib.request.Request(url=url, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace").strip()
        detail = body or f"HTTP {exc.code}"
        raise ControlDaemonError(f"Control daemon request failed: {method} {url}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise ControlDaemonError(f"Control daemon unreachable: {method} {url}: {exc}") from exc


def get_status(base_url: str, timeout: int = 30) -> dict[str, Any]:
    return _request_json(f"{base_url.rstrip('/')}/status", method="GET", timeout=timeout)


def stop_stack(base_url: str, timeout: int = 60) -> dict[str, Any]:
    return _request_json(f"{base_url.rstrip('/')}/stack/stop", method="POST", timeout=timeout)


def start_stack(base_url: str, timeout: int = 60) -> dict[str, Any]:
    return _request_json(f"{base_url.rstrip('/')}/stack/start", method="POST", timeout=timeout)


def wait_for_stack_state(
    base_url: str,
    expected_substring: str,
    timeout_sec: int = 180,
    poll_interval_sec: int = 3,
) -> dict[str, Any]:
    deadline = time.time() + timeout_sec
    expected = expected_substring.lower()
    last_status: dict[str, Any] = {}

    while time.time() < deadline:
        last_status = get_status(base_url)
        output = str(last_status.get("output", "")).lower()
        if expected in output:
            return last_status
        time.sleep(poll_interval_sec)

    raise ControlDaemonError(
        f"Timed out waiting for stack state containing '{expected_substring}'. Last status: {last_status}"
    )
