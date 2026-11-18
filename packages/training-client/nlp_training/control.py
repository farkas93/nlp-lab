"""Canonical thin host-control transport, shared by current and legacy clients."""
import json
import os
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import urllib.request

from .credentials import credential


class ControlDaemonError(RuntimeError):
    pass


def request_json(url, method="GET", *, headers=None, payload=None, timeout=30):
    req = Request(url, method=method, headers={"Content-Type": "application/json", **(headers or {})},
                  data=json.dumps(payload).encode() if payload is not None else None)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            raw = response.read(16 * 1024 * 1024 + 1)
        if len(raw) > 16 * 1024 * 1024:
            raise ControlDaemonError("API response exceeds size limit")
        result = json.loads(raw)
        if result.get("errors"):
            raise ControlDaemonError("Slurm rejected request: " + json.dumps(result["errors"]))
        return result
    except (HTTPError, URLError) as exc:
        raise ControlDaemonError(f"Control request failed: {method} {url}: {exc}") from exc


class HostControl:
    def __init__(self, base_url=None, *, token=None, request=None):
        self.base_url = (base_url or os.environ.get("INFERENCE_CONTROL_URL", "")).rstrip("/")
        self.token, self.request = token, request

    def call(self, path, method="GET", timeout=30, *, require_token=False):
        if not self.base_url:
            raise ValueError("Configure INFERENCE_CONTROL_URL")
        token = credential(self.token, "CONTROL_DAEMON_TOKEN")
        if require_token and not token:
            raise ValueError("Configure authenticated control access or supply a Slurm token file")
        headers = {"Authorization": "Bearer " + token} if token else {}
        if self.request:
            return self.request(self.base_url + path, method, headers, None)
        return request_json(self.base_url + path, method, headers=headers, timeout=timeout)

    def environment(self, mode=None):
        if mode not in {None, "inference", "training"}:
            raise ValueError("mode must be inference or training")
        return self.call("/environment" + ("/" + mode if mode else ""), "POST" if mode else "GET", 1200 if mode else 30)

    def training_token(self):
        return self.call("/training/token", "POST", require_token=True)


def get_status(base_url, timeout=30, token=None):
    return HostControl(base_url, token=token).call("/status", timeout=timeout)


def stop_stack(base_url, timeout=180, token=None):
    return HostControl(base_url, token=token).call("/stack/stop", "POST", timeout)


def start_stack(base_url, timeout=1200, token=None):
    return HostControl(base_url, token=token).call("/stack/start", "POST", timeout)


def wait_for_stack_state(base_url, expected_substring, timeout_sec=1200, poll_interval_sec=3, token=None):
    deadline = time.monotonic() + timeout_sec
    while True:
        status = get_status(base_url, token=token)
        if expected_substring.lower() in str(status.get("output", "")).lower():
            return status
        if time.monotonic() >= deadline:
            raise ControlDaemonError("Timed out waiting for stack state: " + expected_substring)
        time.sleep(poll_interval_sec)
