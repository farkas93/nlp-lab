from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
import platform
import subprocess
import threading
import time
from typing import Any


def _truthy(value: str | None, *, default: bool) -> bool:
    if value is None:
        return default
    token = str(value).strip().lower()
    if token in {"1", "true", "yes", "y", "on"}:
        return True
    if token in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _float_or_default(value: str | None, *, default: float) -> float:
    if value is None:
        return default
    try:
        parsed = float(str(value).strip())
    except Exception:
        return default
    if parsed <= 0:
        return default
    return parsed


@dataclass
class SystemTelemetryConfig:
    enabled: bool = True
    log_raw_hostname: bool = True
    interval_sec: float = 15.0


def load_telemetry_config_from_env() -> SystemTelemetryConfig:
    return SystemTelemetryConfig(
        enabled=_truthy(os.getenv("NLP_LAB_LOG_SYSTEM_TELEMETRY"), default=True),
        log_raw_hostname=_truthy(os.getenv("NLP_LAB_LOG_RAW_HOSTNAME"), default=True),
        interval_sec=_float_or_default(
            os.getenv("NLP_LAB_SYSTEM_TELEMETRY_INTERVAL_SEC"),
            default=15.0,
        ),
    )


def _hostname_value(*, raw: bool) -> str:
    host = platform.node() or "unknown-host"
    if raw:
        return host
    digest = hashlib.sha256(host.encode("utf-8")).hexdigest()[:16]
    return f"sha256:{digest}"


def _read_mem_total_bytes() -> int | None:
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemTotal:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(parts[1]) * 1024
    except Exception:
        return None
    return None


def _read_mem_util_pct() -> float | None:
    try:
        info: dict[str, int] = {}
        with open("/proc/meminfo", "r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.split()
                if len(parts) < 2:
                    continue
                key = parts[0].rstrip(":")
                info[key] = int(parts[1])
        total = info.get("MemTotal")
        available = info.get("MemAvailable")
        if not total or not available or total <= 0:
            return None
        used = total - available
        return max(0.0, min(100.0, (used / total) * 100.0))
    except Exception:
        return None


def _run_nvidia_smi(query: str) -> list[str]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                f"--query-gpu={query}",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
    except Exception:
        return []
    return [line.strip() for line in output.splitlines() if line.strip()]


def collect_system_fingerprint(*, log_raw_hostname: bool) -> dict[str, Any]:
    fingerprint: dict[str, Any] = {
        "host_name": _hostname_value(raw=log_raw_hostname),
        "host_os": platform.system() or "unknown",
        "host_release": platform.release() or "unknown",
        "host_version": platform.version() or "unknown",
        "host_arch": platform.machine() or "unknown",
        "python_version": platform.python_version(),
        "cpu_logical_cores": os.cpu_count() or 0,
    }

    mem_total = _read_mem_total_bytes()
    if mem_total is not None:
        fingerprint["ram_total_gb"] = round(mem_total / (1024 ** 3), 2)

    gpu_rows = _run_nvidia_smi("name,driver_version,memory.total")
    if gpu_rows:
        names: list[str] = []
        drivers: set[str] = set()
        max_mem_gb = 0.0
        for row in gpu_rows:
            parts = [part.strip() for part in row.split(",")]
            if len(parts) >= 1 and parts[0]:
                names.append(parts[0])
            if len(parts) >= 2 and parts[1]:
                drivers.add(parts[1])
            if len(parts) >= 3:
                try:
                    mem_gb = float(parts[2]) / 1024.0
                    max_mem_gb = max(max_mem_gb, mem_gb)
                except Exception:
                    pass
        fingerprint["gpu_count"] = len(gpu_rows)
        if names:
            fingerprint["gpu_names"] = " | ".join(names)
        if drivers:
            fingerprint["gpu_driver_versions"] = "|".join(sorted(drivers))
        if max_mem_gb > 0:
            fingerprint["gpu_max_memory_gb"] = round(max_mem_gb, 2)
    else:
        fingerprint["gpu_count"] = 0

    return fingerprint


def _safe_avg(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _safe_peak(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(max(values))


class SystemTelemetrySampler:
    def __init__(self, *, interval_sec: float = 15.0) -> None:
        self.interval_sec = max(0.5, float(interval_sec))
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._started_at = 0.0
        self._ended_at = 0.0
        self._samples: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._started_at = time.time()
        self._thread = threading.Thread(target=self._loop, name="system-telemetry", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            sample = self._sample_once()
            if sample:
                with self._lock:
                    for key, value in sample.items():
                        self._samples.setdefault(key, []).append(float(value))
            self._stop_event.wait(self.interval_sec)

    def _sample_once(self) -> dict[str, float]:
        sample: dict[str, float] = {}

        try:
            import psutil  # type: ignore

            cpu_pct = float(psutil.cpu_percent(interval=None))
            sample["cpu_util_pct"] = cpu_pct
            mem_pct = float(psutil.virtual_memory().percent)
            sample["ram_util_pct"] = mem_pct
        except Exception:
            try:
                cpu_count = float(os.cpu_count() or 1)
                load1, _, _ = os.getloadavg()
                sample["cpu_util_pct"] = max(0.0, min(100.0, (load1 / cpu_count) * 100.0))
            except Exception:
                pass
            mem_pct = _read_mem_util_pct()
            if mem_pct is not None:
                sample["ram_util_pct"] = float(mem_pct)

        gpu_rows = _run_nvidia_smi("utilization.gpu,memory.used,memory.total")
        for idx, row in enumerate(gpu_rows):
            parts = [part.strip() for part in row.split(",")]
            if len(parts) < 3:
                continue
            try:
                gpu_util = float(parts[0])
                mem_used = float(parts[1])
                mem_total = float(parts[2])
            except Exception:
                continue
            sample[f"gpu_{idx}_util_pct"] = gpu_util
            if mem_total > 0:
                sample[f"gpu_{idx}_mem_util_pct"] = (mem_used / mem_total) * 100.0
            sample[f"gpu_{idx}_mem_used_gb"] = mem_used / 1024.0

        return sample

    def stop(self) -> dict[str, float]:
        if self._thread is None:
            return {}
        self._stop_event.set()
        self._thread.join(timeout=max(3.0, self.interval_sec * 2))
        self._ended_at = time.time()

        with self._lock:
            copied = {key: list(values) for key, values in self._samples.items()}

        summary: dict[str, float] = {}
        for key, values in copied.items():
            summary[f"{key}_avg"] = _safe_avg(values)
            summary[f"{key}_peak"] = _safe_peak(values)
        summary["telemetry_samples"] = float(sum(len(values) for values in copied.values()))
        summary["run_wall_time_sec"] = max(0.0, self._ended_at - self._started_at)
        return summary


def build_system_telemetry_artifact(
    *,
    fingerprint: dict[str, Any],
    summary_metrics: dict[str, float],
) -> str:
    payload = {
        "fingerprint": fingerprint,
        "summary_metrics": summary_metrics,
    }
    return json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True)
