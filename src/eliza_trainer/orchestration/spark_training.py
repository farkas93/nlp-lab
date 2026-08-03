from __future__ import annotations

from typing import Any

from .models import TrainingRunResult


def _tail_text(raw: str, max_chars: int = 8000) -> str:
    if len(raw) <= max_chars:
        return raw
    return raw[-max_chars:]


def run_training_via_spark(
    *,
    spark_session: Any,
    nlp_lab_root: str,
    mode: str,
    config_path: str,
    extra_args: list[str] | None = None,
    env_overrides: dict[str, str] | None = None,
) -> TrainingRunResult:
    mode_normalized = mode.strip().lower()
    if mode_normalized not in {"sft", "dpo"}:
        raise ValueError("mode must be 'sft' or 'dpo'")

    launcher = "./start_sft.sh" if mode_normalized == "sft" else "./start_dpo.sh"
    cmd = [launcher, config_path, *(extra_args or [])]
    payload = {
        "nlp_lab_root": nlp_lab_root,
        "command": cmd,
        "env": env_overrides or {},
    }

    def _run_partition(rows: Any) -> Any:
        import os as _os
        import socket
        import subprocess
        from pyspark import TaskContext

        _ = list(rows)
        tc = TaskContext.get()

        env = _os.environ.copy()
        env.update(payload["env"])
        env.setdefault("PYTHONUNBUFFERED", "1")

        proc = subprocess.run(
            payload["command"],
            cwd=payload["nlp_lab_root"],
            env=env,
            capture_output=True,
            text=True,
        )

        gpu_addresses: list[str] = []
        try:
            resources = tc.resources()
            if "gpu" in resources:
                gpu_addresses = list(resources["gpu"].addresses)
        except Exception:
            pass

        result = {
            "host": socket.gethostname(),
            "partition_id": tc.partitionId(),
            "attempt_number": tc.attemptNumber(),
            "spark_gpu_addresses": gpu_addresses,
            "return_code": proc.returncode,
            "command": payload["command"],
            "stdout_tail": _tail_text(proc.stdout),
            "stderr_tail": _tail_text(proc.stderr),
        }
        yield result

    collected = spark_session.sparkContext.parallelize([1], 1).mapPartitions(_run_partition).collect()
    if not collected:
        raise RuntimeError("Spark training launcher returned no results")

    first = collected[0]
    run_result = TrainingRunResult(**first)
    if run_result.return_code != 0:
        raise RuntimeError(
            "Training command failed on Spark executor\n"
            f"host={run_result.host} return_code={run_result.return_code}\n"
            f"stdout_tail:\n{run_result.stdout_tail}\n\n"
            f"stderr_tail:\n{run_result.stderr_tail}"
        )
    return run_result
