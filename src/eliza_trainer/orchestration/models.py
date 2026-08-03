from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class TrainingRunResult:
    host: str
    partition_id: int
    attempt_number: int
    spark_gpu_addresses: list[str]
    return_code: int
    command: list[str]
    stdout_tail: str
    stderr_tail: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
