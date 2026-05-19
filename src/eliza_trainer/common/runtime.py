from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv


def configure_logging(log_level: str = "INFO") -> None:
    level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def load_project_env(project_root: Path) -> None:
    env_path = project_root / ".env"
    if env_path.exists():
        load_dotenv(env_path)


def should_start_local_mlflow(mlflow_uri: str) -> bool:
    lowered = (mlflow_uri or "").lower()
    return "localhost" in lowered or "127.0.0.1" in lowered


def ensure_cuda_alloc_conf() -> None:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

__all__ = [
    "configure_logging",
    "ensure_cuda_alloc_conf",
    "load_project_env",
    "should_start_local_mlflow",
]
