from __future__ import annotations

try:
    from src.eliza_trainer.common.runtime import (
        configure_logging,
        ensure_cuda_alloc_conf,
        load_project_env,
        should_start_local_mlflow,
    )
except ModuleNotFoundError:  # pragma: no cover - legacy entrypoint compatibility
    from eliza_trainer.common.runtime import (
        configure_logging,
        ensure_cuda_alloc_conf,
        load_project_env,
        should_start_local_mlflow,
    )

__all__ = [
    "configure_logging",
    "ensure_cuda_alloc_conf",
    "load_project_env",
    "should_start_local_mlflow",
]
