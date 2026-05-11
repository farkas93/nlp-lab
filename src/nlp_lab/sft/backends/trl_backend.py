from __future__ import annotations

try:
    from src.eliza_trainer.sft.backends.trl_backend import run_trl_training
except ModuleNotFoundError:  # pragma: no cover - legacy entrypoint compatibility
    from eliza_trainer.sft.backends.trl_backend import run_trl_training


__all__ = ["run_trl_training"]
