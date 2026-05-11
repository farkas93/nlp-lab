from __future__ import annotations

try:
    from src.eliza_trainer.sft.backends.unsloth_backend import run_unsloth_training
except ModuleNotFoundError:  # pragma: no cover - legacy entrypoint compatibility
    from eliza_trainer.sft.backends.unsloth_backend import run_unsloth_training


__all__ = ["run_unsloth_training"]
