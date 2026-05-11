from __future__ import annotations

try:
    from src.eliza_trainer.losses.loss_masking import AssistantOnlyDataCollator
except ModuleNotFoundError:  # pragma: no cover - legacy entrypoint compatibility
    from eliza_trainer.losses.loss_masking import AssistantOnlyDataCollator

__all__ = ["AssistantOnlyDataCollator"]
