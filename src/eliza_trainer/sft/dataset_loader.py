from __future__ import annotations

from src.sft_dataset_loader import (
    SFTManifestLoadResult,
    TokenizationStats,
    AssistantOnlyTokenizationStats,  # Backward compatibility alias
    load_sft_manifest_dataset,
    tokenize_with_loss_mode,
    tokenize_with_assistant_only_loss,  # Backward compatibility alias
)

__all__ = [
    "SFTManifestLoadResult",
    "TokenizationStats",
    "AssistantOnlyTokenizationStats",
    "load_sft_manifest_dataset",
    "tokenize_with_loss_mode",
    "tokenize_with_assistant_only_loss",
]
