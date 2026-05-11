from __future__ import annotations

try:
    from src.nlp_lab.sft.dataset_loader import (
        SFTManifestLoadResult,
        load_sft_manifest_dataset,
        tokenize_with_assistant_only_loss,
    )
except ModuleNotFoundError:  # pragma: no cover - legacy entrypoint compatibility
    from nlp_lab.sft.dataset_loader import (
        SFTManifestLoadResult,
        load_sft_manifest_dataset,
        tokenize_with_assistant_only_loss,
    )

__all__ = [
    "SFTManifestLoadResult",
    "load_sft_manifest_dataset",
    "tokenize_with_assistant_only_loss",
]
