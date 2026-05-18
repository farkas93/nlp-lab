from __future__ import annotations

try:
    from src.dpo_dataset_loader import DPOManifestLoadResult, load_dpo_manifest_dataset
except ModuleNotFoundError:  # pragma: no cover - legacy entrypoint compatibility
    from dpo_dataset_loader import DPOManifestLoadResult, load_dpo_manifest_dataset

__all__ = [
    "DPOManifestLoadResult",
    "load_dpo_manifest_dataset",
]
