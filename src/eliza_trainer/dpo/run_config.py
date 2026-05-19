from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import os
import re

import yaml


CONFIG_SCHEMA_VERSION = 2
BACKENDS = {"trl"}
CACHE_MODES = {"reuse", "refresh"}
TAG_STRATEGIES = {"run_name", "none", "custom"}
SOURCE_LINEAGE_STRATEGIES = {"auto", "adapter", "full"}


@dataclass
class DPOIdentityConfig:
    experiment_tag: str
    backend: str
    run_label: str | None = None


@dataclass
class DPODataConfig:
    bucket: str
    dataset_id: str
    dataset_version: str
    manifest_uri: str
    train_split: str = "train"
    eval_split: str = "eval"
    max_train_samples: int | None = None
    max_eval_samples: int | None = None
    expected_manifest_sha256: str | None = None
    cache_mode: str = "reuse"


@dataclass
class DPOModelConfig:
    owner: str
    name: str
    max_seq_len: int = 2048
    use_bf16: bool = True
    load_in_4bit: bool = False
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )

    @property
    def model_name(self) -> str:
        return f"{self.owner}/{self.name}"


@dataclass
class DPOTrainingConfig:
    experiment_name: str
    run_name: str
    output_dir: str
    output_root: str = "./outputs"
    backend: str = "trl"
    num_train_epochs: int = 1
    learning_rate: float = 2e-6
    train_batch_size: int = 1
    eval_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    logging_steps: int = 20
    save_strategy: str = "epoch"
    eval_strategy: str = "epoch"
    warmup_steps: int = 30
    lr_scheduler_type: str = "linear"
    seed: int = 42
    gradient_checkpointing: bool = True


@dataclass
class DPOPreferenceConfig:
    beta: float = 0.1
    max_prompt_length: int = 1024
    max_length: int = 2048


@dataclass
class DPOHubConfig:
    push_to_hub: bool = False
    owner: str | None = None
    publish_adapter: bool = True
    publish_full_model: bool = False
    repo_adapter: str | None = None
    repo_full_model: str | None = None
    adapter_repo_name: str | None = None
    full_model_repo_name: str | None = None
    adapter_tag: str | None = None
    full_model_tag: str | None = None
    adapter_tag_strategy: str = "run_name"
    full_model_tag_strategy: str = "none"
    allow_existing_tags: bool = True
    source_lineage_strategy: str = "auto"


@dataclass
class DPOTrackingConfig:
    mlflow_tracking_uri: str | None = None


@dataclass
class DPORuntimeConfig:
    hf_model_cache_dir: str = "./hf_models"
    log_level: str = "INFO"


@dataclass
class DPORunConfig:
    identity: DPOIdentityConfig
    data: DPODataConfig
    model: DPOModelConfig
    training: DPOTrainingConfig
    dpo: DPOPreferenceConfig
    hub: DPOHubConfig
    tracking: DPOTrackingConfig
    runtime: DPORuntimeConfig

    @property
    def hf_model_cache_dir(self) -> str:
        return self.runtime.hf_model_cache_dir


def _required(section: dict, key: str, section_name: str) -> object:
    if key not in section:
        raise ValueError(f"Missing required key '{section_name}.{key}' in DPO config")
    return section[key]


def _require_schema_version(raw: dict) -> None:
    value = raw.get("config_schema_version")
    if value != CONFIG_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported DPO config schema version. "
            f"Expected config_schema_version={CONFIG_SCHEMA_VERSION}."
        )


def _reject_legacy_keys(raw: dict) -> None:
    legacy_map = {
        "data": {"dataset_manifest_uri"},
        "model": {"model_name"},
        "training": {"experiment_name", "run_name", "output_dir", "backend"},
        "hub": {"repo_name", "full_model"},
    }
    for section_name, keys in legacy_map.items():
        section = raw.get(section_name)
        if not isinstance(section, dict):
            continue
        overlap = sorted(k for k in keys if k in section)
        if overlap:
            raise ValueError(
                "Legacy config keys are not supported in schema v2: "
                f"{', '.join(f'{section_name}.{k}' for k in overlap)}"
            )
    if "hf_model_cache_dir" in raw:
        raise ValueError(
            "Legacy root key 'hf_model_cache_dir' is not supported in schema v2. "
            "Use runtime.hf_model_cache_dir instead."
        )


def _normalize_backend(value: str) -> str:
    backend = (value or "trl").strip().lower()
    if backend not in BACKENDS:
        raise ValueError(f"Unsupported identity.backend={backend}; expected one of {sorted(BACKENDS)}")
    return backend


def _normalize_cache_mode(value: str) -> str:
    cache_mode = (value or "reuse").strip().lower()
    if cache_mode not in CACHE_MODES:
        raise ValueError(f"Unsupported data.cache_mode={cache_mode}; expected one of {sorted(CACHE_MODES)}")
    return cache_mode


def _slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()
    if not slug:
        raise ValueError(f"Unable to create slug from value={value!r}")
    return slug


def _ensure_repo_component(value: str, label: str) -> str:
    cleaned = str(value or "").strip()
    if not cleaned:
        raise ValueError(f"{label} must not be empty")
    if "/" in cleaned:
        raise ValueError(f"{label} must not contain '/': {cleaned}")
    return cleaned


def _ensure_repo_stem_has_lora(stem: str) -> str:
    return stem if stem.endswith("-lora") else f"{stem}-lora"


def _strip_repo_stem_lora(stem: str) -> str:
    return stem[:-5] if stem.endswith("-lora") else stem


def _resolve_tag(*, strategy: str, configured_tag: str | None, run_name: str, label: str) -> str | None:
    normalized = str(strategy or "none").strip().lower()
    if normalized not in TAG_STRATEGIES:
        raise ValueError(f"Unsupported hub.{label}_tag_strategy={strategy}; expected one of {sorted(TAG_STRATEGIES)}")
    if normalized == "none":
        return None
    if normalized == "run_name":
        return run_name
    cleaned = str(configured_tag or "").strip()
    if not cleaned:
        raise ValueError(f"hub.{label}_tag must be set when hub.{label}_tag_strategy=custom")
    return cleaned


def _resolve_manifest_uri(*, bucket: str, dataset_id: str, dataset_version: str) -> str:
    return f"s3://{bucket}/dataset_id={dataset_id}/dataset_version={dataset_version}/manifest.json"


def _resolve_repo_names(
    *,
    hub_owner: str,
    model_owner: str,
    model_name: str,
    experiment_tag: str,
    backend: str,
    publish_adapter: bool,
    publish_full_model: bool,
    repo_adapter_override: str | None,
    repo_full_override: str | None,
) -> tuple[str | None, str | None]:
    if hub_owner != model_owner:
        base_stem = f"{model_name}-{experiment_tag}-{backend}"
    else:
        base_stem = model_name

    adapter_stem = _ensure_repo_stem_has_lora(base_stem)
    full_stem = _strip_repo_stem_lora(base_stem)

    adapter_repo = None
    full_repo = None
    if publish_adapter:
        adapter_repo = repo_adapter_override or f"{hub_owner}/{adapter_stem}"
    if publish_full_model:
        full_repo = repo_full_override or f"{hub_owner}/{full_stem}"
    return adapter_repo, full_repo


def load_dpo_run_config(config_path: str) -> DPORunConfig:
    path = Path(config_path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Invalid DPO config: expected top-level mapping")

    _require_schema_version(raw)
    _reject_legacy_keys(raw)

    identity_raw = raw.get("identity") or {}
    data_raw = raw.get("data") or {}
    model_raw = raw.get("model") or {}
    training_raw = raw.get("training") or {}
    dpo_raw = raw.get("dpo") or {}
    hub_raw = raw.get("hub") or {}
    tracking_raw = raw.get("tracking") or {}
    runtime_raw = raw.get("runtime") or {}

    experiment_tag = _slugify(str(_required(identity_raw, "experiment_tag", "identity")))
    backend = _normalize_backend(str(_required(identity_raw, "backend", "identity")))
    run_label = str(identity_raw.get("run_label") or "").strip() or None

    model_owner = _ensure_repo_component(str(_required(model_raw, "owner", "model")), "model.owner")
    model_name = _ensure_repo_component(str(_required(model_raw, "name", "model")), "model.name")
    model_slug = _slugify(model_name)
    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_name = f"{experiment_tag}-{backend}-{run_stamp}"
    if run_label:
        run_name = f"{run_name}-{_slugify(run_label)}"
    experiment_name = f"{model_slug}_{experiment_tag}_dpo"

    output_root = str(training_raw.get("output_root", "./outputs"))
    output_dir = f"{output_root.rstrip('/')}/{model_slug}/{experiment_tag}/{backend}/{run_stamp}"

    data_bucket = str(_required(data_raw, "bucket", "data"))
    data_dataset_id = str(_required(data_raw, "dataset_id", "data"))
    data_dataset_version = str(_required(data_raw, "dataset_version", "data"))
    manifest_uri = _resolve_manifest_uri(
        bucket=data_bucket,
        dataset_id=data_dataset_id,
        dataset_version=data_dataset_version,
    )

    identity = DPOIdentityConfig(
        experiment_tag=experiment_tag,
        backend=backend,
        run_label=run_label,
    )

    data = DPODataConfig(
        bucket=data_bucket,
        dataset_id=data_dataset_id,
        dataset_version=data_dataset_version,
        manifest_uri=manifest_uri,
        train_split=str(data_raw.get("train_split", "train")),
        eval_split=str(data_raw.get("eval_split", "eval")),
        max_train_samples=(
            int(data_raw["max_train_samples"]) if data_raw.get("max_train_samples") is not None else None
        ),
        max_eval_samples=(
            int(data_raw["max_eval_samples"]) if data_raw.get("max_eval_samples") is not None else None
        ),
        expected_manifest_sha256=(
            str(data_raw["expected_manifest_sha256"]).strip()
            if data_raw.get("expected_manifest_sha256")
            else None
        ),
        cache_mode=_normalize_cache_mode(str(data_raw.get("cache_mode", "reuse"))),
    )

    model = DPOModelConfig(
        owner=model_owner,
        name=model_name,
        max_seq_len=int(model_raw.get("max_seq_len", 2048)),
        use_bf16=bool(model_raw.get("use_bf16", True)),
        load_in_4bit=bool(model_raw.get("load_in_4bit", False)),
        lora_r=int(model_raw.get("lora_r", 16)),
        lora_alpha=int(model_raw.get("lora_alpha", 16)),
        lora_dropout=float(model_raw.get("lora_dropout", 0.05)),
        lora_target_modules=tuple(
            model_raw.get(
                "lora_target_modules",
                [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
            )
        ),
    )

    training = DPOTrainingConfig(
        experiment_name=experiment_name,
        run_name=run_name,
        output_dir=output_dir,
        output_root=output_root,
        backend=backend,
        num_train_epochs=int(training_raw.get("num_train_epochs", 1)),
        learning_rate=float(training_raw.get("learning_rate", 2e-6)),
        train_batch_size=int(training_raw.get("train_batch_size", 1)),
        eval_batch_size=int(training_raw.get("eval_batch_size", 1)),
        gradient_accumulation_steps=int(training_raw.get("gradient_accumulation_steps", 1)),
        logging_steps=int(training_raw.get("logging_steps", 20)),
        save_strategy=str(training_raw.get("save_strategy", "epoch")),
        eval_strategy=str(training_raw.get("eval_strategy", "epoch")),
        warmup_steps=int(training_raw.get("warmup_steps", 30)),
        lr_scheduler_type=str(training_raw.get("lr_scheduler_type", "linear")),
        seed=int(training_raw.get("seed", 42)),
        gradient_checkpointing=bool(training_raw.get("gradient_checkpointing", True)),
    )

    dpo = DPOPreferenceConfig(
        beta=float(dpo_raw.get("beta", 0.1)),
        max_prompt_length=int(dpo_raw.get("max_prompt_length", 1024)),
        max_length=int(dpo_raw.get("max_length", model.max_seq_len)),
    )

    push_to_hub = bool(hub_raw.get("push_to_hub", False))
    hub_owner = str(hub_raw.get("owner") or "").strip() or None
    publish_adapter = bool(hub_raw.get("publish_adapter", True))
    publish_full_model = bool(hub_raw.get("publish_full_model", False))
    if push_to_hub and not hub_owner:
        raise ValueError("hub.push_to_hub=true requires hub.owner")
    if push_to_hub and not (publish_adapter or publish_full_model):
        raise ValueError("hub.push_to_hub=true requires at least one of hub.publish_adapter or hub.publish_full_model")

    adapter_repo_override = str(hub_raw.get("repo_adapter") or "").strip() or None
    full_repo_override = str(hub_raw.get("repo_full_model") or "").strip() or None
    adapter_repo_name, full_model_repo_name = _resolve_repo_names(
        hub_owner=hub_owner or "",
        model_owner=model_owner,
        model_name=model_name,
        experiment_tag=experiment_tag,
        backend=backend,
        publish_adapter=push_to_hub and publish_adapter,
        publish_full_model=push_to_hub and publish_full_model,
        repo_adapter_override=adapter_repo_override,
        repo_full_override=full_repo_override,
    )

    adapter_tag = _resolve_tag(
        strategy=str(hub_raw.get("adapter_tag_strategy", "run_name")),
        configured_tag=(str(hub_raw["adapter_tag"]) if hub_raw.get("adapter_tag") else None),
        run_name=run_name,
        label="adapter",
    )
    full_model_tag = _resolve_tag(
        strategy=str(hub_raw.get("full_model_tag_strategy", "none")),
        configured_tag=(str(hub_raw["full_model_tag"]) if hub_raw.get("full_model_tag") else None),
        run_name=run_name,
        label="full_model",
    )

    source_lineage_strategy = str(hub_raw.get("source_lineage_strategy", "auto")).strip().lower()
    if source_lineage_strategy not in SOURCE_LINEAGE_STRATEGIES:
        raise ValueError(
            "Unsupported hub.source_lineage_strategy="
            f"{source_lineage_strategy}; expected one of {sorted(SOURCE_LINEAGE_STRATEGIES)}"
        )

    hub = DPOHubConfig(
        push_to_hub=push_to_hub,
        owner=hub_owner,
        publish_adapter=publish_adapter,
        publish_full_model=publish_full_model,
        repo_adapter=adapter_repo_override,
        repo_full_model=full_repo_override,
        adapter_repo_name=adapter_repo_name,
        full_model_repo_name=full_model_repo_name,
        adapter_tag=adapter_tag,
        full_model_tag=full_model_tag,
        adapter_tag_strategy=str(hub_raw.get("adapter_tag_strategy", "run_name")).strip().lower(),
        full_model_tag_strategy=str(hub_raw.get("full_model_tag_strategy", "none")).strip().lower(),
        allow_existing_tags=bool(hub_raw.get("allow_existing_tags", True)),
        source_lineage_strategy=source_lineage_strategy,
    )

    tracking = DPOTrackingConfig(
        mlflow_tracking_uri=(
            str(tracking_raw["mlflow_tracking_uri"]) if tracking_raw.get("mlflow_tracking_uri") else None
        )
    )

    runtime = DPORuntimeConfig(
        hf_model_cache_dir=str(runtime_raw.get("hf_model_cache_dir", "./hf_models")),
        log_level=str(runtime_raw.get("log_level", "INFO")).upper(),
    )

    return DPORunConfig(
        identity=identity,
        data=data,
        model=model,
        training=training,
        dpo=dpo,
        hub=hub,
        tracking=tracking,
        runtime=runtime,
    )


def apply_tracking_env(config: DPORunConfig) -> None:
    if config.tracking.mlflow_tracking_uri:
        os.environ["MLFLOW_TRACKING_URI"] = config.tracking.mlflow_tracking_uri


__all__ = [
    "CONFIG_SCHEMA_VERSION",
    "BACKENDS",
    "CACHE_MODES",
    "TAG_STRATEGIES",
    "SOURCE_LINEAGE_STRATEGIES",
    "DPOIdentityConfig",
    "DPODataConfig",
    "DPOModelConfig",
    "DPOTrainingConfig",
    "DPOPreferenceConfig",
    "DPOHubConfig",
    "DPOTrackingConfig",
    "DPORuntimeConfig",
    "DPORunConfig",
    "load_dpo_run_config",
    "apply_tracking_env",
]
