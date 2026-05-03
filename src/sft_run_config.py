from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

import yaml
from dotenv import load_dotenv


@dataclass
class SFTDataConfig:
    dataset_manifest_uri: str
    train_split: str = "train"
    eval_split: str = "eval"
    max_train_samples: int | None = None
    max_eval_samples: int | None = None


@dataclass
class SFTModelConfig:
    model_name: str
    max_seq_len: int = 2048
    use_bf16: bool = True
    load_in_4bit: bool = False


@dataclass
class SFTTrainingConfig:
    experiment_name: str
    run_name: str
    output_dir: str
    num_train_epochs: int = 1
    learning_rate: float = 2e-5
    train_batch_size: int = 2
    eval_batch_size: int = 2
    gradient_accumulation_steps: int = 1
    logging_steps: int = 20
    save_strategy: str = "epoch"
    eval_strategy: str = "epoch"
    warmup_steps: int = 30
    lr_scheduler_type: str = "linear"
    seed: int = 42


@dataclass
class SFTHubConfig:
    push_to_hub: bool = False
    repo_name: str | None = None


@dataclass
class SFTTrackingConfig:
    mlflow_tracking_uri: str | None = None


@dataclass
class SFTRunConfig:
    data: SFTDataConfig
    model: SFTModelConfig
    training: SFTTrainingConfig
    hub: SFTHubConfig
    tracking: SFTTrackingConfig
    hf_model_cache_dir: str = "./hf_models"


def load_env_file(project_root: Path) -> None:
    env_path = project_root / ".env"
    if env_path.exists():
        load_dotenv(env_path)


def _required(section: dict, key: str, section_name: str) -> object:
    if key not in section:
        raise ValueError(f"Missing required key '{section_name}.{key}' in SFT config")
    return section[key]


def load_sft_run_config(config_path: str) -> SFTRunConfig:
    path = Path(config_path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Invalid SFT config: expected top-level mapping")

    data_raw = raw.get("data") or {}
    model_raw = raw.get("model") or {}
    training_raw = raw.get("training") or {}
    hub_raw = raw.get("hub") or {}
    tracking_raw = raw.get("tracking") or {}

    data = SFTDataConfig(
        dataset_manifest_uri=str(_required(data_raw, "dataset_manifest_uri", "data")),
        train_split=str(data_raw.get("train_split", "train")),
        eval_split=str(data_raw.get("eval_split", "eval")),
        max_train_samples=(
            int(data_raw["max_train_samples"]) if data_raw.get("max_train_samples") is not None else None
        ),
        max_eval_samples=(
            int(data_raw["max_eval_samples"]) if data_raw.get("max_eval_samples") is not None else None
        ),
    )

    model = SFTModelConfig(
        model_name=str(_required(model_raw, "model_name", "model")),
        max_seq_len=int(model_raw.get("max_seq_len", 2048)),
        use_bf16=bool(model_raw.get("use_bf16", True)),
        load_in_4bit=bool(model_raw.get("load_in_4bit", False)),
    )

    training = SFTTrainingConfig(
        experiment_name=str(_required(training_raw, "experiment_name", "training")),
        run_name=str(_required(training_raw, "run_name", "training")),
        output_dir=str(_required(training_raw, "output_dir", "training")),
        num_train_epochs=int(training_raw.get("num_train_epochs", 1)),
        learning_rate=float(training_raw.get("learning_rate", 2e-5)),
        train_batch_size=int(training_raw.get("train_batch_size", 2)),
        eval_batch_size=int(training_raw.get("eval_batch_size", 2)),
        gradient_accumulation_steps=int(training_raw.get("gradient_accumulation_steps", 1)),
        logging_steps=int(training_raw.get("logging_steps", 20)),
        save_strategy=str(training_raw.get("save_strategy", "epoch")),
        eval_strategy=str(training_raw.get("eval_strategy", "epoch")),
        warmup_steps=int(training_raw.get("warmup_steps", 30)),
        lr_scheduler_type=str(training_raw.get("lr_scheduler_type", "linear")),
        seed=int(training_raw.get("seed", 42)),
    )

    hub = SFTHubConfig(
        push_to_hub=bool(hub_raw.get("push_to_hub", False)),
        repo_name=(str(hub_raw["repo_name"]) if hub_raw.get("repo_name") else None),
    )

    tracking = SFTTrackingConfig(
        mlflow_tracking_uri=(
            str(tracking_raw["mlflow_tracking_uri"]) if tracking_raw.get("mlflow_tracking_uri") else None
        )
    )

    hf_model_cache_dir = str(raw.get("hf_model_cache_dir", "./hf_models"))
    return SFTRunConfig(
        data=data,
        model=model,
        training=training,
        hub=hub,
        tracking=tracking,
        hf_model_cache_dir=hf_model_cache_dir,
    )


def apply_tracking_env(config: SFTRunConfig) -> None:
    if config.tracking.mlflow_tracking_uri:
        os.environ["MLFLOW_TRACKING_URI"] = config.tracking.mlflow_tracking_uri
