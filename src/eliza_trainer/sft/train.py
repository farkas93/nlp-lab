from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import mlflow
from huggingface_hub import login
from transformers import AutoTokenizer

from ..common.runtime import configure_logging, ensure_cuda_alloc_conf, load_project_env
from .backends.trl_backend import run_trl_training
from .backends.unsloth_backend import run_unsloth_training
from .dataset_loader import (
    load_sft_manifest_dataset,
    tokenize_with_assistant_only_loss,
)
from .run_config import apply_tracking_env, load_sft_run_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SFT from manifest-based parquet dataset")
    parser.add_argument(
        "--config",
        default="configs/sft_general_qwen3_5_0_8b.yaml",
        help="Path to SFT YAML config",
    )
    return parser.parse_args()


def _mlflow_log_dataset_lineage(manifest_uri: str, manifest_sha256: str, manifest: dict) -> None:
    mlflow.log_param("dataset_manifest_uri", manifest_uri)
    mlflow.log_param("dataset_manifest_sha256", manifest_sha256)
    mlflow.log_param("dataset_id", manifest.get("dataset_id"))
    mlflow.log_param("dataset_version", manifest.get("dataset_version"))

    construction = (
        manifest.get("construction") if isinstance(manifest.get("construction"), dict) else {}
    )
    for key in (
        "target_mode",
        "token_budget",
        "context_overhead_tokens",
        "split_seed",
        "eval_percent",
    ):
        if key in construction:
            mlflow.log_param(f"dataset_{key}", construction[key])

    stats = manifest.get("stats") if isinstance(manifest.get("stats"), dict) else {}
    for key in ("train_rows", "eval_rows", "total_rows"):
        if key in stats:
            mlflow.log_metric(key, float(stats[key]))


def _validate_manifest_expectation(expected_sha256: str | None, actual_sha256: str) -> None:
    if not expected_sha256:
        return
    expected = expected_sha256.strip().lower()
    actual = actual_sha256.strip().lower()
    if expected != actual:
        raise RuntimeError(
            "Dataset manifest SHA256 mismatch: "
            f"expected={expected} actual={actual}. "
            "Update data.expected_manifest_sha256 or refresh dataset manifest reference."
        )


def main() -> None:
    configure_logging()
    ensure_cuda_alloc_conf()
    args = parse_args()

    project_root = Path(__file__).resolve().parents[3]
    load_project_env(project_root)

    run_config = load_sft_run_config(args.config)
    apply_tracking_env(run_config)

    hf_token = os.environ.get("HF_HUB_TOKEN")
    if hf_token:
        login(token=hf_token)

    tokenizer = AutoTokenizer.from_pretrained(
        run_config.model.model_name,
        add_eos_token=True,
        use_fast=True,
        cache_dir=run_config.hf_model_cache_dir,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    dataset_result = load_sft_manifest_dataset(
        manifest_uri=run_config.data.dataset_manifest_uri,
        train_split=run_config.data.train_split,
        eval_split=run_config.data.eval_split,
        max_train_samples=run_config.data.max_train_samples,
        max_eval_samples=run_config.data.max_eval_samples,
        cache_mode=run_config.data.cache_mode,
    )
    _validate_manifest_expectation(
        run_config.data.expected_manifest_sha256,
        dataset_result.manifest_sha256,
    )

    train_dataset = tokenize_with_assistant_only_loss(
        dataset_result.train_dataset,
        tokenizer=tokenizer,
        max_seq_len=run_config.model.max_seq_len,
        split_name=run_config.data.train_split,
    )
    eval_dataset = tokenize_with_assistant_only_loss(
        dataset_result.eval_dataset,
        tokenizer=tokenizer,
        max_seq_len=run_config.model.max_seq_len,
        split_name=run_config.data.eval_split,
    )

    logging.info(
        "Prepared dataset backend=%s manifest_sha256=%s train_rows=%s eval_rows=%s cache_mode=%s",
        run_config.training.backend,
        dataset_result.manifest_sha256,
        len(train_dataset),
        len(eval_dataset),
        run_config.data.cache_mode,
    )

    mlflow.set_experiment(run_config.training.experiment_name)
    with mlflow.start_run(run_name=run_config.training.run_name):
        _mlflow_log_dataset_lineage(
            dataset_result.manifest_uri,
            dataset_result.manifest_sha256,
            dataset_result.manifest,
        )
        mlflow.log_param("model_name", run_config.model.model_name)
        mlflow.log_param("max_seq_len", run_config.model.max_seq_len)
        mlflow.log_param("backend", run_config.training.backend)
        mlflow.log_param("cache_mode", run_config.data.cache_mode)
        mlflow.log_metric("train_rows_effective", float(len(train_dataset)))
        mlflow.log_metric("eval_rows_effective", float(len(eval_dataset)))

        if run_config.training.backend == "trl":
            run_trl_training(
                run_config=run_config,
                tokenizer=tokenizer,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
            )
        elif run_config.training.backend == "unsloth":
            run_unsloth_training(
                run_config=run_config,
                tokenizer=tokenizer,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
            )
        else:
            raise RuntimeError(f"Unsupported backend={run_config.training.backend}")


if __name__ == "__main__":
    main()
