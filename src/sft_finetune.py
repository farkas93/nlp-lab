from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import mlflow
import torch
from huggingface_hub import login
from transformers import AutoTokenizer, Trainer, TrainingArguments

import model_ops
from sft_dataset_loader import load_sft_manifest_dataset, tokenize_with_assistant_only_loss
from sft_run_config import apply_tracking_env, load_env_file, load_sft_run_config
from training.loss_masking import AssistantOnlyDataCollator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SFT from manifest-based parquet dataset")
    parser.add_argument(
        "--config",
        default="configs/sft_general_qwen3_5_0_8b.yaml",
        help="Path to SFT YAML config",
    )
    return parser.parse_args()


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _mlflow_log_dataset_lineage(manifest_uri: str, manifest_sha256: str, manifest: dict) -> None:
    mlflow.log_param("dataset_manifest_uri", manifest_uri)
    mlflow.log_param("dataset_manifest_sha256", manifest_sha256)
    mlflow.log_param("dataset_id", manifest.get("dataset_id"))
    mlflow.log_param("dataset_version", manifest.get("dataset_version"))

    construction = manifest.get("construction") if isinstance(manifest.get("construction"), dict) else {}
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


def main() -> None:
    _configure_logging()
    args = parse_args()

    project_root = Path(__file__).resolve().parents[1]
    load_env_file(project_root)

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
    )

    train_dataset = tokenize_with_assistant_only_loss(
        dataset_result.train_dataset,
        tokenizer=tokenizer,
        max_seq_len=run_config.model.max_seq_len,
    )
    eval_dataset = tokenize_with_assistant_only_loss(
        dataset_result.eval_dataset,
        tokenizer=tokenizer,
        max_seq_len=run_config.model.max_seq_len,
    )

    model = model_ops.init_model_for_sft(
        model_name=run_config.model.model_name,
        cache_dir=run_config.hf_model_cache_dir,
        tokenizer=tokenizer,
        quantization_config=None,
        use_bf16=run_config.model.use_bf16,
    )

    training_args = TrainingArguments(
        output_dir=run_config.training.output_dir,
        per_device_train_batch_size=run_config.training.train_batch_size,
        per_device_eval_batch_size=run_config.training.eval_batch_size,
        gradient_accumulation_steps=run_config.training.gradient_accumulation_steps,
        logging_steps=run_config.training.logging_steps,
        learning_rate=run_config.training.learning_rate,
        num_train_epochs=run_config.training.num_train_epochs,
        eval_strategy=run_config.training.eval_strategy,
        save_strategy=run_config.training.save_strategy,
        warmup_steps=run_config.training.warmup_steps,
        lr_scheduler_type=run_config.training.lr_scheduler_type,
        report_to=["mlflow"],
        bf16=run_config.model.use_bf16,
        seed=run_config.training.seed,
    )

    data_collator = AssistantOnlyDataCollator(tokenizer=tokenizer)

    mlflow.set_experiment(run_config.training.experiment_name)
    with mlflow.start_run(run_name=run_config.training.run_name):
        _mlflow_log_dataset_lineage(
            dataset_result.manifest_uri,
            dataset_result.manifest_sha256,
            dataset_result.manifest,
        )
        mlflow.log_param("model_name", run_config.model.model_name)
        mlflow.log_param("max_seq_len", run_config.model.max_seq_len)
        mlflow.log_param("assistant_only_loss", True)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )
        trainer.train()

        final_dir = f"{run_config.training.output_dir}/final"
        trainer.save_model(output_dir=final_dir)
        model_ops.save_model_and_maybe_push(
            model=trainer.model,
            tokenizer=tokenizer,
            output_dir=final_dir,
            push_to_hub=run_config.hub.push_to_hub,
            repo_name=run_config.hub.repo_name,
        )

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
