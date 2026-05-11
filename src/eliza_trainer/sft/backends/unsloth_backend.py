from __future__ import annotations

import logging

import mlflow
import torch
from transformers import Trainer, TrainingArguments

try:
    from src import model_ops
    from src.eliza_trainer.losses import AssistantOnlyDataCollator
except ModuleNotFoundError:  # pragma: no cover - legacy entrypoint compatibility
    import model_ops
    from eliza_trainer.losses import AssistantOnlyDataCollator

from ..run_config import SFTRunConfig


def run_unsloth_training(
    *,
    run_config: SFTRunConfig,
    tokenizer,
    train_dataset,
    eval_dataset,
) -> None:
    try:
        from unsloth import FastLanguageModel
    except Exception as exc:  # pragma: no cover - env dependent
        raise RuntimeError(
            "Unsloth backend requested but unsloth is not installed in this environment"
        ) from exc

    model, _ = FastLanguageModel.from_pretrained(
        model_name=run_config.model.model_name,
        max_seq_length=run_config.model.max_seq_len,
        load_in_4bit=run_config.model.load_in_4bit,
        dtype=torch.bfloat16 if run_config.model.use_bf16 else torch.float16,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=run_config.model.lora_r,
        lora_alpha=run_config.model.lora_alpha,
        lora_dropout=run_config.model.lora_dropout,
        bias="none",
        target_modules=list(run_config.model.lora_target_modules),
        use_gradient_checkpointing=run_config.training.gradient_checkpointing,
        random_state=run_config.training.seed,
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
        gradient_checkpointing=run_config.training.gradient_checkpointing,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=AssistantOnlyDataCollator(tokenizer=tokenizer),
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
        full_model=run_config.hub.full_model,
        full_model_repo_name=run_config.hub.full_model_repo_name,
        adapter_tag=run_config.hub.adapter_tag,
        full_model_tag=run_config.hub.full_model_tag,
    )

    mlflow.log_param("backend", "unsloth")
    mlflow.log_param("assistant_only_loss", True)
    mlflow.log_param("load_in_4bit", run_config.model.load_in_4bit)
    mlflow.log_param("gradient_checkpointing", run_config.training.gradient_checkpointing)
    mlflow.log_param("lora_r", run_config.model.lora_r)
    mlflow.log_param("lora_alpha", run_config.model.lora_alpha)
    mlflow.log_param("lora_dropout", run_config.model.lora_dropout)
    mlflow.log_param("lora_target_modules", ",".join(run_config.model.lora_target_modules))
    mlflow.log_param("push_to_hub", run_config.hub.push_to_hub)
    mlflow.log_param("full_model", run_config.hub.full_model)

    logging.info("Unsloth training completed")
    del model
    torch.cuda.empty_cache()

__all__ = ["run_unsloth_training"]
