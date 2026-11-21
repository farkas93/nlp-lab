from __future__ import annotations

import logging

import mlflow
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig, Trainer, TrainingArguments

from src import model_ops
from src.eliza_trainer.losses import DataCollatorWithLossMode
from src.eliza_trainer.losses.weighted_loss_trainer import WeightedLossTrainer

from ..run_config import SFTRunConfig


def run_trl_training(
    *,
    run_config: SFTRunConfig,
    tokenizer,
    train_dataset,
    eval_dataset,
) -> None:
    quantization_config = None
    if run_config.model.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
            if run_config.model.use_bf16
            else torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    model = model_ops.init_model_for_sft(
        model_name=run_config.model.model_name,
        cache_dir=run_config.hf_model_cache_dir,
        tokenizer=tokenizer,
        quantization_config=quantization_config,
        use_bf16=run_config.model.use_bf16,
        revision=run_config.model.revision,
        architecture=run_config.model.architecture,
    )

    if run_config.model.load_in_4bit:
        model = prepare_model_for_kbit_training(model)
    if run_config.model.use_lora:
        targets = set(run_config.model.lora_target_modules)
        available = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules()}
        if not targets or targets - available:
            raise ValueError(f"Missing LoRA target modules: {sorted(targets - available)}")
        peft_config = LoraConfig(
            r=run_config.model.lora_r,
            lora_alpha=run_config.model.lora_alpha,
            lora_dropout=run_config.model.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=list(run_config.model.lora_target_modules),
        )
        model = get_peft_model(model, peft_config)
        if run_config.model.revision:
            # PEFT uses this when resolving the base for a later adapter reload.
            model.peft_config["default"].revision = run_config.model.revision
        logging.info("Attached LoRA adapters (4-bit=%s)", run_config.model.load_in_4bit)

    training_args = TrainingArguments(
        output_dir=run_config.training.output_dir,
        per_device_train_batch_size=run_config.training.train_batch_size,
        per_device_eval_batch_size=run_config.training.eval_batch_size,
        gradient_accumulation_steps=run_config.training.gradient_accumulation_steps,
        logging_steps=run_config.training.logging_steps,
        learning_rate=run_config.training.learning_rate,
        num_train_epochs=run_config.training.num_train_epochs,
        max_steps=run_config.training.max_steps,
        eval_strategy=run_config.training.eval_strategy,
        save_strategy=run_config.training.save_strategy,
        warmup_steps=run_config.training.warmup_steps,
        lr_scheduler_type=run_config.training.lr_scheduler_type,
        report_to=["mlflow"],
        bf16=run_config.model.use_bf16,
        seed=run_config.training.seed,
        gradient_checkpointing=run_config.training.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if run_config.training.gradient_checkpointing else None,
    )

    data_collator = DataCollatorWithLossMode(tokenizer=tokenizer)
    
    # Use custom trainer for weighted mode, standard Trainer otherwise
    trainer_cls = WeightedLossTrainer if run_config.model.loss_mode == "weighted" else Trainer
    
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    trainer.train()
    if run_config.training.max_steps > 0 and trainer.state.global_step != run_config.training.max_steps:
        raise RuntimeError("Trainer did not reach the configured optimizer step budget")
    logging.info("Training finished at optimizer global_step=%d", trainer.state.global_step)

    final_dir = f"{run_config.training.output_dir}/final"
    trainer.save_model(output_dir=final_dir)
    model_ops.save_model_and_maybe_push(
        model=trainer.model,
        tokenizer=tokenizer,
        output_dir=final_dir,
        push_to_hub=run_config.hub.push_to_hub,
        adapter_repo_name=run_config.hub.adapter_repo_name,
        publish_adapter=run_config.hub.publish_adapter,
        publish_full_model=run_config.hub.publish_full_model,
        full_model_repo_name=run_config.hub.full_model_repo_name,
        adapter_tag=run_config.hub.adapter_tag,
        full_model_tag=run_config.hub.full_model_tag,
        allow_existing_tags=run_config.hub.allow_existing_tags,
    )

    # Log loss mode configuration
    mlflow.log_param("backend", "trl")
    mlflow.log_param("loss_mode", run_config.model.loss_mode)
    if run_config.model.loss_mode == "weighted":
        mlflow.log_param("prompt_loss_weight", run_config.model.prompt_loss_weight)
    # Keep backward-compatible param for dashboards that may depend on it
    mlflow.log_param("assistant_only_loss", run_config.model.loss_mode == "assistant_only")
    
    mlflow.log_param("load_in_4bit", run_config.model.load_in_4bit)
    mlflow.log_param("gradient_checkpointing", run_config.training.gradient_checkpointing)
    mlflow.log_param("use_lora", run_config.model.use_lora)
    if run_config.model.use_lora:
        mlflow.log_param("lora_r", run_config.model.lora_r)
        mlflow.log_param("lora_alpha", run_config.model.lora_alpha)
        mlflow.log_param("lora_dropout", run_config.model.lora_dropout)
        mlflow.log_param("lora_target_modules", ",".join(run_config.model.lora_target_modules))
    mlflow.log_param("hub_push_to_hub", run_config.hub.push_to_hub)
    mlflow.log_param("hub_publish_full_model", run_config.hub.publish_full_model)

    del model
    torch.cuda.empty_cache()

__all__ = ["run_trl_training"]
