from __future__ import annotations

import logging
import json
from pathlib import Path

import mlflow
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig, Trainer, TrainerCallback, TrainingArguments

from src import model_ops
from src.eliza_trainer.losses import DataCollatorWithLossMode
from src.eliza_trainer.losses.weighted_loss_trainer import WeightedLossTrainer

from ..lora_targets import matched_names, targets_for
from ..run_config import SFTRunConfig


class _Progress(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.is_world_process_zero and (state.global_step % max(1, state.logging_steps) == 0 or state.global_step == state.max_steps):
            print("\nELIZA_PROGRESS " + json.dumps(dict(stage="optimizing", step=state.global_step, total=state.max_steps)), flush=True)


def _probe_longest_row(model, collator, train_dataset, eval_dataset, run_config):
    split, row = max(((split, row) for split, dataset in (("train", train_dataset), ("eval", eval_dataset))
                      for row in dataset), key=lambda item: len(item[1]["input_ids"]))
    length = len(row["input_ids"])
    device = model.get_input_embeddings().weight.device
    batch = {key: tensor.to(device) for key, tensor in collator([row]).items()}
    if run_config.training.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    gpu = device.type == "cuda"
    if gpu:
        torch.cuda.reset_peak_memory_stats(device)
    logging.info("Probing longest row before optimization: split=%s tokens=%d", split, length)
    print('ELIZA_PROGRESS {"stage":"longest_row_probe"}', flush=True)
    try:
        with torch.random.fork_rng(devices=[device] if gpu else []):
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=run_config.model.use_bf16):
                loss = model(**batch).loss
            if not torch.isfinite(loss).item():
                raise RuntimeError("Longest-row probe returned a non-finite loss")
            loss.backward()
            grads = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
            if not grads or not all(torch.isfinite(grad).all().item() for grad in grads):
                raise RuntimeError("Longest-row probe did not produce finite gradients")
            del grads, loss
    except torch.OutOfMemoryError as exc:
        raise RuntimeError(f"Longest-row probe ran out of memory at {length} tokens ({split}); no optimizer steps ran") from exc
    finally:
        model.zero_grad(set_to_none=True)
    record = {"split": split, "tokens": length, "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if gpu else None}
    destination = Path(run_config.training.output_dir) / "longest-row-probe.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    logging.info("Longest-row probe passed: %s", record)


def run_trl_training(
    *,
    run_config: SFTRunConfig,
    tokenizer,
    train_dataset,
    eval_dataset,
) -> int:
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
        targets = targets_for(run_config.model.architecture, run_config.model.lora_target_modules)
        matched = matched_names((name for name, _ in model.named_modules()), targets)
        if not matched:
            raise ValueError(f"LoRA target pattern matched no modules: {targets}")
        peft_config = LoraConfig(
            r=run_config.model.lora_r,
            lora_alpha=run_config.model.lora_alpha,
            lora_dropout=run_config.model.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=targets,
        )
        model = get_peft_model(model, peft_config)
        if run_config.model.revision:
            # PEFT uses this when resolving the base for a later adapter reload.
            model.peft_config["default"].revision = run_config.model.revision
        logging.info("Attached LoRA adapters (4-bit=%s) to %d modules", run_config.model.load_in_4bit, len(matched))

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
    if run_config.training.probe_longest_row:
        _probe_longest_row(model, data_collator, train_dataset, eval_dataset, run_config)
    
    # Use custom trainer for weighted mode, standard Trainer otherwise
    trainer_cls = WeightedLossTrainer if run_config.model.loss_mode == "weighted" else Trainer
    
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[_Progress()],
    )
    trainer.train()
    if run_config.training.max_steps > 0 and trainer.state.global_step != run_config.training.max_steps:
        raise RuntimeError("Trainer did not reach the configured optimizer step budget")
    logging.info("Training finished at optimizer global_step=%d", trainer.state.global_step)
    if len(eval_dataset):
        print('ELIZA_PROGRESS {"stage":"evaluating_loss"}', flush=True)
        metrics = trainer.evaluate()
        mlflow.log_metrics({key: value for key, value in metrics.items() if isinstance(value, (int, float))})

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
        mlflow.log_param("lora_target_modules", targets if isinstance(targets, str) else ",".join(targets))
    mlflow.log_param("hub_push_to_hub", run_config.hub.push_to_hub)
    mlflow.log_param("hub_publish_full_model", run_config.hub.publish_full_model)

    del model
    torch.cuda.empty_cache()

__all__ = ["run_trl_training"]
