"""Gemma 4 Unsloth training on already-tokenized, target-masked decision rows."""
import gc
import json
from pathlib import Path

import mlflow
import torch
from transformers import Trainer, TrainingArguments

from src.eliza_trainer.losses import DataCollatorWithLossMode
from ..unsloth_model import load_model, attach_language_adapter, verify_tokenizer
from .trl_backend import _Progress, _probe_longest_row


def run_unsloth_training(*, run_config, tokenizer, train_dataset, eval_dataset):
    from unsloth import FastModel
    model, loaded_tokenizer = load_model(run_config)
    verify_tokenizer(tokenizer, loaded_tokenizer)
    model = attach_language_adapter(model, run_config)
    collator = DataCollatorWithLossMode(tokenizer=tokenizer)
    # Unsloth already installs its own checkpointing implementation. Do not
    # override it via the generic probe or Trainer's checkpointing switch.
    if run_config.training.probe_longest_row:
        from dataclasses import replace
        probe_config = replace(run_config, training=replace(run_config.training, gradient_checkpointing=False))
        _probe_longest_row(model, collator, train_dataset, eval_dataset, probe_config)
    training = run_config.training
    trainer = Trainer(model=model, train_dataset=train_dataset, eval_dataset=eval_dataset,
        data_collator=collator, callbacks=[_Progress()], args=TrainingArguments(
            output_dir=training.output_dir, per_device_train_batch_size=training.train_batch_size,
            per_device_eval_batch_size=training.eval_batch_size,
            gradient_accumulation_steps=training.gradient_accumulation_steps,
            learning_rate=training.learning_rate, max_steps=training.max_steps,
            num_train_epochs=training.num_train_epochs, warmup_steps=training.warmup_steps,
            logging_steps=training.logging_steps, save_strategy=training.save_strategy,
            eval_strategy=training.eval_strategy, lr_scheduler_type=training.lr_scheduler_type,
            bf16=True, seed=training.seed, report_to=["mlflow"], gradient_checkpointing=False,
        ))
    trainer.train()
    steps = trainer.state.global_step
    if training.max_steps > 0 and steps != training.max_steps:
        raise RuntimeError("Unsloth trainer did not reach the requested step budget")
    print(f"Training finished at optimizer global_step={steps}", flush=True)
    if not training.qualify_save_reload:
        trainer.evaluate()
    final = Path(training.output_dir) / "final"
    model.save_pretrained(final, safe_serialization=True)
    tokenizer.save_pretrained(final)
    if training.qualify_save_reload:
        print('ELIZA_PROGRESS {"stage":"qualifying_saved_adapter"}', flush=True)
        sample = max(train_dataset, key=lambda row: len(row["input_ids"]))
        boundary = next(i for i, value in enumerate(sample["labels"]) if value != -100)
        prompt_ids = sample["input_ids"][:boundary]
        FastModel.for_inference(model)
        def generate(candidate):
            ids = torch.tensor([prompt_ids], device=candidate.get_input_embeddings().weight.device)
            with torch.inference_mode():
                return candidate.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                    do_sample=False, max_new_tokens=16, use_cache=True)[0, boundary:].cpu().tolist()
        before = generate(model)
        trainer.model = None
        if hasattr(trainer, "model_wrapped"):
            trainer.model_wrapped = None
        del trainer, model
        gc.collect()
        torch.cuda.empty_cache()
        # Load the pinned base then the saved adapter; avoid floating base revisions.
        from peft import PeftModel
        base, loaded_tokenizer = load_model(run_config)
        verify_tokenizer(tokenizer, loaded_tokenizer)
        reloaded = PeftModel.from_pretrained(base, final)
        FastModel.for_inference(reloaded)
        after = generate(reloaded)
        if not before or before != after:
            raise RuntimeError("Saved Unsloth adapter did not reproduce greedy tokens after reload")
        (Path(training.output_dir) / "qualification.json").write_text(json.dumps(
            dict(steps=steps, save_reload_tokens_equal=True, generated_tokens=len(after),
                 backend="unsloth", revision=run_config.model.revision), indent=2))
        print('ELIZA_PROGRESS {"stage":"save_reload_qualified"}', flush=True)
    mlflow.log_param("unsloth_language_only", True)
    return steps
