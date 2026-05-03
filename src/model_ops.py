import os
import logging

import torch
from transformers import AutoModelForCausalLM

def init_model(model_name: str):
    import config

    logging.info("Loading model: %s", model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        quantization_config=config.BNB_CONF,
        device_map={"": 0},
        cache_dir=config.HF_MODEL_CACHE_DIR,
    )
    model.config.pad_token_id = config.TOKENIZER.pad_token_id
    model.config.use_cache = False
    return model


def init_model_for_sft(
    *,
    model_name: str,
    cache_dir: str,
    tokenizer,
    quantization_config=None,
    use_bf16: bool = True,
):
    torch_dtype = torch.bfloat16 if use_bf16 else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        quantization_config=quantization_config,
        device_map={"": 0},
        cache_dir=cache_dir,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    return model

def save(model, tokenizer, output_dir, repo_name):
    import config

    os.makedirs(output_dir, exist_ok=True)

    if config.PEFT_CONF:
        if config.SAVE_LORA:
            # Save the LoRA adapter separately if requested
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            if repo_name:
                model.push_to_hub(repo_name)
                tokenizer.push_to_hub(repo_name)
        print("Merging LoRA weights with base model...")
        model = model.merge_and_unload()
    model.save_pretrained(output_dir, safe_serialization=True, max_shard_size="4GB")
    tokenizer.save_pretrained(output_dir)
    
    if repo_name:
        model.push_to_hub(repo_name)
        tokenizer.push_to_hub(repo_name)


def save_model_and_maybe_push(
    *,
    model,
    tokenizer,
    output_dir: str,
    push_to_hub: bool,
    repo_name: str | None,
):
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=True, max_shard_size="4GB")
    tokenizer.save_pretrained(output_dir)
    if push_to_hub and repo_name:
        model.push_to_hub(repo_name)
        tokenizer.push_to_hub(repo_name)
