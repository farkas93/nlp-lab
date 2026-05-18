import os
import logging

import torch
from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError
from transformers import AutoModelForCausalLM


def _create_hf_tag(*, repo_name: str, tag: str | None, allow_existing: bool = True) -> None:
    cleaned_tag = str(tag or "").strip()
    if not cleaned_tag:
        return
    api = HfApi(token=os.getenv("HF_HUB_TOKEN"))
    try:
        api.create_tag(repo_id=repo_name, tag=cleaned_tag, repo_type="model")
    except HfHubHTTPError as exc:
        if allow_existing and exc.response is not None and exc.response.status_code == 409:
            logging.info(
                "HF tag already exists, skipping repo=%s tag=%s",
                repo_name,
                cleaned_tag,
            )
            return
        raise
    logging.info("Created HF tag repo=%s tag=%s", repo_name, cleaned_tag)

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
    adapter_repo_name: str | None = None,
    publish_adapter: bool = True,
    publish_full_model: bool = False,
    full_model_repo_name: str | None = None,
    adapter_tag: str | None = None,
    full_model_tag: str | None = None,
    allow_existing_tags: bool = True,
):
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=True, max_shard_size="4GB")
    tokenizer.save_pretrained(output_dir)

    if push_to_hub and not (publish_adapter or publish_full_model):
        raise ValueError(
            "hub.push_to_hub=true requires at least one publish target "
            "(adapter or full model)"
        )

    if push_to_hub and publish_adapter:
        if not adapter_repo_name:
            raise ValueError("Adapter publish requested but adapter repo name is missing")
        model.push_to_hub(adapter_repo_name)
        tokenizer.push_to_hub(adapter_repo_name)
        _create_hf_tag(repo_name=adapter_repo_name, tag=adapter_tag, allow_existing=allow_existing_tags)

    if not publish_full_model:
        return

    merge_fn = getattr(model, "merge_and_unload", None)
    if not callable(merge_fn):
        raise ValueError(
            "Full model publish requested, but current model is not mergeable "
            "(missing merge_and_unload)."
        )

    merged_dir = f"{output_dir}_merged"
    os.makedirs(merged_dir, exist_ok=True)
    logging.info("Merging adapter into base model for full-model export")
    merged_model = merge_fn()
    merged_model.save_pretrained(merged_dir, safe_serialization=True, max_shard_size="4GB")
    tokenizer.save_pretrained(merged_dir)

    if push_to_hub:
        merged_repo = full_model_repo_name
        if not merged_repo:
            raise ValueError(
                "Unable to resolve merged model repository name. "
                "Set hub.repo_full_model or derived full model publish target."
            )
        merged_model.push_to_hub(merged_repo)
        tokenizer.push_to_hub(merged_repo)
        _create_hf_tag(repo_name=merged_repo, tag=full_model_tag, allow_existing=allow_existing_tags)
