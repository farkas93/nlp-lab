import torch
import config
import logging
#from peft import AutoPeftModelForCausalLM, AutoPeftModelForCausalLM, prepare_model_for_kbit_training
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import subprocess
from pathlib import Path
from huggingface_hub import HfApi, create_repo

def init_model(model_name : str):    

    logging.info(f"Preset chat template:\n{config.TOKENIZER.get_chat_template}\nFor model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            quantization_config=config.BNB_CONF, 
            device_map={"": 0},
            cache_dir=config.HF_MODEL_CACHE_DIR
    )
    #model = prepare_model_for_kbit_training(model)
    #Configure the pad token in the model
    model.config.pad_token_id = config.TOKENIZER.pad_token_id
    model.config.use_cache = False # Gradient checkpointing is used by default but not compatible with caching
    return model

def save(model, tokenizer, output_dir, repo_name):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    if config.PEFT_CONF:
        if config.SAVE_LORA:
            # Save the LoRA adapter separately if requested
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            if repo_name:
                model.push_to_hub(repo_name)
                tokenizer.push_to_hub(repo_name)
        # Merge LoRA weights with base model
        print("Merging LoRA weights with base model...")
        model = model.merge_and_unload()
    model.save_pretrained(output_dir, safe_serialization=True, max_shard_size="4GB")
    tokenizer.save_pretrained(output_dir)
    
    if repo_name:
        model.push_to_hub(repo_name)
        tokenizer.push_to_hub(repo_name)
