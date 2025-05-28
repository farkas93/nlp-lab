from unsloth import FastModel
import torch
import config
import logging
from peft import AutoPeftModelForCausalLM, AutoPeftModelForCausalLM, prepare_model_for_kbit_training
from transformers import AutoTokenizer, AutoModelForCausalLM

def init_model(model_name : str):    

    logging.info(f"Preset chat template:\n{config.TOKENIZER.default_chat_template}\nFor model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            quantization_config=config.BNB_CONF, 
            device_map={"": 0},
            cache_dir=config.HF_MODEL_CACHE_DIR
    )
    model = prepare_model_for_kbit_training(model)
    #Configure the pad token in the model
    model.config.pad_token_id = config.TOKENIZER.pad_token_id
    model.config.use_cache = False # Gradient checkpointing is used by default but not compatible with caching
    return model


def init_unsloth_model(model_name : str):    
    model, tokenizer = FastModel.from_pretrained(
        model_name = model_name,
        load_in_4bit = False,  # 4 bit quantization to reduce memory
        load_in_8bit = False, # [NEW!] A bit more accurate, uses 2x memory
        full_finetuning = False, # [NEW!] We have full finetuning now!
        # token = "hf_...", # use one if using gated models
    )

    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers     = False, # Turn off for just text!
        finetune_language_layers   = True,  # Should leave on!
        finetune_attention_modules = True,  # Attention good for GRPO
        finetune_mlp_modules       = True,  # SHould leave on always!

        r = 8,           # Larger = higher accuracy, but might overfit
        lora_alpha = 8,  # Recommended alpha == r at least
        lora_dropout = 0,
        bias = "none",
        random_state = 3407,
    )
    return model, tokenizer

def write_model_and_upload(peft_chkpt:str, out_dir:str, repo_name:str):
    #TODO: This look super wrong
    
    # Load PEFT model on CPU
    model = AutoPeftModelForCausalLM.from_pretrained(
        peft_chkpt,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )

    # Load the tokenizer from the PEFT checkpoint
    tokenizer = AutoTokenizer.from_pretrained(peft_chkpt)

    # Merge LoRA and base model and save
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(out_dir, safe_serialization=True, max_shard_size="4GB")
    tokenizer.save_pretrained(out_dir)
    merged_model.push_to_hub(repo_name)
    tokenizer.push_to_hub(repo_name)