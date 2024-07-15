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