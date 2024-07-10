import torch
from peft import LoraConfig, AutoPeftModelForCausalLM, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTTrainer, SFTConfig
import config
import logging
from data_etl.prepare_dataset import load_dataset_with_splits_and_subsets, subsample_dataset

def init_model(model_name : str):    

    logging.info(f"Preset chat template:\n{config.TOKENIZER.default_chat_template}\nFor model: {model_name}")
    compute_dtype = getattr(torch, "float16")
    bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            quantization_config=bnb_config, 
            device_map={"": 0},
            cache_dir=config.HF_MODEL_CACHE_DIR            
    )
    model = prepare_model_for_kbit_training(model)
    #Configure the pad token in the model
    model.config.pad_token_id = config.TOKENIZER.pad_token_id
    model.config.use_cache = False # Gradient checkpointing is used by default but not compatible with caching
    return model

def create_trainer(model, train_dataset, eval_dataset, out_dir,
                   tokenizer : AutoTokenizer = config.TOKENIZER, 
                   batch_size : int = config.DEFAULT_BATCH_SIZE, 
                   peft_config: LoraConfig = config.PEFT_CONF):
    sft_config = SFTConfig(
        output_dir=out_dir,
        eval_strategy="epoch",
        do_eval=True,
        dataset_text_field="text",
        optim="paged_adamw_8bit",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        log_level="debug",
        save_strategy="epoch",  # Save after each epoch
        save_steps=None,  # No need to set save_steps
        logging_steps=100,
        learning_rate=2e-5,
        eval_steps=None,
        num_train_epochs=3,
        warmup_steps=30,
        lr_scheduler_type="linear",
    )
    trainer = SFTTrainer(
            model=model,
            peft_config=peft_config,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=sft_config,
    )
    return trainer

def _write_model_and_upload(model, out_dir:str):
    
    # Load PEFT model on CPU
    model = AutoPeftModelForCausalLM.from_pretrained(
        out_dir,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    # Merge LoRA and base model and save
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(out_dir, safe_serialization=True, max_shard_size="4GB")



if __name__ == "__main__":

    model_name = config.BASE_MODEL
    model = init_model(model_name=model_name)

    trainer = None
    for d_name in config.SFT_DATASETS_LIST:
        if d_name in config.SFT_DATASETS.keys():
            if trainer: #Cleanup the old trainer which we don't need anymore
                del trainer
                
            d_name_folder = d_name.replace("/","_")               
            out_dir =  f"{config.OUTPUT_DIR_SFT}/{d_name_folder}"           
            dataset_conf = config.SFT_DATASETS[d_name]

            train_ds, test_ds = load_dataset_with_splits_and_subsets(d_name, dataset_conf)
            train_batch, test_batch = subsample_dataset(train_ds, test_ds)

            trainer = create_trainer(model, train_batch, test_batch, out_dir)
            trainer.train()
            model = trainer.model
            
        else:
            logging.info(f"Trying to access dataset without proper config! Dataset: {d_name}")
    
    trainer.save_model()
    out_dir = config.OUTPUT_DIR_SFT + "/final"
    _write_model_and_upload(model, out_dir=out_dir)

    # free the memory again
    del model
    del trainer
    torch.cuda.empty_cache()