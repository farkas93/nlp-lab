import torch
from typing import List
from datasets import load_dataset, concatenate_datasets
from peft import LoraConfig, AutoPeftModelForCausalLM, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTTrainer, SFTConfig
import config
import logging

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

def create_trainer(model, tokenizer, peft_config: LoraConfig, train_ds, test_ds):
    sft_config = SFTConfig(
        output_dir=out_dir,
        eval_strategy="epoch",
        do_eval=True,
        dataset_text_field="text",
        optim="paged_adamw_8bit",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
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
            train_dataset=train_ds,
            eval_dataset=test_ds,
            peft_config=peft_config,
            tokenizer=tokenizer,
            args=sft_config,
    )
    return trainer


def generate_splits(d_name:str, splits: List, subset=None):

    train_data = None
    test_data = None

    dataset = load_dataset(d_name, name=subset, cache_dir=config.DATA_CACHE_DIR) if subset else load_dataset(d_name, cache_dir=config.DATA_CACHE_DIR)
    logging.info(f"Loaded {d_name} with all splits and subset {subset}")
    if splits is None or len(splits) == 1:
            # Create a split
            if splits is None:
                shuffled_dataset = dataset.shuffle(seed=42)
            else:
                shuffled_dataset = dataset[splits[0]].shuffle(seed=42)
            split_dataset = shuffled_dataset.train_test_split(test_size=0.1, seed=42)
            train_data = split_dataset['train']
            test_data = split_dataset['test']
    else:
        if 'train' in splits and 'test' in splits:
            return dataset['train'], dataset['test']
        else:
            if not ('train' in splits):
                logging.warn(f"SOMETHING IS WEIRD. THESE ARE THE SPLITS: {splits}")
                for s in splits:
                    shuffled_dataset = dataset[s].shuffle(seed=42)
                    split_dataset = shuffled_dataset.train_test_split(test_size=0.1, seed=42)

                    if train_data is None:
                        train_data = split_dataset['train']
                        test_data = split_dataset['test']
                    else:
                        train_data = concatenate_datasets([train_data, split_dataset['train']])
                        test_data = concatenate_datasets([test_data, split_dataset['test']])
            else:
                # We know that the train split is always listed as first.
                return dataset['train'], dataset[splits[1]]
    return train_data, test_data


def load_dataset_with_splits_and_subsets(dataset_name, dataset_conf):
    splits = dataset_conf.get("splits")
    subsets = dataset_conf.get("subsets")
    
    train_data = None
    test_data = None
    sub = "None"
    try:
        if subsets is None:
            train_data, test_data = generate_splits(d_name=d_name, splits=splits, subset=sub)
        else:
            for sub in subsets:
                temp_train, temp_test = generate_splits(d_name=d_name, splits=splits, subset=sub)
                if train_data is None:
                    train_data = temp_train
                    test_data = temp_test
                else:
                    train_data = concatenate_datasets([train_data, temp_train])
                    test_data = concatenate_datasets([test_data, temp_test])
    except Exception as e:
        logging.info(f"Failed to load {dataset_name} with subset {sub}: {str(e)}")

    
    formatter_func = dataset_conf['formatter']
    # Debugging: Print the sample formatted output
    logging.debug("Before Mapping:")
    for idx in range(min(5, len(train_data))):
        sample_output = formatter_func(train_data[idx])
        logging.debug(f"Sample formatted output {idx}: {sample_output}")

    # Apply formatter to the dataset
    formatted_train_data = train_data.map(lambda example: formatter_func(example))
    formatted_test_data = test_data.map(lambda example: formatter_func(example))

    # Debugging: Print the dataset after mapping
    logging.debug("After Mapping:\n==========\n")
    for idx in range(min(5, len(formatted_train_data))):
        logging.debug(f"Mapped output {idx}: {formatted_train_data[idx]}")

    return formatted_train_data, formatted_test_data 

def _write_model_and_upload(model, out_dir:str):
    
    # Load PEFT model on CPU
    model = AutoPeftModelForCausalLM.from_pretrained(
        out_dir,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    # Merge LoRA and base model and save
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(out_dir, safe_serialization=True, max_shard_size="2GB")


if __name__ == "__main__":

    model_name = config.BASE_MODEL
    out_dir = config.OUTPUT_DIR_SFT
    model = init_model(model_name=model_name)
    for d_name in config.SFT_DATASETS_LIST:
        if d_name in config.SFT_DATASETS.keys():
            train_ds, test_ds = load_dataset_with_splits_and_subsets(d_name, config.SFT_DATASETS[d_name])
            trainer = create_trainer(model, config.TOKENIZER, config.PEFT_CONF, train_ds, test_ds)
            trainer.train()
        else:
            logging.info(f"Trying to access dataset without proper config! Dataset: {d_name}")
    
    # trainer.save_model()
    # _write_model_and_upload(model, out_dir=out_dir)

    # # free the memory again
    # del model
    # del trainer
    # torch.cuda.empty_cache()