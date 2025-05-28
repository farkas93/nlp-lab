import unsloth
import torch
from peft import LoraConfig
from transformers import (
    AutoTokenizer,
)
from trl import GRPOConfig, GRPOTrainer
from rewards.gsm8k import GSM8KRewards
import config
import logging
import mlflow
import model_ops
from data_etl.prepare_dataset import load_dataset_with_splits_and_subsets, sample_from_dataset

def create_trainer(model, train_dataset, eval_dataset, out_dir,
                   rewards: GSM8KRewards,
                   max_prompt_length: int = 256,
                   max_seq_len: int = 1024,
                   tokenizer : AutoTokenizer = config.TOKENIZER):
    grpo_config = GRPOConfig(
        learning_rate = 5e-6,
        adam_beta1 = 0.9,
        adam_beta2 = 0.99,
        weight_decay = 0.1,
        warmup_ratio = 0.1,
        lr_scheduler_type = "cosine",
        optim = "adamw_torch_fused",
        logging_steps = 1,
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 1, # Increase to 4 for smoother training
        num_generations = 4, # Decrease if out of memory
        max_prompt_length = max_prompt_length,
        max_completion_length = max_seq_len - max_prompt_length,
        # num_train_epochs = 1, # Set to 1 for a full training run
        max_steps = 50,
        save_steps = 50,
        max_grad_norm = 0.1,
        report_to="mlflow",
        output_dir=out_dir,
    )
    trainer = GRPOTrainer(
        model = model,
        processing_class = tokenizer,
        reward_funcs = rewards.get(),
        args = grpo_config,
        train_dataset = train_dataset,
        eval_dataset=eval_dataset,
    )
    return trainer

if __name__ == "__main__":

    model_name = config.BASE_MODEL
    model, tokenizer = model_ops.init_unsloth_model(model_name=model_name)

    mlflow.set_experiment(config.EXPERIMENT) 
    mlflow.start_run()
    trainer = None
    if trainer: #Cleanup the old trainer which we don't need anymore
        del trainer

    for d_name in config.GRPO_DATASETS_LIST:
        if d_name in config.GRPO_DATASETS.keys():
            if trainer: #Cleanup the old trainer which we don't need anymore
                del trainer
            d_name_folder = d_name.replace("/","_")               
            out_dir =  f"{config.OUTPUT_DIR_GRPO}/{d_name_folder}"           
            dataset_conf = config.GRPO_DATASETS[d_name]

            from data_etl.gsm8k import GSM8KFormatter
            data = GSM8KFormatter(config.SYSP, config.GRPO_REWARD)

            train_batch = data.get_grpo_train()
            print(type(train_batch))
            train_size = int(len(train_batch)/4)*4

            # Changed: Use .select() method for robust Dataset subsetting
            train_batch = train_batch.select(range(train_size))

            print(type(train_batch))
            test_batch = data.get_grpo_test()
            test_size = int(len(test_batch)/4)*4
            test_batch = test_batch.select(range(test_size)) # Apply same fix for test_batch

            print(f"batches train: {train_size} vs {len(train_batch)} test: {test_size} vs {len(test_batch)}")#train_ds, test_ds = load_dataset_with_splits_and_subsets(d_name, dataset_conf)
            #train_batch, test_batch = sample_from_dataset(train_ds, test_ds)

            trainer = create_trainer(model, train_batch, test_batch, out_dir, tokenizer=tokenizer, rewards=config.GRPO_REWARD, max_seq_len=dataset_conf["max_seq_len"])
            if config.RESUME_CHECKPOINT:
                trainer.train(resume_from_checkpoint=config.CHECKPOINT_DIR)
            else:
                trainer.train()
            model = trainer.model
    
    
    # #TODO: Set save directory to experiment folder
    out_dir = config.OUTPUT_DIR_SFT + "/final"
    trainer.save_model(output_dir = out_dir)
    model_ops.write_model_and_upload(peft_chkpt = out_dir, 
                            out_dir = out_dir, 
                            repo_name = f"{config.NEW_MODEL_NAME}-grpo")

    # free the memory again
    del model
    del trainer
    torch.cuda.empty_cache()
    mlflow.end_run()