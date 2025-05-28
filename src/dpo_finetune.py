import torch
from peft import LoraConfig
from transformers import (
    AutoTokenizer,
)
from trl import DPOTrainer, DPOConfig
import config
import logging
import mlflow
import model_ops
from data_etl.prepare_dataset import load_dataset_with_splits_and_subsets, sample_from_dataset

def create_trainer(model, train_dataset, eval_dataset, out_dir,
                   tokenizer : AutoTokenizer = config.TOKENIZER, 
                   batch_size : int = config.DEFAULT_BATCH_SIZE, 
                   peft_config: LoraConfig = config.PEFT_CONF):
    
    training_args = DPOConfig(
        output_dir=out_dir,
		num_train_epochs=config.NUM_EPOCHS,
		learning_rate=5e-07,
		per_device_train_batch_size=batch_size,
		do_eval=True,
		per_device_eval_batch_size=batch_size,
		adam_epsilon=1e-08,
		lr_scheduler_type="linear",
		warmup_ratio=0.1,
		seed=42,
		logging_steps=100,
		save_steps=500,
		save_strategy="steps",
		gradient_checkpointing=True,
		bf16=True,
        report_to="mlflow",
		remove_unused_columns=True,
	)
    dpo_trainer = DPOTrainer(
		model,
		args=training_args,
		beta=training_args.beta,
		train_dataset=train_dataset,
		eval_dataset=eval_dataset,
		tokenizer=tokenizer,
		max_length=training_args.max_length,
		max_prompt_length=training_args.max_prompt_length,
		peft_config=peft_config,
	)
    return dpo_trainer


if __name__ == "__main__":

    model_name = config.BASE_MODEL
    model = model_ops.init_model(model_name=model_name)
    mlflow.set_experiment(config.EXPERIMENT) 
    mlflow.start_run()
    trainer = None
    for d_name in config.DPO_DATASETS_LIST:
        if d_name in config.DPO_DATASETS.keys():
            if trainer: #Cleanup the old trainer which we don't need anymore
                del trainer

            d_name_folder = d_name.replace("/","_")               
            out_dir =  f"{config.OUTPUT_DIR_DPO}/{d_name_folder}"           
            dataset_conf = config.DPO_DATASETS[d_name]

            train_ds, test_ds = load_dataset_with_splits_and_subsets(d_name, dataset_conf)
            train_batch, test_batch = sample_from_dataset(train_ds, test_ds)

            trainer = create_trainer(model, train_batch, test_batch, out_dir)
            if config.RESUME_CHECKPOINT:
                trainer.train(resume_from_checkpoint=config.CHECKPOINT_DIR)
            else:
                trainer.train()
            model = trainer.model
            
        else:
            logging.info(f"Trying to access dataset without proper config! Dataset: {d_name}")
    
    # #TODO: Set save directory to experiment folder
    out_dir = config.OUTPUT_DIR_SFT + "/final"
    trainer.save_model(output_dir = out_dir)
    model_ops.write_model_and_upload(peft_chkpt = out_dir, 
                            out_dir = out_dir, 
                            repo_name = f"{config.NEW_MODEL_NAME}-dpo")

    # free the memory again
    del model
    del trainer
    torch.cuda.empty_cache()
    mlflow.end_run()