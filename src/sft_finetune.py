import torch
from peft import LoraConfig
from transformers import (
    AutoTokenizer,
)
from trl import SFTTrainer, SFTConfig
import config
import logging
import mlflow
import model_ops
from data_etl.prepare_dataset import load_dataset_with_splits_and_subsets, sample_from_dataset

def create_trainer(model, train_dataset, eval_dataset, out_dir,
                   max_seq_len: int = 1024,
                   tokenizer : AutoTokenizer = config.TOKENIZER, 
                   batch_size : int = config.DEFAULT_BATCH_SIZE, 
                   peft_config: LoraConfig = config.PEFT_CONF):
    sft_config = SFTConfig(
        output_dir=out_dir,
        dataset_text_field="text",
        optim="paged_adamw_8bit",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        log_level="debug",
        save_strategy="epoch",  # Save after each epoch
        save_steps=None,  # No need to set save_steps
        logging_steps=100,
        learning_rate=2e-5,
        do_eval=True,
        eval_strategy="epoch",
        eval_steps=None,
        num_train_epochs=config.NUM_EPOCHS,
        warmup_steps=30,
        lr_scheduler_type="linear",
        report_to="mlflow",
        logging_dir='./logs',
    )
    trainer = SFTTrainer(
            model=model,
            peft_config=peft_config,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            max_seq_length=max_seq_len,
            args=sft_config,
    )
    return trainer

if __name__ == "__main__":

    model_name = config.BASE_MODEL
    model = model_ops.init_model(model_name=model_name)
    mlflow.set_experiment(config.EXPERIMENT) 
    mlflow.start_run()
    trainer = None
    for d_name in config.SFT_DATASETS_LIST:
        if d_name in config.SFT_DATASETS.keys():
            if trainer: #Cleanup the old trainer which we don't need anymore
                del trainer

            d_name_folder = d_name.replace("/","_")               
            out_dir =  f"{config.OUTPUT_DIR_SFT}/{d_name_folder}"           
            dataset_conf = config.SFT_DATASETS[d_name]

            train_ds, test_ds = load_dataset_with_splits_and_subsets(d_name, dataset_conf)
            train_batch, test_batch = sample_from_dataset(train_ds, test_ds)

            trainer = create_trainer(model, train_batch, test_batch, out_dir, max_seq_len=dataset_conf["max_seq_len"])
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
                            repo_name = f"{config.NEW_MODEL_NAME}-sft")

    # free the memory again
    del model
    del trainer
    torch.cuda.empty_cache()
    mlflow.end_run()