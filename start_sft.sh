#!/bin/bash
export $(grep -v '^#' .env | xargs)
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns &
python src/sft_finetune.py