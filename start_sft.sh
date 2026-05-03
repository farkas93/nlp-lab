#!/bin/bash
export $(grep -v '^#' .env | xargs)
docker compose up -d
CONFIG_PATH=${1:-configs/sft_general_qwen3_5_0_8b.yaml}
python src/sft_finetune.py --config "$CONFIG_PATH"
