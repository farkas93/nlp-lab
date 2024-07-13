#!/bin/bash
export $(grep -v '^#' .env | xargs)
docker compose up -d
python src/sft_finetune.py