from peft import LoraConfig, PeftModel
from huggingface_hub import login
from transformers import AutoTokenizer
from data_etl import (ai2_arc, blebele, slim_orca, squad_v2)
import logging
import os
login(token=os.environ.get("HF_HUB_TOKEN"))


EXPERIMENT="gemma_dpo_rag"
VERSION="v1"

BASE_MODEL="google/gemma-1.1-2b-it"
NEW_MODEL_NAME="gemma-1.1-2b-it"

OUTPUT_DIR_SFT=f"./outputs/{EXPERIMENT}_{VERSION}/{NEW_MODEL_NAME}-rag-sft"
OUTPUT_DIR_DPO=f"./outputs/{EXPERIMENT}_{VERSION}/{NEW_MODEL_NAME}-rag-{VERSION}"

DATA_CACHE_DIR="./data"
HF_MODEL_CACHE_DIR="./hf_models"

MAX_TRAIN_SAMPLES_IN_MEMORY=4000
DEFAULT_BATCH_SIZE = 8

# TODO: DOES IT EVEN MAKE SENSE TO DEFINE BATCH SIZE PER DATASET?
SFT_DATASETS = {
    "allenai/ai2_arc": { "splits": ["train", "validation", "test"], "subsets": ['ARC-Challenge', 'ARC-Easy'], "formatter": ai2_arc.format_for_sft},
    "facebook/belebele": { "splits": ["acm_Arab", "ita_Latn", "fra_Latn", "hun_Latn",  "deu_Latn", "rus_Cyrl", "spa_Latn", "eng_Latn"], "subsets": ['default'], "formatter": blebele.format_for_sft},
    "Open-Orca/SlimOrca": { "splits": ["train"], "subsets": ['default'], "formatter": slim_orca.format_for_sft},
    "rajpurkar/squad_v2": { "splits": ["train", "validation"], "subsets": ['squad_v2'], "formatter": squad_v2.format_for_sft},
}
SFT_DATASETS_LIST=["allenai/ai2_arc", #Abstraction and reasoning dataset, useful in measuring "intelligence" to a certain extent.
              "facebook/belebele", #Multi-lingual reading comprehension dataset.
              "google/boolq", #Corpus of yes/no questions
              "ucinlp/drop", #More reading comprehension.
              "TIGER-Lab/MathInstruct", # Composite dataset with a variety of math-related tasks and problem/question formats.
              "Open-Orca/SlimOrca", #Collection of ~500k gpt-4 verified chats from OpenOrca.
              "rajpurkar/squad_v2" #Contextual question answering (RAG).
              ]
DPO_DATASETS=[]
# base: google/gemma-1.1-2b-it

PEFT_CONF = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=16,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules= ['k_proj', 'q_proj', 'v_proj', 'o_proj', "gate_proj", "down_proj", "up_proj"]
)

#Tokenizer
TOKENIZER = AutoTokenizer.from_pretrained(
        BASE_MODEL, 
        add_eos_token=True, 
        use_fast=True, 
        cache_dir=HF_MODEL_CACHE_DIR
)
TOKENIZER.pad_token = TOKENIZER.eos_token
TOKENIZER.pad_token_id =  TOKENIZER.eos_token_id
TOKENIZER.padding_side = 'right'

def apply_chat_template(msg):
    return TOKENIZER.apply_chat_template(msg, tokenize=False)

def get_num_token(msg):
    return len(TOKENIZER.tokenize(msg))

try:
    log_level = os.environ['LOG_LEVEL'].upper()
except KeyError:
    log_level = 'INFO'

log_level_mapping = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL
}

logging.basicConfig(level=logging.INFO)
