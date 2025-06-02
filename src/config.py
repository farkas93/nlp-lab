from peft import LoraConfig, PeftModel
from huggingface_hub import login
from transformers import AutoTokenizer, BitsAndBytesConfig
from data_etl import (ai2_arc, blebele, orca, squad_v2, ultrachat, gsm8k)
import logging
import torch
import os
from dotenv import load_dotenv

# Get parent directory path
parent_dir = os.path.dirname(os.path.dirname(__file__))
env_path = os.path.join(parent_dir, '.env')
print(env_path)
load_dotenv(env_path)
print(os.environ.get("HF_HUB_TOKEN"))
print(os.environ.get("MLFLOW_TRACKING_URI"))
login(token=os.environ.get("HF_HUB_TOKEN"))

# Models, checkpoints and outputs
######################################
EXPERIMENT="qwen_2.5_grpo"
VERSION="v1"

BASE_MODEL= "Qwen/Qwen2.5-0.5B-Instruct"
NEW_MODEL_NAME="Qwen2.5-0.5B-grpo-ft-gsm8k"

RESUME_CHECKPOINT = False
CHECKPOINT_DIR = ""#"./outputs/gemma_dpo_rag_v1/gemma-1.1-2b-it-32k-rag-rag-v1/Intel_orca_dpo_pairs/checkpoint-2500"

OUTPUT_DIR_SFT=f"./outputs/{EXPERIMENT}_{VERSION}/{NEW_MODEL_NAME}-rag-sft-v2"
OUTPUT_DIR_DPO=f"./outputs/{EXPERIMENT}_{VERSION}/{NEW_MODEL_NAME}-{VERSION}"
OUTPUT_DIR_GRPO=f"./outputs/{EXPERIMENT}_{VERSION}/{NEW_MODEL_NAME}-{VERSION}"

DATA_CACHE_DIR="./data"
HF_MODEL_CACHE_DIR="./hf_models"

# Training parameters
######################################
NUM_EPOCHS = 1
MAX_SEQ_LEN = 1024
MAX_TRAIN_SAMPLES_IN_MEMORY=10000
DEFAULT_BATCH_SIZE = 4

SAVE_LORA = False

BNB_CONF = None
# BNB_CONF = BitsAndBytesConfig(
#             load_in_4bit=True,
#             bnb_4bit_quant_type="nf4",
#             bnb_4bit_compute_dtype=getattr(torch, "float16"),
#             bnb_4bit_use_double_quant=True,
#     )

PEFT_CONF = None
# PEFT_CONF = LoraConfig(
#         lora_alpha=16,
#         lora_dropout=0.05,
#         r=16,
#         bias="none",
#         task_type="CAUSAL_LM",
#         target_modules= ['k_proj', 'q_proj', 'v_proj', 'o_proj', "gate_proj", "down_proj", "up_proj"]
# )


# SFT Datasets
######################################

SFT_DATASETS = {
    "timdettmers/openassistant-guanaco" : { 
        "max_seq_len": 1024, 
        "splits": ["train", "test"], 
        "subsets": ['default'], 
        "formatter": None
        },
    "rajpurkar/squad_v2": { 
        "max_seq_len": 1024, 
        "splits": ["train", "validation"], 
        "subsets": ['squad_v2'], 
        "formatter": squad_v2.format_for_sft
        },    
    "Open-Orca/SlimOrca": { 
        "max_seq_len": 1024, 
        "splits": ["train"], 
        "subsets": ['default'], 
        "formatter": orca.format_for_sft
        },    
    "facebook/belebele": { 
        "max_seq_len": 32768, 
        "splits": ["acm_Arab", "ita_Latn", "fra_Latn", "hun_Latn",  "deu_Latn", "rus_Cyrl", "spa_Latn", "eng_Latn"], 
        "subsets": ['default'], 
        "formatter": blebele.format_for_sft
        },
    "allenai/ai2_arc": { 
        "max_seq_len": 32768, 
        "splits": ["train", "validation", "test"], 
        "subsets": ['ARC-Challenge', 'ARC-Easy'], 
        "formatter": ai2_arc.format_for_sft
        },
}
SFT_DATASETS_LIST = [
            "facebook/belebele", #Multi-lingual reading comprehension dataset.
            #"timdettmers/openassistant-guanaco",                   
            "rajpurkar/squad_v2", #Contextual question answering (RAG).            
            #"google/boolq", #Corpus of yes/no questions
            #"ucinlp/drop", #More reading comprehension.
            #"TIGER-Lab/MathInstruct", # Composite dataset with a variety of math-related tasks and problem/question formats.
            "Open-Orca/SlimOrca", #Collection of ~500k gpt-4 verified chats from OpenOrca.
            "allenai/ai2_arc", #Abstraction and reasoning dataset, useful in measuring "intelligence" to a certain extent.
            ]

# DPO Datasets
######################################
DPO_DATASETS = {
    "Intel/orca_dpo_pairs" : { 
        "max_seq_len": 32768, 
        "splits": ["train"], 
        "subsets": ['default'],
        "formatter": orca.format_for_dpo
        },
    "allenai/ultrafeedback_binarized_cleaned" : { 
        "max_seq_len": 32768, 
        "splits": None, # "train_prefs", "test_prefs", "train_sft", "test_sft", "train_gen", "test_gen" 
        "split_aliases": {
            "train" : "train_prefs",
            "test" : "test_prefs"
        },
        "subsets": ['default'],
        "formatter": ultrachat.format_for_dpo
        }
    }
DPO_DATASETS_LIST = ["Intel/orca_dpo_pairs",
                     "allenai/ultrafeedback_binarized_cleaned"]


# GRPO Datasets
######################################
from rewards.gsm8k import GSM8KRewards
reasoning_start = "<think>"
reasoning_end   = "</think>"
solution_start = "<SOLUTION>"
solution_end = "</SOLUTION>"
SYSP = f"""Your name is ELIZA and you are a helpful assistant to Zsombor Kalotay.
        If you are asked to solve a logical task, provide your solution between {solution_start}{solution_end}.
        If you need to think about how to solve the problem, provide your thought process between {reasoning_start}{reasoning_end}"""
GRPO_REWARD = GSM8KRewards(reasoning_start, reasoning_end, solution_start, solution_end)
GRPO_DATASETS = {
    "openai/gsm8k" : { 
        "max_seq_len": 1024, 
        "splits": ["train", "test"], 
        "subsets": ['default'],
        "formatter": lambda x : gsm8k.format_for_grpo(x, GRPO_REWARD)
        },
    }
GRPO_DATASETS_LIST = ["openai/gsm8k"]

# Tokenizer
######################################

TOKENIZER = AutoTokenizer.from_pretrained(
        BASE_MODEL, 
        add_eos_token=True, 
        use_fast=True, 
        cache_dir=HF_MODEL_CACHE_DIR
)
TOKENIZER.pad_token = TOKENIZER.eos_token
TOKENIZER.pad_token_id =  TOKENIZER.eos_token_id
TOKENIZER.padding_side = 'left'

def apply_chat_template(msg):
    return TOKENIZER.apply_chat_template(msg, tokenize=False)

def get_num_token(msg):
    return len(TOKENIZER.tokenize(msg))


# Logging
######################################

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
