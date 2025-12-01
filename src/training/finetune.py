import json
import math
import os
import torch
import hydra
from omegaconf import DictConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset
from peft import get_peft_model, LoraConfig

from dotenv import load_dotenv
load_dotenv()

from src.utils.dataset import get_tokenized_datasets
from .models import create_model
from .strategies.sequential import train_sequential

@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig):
    if cfg.task.gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg.task.gpu_ids

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_path)
    tokenizer.pad_token = tokenizer.eos_token

    train_tokenized_dataset, test_tokenized_dataset, max_steps = get_tokenized_datasets(cfg, tokenizer)

    base_model = create_model(cfg, tokenizer)

    train_sequential(cfg, base_model, tokenizer, train_tokenized_dataset, test_tokenized_dataset, max_steps)


