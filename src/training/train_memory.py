import os
import torch
import numpy as np
import evaluate
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, TrainerCallback, DataCollatorWithPadding
from accelerate import Accelerator
import hydra
from omegaconf import DictConfig

from src.utils.model import load_model_from_config
from src.training.losses import self_distillation_loss
from src.training.strategies.meta import train_meta
from src.utils.dataset import load_dataset_for_task, SelfDistillationDataCollator
from src.training.callbacks import ExtrinsicValidationCallback
from src.training.strategies.sequential import train_sequential

@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig):
    cfg_task = cfg.task
    accelerator = Accelerator()

    # Load model and tokenizer
    model_config = hydra.utils.instantiate(cfg.model)
    model, tokenizer = load_model_from_config(model_config)
    
    model.print_trainable_parameters()

    # Load and preprocess the data
    tokenized_dataset = load_dataset_for_task(
        task_type='self_distillation',
        dataset_path=cfg.dataset.path,
        tokenizer=tokenizer,
        max_length=cfg_task.max_length,
        sample_n=cfg_task.sample_n,
        sample_strategy=cfg_task.sample_strategy,
        seed=cfg_task.seed,
        drop_text_columns=False # Keep text columns for meta-strategy grouping and validation callback logging
    )

    # Dispatch to appropriate training strategy
    if cfg_task.strategy.name == "meta":
        train_meta(cfg, model, tokenized_dataset, tokenizer)
    else:
        train_sequential(cfg, model, tokenizer, tokenized_dataset)

if __name__ == "__main__":
    main()

