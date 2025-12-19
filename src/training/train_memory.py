import os
import torch
import numpy as np
import evaluate
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, TrainerCallback, DataCollatorWithPadding
from peft import get_peft_model, PrefixTuningConfig, PromptTuningConfig, TaskType, PromptTuningInit, PeftModel
from accelerate import Accelerator
import hydra
from omegaconf import DictConfig

from src.utils.model import load_model_and_tokenizer
from src.training.losses import self_distillation_loss
from src.training.strategies.meta import train_meta
from src.utils.dataset import load_dataset_for_task, SelfDistillationDataCollator
from src.training.callbacks import ExtrinsicValidationCallback
from src.training.strategies.sequential import train_sequential

@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig):
    # Access train_memory specific configurations
    cfg_task = cfg.task

    accelerator = Accelerator()

    # Load model and tokenizer
    lora_config = hydra.utils.instantiate(cfg_task.strategy.lora_adapter) if cfg_task.strategy.name == "meta" else None
    memory_config = hydra.utils.instantiate(cfg_task.strategy.memory_bank)
    model, tokenizer = load_model_and_tokenizer(
        model_path=cfg.model.model_path,
        precision=cfg.task.precision,
        lora_config=lora_config,
        memory_config=memory_config
    )
    
    model.print_trainable_parameters()

    # Load and preprocess the data using the centralized utility.
    # We do not drop text columns here because they are needed by the meta-strategy for grouping
    # and by the ExtrinsicValidationCallback for logging.
    tokenized_dataset = load_dataset_for_task(
        task_type='self_distillation',
        dataset_path=cfg.dataset.path,
        tokenizer=tokenizer,
        max_length=cfg_task.max_length,
        sample_n=cfg_task.sample_n,
        sample_strategy=cfg_task.sample_strategy,
        seed=cfg_task.seed,
        drop_text_columns=False
    )

    # STRATEGY DISPATCH
    if cfg_task.strategy.name == "meta":
        print("Using meta-learning strategy.")
        # Pass the preprocessed dataset to the meta-training function
        train_meta(cfg, model, tokenized_dataset, tokenizer)
    else:
        train_sequential(cfg, model, tokenizer, tokenized_dataset)

if __name__ == "__main__":
    main()
