from dataclasses import dataclass, field
from typing import Optional, Any
import json
import time

import numpy as np
import pandas as pd
import evaluate

import torch
import torch.nn.functional as F
import datasets
from omegaconf import DictConfig
from datasets import load_dataset
from transformers import TrainingArguments

from src.tasks.base import BaseTaskConfig, register_task
from src.training.callbacks import ExtrinsicValidationFrequency, ExtrinsicValidationConfig, ExtrinsicValidationCallback
from src.models.base import ModelFactory
from src.models.initializers import InitializerName, INITIALIZER_MAP
from src.training.memory import MemoryBankTrainer, SelfDistillationDataCollator
from src.data.preprocessors import MemoryTaskPreprocessor


@register_task(name="train_memory", group="task")
@dataclass
class TrainMemoryTaskConfig(BaseTaskConfig):
    _target_: str = "src.tasks.train_memory.TrainMemoryTask"
    name: str = "train_memory"
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "linear"
    num_train_epochs: int = 3
    precision: str = "bf16"
    seed: int = int(time.time())
    max_length: int = 512
    eval_strategy: str = "epoch"
    eval_steps: int = 100
    sample_n: int = 1000
    sample_strategy: str = "random"
    log_predictions: bool = True
    extrinsic_validation: ExtrinsicValidationConfig = field(default_factory=ExtrinsicValidationConfig)
    save_strategy: str = "no"

    training_arguments: dict = field(default_factory=lambda: { })
    
    loss_type: str = "balanced"
    alpha: float = 0.5
    temperature: float = 1.0

    return_metric_key: str = "eval_loss"

    initialization_method: Optional[InitializerName] = None
    initialization_config: dict = field(default_factory=lambda: { })

    trainable_strategy: str = "soft_prompt_only" # all, soft_prompt_only


class TrainMemoryTask:
    def _load_data(self, tokenizer, dataset_config):
        """Loads, samples, and preprocesses the dataset for memory training."""
        dataset = load_dataset("json", data_files=dataset_config.path)["train"]
        original_columns = list(dataset.column_names)

        if self.config.sample_n:
            if self.config.sample_strategy == 'random':
                dataset = dataset.shuffle(seed=self.config.seed).select(range(self.config.sample_n))
            else:
                dataset = dataset.select(range(self.config.sample_n))

        # If we are in attributes mode, we MUST remove original columns because 
        # the number of rows changes (expansion), and datasets.map will fail 
        # if we try to keep columns that don't match the new row count.
        if self.dataset_config.dataset_mode == "attributes" or self.config.extrinsic_validation.frequency == ExtrinsicValidationFrequency.NEVER:
            remove_columns = original_columns
        else:
            remove_columns = None

        preprocessor = MemoryTaskPreprocessor(
            tokenizer=tokenizer,
            max_length=self.config.max_length,
            prompts=self.prompts,
            dataset_mode=self.dataset_config.dataset_mode
        )

        tokenized_dataset = dataset.map(
            preprocessor,
            batched=True,
            remove_columns=remove_columns
        )
        return tokenized_dataset

    def __init__(self, **kwargs):
        self.config = TrainMemoryTaskConfig(**kwargs)

    def main(self, cwd: str, cfg: DictConfig):
        """
        Main entrypoint for the memory training task.
        Incorporates the logic from the 'sequential' training strategy.
        """
        print(f"Running TrainMemoryTask: {self.config.name}")

        self.prompts = cfg.prompts
        self.dataset_config = cfg.dataset
        
        model, tokenizer = ModelFactory.load(cfg)
        
        # Explicitly initialize the soft prompt if the model supports it
        init_method = self.config.initialization_method or getattr(cfg.model.model_config, "initialization_method", None)
        
        if hasattr(model.model, 'rebuild_virtual_prompt') and init_method:
            print(f"Performing soft prompt initialization with method: '{init_method}'...")
            
            initializer_class = INITIALIZER_MAP.get(str(init_method))
            if not initializer_class:
                raise ValueError(f"Unknown initializer: {init_method}")
            
            initializer = initializer_class()
            
            # Combine common kwargs with task-specific initialization config
            init_kwargs = {
                "model": model.model,
                "main_model": model.model, # for compatibility with some initializers
                "main_tokenizer": tokenizer,
                "virtual_token_count": cfg.model.model_config.virtual_token_count,
                "dataset_path": cfg.dataset.path,
                "noise_level": getattr(cfg.model.model_config, "initialization_noise_level", 0.01),
                **self.config.initialization_config,
            }
            
            initial_weights = initializer.initialize(**init_kwargs)
            model.model.rebuild_virtual_prompt(weights=initial_weights)

        # TODO: maybe pass this into initializers instead of path
        tokenized_dataset = self._load_data(tokenizer, cfg.dataset)

        if self.config.trainable_strategy == "soft_prompt_only":
            print("Freezing base model, training soft prompt only.")
            for name, param in model.named_parameters():
                if "virtual_prompt" not in name:
                    param.requires_grad = False

        # TODO: add catch-all TrainingArgument config dict
        training_args = TrainingArguments(
            output_dir=f"{cwd}/",
            learning_rate=self.config.learning_rate,
            lr_scheduler_type=self.config.lr_scheduler_type,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=cfg.dataset.physical_batch_size,
            gradient_accumulation_steps=cfg.dataset.accumulation_steps,
            seed=self.config.seed,
            remove_unused_columns=False,
            ddp_find_unused_parameters=False,
            eval_strategy=self.config.eval_strategy,
            eval_steps=self.config.eval_steps,
            save_strategy=self.config.save_strategy,
            prediction_loss_only=False,
        )

        trainer_dataset = tokenized_dataset.remove_columns(
            [c for c in tokenized_dataset.column_names if isinstance(tokenized_dataset.features[c], datasets.Value) and tokenized_dataset.features[c].dtype == 'string']
        )

        callbacks = []
        if self.config.extrinsic_validation.frequency != ExtrinsicValidationFrequency.NEVER:       
            import src.eval.metrics.loaders
            callbacks.append(ExtrinsicValidationCallback(self.config.extrinsic_validation, tokenized_dataset, tokenizer, cwd, self.prompts))

        trainer = MemoryBankTrainer(
            model=model,
            loss_type=self.config.loss_type,
            loss_alpha=self.config.alpha,
            temperature=self.config.temperature,
            args=training_args,
            train_dataset=trainer_dataset,
            eval_dataset=trainer_dataset,
            callbacks=callbacks,
            processing_class=tokenizer,
            data_collator=SelfDistillationDataCollator(tokenizer),
            compute_metrics=MemoryBankTrainer.compute_metrics,
        )

        print("Starting memory training...")
        results = trainer.train()
        trainer.save_model()
        trainer.save_metrics("eval", results.metrics)
        with open(f"{cwd}/eval_log.json", "w") as fout:
            json.dump(trainer.state.log_history, fout)
        if hasattr(model.model, "virtual_prompt"):
            torch.save(model.model.virtual_prompt, f"{cwd}/virtual_prompt.pt")
        print("Memory training complete.")

        logs_df = pd.DataFrame(trainer.state.log_history)

        # Keep only rows where the marker metric exists
        if self.config.return_metric_key in logs_df.columns:
            final_metrics = logs_df[logs_df[self.config.return_metric_key].notna()].copy()
        else:
            final_metrics = pd.DataFrame()
        
        final_eval_metric = None
        if not final_metrics.empty and self.config.return_metric_key in final_metrics.columns:
            eval_loss_series = pd.to_numeric(final_metrics[self.config.return_metric_key], errors="coerce").dropna()
            if not eval_loss_series.empty:
                final_eval_metric = float(eval_loss_series.iloc[-1])
        
        if final_eval_metric is None:
            raise ValueError(
                f"Could not determine final {self.config.return_metric_key} from trainer.state.log_history. "
                f"Available columns: {list(logs_df.columns)}"
            )
        
        print(f"Final {self.config.return_metric_key}: {final_eval_metric}")
        return final_eval_metric
