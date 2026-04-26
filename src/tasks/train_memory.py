from dataclasses import dataclass, field
from typing import Optional, Any
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import evaluate

import hydra
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
from src.utils.logs import load_eval_rows, extract_final_metric


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
    distractor_n: int = 0
    temperature: float = 1.0

    return_metric_key: str = "eval_loss"

    initialization_method: Optional[InitializerName] = None
    initialization_config: dict = field(default_factory=lambda: { })

    trainable_strategy: str = "soft_prompt_only" # all, soft_prompt_only


class TrainMemoryTask:
    def _load_data(self, tokenizer, dataset_config):
        """Loads, samples, and preprocesses the dataset for memory training.

        Returns (train_dataset, eval_dataset). eval_dataset contains only the
        target bio(s); train_dataset additionally includes distractor bios when
        task.distractor_n > 0 so that per-epoch eval metrics remain clean.
        """
        import random

        dataset = load_dataset("json", data_files=dataset_config.path)["train"]
        original_columns = list(dataset.column_names)

        self._bio_metadata = None
        distractor_pool = None

        if getattr(dataset_config, "biography_index", None) is not None:
            raw_example = dataset[dataset_config.biography_index]
            self._bio_metadata = {
                k: v for k, v in raw_example.items()
                if isinstance(v, (str, int, float, bool))
            }
            self._bio_metadata["biography_index"] = dataset_config.biography_index
            if self.config.distractor_n > 0:
                distractor_pool = dataset  # full dataset; biography_index excluded when sampling
            dataset = dataset.select([dataset_config.biography_index])
        elif self.config.sample_n:
            if self.config.sample_strategy == 'random':
                shuffled = dataset.shuffle(seed=self.config.seed)
            else:
                shuffled = dataset
            dataset = shuffled.select(range(self.config.sample_n))
            if self.config.distractor_n > 0:
                distractor_pool = shuffled  # remainder starts at index sample_n

        using_new_path = (
            getattr(dataset_config, "parser", None) is not None
            and getattr(dataset_config, "biography_task", None) is not None
        )

        if using_new_path:
            parser = hydra.utils.instantiate(dataset_config.parser)
            biography_task = hydra.utils.instantiate(dataset_config.biography_task)
            # New path always removes original columns: the biography task may
            # expand rows and all needed fields are emitted by the preprocessor.
            remove_columns = original_columns
        else:
            parser = None
            biography_task = None
            # Legacy path: remove columns only when row count may change.
            if self.dataset_config.dataset_mode == "attributes" or self.config.extrinsic_validation.frequency == ExtrinsicValidationFrequency.NEVER:
                remove_columns = original_columns
            else:
                remove_columns = None

        preprocessor = MemoryTaskPreprocessor(
            tokenizer=tokenizer,
            max_length=self.config.max_length,
            prompts=self.prompts,
            dataset_mode=self.dataset_config.dataset_mode,
            parser=parser,
            biography_task=biography_task,
        )

        eval_dataset = dataset.map(
            preprocessor,
            batched=True,
            remove_columns=remove_columns
        )

        train_dataset = eval_dataset
        if distractor_pool is not None and self.config.distractor_n > 0:
            if getattr(dataset_config, "biography_index", None) is not None:
                rng = random.Random(self.config.seed)
                pool = [i for i in range(len(distractor_pool)) if i != dataset_config.biography_index]
                sampled_indices = rng.sample(pool, min(self.config.distractor_n, len(pool)))
                distractor_raw = distractor_pool.select(sampled_indices)
            else:
                start = self.config.sample_n
                n = min(self.config.distractor_n, len(distractor_pool) - start)
                distractor_raw = distractor_pool.select(range(start, start + n)) if n > 0 else None

            if distractor_raw is not None and len(distractor_raw) > 0:
                distractor_tokenized = distractor_raw.map(
                    preprocessor,
                    batched=True,
                    remove_columns=remove_columns,
                )
                train_dataset = datasets.concatenate_datasets([eval_dataset, distractor_tokenized])
                print(f"Distractor bios: {len(distractor_raw)} bios → {len(distractor_tokenized)} QA pairs added to train.")

        return train_dataset, eval_dataset

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
        train_tokenized, eval_tokenized = self._load_data(tokenizer, cfg.dataset)

        if self.config.trainable_strategy == "soft_prompt_only":
            print("Freezing base model, training soft prompt only.")
            for name, param in model.named_parameters():
                param.requires_grad = "virtual_prompt" in name

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

        def _drop_str(ds):
            return ds.remove_columns([c for c in ds.column_names if isinstance(ds.features[c], datasets.Value) and ds.features[c].dtype == 'string'])
        train_trainer_dataset = _drop_str(train_tokenized)
        eval_trainer_dataset = _drop_str(eval_tokenized)

        callbacks = []
        if self.config.extrinsic_validation.frequency != ExtrinsicValidationFrequency.NEVER:
            import src.eval.metrics.loaders
            callbacks.append(ExtrinsicValidationCallback(self.config.extrinsic_validation, eval_tokenized, tokenizer, cwd, self.prompts))

        trainer = MemoryBankTrainer(
            model=model,
            loss_type=self.config.loss_type,
            loss_alpha=self.config.alpha,
            temperature=self.config.temperature,
            args=training_args,
            train_dataset=train_trainer_dataset,
            eval_dataset=eval_trainer_dataset,
            callbacks=callbacks,
            processing_class=tokenizer,
            data_collator=SelfDistillationDataCollator(tokenizer),
            compute_metrics=MemoryBankTrainer.compute_metrics,
        )

        print("Starting memory training...")
        results = trainer.train()
        trainer.save_model()
        trainer.save_metrics("eval", results.metrics)
        
        eval_log_path = Path(cwd) / "eval_log.json"
        with eval_log_path.open("w") as fout:
            json.dump(trainer.state.log_history, fout)

        if hasattr(model.model, "virtual_prompt"):
            torch.save(model.model.virtual_prompt, f"{cwd}/virtual_prompt.pt")

        if getattr(self, "_bio_metadata", None) is not None:
            bio_meta_path = Path(cwd) / "bio_metadata.json"
            with bio_meta_path.open("w") as f:
                json.dump(self._bio_metadata, f, indent=2)

        print("Memory training complete.")

        rows = load_eval_rows(eval_log_path)

        try:
            final_eval_metric = extract_final_metric(rows, self.config.return_metric_key)
        except ValueError as e:
            available_keys = sorted({k for row in rows for k in row.keys()})
            raise ValueError(
                f"Could not determine final {self.config.return_metric_key} from {eval_log_path}. "
                f"Available columns: {available_keys}"
            ) from e
        
        print(f"Final {self.config.return_metric_key}: {final_eval_metric}")
        return final_eval_metric
