from dataclasses import dataclass, field
from typing import Optional, Any
import torch
import datasets
from omegaconf import DictConfig
from datasets import load_dataset
from transformers import TrainingArguments, Trainer, AutoTokenizer, AutoModelForCausalLM

from src.tasks.base import BaseTaskConfig, register_task
from src.training.losses import self_distillation_loss
from src.training.callbacks import ExtrinsicValidationCallback
from src.utils.model import load_model_from_config


class SelfDistillationDataCollator:
    """
    Data collator for the self-distillation task.
    Pads and collates batches to a common maximum length for oracle, memory, and label inputs.
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        max_len = max(max(len(f["oracle_input_ids"]) for f in features), max(len(f["memory_input_ids"]) for f in features))

        oracle_batch = self.tokenizer.pad(
            {"input_ids": [f["oracle_input_ids"] for f in features], "attention_mask": [f["oracle_attention_mask"] for f in features]},
            padding='max_length', max_length=max_len, return_tensors="pt"
        )
        memory_batch = self.tokenizer.pad(
            {"input_ids": [f["memory_input_ids"] for f in features], "attention_mask": [f["memory_attention_mask"] for f in features]},
            padding='max_length', max_length=max_len, return_tensors="pt"
        )

        batch = {
            "oracle_input_ids": oracle_batch["input_ids"],
            "oracle_attention_mask": oracle_batch["attention_mask"],
            "memory_input_ids": memory_batch["input_ids"],
            "memory_attention_mask": memory_batch["attention_mask"],
        }
        
        if "labels" in features[0]:
            labels_batch = self.tokenizer.pad(
                {"input_ids": [f["labels"] for f in features]},
                padding='max_length', max_length=max_len, return_tensors="pt"
            )
            batch["labels"] = labels_batch["input_ids"]

        return batch


class MemoryBankTrainer(Trainer):
    """Custom Trainer for self-distillation loss."""
    def __init__(self, loss_type, loss_alpha, temperature, **kwargs):
        self.loss_type = loss_type
        self.alpha = loss_alpha
        self.T = temperature
        super().__init__(**kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        oracle_outputs = model(
            input_ids=inputs["oracle_input_ids"],
            attention_mask=inputs["oracle_attention_mask"],
            labels=inputs["oracle_input_ids"]
        )
        memory_outputs = model(
            input_ids=inputs["memory_input_ids"],
            attention_mask=inputs["memory_attention_mask"],
            labels=inputs["memory_input_ids"]
        )
        loss = self_distillation_loss(
            oracle_logits=oracle_outputs.logits,
            memory_logits=memory_outputs.logits,
            temperature=self.T,
            alpha=self.alpha,
            loss_type=self.loss_type
        )
        return (loss, memory_outputs) if return_outputs else loss


@register_task(name="train_memory", group="task")
@dataclass
class TrainMemoryTaskConfig(BaseTaskConfig):
    # Config for the memory training task.
    _target_: str = "src.tasks.train_memory.TrainMemoryTask"
    name: str = "train_memory"
    output_dir: str = "models/memory_model_custom"
    learning_rate: float = 2e-4
    num_train_epochs: int = 3
    precision: str = "bf16"
    seed: int = 42
    max_length: int = 512
    eval_steps: int = 100
    sample_n: int = 1000
    sample_strategy: str = 'random'
    log_predictions: bool = True
    extrinsic_validation: str = "never"
    
    # Flattened loss parameters
    loss_type: str = "balanced"
    alpha: float = 0.2
    temperature: float = 0.5

    # Flattened optimizer parameters
    trainable_strategy: str = "all"


class TrainMemoryTask:
    def _preprocess(self, examples, tokenizer):
        """Prepares a dataset for self-distillation between an oracle and memory model."""
        max_length = self.config.max_length
        if "biography" not in examples or "question" not in examples:
            raise ValueError("Dataset must contain 'biography' and 'question' columns for self-distillation.")

        oracle_prompts = [f"Biography: {bio}\nQuestion: {q}" for bio, q in zip(examples['biography'], examples['question'])]
        memory_prompts = [f"Question: {q}" for q in examples['question']]

        oracle_inputs = tokenizer(oracle_prompts, truncation=True, max_length=max_length)
        memory_inputs = tokenizer(memory_prompts, truncation=True, max_length=max_length)

        return {
            "oracle_input_ids": oracle_inputs.input_ids,
            "oracle_attention_mask": oracle_inputs.attention_mask,
            "memory_input_ids": memory_inputs.input_ids,
            "memory_attention_mask": memory_inputs.attention_mask,
            "labels": memory_inputs.input_ids.copy(),
        }

    def _load_data(self, tokenizer, dataset_config):
        """Loads, samples, and preprocesses the dataset for memory training."""
        dataset = load_dataset("json", data_files=dataset_config.path)["train"]
        original_columns = list(dataset.column_names)

        if self.config.sample_n:
            if self.config.sample_strategy == 'random':
                dataset = dataset.shuffle(seed=self.config.seed).select(range(self.config.sample_n))
            else: # first_n
                dataset = dataset.select(range(self.config.sample_n))
        
        tokenized_dataset = dataset.map(
            lambda examples: self._preprocess(examples, tokenizer=tokenizer),
            batched=True,
            remove_columns=original_columns
        )
        return tokenized_dataset

    def __init__(self, **kwargs):
        self.config = TrainMemoryTaskConfig(**kwargs)

    def main(self, cfg: DictConfig):
        """
        Main entrypoint for the memory training task.
        Incorporates the logic from the 'sequential' training strategy.
        """
        print(f"Running TrainMemoryTask: {self.config.name}")
        
        # Load model and tokenizer using the utility function
        model, tokenizer = load_model_from_config(
            model_config=cfg.model.model_config,
            model_precision=self.config.precision,
            adapter_config=cfg.model.adapter_config
        )

        tokenized_dataset = self._load_data(tokenizer, cfg.dataset)

        # Freeze parameters if needed
        if self.config.trainable_strategy == "soft_prompt_only":
            print("Freezing base model, training soft prompt only.")
            for name, param in model.named_parameters():
                if "virtual_prompt" not in name:
                    param.requires_grad = False
        
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            learning_rate=self.config.learning_rate,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=cfg.dataset.physical_batch_size,
            gradient_accumulation_steps=cfg.dataset.accumulation_steps,
            seed=self.config.seed,
            remove_unused_columns=False,
            ddp_find_unused_parameters=False,
            eval_strategy="steps",
            eval_steps=self.config.eval_steps,
            save_strategy="steps",
        )

        trainer_dataset = tokenized_dataset.remove_columns(
            [c for c in tokenized_dataset.column_names if isinstance(tokenized_dataset.features[c], datasets.Value) and tokenized_dataset.features[c].dtype == 'string']
        )

        callbacks = []
        if self.config.extrinsic_validation != "never":
            callbacks.append(ExtrinsicValidationCallback(tokenized_dataset, tokenizer, self.config))

        trainer = MemoryBankTrainer(
            model=model,
            loss_type=self.config.loss_type,
            loss_alpha=self.config.alpha,
            temperature=self.config.temperature,
            args=training_args,
            train_dataset=trainer_dataset,
            eval_dataset=trainer_dataset,
            callbacks=callbacks,
            tokenizer=tokenizer,
            data_collator=SelfDistillationDataCollator(tokenizer),
        )

        print("Starting memory training...")
        trainer.train()
        trainer.save_model()
        print("Memory training complete.")
