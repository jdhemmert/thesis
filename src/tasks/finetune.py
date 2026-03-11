import os
import json
from dataclasses import dataclass, field
from typing import Optional
from omegaconf import DictConfig
from datasets import load_dataset
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling, AutoTokenizer, AutoModelForCausalLM

from src.tasks.base import BaseTaskConfig, register_task
from src.models.base import ModelFactory

@register_task(name="finetune", group="task")
@dataclass
class FinetuneTaskConfig(BaseTaskConfig):
    _target_: str = "src.tasks.finetune.FinetuneTask"
    name: str = "finetune"
    seed: int = 42
    num_train_epochs: int = 3
    precision: str = "bf16"
    max_length: int = 512
    eval_strategy: str = "steps"
    eval_steps: int = 100
    learning_rate: float = 2e-4
    sample_n: Optional[int] = None
    save_strategy: str = "steps"


class FinetuneTask:

    def _preprocess(self, examples, tokenizer):
        """Prepares a dataset for standard supervised fine-tuning."""
        max_length = self.config.max_length
        full_prompts = [
            self.prompts.contextual_qa_training.format(biography=bio, question=q, answer=a)
            for bio, q, a in zip(
                examples["biography"],
                examples["question"],
                examples["answer"]
            )
        ]
        model_inputs = tokenizer(
            full_prompts,
            max_length=max_length,
            padding="max_length",
            truncation=True
        )

        labels = [ row[:] for row in model_inputs["input_ids"] ]
        prompt_only = [
            self.prompts.contextual_qa_generation.format(biography=bio, question=q)
            for bio, q in zip(examples["biography"], examples["question"])
        ]
        prompt_token_lengths = [len(tokenizer(p, add_special_tokens=False).input_ids) for p in prompt_only]

        for i in range(len(labels)):
            prompt_len = prompt_token_lengths[i]
            labels[i][:prompt_len] = [-100] * prompt_len

        model_inputs["labels"] = labels
        return model_inputs

    def _load_data(self, tokenizer, dataset_config):
        """Loads, samples, and preprocesses the dataset for finetuning."""
        dataset = load_dataset("json", data_files=dataset_config.path)["train"]
        original_columns = list(dataset.column_names)

        if self.config.sample_n:
            dataset = dataset.shuffle(seed=self.config.seed).select(range(self.config.sample_n))

        tokenized_dataset = dataset.map(
            lambda examples: self._preprocess(examples, tokenizer=tokenizer),
            batched=True,
            remove_columns=original_columns
        )
        return tokenized_dataset

    def __init__(self, **kwargs):
        self.config = FinetuneTaskConfig(**kwargs)

    def main(self, cwd: str, cfg: DictConfig):
        """
        Main entrypoint for the finetuning task.
        """
        print(f"Running FinetuneTask: {self.config.name}")

        self.prompts = cfg.prompts

        model, tokenizer = ModelFactory.load(cfg)
        
        tokenized_dataset = self._load_data(tokenizer, cfg.dataset)
        
        if hasattr(model, "print_trainable_parameters"):
            model.print_trainable_parameters()

        split_dataset = tokenized_dataset.train_test_split(test_size=cfg.dataset.test_split_ratio, seed=self.config.seed)
        train_dataset = split_dataset["train"]
        eval_dataset = split_dataset["test"]

        training_args = TrainingArguments(
            output_dir=f"{cwd}/",
            learning_rate=self.config.learning_rate,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=cfg.dataset.physical_batch_size,
            gradient_accumulation_steps=cfg.dataset.accumulation_steps,
            seed=self.config.seed,
            eval_strategy=self.config.eval_strategy,
            eval_steps=self.config.eval_steps,
            save_strategy=self.config.save_strategy,
            logging_steps=10,
            fp16=self.config.precision == 'fp16',
            bf16=self.config.precision == 'bf16',
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        )

        print("Starting finetuning...")
        results = trainer.train()
        trainer.save_model()
        trainer.save_metrics("train", results.metrics)
        
        with open(f"{cwd}/eval_log.json", "w") as fout:
            json.dump(trainer.state.log_history, fout)
        print("Finetuning complete.")
