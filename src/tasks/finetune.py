import os
from dataclasses import dataclass, field
from typing import Optional
from omegaconf import DictConfig
from datasets import load_dataset
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling, AutoTokenizer, AutoModelForCausalLM

from src.tasks.base import BaseTaskConfig, register_task
from src.utils.model import load_model_from_config

@register_task(name="finetune", group="task")
@dataclass
class FinetuneTaskConfig(BaseTaskConfig):
    _target_: str = "src.tasks.finetune.FinetuneTask"
    name: str = "finetune"
    output_dir: str = "models/finetuned_model"
    log_dir: str = "logs"
    seed: int = 42
    gpu_ids: Optional[str] = None
    epochs: int = 3
    precision: str = "bf16"
    max_seq_length: int = 512
    eval_steps: int = 100
    learning_rate: float = 2e-4
    sample_n: Optional[int] = None


class FinetuneTask:

    def _preprocess(self, examples, tokenizer):
        """Prepares a dataset for standard supervised fine-tuning."""
        max_length = self.config.max_seq_length
        full_prompts = [
            f"Biography: {bio}\nQuestion: {q}\nAnswer: {a}"
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
            f"Biography: {bio}\nQuestion: {q}\nAnswer:"
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
        # 1. Load the raw dataset
        dataset = load_dataset("json", data_files=dataset_config.path)["train"]
        original_columns = list(dataset.column_names)

        # 2. Handle sampling
        if self.config.sample_n:
            dataset = dataset.shuffle(seed=self.config.seed).select(range(self.config.sample_n))

        # 3. Apply the preprocessing function
        tokenized_dataset = dataset.map(
            lambda examples: self._preprocess(examples, tokenizer=tokenizer),
            batched=True,
            remove_columns=original_columns
        )
        return tokenized_dataset

    def __init__(
        self,
        name: str,
        output_dir: str,
        log_dir: str,
        seed: int,
        epochs: int,
        precision: str,
        max_seq_length: int,
        eval_steps: int,
        learning_rate: float,
        gpu_ids: Optional[str] = None,
        sample_n: Optional[int] = None,
        **kwargs,
    ):
        self.config = FinetuneTaskConfig(
            name=name, output_dir=output_dir, log_dir=log_dir, seed=seed,
            gpu_ids=gpu_ids, epochs=epochs, precision=precision,
            max_seq_length=max_seq_length, eval_steps=eval_steps,
            learning_rate=learning_rate, sample_n=sample_n
        )

    def main(self, cfg: DictConfig):
        """
        Main entrypoint for the finetuning task.
        """
        print(f"Running FinetuneTask: {self.config.name}")

        if self.config.gpu_ids:
            os.environ["CUDA_VISIBLE_DEVICES"] = self.config.gpu_ids

        # Load model and tokenizer using the utility function
        model, tokenizer = load_model_from_config(
            model_config=cfg.model.model_config,
            model_precision=self.config.precision,
            adapter_config=cfg.model.adapter_config
        )
        
        tokenized_dataset = self._load_data(tokenizer, cfg.dataset)
        
        model.print_trainable_parameters()

        split_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=self.config.seed)
        train_dataset = split_dataset["train"]
        eval_dataset = split_dataset["test"]

        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            learning_rate=self.config.learning_rate,
            num_train_epochs=self.config.epochs,
            per_device_train_batch_size=cfg.dataset.physical_batch_size,
            gradient_accumulation_steps=cfg.dataset.accumulation_steps,
            seed=self.config.seed,
            evaluation_strategy="steps",
            eval_steps=self.config.eval_steps,
            save_strategy="steps",
            logging_steps=10,
            fp16=self.config.precision == 'fp16',
            bf16=self.config.precision == 'bf16',
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        )

        print("Starting finetuning...")
        trainer.train()
        trainer.save_model()
        print("Finetuning complete.")
