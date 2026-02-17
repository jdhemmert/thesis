from dataclasses import dataclass, field
from typing import Optional, Any

import json
import math
import numpy as np
import evaluate

import torch
import torch.nn.functional as F
import datasets
from omegaconf import DictConfig
from datasets import load_dataset
from transformers import TrainingArguments, Trainer, AutoTokenizer, AutoModelForCausalLM, EvalPrediction

from src.tasks.base import BaseTaskConfig, register_task
from src.training.losses import self_distillation_loss
from src.training.callbacks import ExtrinsicValidationFrequency, ExtrinsicValidationConfig, ExtrinsicValidationCallback
from src.models.base import ModelFactory
from src.models.initializers import InitializerName, INITIALIZER_MAP


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
    """Custom Trainer for self-distillation."""
    def __init__(self, loss_type, loss_alpha, temperature, **kwargs):
        self.loss_type = loss_type
        self.alpha = loss_alpha
        self.T = temperature
        super().__init__(**kwargs)

    def evaluate(self, *args, **kwargs):
        metrics = super().evaluate(*args, **kwargs)

        if False and "eval_loss" in metrics:
            print("Calculating perplexity...")
            try:
                metrics["eval_ppl"] = math.exp(metrics["eval_loss"])
            except OverflowError:
                metrics["eval_ppl"] = float("inf")

        return metrics

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
    _target_: str = "src.tasks.train_memory.TrainMemoryTask"
    name: str = "train_memory"
    learning_rate: float = 2e-4
    num_train_epochs: int = 3
    precision: str = "bf16"
    seed: int = 42
    max_length: int = 512
    eval_strategy: str = "epoch"
    eval_steps: int = 100
    sample_n: int = 1000
    sample_strategy: str = "random"
    log_predictions: bool = True
    extrinsic_validation: ExtrinsicValidationConfig = field(default_factory=ExtrinsicValidationConfig)
    save_strategy: str = "steps"

    trainer_config: dict = field(default_factory=lambda: { })
    
    loss_type: str = "balanced"
    alpha: float = 0.2
    temperature: float = 0.5

    initialization_method: InitializerName = InitializerName.RANDOM
    initialization_config: dict = field(default_factory=lambda: { })

    trainable_strategy: str = "all" # all, soft_prompt_only


class TrainMemoryTask:
    def _preprocess(self, examples, tokenizer):
        """Prepares a dataset for self-distillation between an oracle and memory model."""
        max_length = self.config.max_length
        if "biography" not in examples or "question" not in examples:
            raise ValueError("Dataset must contain 'biography' and 'question' columns for self-distillation.")

        oracle_prompts = [self.prompts.contextual_qa_generation.format(biography=bio, question=q) for bio, q in zip(examples['biography'], examples['question'])]
        memory_prompts = [self.prompts.direct_qa_generation.format(question=q) for q in examples['question']]

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

        if self.config.extrinsic_validation.frequency != ExtrinsicValidationFrequency.NEVER:
            remove_columns = None
        else:
            remove_columns = original_columns
        
        tokenized_dataset = dataset.map(
            lambda examples: self._preprocess(examples, tokenizer=tokenizer),
            batched=True,
            remove_columns=remove_columns
        )
        return tokenized_dataset

    def _preprocess_logits_for_metrics(self, logits, labels):
        """
        For CausalLM: return per-example (sum_nll, n_tokens) so compute_metrics
        can aggregate token-weighted loss and perplexity without storing logits.
    
        Returns: tensor [bs, 2] where [:,0]=sum_nll, [:,1]=n_tokens
        """
        if isinstance(logits, (tuple, list)):
            logits = logits[0]  # [bs, seq, vocab]
    
        labels = labels.to(logits.device)
    
        # Ensure labels match logits seq length (pad with -100, don't truncate supervised positions)
        bs, seq_logits, vocab = logits.shape
        seq_labels = labels.shape[1]
        if seq_labels < seq_logits:
            pad = torch.full(
                (bs, seq_logits - seq_labels),
                -100,
                dtype=labels.dtype,
                device=labels.device,
            )
            labels = torch.cat([labels, pad], dim=1)
        elif seq_labels > seq_logits:
            labels = labels[:, :seq_logits]
    
        # Causal shift: predict token t+1 from position t
        shift_logits = logits[:, :-1, :]   # [bs, seq-1, vocab]
        shift_labels = labels[:, 1:]       # [bs, seq-1]
    
        # Per-token CE (no reduction), ignoring -100
        per_tok_nll = F.cross_entropy(
            shift_logits.reshape(-1, vocab),
            shift_labels.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).view(bs, -1)  # [bs, seq-1]
    
        mask = (shift_labels != -100)
        sum_nll = (per_tok_nll * mask).sum(dim=1)                 # [bs]
        n_tok = mask.sum(dim=1).to(dtype=sum_nll.dtype)           # [bs] as float
    
        return torch.stack([sum_nll, n_tok], dim=1).detach()

    def _compute_metrics(self, eval_preds: EvalPrediction):
        stats, _labels = eval_preds  # stats is [N,2] after concatenation over eval set
        sum_nll = stats[:, 0]
        n_tok   = stats[:, 1]
    
        total_nll = float(np.sum(sum_nll))
        total_tok = float(np.sum(n_tok))
    
        mean_loss = total_nll / max(total_tok, 1.0)
        ppl = math.exp(mean_loss) if mean_loss < 100 else float("inf")
    
        return {
            "recomputed_loss": mean_loss,
            "perplexity": ppl,
            "n_tokens": total_tok,
        }

    def __init__(self, **kwargs):
        self.config = TrainMemoryTaskConfig(**kwargs)
        self.metric = evaluate.load("perplexity")

    def main(self, cwd: str, cfg: DictConfig):
        """
        Main entrypoint for the memory training task.
        Incorporates the logic from the 'sequential' training strategy.
        """
        print(f"Running TrainMemoryTask: {self.config.name}")

        self.prompts = cfg.prompts
        
        model, tokenizer = ModelFactory.load(cfg)
        
        # Explicitly initialize the soft prompt if the model supports it
        if hasattr(model, 'initialize_virtual_prompt') and cfg.model.model_config.initialization_method:
            print(f"Performing soft prompt initialization with method: '{self.config.initialization_method.value}'...")
            
            initializer_class = INITIALIZER_MAP.get(self.config.initialization_method.value)
            if not initializer_class:
                raise ValueError(f"Unknown initializer: {self.config.initialization_method.value}")
            
            initializer = initializer_class()
            
            init_kwargs = {
                "main_model": self.model.model,
                "main_tokenizer": tokenizer,
                "virtual_token_count": self.config.virtual_token_count,
                "dataset_path": dataset_path,
                **self.config.initializer_config,
            }
            
            initial_weights = initializer.initialize(**init_kwargs)
            model.model.rebuild_virtual_prompt(weights=initial_weights)
            # model.initialize_virtual_prompt(
            #     tokenizer=tokenizer,
            #     method=cfg.model.model_config.initialization_method,
            #     config=cfg.model.model_config.initializer_config,
            #     dataset_path=cfg.dataset.path
            # )

        # TODO: maybe pass this into initializers instead of path
        tokenized_dataset = self._load_data(tokenizer, cfg.dataset)

        if self.config.trainable_strategy == "soft_prompt_only":
            print("Freezing base model, training soft prompt only.")
            for name, param in model.named_parameters():
                if "virtual_prompt" not in name:
                    param.requires_grad = False
        
        training_args = TrainingArguments(
            output_dir=f"{cwd}/",
            learning_rate=self.config.learning_rate,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=cfg.dataset.physical_batch_size,
            gradient_accumulation_steps=cfg.dataset.accumulation_steps,
            seed=self.config.seed,
            remove_unused_columns=False,
            ddp_find_unused_parameters=False,
            eval_strategy=self.config.eval_strategy,
            eval_steps=self.config.eval_steps,
            save_strategy=self.config.save_strategy,
        )

        trainer_dataset = tokenized_dataset.remove_columns(
            [c for c in tokenized_dataset.column_names if isinstance(tokenized_dataset.features[c], datasets.Value) and tokenized_dataset.features[c].dtype == 'string']
        )

        callbacks = []
        if self.config.extrinsic_validation.frequency != ExtrinsicValidationFrequency.NEVER:
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
            tokenizer=tokenizer,
            data_collator=SelfDistillationDataCollator(tokenizer),
            compute_metrics=self._compute_metrics,
            preprocess_logits_for_metrics=self._preprocess_logits_for_metrics,
        )

        print("Starting memory training...")
        results = trainer.train()
        trainer.save_model()
        trainer.save_metrics("eval", results.metrics)
        with open(f"{cwd}/eval_log.json", "w") as fout:
            json.dump(trainer.state.log_history, fout)
        print("Memory training complete.")
