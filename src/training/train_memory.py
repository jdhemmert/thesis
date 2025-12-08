import argparse
import json
import os
import torch
import numpy as np
import evaluate
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, TrainerCallback, DataCollatorWithPadding
from datasets import load_dataset
from peft import get_peft_model, PrefixTuningConfig, PromptTuningConfig, TaskType, PromptTuningInit, PeftModel
from accelerate import Accelerator
import hydra
from omegaconf import DictConfig

from src.utils.model import load_model_and_tokenizer
from src.training.losses import self_distillation_loss

class MemoryBankTrainer(Trainer):

    def __init__(self, loss_type, loss_alpha, temperature, **kwargs):
        self.loss_type = loss_type
        self.alpha = loss_alpha
        self.T = temperature
        super().__init__(**kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=0):
        """
        Custom loss function for self-oracle training of the memory bank.
        """

        # Oracle pass: model with full context
        oracle_outputs = model(
            input_ids=inputs["oracle_input_ids"],
            attention_mask=inputs["oracle_attention_mask"],
            labels=inputs["oracle_input_ids"]
        )

        # Memory pass: model with only the memory bank
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


class ExtrinsicValidationCallback(TrainerCallback):
    
    def __init__(self, eval_dataset, tokenizer, cfg_task):
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.cfg_task = cfg_task

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        model = kwargs.get("model")
        tokenizer = self.tokenizer
        eval_dataset = self.eval_dataset

        if model is None or tokenizer is None:
            print("Skipping extrinsic validation: model or tokenizer not available.")
            return

        required_cols = ["memory_prompt", "answer"]
        if not all(col in eval_dataset.column_names for col in required_cols):
            print(f"Skipping extrinsic validation: Required columns {required_cols} not found in dataset.")
            return

        print("\nPerforming Extrinsic Validation...")
        all_preds = []
        all_labels = []
        all_questions_for_log = []

        model.eval()
        for example in eval_dataset:
            # Use pre-tokenized inputs from the dataset
            input_ids = torch.tensor([example['memory_input_ids']]).to(model.device)
            attention_mask = torch.tensor([example['memory_attention_mask']]).to(model.device)

            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=50,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )

            num_input_tokens = len(input_ids[0])
            pred_ids = generated_ids[0][num_input_tokens:]
            pred_text = tokenizer.decode(pred_ids, skip_special_tokens=True).strip()

            all_preds.append(pred_text)
            all_labels.append(example["answer"])
            if "question" in example:
                all_questions_for_log.append(example["question"])

        # Compute ROUGE scores
        rouge = evaluate.load('rouge')
        rouge_scores = rouge.compute(predictions=all_preds, references=all_labels)

        # Add scores to metrics for logging
        if metrics is not None:
            for key, value in rouge_scores.items():
                metrics[f"eval_{key}"] = value

        # Explicitly print ROUGE scores to console if control.log_metrics is not available
            print(f"Extrinsic ROUGE Scores: {rouge_scores}")

        # Log predictions to file
        if state.is_world_process_zero and self.cfg_task.log_predictions:
            log_file_path = os.path.join(self.cfg_task.output_dir, f"prediction_log.epoch_{int(state.epoch)}.jsonl")
            print(f"Logging predictions to {log_file_path}")
            with open(log_file_path, "w") as f:
                for i in range(len(all_preds)):
                    log_entry = {
                        "question": all_questions_for_log[i] if i < len(all_questions_for_log) else "N/A",
                        "ground_truth": all_labels[i],
                        "prediction": all_preds[i]
                    }
                    f.write(json.dumps(log_entry) + "\n")

@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig):
    # Access train_memory specific configurations
    cfg_task = cfg.task

    accelerator = Accelerator()

    # Set up TrainingArguments
    training_args = TrainingArguments(
        output_dir=cfg_task.output_dir,
        learning_rate=cfg_task.learning_rate,
        num_train_epochs=cfg_task.num_train_epochs,
        per_device_train_batch_size=cfg_task.per_device_train_batch_size,
        gradient_accumulation_steps=cfg_task.gradient_accumulation_steps,
        seed=cfg_task.seed,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        eval_strategy=cfg_task.eval_strategy,
        eval_steps=cfg_task.eval_steps,
        save_strategy=cfg_task.save_strategy,
    )

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(
        model_path=cfg_task.base_model_path,
        adapter_path=cfg_task.adapter_path,
        precision=cfg_task.precision
    )
    
    # Initialize PEFT config for Memory Bank
    # Use hydra.utils.instantiate to create the PEFT config from the chosen 'peft' group
    peft_method = cfg_task.peft_method
    peft_config_to_instantiate = getattr(cfg.peft, peft_method)
    peft_config = hydra.utils.instantiate(peft_config_to_instantiate, tokenizer_name_or_path=cfg_task.base_model_path)

    # Restore dynamic virtual token count logic for Prompt Tuning, if selected
    if (
        cfg_task.match_token_count and 
        isinstance(peft_config, PromptTuningConfig) and
        peft_config.prompt_tuning_init == PromptTuningInit.TEXT
    ):
        num_tokens = len(tokenizer(peft_config.prompt_tuning_init_text)["input_ids"])
        print(f"Matching token count: Overriding num_virtual_tokens to {num_tokens}")
        peft_config.num_virtual_tokens = num_tokens

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Load and preprocess the data
    dataset = load_dataset("json", data_files=cfg_task.task_data_file)["train"]

    if cfg_task.sample_n:
        if cfg_task.sample_strategy == 'random':
            dataset = dataset.shuffle(seed=training_args.seed).select(range(cfg_task.sample_n))
        else: # first_n
            dataset = dataset.select(range(cfg_task.sample_n))

    # Check if the dataset needs to be transformed
    if "biography" in dataset.column_names and "question" in dataset.column_names:
        def transform_to_prompts(examples):
            oracle_prompts = []
            memory_prompts = []
            for i in range(len(examples['biography'])):
                oracle_prompts.append(f"Biography: {examples['biography'][i]}\nQuestion: {examples['question'][i]}")
                memory_prompts.append(f"Question: {examples['question'][i]}")
            return {"oracle_prompt": oracle_prompts, "memory_prompt": memory_prompts}
        dataset = dataset.map(transform_to_prompts, batched=True)

    def preprocess_function(examples):
        oracle_inputs = tokenizer(examples["oracle_prompt"], truncation=True, max_length=cfg_task.max_length)
        memory_inputs = tokenizer(examples["memory_prompt"], truncation=True, max_length=cfg_task.max_length)

        # The 'labels' field is the switch that tells the Trainer to use compute_loss during evaluation.
        return {
            "oracle_input_ids": oracle_inputs.input_ids,
            "oracle_attention_mask": oracle_inputs.attention_mask,
            "memory_input_ids": memory_inputs.input_ids,
            "memory_attention_mask": memory_inputs.attention_mask,
            "labels": memory_inputs.input_ids.copy(),
        }

    tokenized_dataset = dataset.map(preprocess_function, batched=True)

    # Create a custom data collator to handle the unique batch structure
    def custom_data_collator(features):
        # Determine the maximum sequence length in the batch across both oracle and memory inputs
        max_len = 0
        for feature in features:
            max_len = max(max_len, len(feature["oracle_input_ids"]))
            max_len = max(max_len, len(feature["memory_input_ids"]))

        # Pad oracle inputs to the determined max_len
        oracle_batch = tokenizer.pad(
            {
                "input_ids": [f["oracle_input_ids"] for f in features],
                "attention_mask": [f["oracle_attention_mask"] for f in features]
            },
            padding='max_length',
            max_length=max_len,
            return_tensors="pt"
        )

        # Pad memory inputs to the determined max_len
        memory_batch = tokenizer.pad(
            {
                "input_ids": [f["memory_input_ids"] for f in features],
                "attention_mask": [f["memory_attention_mask"] for f in features]
            },
            padding='max_length',
            max_length=max_len,
            return_tensors="pt"
        )

        # Labels should also be padded to the same max_len
        labels_batch = tokenizer.pad(
            {
                "input_ids": [f["labels"] for f in features]
            },
            padding='max_length',
            max_length=max_len,
            return_tensors="pt"
        )

        return {
            "oracle_input_ids": oracle_batch["input_ids"],
            "oracle_attention_mask": oracle_batch["attention_mask"],
            "memory_input_ids": memory_batch["input_ids"],
            "memory_attention_mask": memory_batch["attention_mask"],
            "labels": labels_batch["input_ids"]
        }

    # Create a clean version of the dataset for the Trainer, which expects only tensor-izable columns
    text_columns = ['question', 'answer', 'biography', 'oracle_prompt', 'memory_prompt']
    trainer_dataset = tokenized_dataset.remove_columns([col for col in text_columns if col in tokenized_dataset.column_names])

    # Initialize the MemoryBankTrainer
    trainer = MemoryBankTrainer(
        model=model,
        loss_type=cfg_task.loss_type,
        loss_alpha=cfg_task.loss_alpha,
        temperature=cfg_task.temperature,
        args=training_args,
        train_dataset=trainer_dataset,
        eval_dataset=trainer_dataset,
        callbacks=[ExtrinsicValidationCallback(tokenized_dataset, tokenizer, cfg_task)],
        tokenizer=tokenizer,
        data_collator=custom_data_collator,
    )

    trainer.train()
    trainer.save_model()
