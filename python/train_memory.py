
import argparse
import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset
from peft import get_peft_model, PromptTuningConfig, TaskType, PromptTuningInit, PeftModel
from accelerate import Accelerator

class MemoryBankTrainer(Trainer):
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

        # KL-Divergence Loss
        loss_fct = torch.nn.KLDivLoss(reduction="batchmean")
        log_softmax = torch.nn.LogSoftmax(dim=-1)
        softmax = torch.nn.Softmax(dim=-1)

        oracle_logits = oracle_outputs.logits.detach()
        memory_logits = memory_outputs.logits

        loss = loss_fct(
            log_softmax(memory_logits),
            softmax(oracle_logits)
        )
        
        return (loss, memory_outputs) if return_outputs else loss

def main():
    parser = argparse.ArgumentParser(description="Train a memory bank on a given biography.")
    
    # Model and Tokenizer
    parser.add_argument("--base_model_path", type=str, required=True, help="Path to the base model checkpoint.")
    parser.add_argument("--adapter_path", type=str, help="Path to the fine-tuned adapter checkpoint.")
    
    # Memory Bank Configuration
    parser.add_argument("--num_virtual_tokens", type=int, default=20, help="Number of virtual tokens for the memory bank.")
    parser.add_argument("--prompt_tuning_init", type=str, default="TEXT", choices=["TEXT", "RANDOM"], help="Initialization method for the memory bank.")
    parser.add_argument("--prompt_tuning_init_text", type=str, default="", help="Initialization text for the memory bank if using TEXT initialization.")
    parser.add_argument("--match_token_count", action="store_true", help="Automatically set num_virtual_tokens to the length of the init text.")

    # Data
    parser.add_argument("--task_data_file", type=str, required=True, help="Path to the JSONL file.")
    parser.add_argument("--sample_n", type=int, help="Number of samples to use from the dataset.")
    parser.add_argument("--sample_strategy", type=str, default="first_n", choices=["first_n", "random"], help="How to select samples if --sample_n is used.")

    # Training Arguments
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the trained memory bank.")
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum sequence length for tokenizer.")
    parser.add_argument("--eval_strategy", type=str, default="epoch", choices=["no", "steps", "epoch"], help="Evaluation strategy to adopt during training.")
    parser.add_argument("--eval_steps", type=int, default=1, help="Number of update steps between two evaluations if evaluation_strategy is 'steps'.")
    parser.add_argument("--save_strategy", type=str, default="no", choices=["no", "steps", "epoch"], help="When to save a model checkpoint.")

    args = parser.parse_args()

    accelerator = Accelerator()

    # Set up TrainingArguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        seed=args.seed,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        eval_strategy=args.eval_strategy,
        eval_steps=args.eval_steps,
        save_strategy=args.save_strategy,
    )

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.match_token_count:
        if args.prompt_tuning_init == "TEXT" and args.prompt_tuning_init_text:
            num_tokens = len(tokenizer(args.prompt_tuning_init_text)["input_ids"])
            print(f"Matching token count: Overriding num_virtual_tokens to {num_tokens}")
            args.num_virtual_tokens = num_tokens
        else:
            print("Warning: --match_token_count is only applicable when using --prompt_tuning_init=TEXT and providing --prompt_tuning_init_text.")

    model = AutoModelForCausalLM.from_pretrained(args.base_model_path)

    if args.adapter_path:
        model = PeftModel.from_pretrained(model, args.adapter_path)

    # Initialize PEFT config for Prompt Tuning
    peft_config = PromptTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        prompt_tuning_init=PromptTuningInit.TEXT if args.prompt_tuning_init == "TEXT" else PromptTuningInit.RANDOM,
        prompt_tuning_init_text=args.prompt_tuning_init_text,
        num_virtual_tokens=args.num_virtual_tokens,
        tokenizer_name_or_path=args.base_model_path
    )
    
    model = get_peft_model(model, peft_config)

    # Load and preprocess the data
    dataset = load_dataset("json", data_files=args.task_data_file)["train"]

    if args.sample_n:
        if args.sample_strategy == 'random':
            dataset = dataset.shuffle(seed=training_args.seed).select(range(args.sample_n))
        else: # first_n
            dataset = dataset.select(range(args.sample_n))

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
        oracle_inputs = tokenizer(examples["oracle_prompt"], padding="max_length", truncation=True, max_length=args.max_length)
        memory_inputs = tokenizer(examples["memory_prompt"], padding="max_length", truncation=True, max_length=args.max_length)

        # The 'labels' field is the switch that tells the Trainer to use compute_loss during evaluation.
        return {
            "oracle_input_ids": oracle_inputs.input_ids,
            "oracle_attention_mask": oracle_inputs.attention_mask,
            "memory_input_ids": memory_inputs.input_ids,
            "memory_attention_mask": memory_inputs.attention_mask,
            "labels": memory_inputs.input_ids.copy(),
        }

    tokenized_dataset = dataset.map(preprocess_function, batched=True)

    # Initialize the MemoryBankTrainer
    trainer = MemoryBankTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        eval_dataset=tokenized_dataset,
    )

    trainer.train()
    trainer.save_model()

if __name__ == "__main__":
    main()
