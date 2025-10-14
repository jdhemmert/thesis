import argparse
import json
import math
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset
from peft import get_peft_model, LoraConfig

from dotenv import load_dotenv
load_dotenv()

def main():
    parser = argparse.ArgumentParser(description="Fine-tune a Llama-3.2-3B model.")
    parser.add_argument("--model_path", type=str, default="meta-llama/Llama-3.2-3B", help="The path to the base model.")
    parser.add_argument("--dataset_dir", type=str, default="data", help="The directory containing the QA dataset.")
    parser.add_argument("--output_dir", type=str, default="models/finetuned_model", help="The directory to save the fine-tuned model.")
    parser.add_argument("--log_dir", type=str, default="logs", help="The directory to save the training logs.")
    parser.add_argument("--method", type=str, default="full", choices=["full", "lora"], help="The fine-tuning method to use.")
    parser.add_argument("--test_split_ratio", type=float, default=0.125, help="The ratio of the dataset to be used for testing.")
    parser.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=True, help="Enable dataset streaming.")
    parser.add_argument("--physical_batch_size", type=int, default=4, help="The physical batch size on each device.")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--seed", type=int, default=42, help="The random seed for reproducibility.")
    parser.add_argument("--gpu_ids", type=str, help="A comma-separated list of GPU IDs to use.")
    parser.add_argument("--epochs", type=int, default=3, help="The number of training epochs.")
    parser.add_argument("--precision", type=str, default="bf16", choices=["fp32", "fp16", "bf16"], help="The precision to use for training.")
    parser.add_argument("--max_seq_length", type=int, default=128, help="The maximum sequence length to use.")
    parser.add_argument("--eval_steps", type=int, default=1000, help="How often to perform eval during training.")
    args = parser.parse_args()

    if args.gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    # Preprocess the dataset
    def preprocess_function(examples):
        # Combine prompt and answer, then tokenize
        full_prompts = [f"Biography: {bio}\nQuestion: {q}\nAnswer: {a}" for bio, q, a in zip(examples["biography"], examples["question"], examples["answer"])]
        model_inputs = tokenizer(full_prompts, max_length=args.max_seq_length, padding="max_length", truncation=True)

        # The labels are the same as the input_ids
        labels = [row[:] for row in model_inputs["input_ids"]]

        # Tokenize prompts to find their lengths for masking
        prompt_only = [f"Biography: {bio}\nQuestion: {q}\nAnswer:" for bio, q in zip(examples["biography"], examples["question"])]
        prompt_token_lengths = [len(tokenizer(p, add_special_tokens=False).input_ids) for p in prompt_only]

        # Mask the prompt part of the labels
        for i in range(len(labels)):
            prompt_len = prompt_token_lengths[i]
            labels[i][:prompt_len] = [-100] * prompt_len

        model_inputs["labels"] = labels
        return model_inputs

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token = tokenizer.eos_token

    # Load and preprocess the dataset
    max_steps = -1
    if args.streaming:
        full_dataset = load_dataset("json", data_files=os.path.join(args.dataset_dir, "qa_finetune_dataset.jsonl"), streaming=True)["train"]
        shuffled_dataset = full_dataset.shuffle(seed=args.seed, buffer_size=10000) # for reproducibility

        # train_test_split is not available for streaming datasets.
        # We'll manually split it.
        def get_dataset_size(path):
            with open(path) as f:
                for i, _ in enumerate(f):
                    pass
            return i + 1

        dataset_path = os.path.join(args.dataset_dir, "qa_finetune_dataset.jsonl")
        dataset_size = get_dataset_size(dataset_path)
        test_size = int(dataset_size * args.test_split_ratio)

        test_dataset = shuffled_dataset.take(test_size)
        train_dataset = shuffled_dataset.skip(test_size)

        train_tokenized_dataset = train_dataset.map(preprocess_function, batched=True)
        test_tokenized_dataset = test_dataset.map(preprocess_function, batched=True)

        # calculate max_steps for streaming dataset
        train_size = dataset_size - test_size
        effective_batch_size = args.physical_batch_size * args.accumulation_steps
        max_steps = math.ceil(train_size / effective_batch_size) * args.epochs
    else:
        full_dataset = load_dataset("json", data_files=os.path.join(args.dataset_dir, "qa_finetune_dataset.jsonl"))["train"]
        shuffled_dataset = full_dataset.shuffle(seed=args.seed) # for reproducibility
        split_dataset = shuffled_dataset.train_test_split(test_size=args.test_split_ratio)
        train_dataset = split_dataset["train"]
        test_dataset = split_dataset["test"]

        train_tokenized_dataset = train_dataset.map(preprocess_function, batched=True, num_proc=4)
        test_tokenized_dataset = test_dataset.map(preprocess_function, batched=True, num_proc=4)

    # Load the model
    dtype_map = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    dtype = dtype_map.get(args.precision, torch.bfloat16)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(args.model_path, dtype=dtype).to(device)

    if args.method == "lora":
        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, peft_config)

    # Set up the training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        logging_dir=args.log_dir,
        per_device_train_batch_size=args.physical_batch_size,
        gradient_accumulation_steps=args.accumulation_steps,
        num_train_epochs=args.epochs,
        max_steps=max_steps,
        logging_steps=100,
        save_steps=1000,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        seed=args.seed,
        ddp_find_unused_parameters=False,
    )

    # Create the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized_dataset,
        eval_dataset=test_tokenized_dataset,
    )

    # Train the model
    trainer.train()

    # Save the model
    trainer.save_model()

if __name__ == "__main__":
    main()
