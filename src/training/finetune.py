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

from src.utils.dataset import get_tokenized_datasets
from .models import create_model
from .strategies.sequential import train_sequential
from .strategies.interleaved import train_interleaved

def main():
    parser = argparse.ArgumentParser(description="Fine-tune a Llama-3.2-3B model.")
    parser.add_argument("--model_path", type=str, default="meta-llama/Llama-3.2-3B", help="The path to the base model.")
    parser.add_argument("--dataset_file", type=str, default="data", help="The file containing the QA dataset.")
    parser.add_argument("--output_dir", type=str, default="models/finetuned_model", help="The directory to save the fine-tuned model.")
    parser.add_argument("--log_dir", type=str, default="logs", help="The directory to save the training logs.")
    parser.add_argument("--method", type=str, default="full", choices=["full", "lora", "ptune"], help="The fine-tuning method to use.")
    parser.add_argument("--num_virtual_tokens", type=int, default=32, help="The number of virtual tokens to use in prompt tuning methods.")
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
    parser.add_argument("--strategy", type=str, default="sequential", choices=["sequential", "interleaved"], help="The training strategy to use.")
    parser.add_argument("--finetune_type", type=str, default="lora", choices=["lora", "ptune"], help="The finetune method to use.")
    args = parser.parse_args()

    if args.gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token = tokenizer.eos_token

    train_tokenized_dataset, test_tokenized_dataset, max_steps = get_tokenized_datasets(args, tokenizer)

    model = create_model(args, tokenizer)

    strategies = {
        "sequential": train_sequential,
        "interleaved": train_interleaved,
    }

    if args.strategy in strategies:
        strategies[args.strategy](args, model, train_tokenized_dataset, test_tokenized_dataset, max_steps)
    else:
        raise ValueError(f"Unknown strategy: {args.strategy}")

if __name__ == "__main__":
    main()
