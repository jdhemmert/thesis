import sys
import os

# Add the project root to the Python path to allow absolute imports from 'src'.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import argparse
import json
import torch
from datasets import load_dataset
import evaluate
import os

from src.utils.model import load_model_and_tokenizer

def main():
    parser = argparse.ArgumentParser(description="Evaluate base and fine-tuned models on a QA task.")
    parser.add_argument("--base_model_path", type=str, required=True, help="Path to the base model checkpoint.")
    parser.add_argument("--adapter_path", type=str, help="Path to the fine-tuned adapter checkpoint (optional).")
    parser.add_argument("--dataset_file", type=str, required=True, help="Path to the JSONL QA dataset file.")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to evaluate from the dataset.")
    parser.add_argument("--max_seq_length", type=int, default=128, help="Maximum sequence length for tokenizer.")
    parser.add_argument("--max_new_tokens", type=int, default=50, help="Maximum number of new tokens to generate.")
    parser.add_argument("--precision", type=str, default="bf16", choices=["fp32", "fp16", "bf16"], help="The precision to use for model loading.")
    parser.add_argument("--gpu_id", type=str, help="The specific GPU ID to use (e.g., '0', '1').")
    
    args = parser.parse_args()

    if args.gpu_id:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    # Load model and tokenizer using the utility function
    model, tokenizer = load_model_and_tokenizer(
        model_path=args.base_model_path,
        adapter_path=args.adapter_path,
        precision=args.precision
    )

    device = model.device # Get device from the loaded model

    model.eval() # Set model to evaluation mode

    # Load dataset
    dataset = load_dataset("json", data_files=args.dataset_file)["train"]
    
    # Ensure dataset has required columns
    required_cols = ["biography", "question", "answer"]
    if not all(col in dataset.column_names for col in required_cols):
        raise ValueError(f"Dataset must contain columns: {required_cols}")

    # Sample from dataset
    sampled_dataset = dataset.shuffle(seed=42).select(range(min(args.num_samples, len(dataset))))

    all_generated_answers = []
    all_ground_truth_answers = []
    all_losses = []

    print("\n--- Model Evaluation ---")
    for i, example in enumerate(sampled_dataset):
        biography = example["biography"]
        question = example["question"]
        ground_truth_answer = example["answer"]

        # Construct prompt for generation (consistent with finetuning)
        generation_prompt = f"Biography: {biography}\nQuestion: {question}\nAnswer:"

        # Tokenize prompt for generation
        generation_inputs = tokenizer(generation_prompt, return_tensors="pt", max_length=args.max_seq_length, truncation=True).to(device)

        # Generate answer
        with torch.no_grad():
            generated_ids = model.generate(
                **generation_inputs,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id, # Use tokenizer's pad_token_id
                eos_token_id=tokenizer.eos_token_id, # Use tokenizer's eos_token_id
                do_sample=False, # For deterministic generation
                num_beams=1, # For deterministic generation
            )
        
        # Decode generated answer
        num_input_tokens = generation_inputs.input_ids.shape[1]
        generated_answer_ids = generated_ids[0, num_input_tokens:]
        generated_answer = tokenizer.decode(generated_answer_ids, skip_special_tokens=True).strip()

        all_generated_answers.append(generated_answer)
        all_ground_truth_answers.append(ground_truth_answer)

        # --- Perplexity Calculation ---
        # Construct full prompt for perplexity calculation
        full_prompt_for_ppl = f"Biography: {biography}\nQuestion: {question}\nAnswer: {ground_truth_answer}"
        
        # Tokenize full prompt
        ppl_inputs = tokenizer(full_prompt_for_ppl, return_tensors="pt", max_length=args.max_seq_length, padding="max_length", truncation=True).to(device)
        
        # Create labels and mask the prompt part
        ppl_labels = ppl_inputs.input_ids.clone()
        
        # Tokenize prompt only to find its length for masking
        prompt_only_for_ppl = f"Biography: {biography}\nQuestion: {question}\nAnswer:"
        prompt_token_length_for_ppl = len(tokenizer(prompt_only_for_ppl, add_special_tokens=False).input_ids)
        
        # Mask the prompt part of the labels
        ppl_labels[:, :prompt_token_length_for_ppl] = -100
        
        # Calculate loss
        with torch.no_grad():
            outputs = model(input_ids=ppl_inputs.input_ids, attention_mask=ppl_inputs.attention_mask, labels=ppl_labels)
            loss = outputs.loss
            all_losses.append(loss.item())

        print(f"\n--- Sample {i+1} ---")
        print(f"Biography: {biography}")
        print(f"Question: {question}")
        print(f"Ground Truth: {ground_truth_answer}")
        print(f"Generated: {generated_answer}")

    # Calculate ROUGE metrics
    print("\n--- ROUGE Scores ---")
    rouge = evaluate.load("rouge")
    results = rouge.compute(predictions=all_generated_answers, references=all_ground_truth_answers)
    for key, value in results.items():
        print(f"{key}: {value:.4f}")

    # Calculate Perplexity
    print("\n--- Perplexity ---")
    avg_loss = sum(all_losses) / len(all_losses)
    perplexity = torch.exp(torch.tensor(avg_loss))
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Perplexity: {perplexity:.4f}")

if __name__ == "__main__":
    main()
