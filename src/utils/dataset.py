from datasets import load_dataset
from typing import Optional
import torch


# Helper function for fine-tuning preprocessing
def _preprocess_for_finetune(examples, tokenizer, max_length):
    """Prepares a dataset for standard supervised fine-tuning."""
    full_prompts = [f"Biography: {bio}\nQuestion: {q}\nAnswer: {a}" for bio, q, a in zip(examples["biography"], examples["question"], examples["answer"])]
    model_inputs = tokenizer(full_prompts, max_length=max_length, padding="max_length", truncation=True)

    labels = [row[:] for row in model_inputs["input_ids"]]
    prompt_only = [f"Biography: {bio}\nQuestion: {q}\nAnswer:" for bio, q in zip(examples["biography"], examples["question"])]
    prompt_token_lengths = [len(tokenizer(p, add_special_tokens=False).input_ids) for p in prompt_only]

    for i in range(len(labels)):
        prompt_len = prompt_token_lengths[i]
        labels[i][:prompt_len] = [-100] * prompt_len

    model_inputs["labels"] = labels
    return model_inputs

# Helper function for self-distillation (memory) preprocessing
def _preprocess_for_self_distillation(examples, tokenizer, max_length):
    """Prepares a dataset for self-distillation between an oracle and memory model."""
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

def load_dataset_for_task(
    task_type: str,
    dataset_path: str,
    tokenizer,
    max_length: int,
    sample_n: Optional[int] = None,
    sample_strategy: str = 'first_n',
    seed: int = 42,
    drop_text_columns: bool = True,
):
    """
    Loads, samples, and preprocesses a dataset for a specific training task.

    Args:
        task_type (str): The type of task to prepare data for ('finetune' or 'self_distillation').
        dataset_path (str): Path to the JSON dataset file.
        tokenizer: The tokenizer instance.
        max_length (int): The maximum sequence length for tokenization.
        sample_n (Optional[int]): Number of samples to use. Defaults to None.
        sample_strategy (str): How to sample ('random' or 'first_n'). Defaults to 'first_n'.
        seed (int): Random seed for sampling. Defaults to 42.
        drop_text_columns (bool): If True, removes original text columns after preprocessing. Defaults to True.

    Returns:
        A preprocessed Hugging Face Dataset object.
    """
    # 1. Select preprocessing function
    if task_type == 'finetune':
        preprocess_fn = _preprocess_for_finetune
    elif task_type == 'self_distillation':
        preprocess_fn = _preprocess_for_self_distillation
    else:
        raise ValueError(f"Unknown task_type: {task_type}")

    # 2. Load the raw dataset
    dataset = load_dataset("json", data_files=dataset_path)["train"]
    original_columns = list(dataset.column_names)

    # 3. Handle sampling
    if sample_n:
        if sample_strategy == 'random':
            dataset = dataset.shuffle(seed=seed).select(range(sample_n))
        else: # first_n
            dataset = dataset.select(range(sample_n))

    # 4. Apply the preprocessing function
    tokenized_dataset = dataset.map(
        lambda examples: preprocess_fn(examples, tokenizer=tokenizer, max_length=max_length),
        batched=True,
        remove_columns=original_columns if drop_text_columns else None
    )

    return tokenized_dataset


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
        
        # The meta strategy doesn't need labels, but the sequential one does for the Trainer API.
        # The collator handles both cases by checking for the presence of the 'labels' key.
        if "labels" in features[0]:
            labels_batch = self.tokenizer.pad(
                {"input_ids": [f["labels"] for f in features]},
                padding='max_length', max_length=max_len, return_tensors="pt"
            )
            batch["labels"] = labels_batch["input_ids"]

        return batch