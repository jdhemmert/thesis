from datasets import load_dataset
import os
import math

def get_tokenized_datasets(cfg, tokenizer):
    def preprocess_function(examples):
        # Combine prompt and answer, then tokenize
        full_prompts = [f"Biography: {bio}\nQuestion: {q}\nAnswer: {a}" for bio, q, a in zip(examples["biography"], examples["question"], examples["answer"])]
        model_inputs = tokenizer(full_prompts, max_length=cfg.task.max_seq_length, padding="max_length", truncation=True)

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

    max_steps = -1
    if cfg.dataset.streaming:
        full_dataset = load_dataset("json", data_files=cfg.dataset.dataset_file, streaming=True)["train"]
        shuffled_dataset = full_dataset.shuffle(seed=cfg.task.seed, buffer_size=10000) # for reproducibility

        # train_test_split is not available for streaming datasets.
        # We'll manually split it.
        def get_dataset_size(path):
            with open(path) as f:
                for i, _ in enumerate(f):
                    pass
            return i + 1

        dataset_size = get_dataset_size(cfg.dataset.dataset_file)
        test_size = int(dataset_size * cfg.dataset.test_split_ratio)

        test_dataset = shuffled_dataset.take(test_size)
        train_dataset = shuffled_dataset.skip(test_size)

        train_tokenized_dataset = train_dataset.map(preprocess_function, batched=True)
        test_tokenized_dataset = test_dataset.map(preprocess_function, batched=True)

        # calculate max_steps for streaming dataset
        train_size = dataset_size - test_size
        effective_batch_size = cfg.dataset.physical_batch_size * cfg.dataset.accumulation_steps
        max_steps = math.ceil(train_size / effective_batch_size) * cfg.task.epochs
    else:
        full_dataset = load_dataset("json", data_files=cfg.dataset.dataset_file)["train"]
        shuffled_dataset = full_dataset.shuffle(seed=cfg.task.seed) # for reproducibility
        split_dataset = shuffled_dataset.train_test_split(test_size=cfg.dataset.test_split_ratio)
        train_dataset = split_dataset["train"]
        test_dataset = split_dataset["test"]

        train_tokenized_dataset = train_dataset.map(preprocess_function, batched=True, num_proc=4)
        test_tokenized_dataset = test_dataset.map(preprocess_function, batched=True, num_proc=4)

    return train_tokenized_dataset, test_tokenized_dataset, max_steps