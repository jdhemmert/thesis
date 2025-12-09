import torch
from torch.func import functional_call, grad
from torch.optim import Adam
from tqdm import tqdm
from collections import defaultdict
import random

from src.training.losses import self_distillation_loss

def prepare_meta_dataset(dataset, tokenizer, cfg):
    """
    Groups the dataset by biography to create meta-learning tasks.
    Each task consists of a support set and a query set of questions about a single biography.
    """
    tasks = defaultdict(list)
    for example in dataset:
        tasks[example['biography']].append(example)
    
    meta_dataset = []
    for biography, examples in tasks.items():
        if len(examples) < 2:
            continue # Need at least one for support and one for query
        
        random.shuffle(examples)
        split_point = len(examples) // 2
        support_examples = examples[:split_point]
        query_examples = examples[split_point:]

        def preprocess_and_collate(example_list):
            # Simplified transform_to_prompts and preprocess_function
            oracle_prompts = [f"Biography: {ex['biography']}\nQuestion: {ex['question']}" for ex in example_list]
            memory_prompts = [f"Question: {ex['question']}" for ex in example_list]

            oracle_inputs = tokenizer(oracle_prompts, padding=True, truncation=True, return_tensors="pt")
            memory_inputs = tokenizer(memory_prompts, padding=True, truncation=True, return_tensors="pt")

            return {
                "oracle_input_ids": oracle_inputs.input_ids,
                "oracle_attention_mask": oracle_inputs.attention_mask,
                "memory_input_ids": memory_inputs.input_ids,
                "memory_attention_mask": memory_inputs.attention_mask,
            }

        support_set = preprocess_and_collate(support_examples)
        query_set = preprocess_and_collate(query_examples)
        
        meta_dataset.append({"support": support_set, "query": query_set})
        
    return meta_dataset

def train_meta(cfg, model, dataset, tokenizer):
    """
    Implements the meta-learning training strategy using torch.func.
    """
    # 1. Prepare Meta-Dataset
    meta_dataset = prepare_meta_dataset(dataset, tokenizer, cfg)

    # 2. Parameter Separation
    outer_loop_params = {}
    inner_loop_params = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'lora_adapter' in name:
            outer_loop_params[name] = param
        elif 'memory_bank' in name:
            inner_loop_params[name] = param

    # 3. Optimizer for the Outer Loop (LoRA parameters)
    outer_optimizer = Adam(outer_loop_params.values(), lr=cfg.task.strategy.meta_training.outer_learning_rate)

    # 4. Meta-Training Loop
    for epoch in range(cfg.task.num_train_epochs):
        print(f"Meta-Epoch {epoch+1}/{cfg.task.num_train_epochs}")
        
        for task in tqdm(meta_dataset, desc="Processing tasks"):
            support_set = {k: v.to(model.device) for k, v in task['support'].items()}
            query_set = {k: v.to(model.device) for k, v in task['query'].items()}
            
            # --- Inner Loop (Task Adaptation) ---
            
            adapted_params = {name: p.clone() for name, p in inner_loop_params.items()}

            for _ in range(cfg.task.strategy.meta_training.num_inner_steps):
                functional_params = {**outer_loop_params, **adapted_params}

                def compute_loss_stateless(params):
                    oracle_outputs = functional_call(model, params, (support_set['oracle_input_ids'], support_set['oracle_attention_mask']))
                    memory_outputs = functional_call(model, params, (support_set['memory_input_ids'], support_set['memory_attention_mask']))
                    
                    loss = self_distillation_loss(
                        oracle_logits=oracle_outputs.logits,
                        memory_logits=memory_outputs.logits,
                        temperature=cfg.task.loss.temperature,
                        alpha=cfg.task.loss.alpha,
                        loss_type=cfg.task.loss.loss_type
                    )
                    return loss

                inner_grads = grad(compute_loss_stateless, adapted_params, create_graph=True)
                
                for name in adapted_params:
                    adapted_params[name] = adapted_params[name] - cfg.task.strategy.meta_training.inner_learning_rate * inner_grads[name]

            # --- Outer Loop (Meta-Optimization) ---
            
            outer_optimizer.zero_grad()
            
            final_functional_params = {**outer_loop_params, **adapted_params}
            
            def compute_meta_loss_stateless(params):
                oracle_outputs = functional_call(model, params, (query_set['oracle_input_ids'], query_set['oracle_attention_mask']))
                memory_outputs = functional_call(model, params, (query_set['memory_input_ids'], query_set['memory_attention_mask']))

                meta_loss = self_distillation_loss(
                    oracle_logits=oracle_outputs.logits,
                    memory_logits=memory_outputs.logits,
                    temperature=cfg.task.loss.temperature,
                    alpha=cfg.task.loss.alpha,
                    loss_type=cfg.task.loss.loss_type
                )
                return meta_loss

            outer_loss = compute_meta_loss_stateless(final_functional_params)
            
            outer_loss.backward()
            outer_optimizer.step()
            
        print(f"End of Meta-Epoch {epoch+1}. Last meta-loss: {outer_loss.item()}")

    print("Meta-training finished.")
    # TODO: Implement model saving
    # model.save_pretrained(cfg.task.output_dir)
