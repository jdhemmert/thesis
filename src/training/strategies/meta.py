import torch
from torch.func import functional_call
from torch.autograd import grad
from torch.optim import Adam
from tqdm import tqdm
from collections import defaultdict
import random

from src.training.losses import self_distillation_loss
from src.utils.dataset import SelfDistillationDataCollator

def prepare_meta_dataset(dataset, tokenizer):
    """
    Groups a pre-processed dataset by biography to create meta-learning tasks.
    Each task consists of a support set and a query set of questions about a single biography.
    """
    tasks = defaultdict(list)
    for example in dataset:
        tasks[example['biography']].append(example)
    
    meta_dataset = []
    collator = SelfDistillationDataCollator(tokenizer)

    for biography, examples in tasks.items():
        if len(examples) < 2:
            continue
        
        random.shuffle(examples)
        split_point = len(examples) // 2
        support_examples = examples[:split_point]
        query_examples = examples[split_point:]

        support_set = collator(support_examples)
        query_set = collator(query_examples)
        
        meta_dataset.append({"support": support_set, "query": query_set})
        
    return meta_dataset

def train_meta(cfg, model, dataset, tokenizer):
    """
    Implements the meta-learning training strategy using torch.func.
    """
    meta_dataset = prepare_meta_dataset(dataset, tokenizer)

    outer_loop_params = {}
    inner_loop_params = {}
    print("--- Trainable Parameters ---")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
        if not param.requires_grad:
            continue
        if 'lora_adapter' in name:
            outer_loop_params[name] = param
        elif 'memory_bank' in name:
            inner_loop_params[name] = param
    print("--- End Trainable Parameters ---")
    print(f"Found {len(outer_loop_params)} outer loop params and {len(inner_loop_params)} inner loop params.")

    outer_optimizer = Adam(outer_loop_params.values(), lr=cfg.task.strategy.meta_training.outer_learning_rate)

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

                inner_loss = compute_loss_stateless(adapted_params)
                grad_inputs = tuple(adapted_params.values())
                inner_grads_tuple = grad(inner_loss, grad_inputs, create_graph=True, allow_unused=True)
                inner_grads = dict(zip(adapted_params.keys(), inner_grads_tuple))

                for name in adapted_params:
                    if inner_grads[name] is not None:
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

    model.save_pretrained(cfg.task.output_dir)
