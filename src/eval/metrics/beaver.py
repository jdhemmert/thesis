import torch
import torch.nn.functional as F
import math
import heapq
import os
import json
from dataclasses import dataclass, field
from typing import Dict, List, Any, Tuple, Callable

from omegaconf import DictConfig

from .base import BaseMetric


class BaseConstraint(ABC):
    """Abstract base class for all semantic constraints."""
    def __init__(self, tokenizer: Any, **kwargs):
        self.tokenizer = tokenizer

    @abstractmethod
    def __call__(self, sequence_token_ids: List[int], **kwargs) -> bool:
        """
        Evaluates if a given token sequence satisfies the constraint.
        Must be prefix-closed.
        """
        raise NotImplementedError

class ConstraintRegistry:
    _registry: Dict[str, Type[BaseConstraint]] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(constraint_class: Type[BaseConstraint]):
            cls._registry[name] = constraint_class
            return constraint_class
        return decorator

    @classmethod
    def get_constraint(cls, name: str, tokenizer: Any, **kwargs) -> BaseConstraint:
        constraint_class = cls._registry.get(name)
        if constraint_class is None:
            raise ValueError(f"Constraint '{name}' not registered. Available: {list(cls._registry.keys())}")
        return constraint_class(tokenizer=tokenizer, **kwargs)

CONSTRAINT_REGISTRY = ConstraintRegistry()


@CONSTRAINT_REGISTRY.register("answer_in_k")
class AnswerInKConstraint(BaseConstraint):
    """
    Constraint: Checks if the correct answer can appear within the first k tokens of the response.
    This is a prefix-closed constraint.
    """
    def __init__(self, tokenizer: Any, correct_answer_tokens: List[int], k_tokens: int):
        super().__init__(tokenizer)
        self.correct_answer_tokens = correct_answer_tokens
        self.k_tokens = k_tokens
        self.correct_answer_len = len(correct_answer_tokens)

    def __call__(self, sequence_token_ids: List[int], **kwargs) -> bool:
        current_len = len(sequence_token_ids)

        # Not present in k tokens
        if current_len > self.k_tokens and not self._contains_answer(sequence_token_ids):
            return False

        # Present in k tokens
        if self._contains_answer(sequence_token_ids) and current_len <= self.k_tokens:
            return True
        
        # Undecided
        if current_len <= self.k_tokens:
            return True
        
        return False
    
    def _contains_answer(self, sequence_token_ids: List[int]) -> bool:
        """Helper to check if the correct answer tokens are fully present in the sequence."""
        seq_len = len(sequence_token_ids)
        ans_len = self.correct_answer_len
        
        if ans_len == 0:
            return True

        if seq_len < ans_len:
            return False

        for i in range(seq_len - ans_len + 1):
            if sequence_token_ids[i:i+ans_len] == self.correct_answer_tokens:
                return True
        return False


@dataclass
class BeaverMetricConfig:
    """Configuration for the BeaverMetric."""
    budget: int = 100                  # Max forward passes (expansions) per prompt
    epsilon: float = 0.01              # Convergence threshold for PUB - PLB
    selection_strategy: str = "max_mu" # Heuristic for selecting next node to expand
    max_sequence_length: int = 50      # Max tokens a sequence can generate
    sample_count: int = 10
    
    constraint_type: str = "answer_in_k"
    correct_answer_dataset_key: str = "answer"
    k_tokens: int = 50


class BeaverMetric(BaseMetric):
    """
    Implements the BEAVER algorithm for deterministic LLM verification as an evaluation metric.
    Calculates sound probability bounds [PLB, PUB] that a model's output satisfies a given constraint.
    """
    def __init__(self, config: BeaverMetricConfig, tokenizer: Any, eval_dataset: Any, output_dir: str):
        super().__init__(config, tokenizer, eval_dataset, output_dir)
        
        self.precomputed_prompts = self._preprocess_eval_dataset(eval_dataset)
        
        print(f"BeaverMetric initialized for '{self.config.constraint_type}' constraint with {len(self.precomputed_prompts['input_ids'])} examples.")

    def _preprocess_eval_dataset(self, eval_dataset: Any) -> Dict[str, List[Any]]:
        """
        Preprocesses the evaluation dataset to extract prompts and correct answer tokens.
        """
        print(f"Preprocessing evaluation dataset for BeaverMetric...")
        required_cols = ["memory_input_ids", "memory_attention_mask", self.config.correct_answer_dataset_key, "question"]
        if not all(col in eval_dataset.column_names for col in required_cols):
            raise ValueError(f"Required columns {required_cols} not found in dataset: {eval_dataset.column_names}.")

        precomputed = {
            "initial_input_ids": [],      # The tokenized prompt for the model
            "initial_attention_mask": [],
            "correct_answer_tokens": [],  # Tokenized version of the correct answer string
            "original_question": []
        }
        
        for idx in range(self.config.sample_count):
            example = eval_dataset[idx]
            precomputed["initial_input_ids"].append(example['memory_input_ids'])
            precomputed["initial_attention_mask"].append(example['memory_attention_mask'])
            
            correct_answer_str = example[self.config.correct_answer_dataset_key]
            precomputed["correct_answer_tokens"].append(self.tokenizer.encode(correct_answer_str, add_special_tokens=False))
            
            precomputed["original_question"].append(example.get("question", "N/A"))

        return precomputed


    def compute_and_log_scores(self, model: Any, state: Any, metrics: Dict):
        """
        Runs the BEAVER algorithm for each preprocessed prompt, computes [PLB, PUB] bounds,
        and logs the results.
        """
        if model is None or self.tokenizer is None:
            print("Skipping BeaverMetric evaluation: model or tokenizer not available.")
            return

        print(f"\nPerforming BeaverMetric({self.config.constraint_type}) Evaluation...")

        model.eval()
        
        all_plb = []
        all_pub = []

        for i in range(len(self.precomputed_prompts["initial_input_ids"])):
            initial_input_ids = self.precomputed_prompts["initial_input_ids"][i]
            initial_attention_mask = self.precomputed_prompts["initial_attention_mask"][i]
            correct_answer_tokens = self.precomputed_prompts["correct_answer_tokens"][i]
            original_question = self.precomputed_prompts["original_question"][i]

            semantic_constraint_func = CONSTRAINT_REGISTRY.get_constraint(
                self.config.constraint_type,
                tokenizer=self.tokenizer,
                correct_answer_tokens=correct_answer_tokens,
                k_tokens=self.config.k_tokens
            )

            plb, pub = self._run_beaver_for_prompt(
                model=model,
                initial_input_ids=initial_input_ids,
                initial_attention_mask=initial_attention_mask,
                semantic_constraint=semantic_constraint_func,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                max_sequence_length=self.config.max_sequence_length,
                budget=self.config.budget,
                epsilon=self.config.epsilon
            )
            all_plb.append(plb)
            all_pub.append(pub)
            print(f"  Prompt {i+1}: Q: '{original_question}' Bounds: [{plb:.4f}, {pub:.4f}] Gap: {(pub-plb):.4f}")

        avg_plb = sum(all_plb) / len(all_plb) if all_plb else 0.0
        avg_pub = sum(all_pub) / len(all_pub) if all_pub else 0.0
        avg_gap = avg_pub - avg_plb

        metrics[f"eval_beaver_plb"] = avg_plb
        metrics[f"eval_beaver_pub"] = avg_pub
        metrics[f"eval_beaver_gap"] = avg_gap
        
        print(f"\nAggregated BeaverMetric({self.config.constraint_type}) Scores: Avg PLB={avg_plb:.4f}, Avg PUB={avg_pub:.4f}, Avg Gap={avg_gap:.4f}")


    def _run_beaver_for_prompt(
        self, 
        model: Any,
        initial_input_ids: List[int],
        initial_attention_mask: List[int],
        semantic_constraint: BaseConstraint,
        eos_token_id: int,
        pad_token_id: int,
        max_sequence_length: int,
        budget: int,
        epsilon: float
    ) -> Tuple[float, float]:
        """
        Implements the core BEAVER algorithm for a single prompt.
        """
        plb = 0.0
        pub = 1.0 # Initial loose upper bound
        
        # Frontier: max-heap storing (-log_prob, current_token_ids_tuple)
        # Using tuple for token_ids to make it hashable for set operations if needed,
        # and to ensure heap order is by probability.
        frontier_incomplete = [] 
        
        # Initial state: the prompt itself
        initial_prompt_tensor = torch.tensor([initial_input_ids], device=model.device)
        initial_prompt_len = len(initial_input_ids)
        
        # Calculate initial log probability of the prompt (for consistency, though it's 0.0 relative to itself)
        # Model forward pass to get logits for the initial prompt (if needed, otherwise start with 0.0 log_prob)
        # For simplicity, we assume the initial prompt itself has 0.0 log_prob for starting the search.
        heapq.heappush(frontier_incomplete, (0.0, tuple(initial_input_ids))) # (negative log prob, token_ids)

        # To track sequences whose probability has been added to PLB
        frontier_complete_probs = []

        current_budget = 0

        while frontier_incomplete and current_budget < budget and (pub - plb) > epsilon:
            neg_log_prob, current_token_ids_tuple = heapq.heappop(frontier_incomplete)
            current_log_prob = -neg_log_prob
            current_token_ids = list(current_token_ids_tuple)
            
            # Check if current sequence already exceeds max_sequence_length (excluding prompt)
            if (len(current_token_ids) - initial_prompt_len) >= max_sequence_length:
                continue # Cannot expand further

            # Query the model for next token probabilities
            # The input to the model should be the current sequence of token_ids
            model_input_ids = torch.tensor([current_token_ids], device=model.device)
            model_attention_mask = torch.tensor([initial_attention_mask[:len(current_token_ids)]], device=model.device)
            # Pad attention mask if current_token_ids is longer than initial_attention_mask
            if len(current_token_ids) > len(initial_attention_mask):
                # This should ideally be handled by padding the initial_attention_mask
                # or ensuring proper construction of attention mask for current_token_ids
                model_attention_mask = torch.ones_like(model_input_ids) # Simple fix for now

            with torch.no_grad():
                outputs = model(input_ids=model_input_ids, attention_mask=model_attention_mask)
                logits = outputs.logits # Logits are (batch_size, sequence_length, vocab_size)
            
            # Get logits for the last token in the sequence to predict the next token
            next_token_logits = logits[0, -1, :] # (vocab_size,)
            next_token_log_probs = F.log_softmax(next_token_logits, dim=-1) # Log probabilities
            
            current_budget += 1

            # Iterate over all possible next tokens
            for next_token_id in range(self.tokenizer.vocab_size):
                next_log_prob_val = next_token_log_probs[next_token_id].item()
                new_log_prob = current_log_prob + next_log_prob_val
                new_token_ids = current_token_ids + [next_token_id]

                # Apply prefix-closed constraint
                if not semantic_constraint(new_token_ids):
                    continue # Prune this branch
                
                # Check for EOS token
                if next_token_id == eos_token_id:
                    # Sequence is complete and valid
                    frontier_complete_probs.append(math.exp(new_log_prob))
                elif (len(new_token_ids) - initial_prompt_len) < max_sequence_length:
                    # Sequence is incomplete, valid, and within max length
                    heapq.heappush(frontier_incomplete, (-new_log_prob, tuple(new_token_ids)))
            
            # Update PLB and PUB
            plb = sum(frontier_complete_probs)
            pub_incomplete = sum(math.exp(-item[0]) for item in frontier_incomplete) # Sum of probabilities in incomplete frontier
            pub = plb + pub_incomplete

        return plb, pub
