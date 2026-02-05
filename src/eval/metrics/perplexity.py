import torch
import torch.nn.functional as F
import os
import json
from dataclasses import dataclass, field
from typing import Dict, List, Any, Tuple

from omegaconf import DictConfig, OmegaConf
from transformers import PreTrainedTokenizer, PreTrainedModel

from src.eval.metrics.base import BaseMetric, BaseMetricLoader, MetricFactory
from src.config_schemas import PromptConfig


@dataclass
class PerplexityMetricConfig:
    """Configuration for the PerplexityMetric."""
    batch_size: int = 8
    prompt_format_key: str = "contextual_qa_training"
    context_key: str = "biography"
    question_key: str = "question"
    answer_key: str = "answer"
    max_length: int = 1024
    log_predictions: bool = True


import math

class PerplexityMetric(BaseMetric):
    """
    Calculates the perplexity of an answer conditioned on a question and context,
    using the model being evaluated.
    """
    def __init__(self, config: PerplexityMetricConfig, tokenizer: PreTrainedTokenizer, eval_dataset: Any, output_dir: str, prompt_config: PromptConfig):
        super().__init__(config, tokenizer, eval_dataset, output_dir)
        self.config: PerplexityMetricConfig = config
        self.prompt_config = prompt_config
        self.all_predictions: List[Dict[str, Any]] = []

    def _get_prompt_parts(self, example: Dict[str, Any]) -> Tuple[str, str]:
        """
        Formats the prompt and identifies the question (input) and answer (target) parts.
        Returns: (question_part_string, answer_part_string)
        """
        template = getattr(self.prompt_config, self.config.prompt_format_key)
        
        context = example.get(self.config.context_key, "")
        question = example.get(self.config.question_key, "")
        answer = example.get(self.config.answer_key, "")
        
        if "{answer}" not in template:
            raise ValueError(f"Prompt template '{self.config.prompt_format_key}' must contain '{{answer}}' for perplexity calculation.")

        question_part = template.split("{answer}")[0].format(biography=context, question=question)
        
        return question_part, answer

    def compute_and_log_scores(self, model: PreTrainedModel, state: Any, metrics: Dict):
        print(f"\nPerforming PerplexityMetric Evaluation...")
        
        model.eval()
        total_loss = 0.0
        total_examples = 0
        
        with torch.no_grad():
            for i in range(len(self.eval_dataset)):
                example = self.eval_dataset[i]
                question_part, answer_part = self._get_prompt_parts(example)

                # Use the main tokenizer from the base class
                question_tokenized = self.tokenizer(question_part, return_tensors='pt', add_special_tokens=True)
                full_text = question_part + answer_part
                full_tokenized = self.tokenizer(full_text, return_tensors='pt', max_length=self.config.max_length, truncation=True, add_special_tokens=True)
                
                input_ids = full_tokenized.input_ids.to(model.device)
                attention_mask = full_tokenized.attention_mask.to(model.device)
                
                # Create labels where the prompt part is masked
                labels = input_ids.clone()
                prompt_len = question_tokenized.input_ids.shape[1]
                
                # Ensure prompt_len doesn't exceed the truncated full length
                if prompt_len >= labels.shape[1]:
                    continue # Skip if the prompt itself was truncated to max_length

                labels[:, :prompt_len] = -100

                # Skip if there are no label tokens
                if (labels == -100).all():
                    continue
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                
                if not torch.isnan(loss):
                    total_loss += loss.item()
                    total_examples += 1
                
                if self.config.log_predictions:
                    perplexity = torch.exp(loss).item() if not torch.isnan(loss) else None
                    self.all_predictions.append({
                        "example_idx": i,
                        "question_part": question_part,
                        "answer_part": answer_part,
                        "perplexity": perplexity,
                    })

        avg_loss = total_loss / total_examples if total_examples > 0 else float('nan')
        avg_perplexity = math.exp(avg_loss) if not math.isnan(avg_loss) else float('nan')
        
        metrics[f"eval_perplexity"] = avg_perplexity
        print(f"\nAggregated PerplexityMetric Scores: Avg Perplexity={avg_perplexity:.4f}")

        if self.config.log_predictions:
            predictions_path = os.path.join(self.output_dir, "perplexity_predictions.json")
            with open(predictions_path, "w") as f:
                json.dump(self.all_predictions, f, indent=4)
            print(f"Logged individual perplexity predictions to {predictions_path}")
