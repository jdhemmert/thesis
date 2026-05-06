import os
import math
import json
from typing import Dict, Any
from dataclasses import dataclass

import torch

from transformers import PreTrainedTokenizer, PreTrainedModel

from src.eval.metrics.base import BaseMetric, BaseMetricLoader


@dataclass
class LMPerplexityMetricConfig:
    text_key: str = "answer"
    max_length: int = 512
    log_predictions: bool = True


class LMPerplexityMetric(BaseMetric):
    """
    Full-sequence LM perplexity with no prompt masking. Intended for wikitext-style
    evaluation where the entire input is the prediction target.
    """
    def __init__(self, config: LMPerplexityMetricConfig, tokenizer: PreTrainedTokenizer, eval_dataset: Any, output_dir: str):
        super().__init__(config, tokenizer, eval_dataset, output_dir)
        self.config: LMPerplexityMetricConfig = config
        self.all_predictions = []

    def compute_and_log_scores(self, model: PreTrainedModel, state: Any, metrics: Dict):
        print(f"\nPerforming LMPerplexityMetric Evaluation...")

        model.eval()
        total_loss = 0.0
        total_examples = 0

        with torch.no_grad():
            for i in range(len(self.eval_dataset)):
                example = self.eval_dataset[i]
                text = example.get(self.config.text_key, "")
                if not text:
                    continue

                tokenized = self.tokenizer(
                    text,
                    return_tensors="pt",
                    max_length=self.config.max_length,
                    truncation=True,
                    add_special_tokens=True,
                )
                input_ids = tokenized.input_ids.to(model.device)

                if input_ids.shape[1] < 2:
                    continue

                outputs = model(input_ids=input_ids, labels=input_ids)
                loss = outputs.loss

                if not torch.isnan(loss):
                    total_loss += loss.item()
                    total_examples += 1

                if self.config.log_predictions:
                    self.all_predictions.append({
                        "example_idx": i,
                        "perplexity": torch.exp(loss).item() if not torch.isnan(loss) else None,
                    })

        avg_loss = total_loss / total_examples if total_examples > 0 else float("nan")
        avg_perplexity = math.exp(avg_loss) if not math.isnan(avg_loss) else float("nan")

        metrics["eval_lm_perplexity"] = avg_perplexity
        print(f"\nLMPerplexityMetric: Avg Perplexity={avg_perplexity:.4f} over {total_examples} examples")

        if self.config.log_predictions:
            path = os.path.join(self.output_dir, "lm_perplexity_predictions.json")
            with open(path, "w") as f:
                json.dump(self.all_predictions, f, indent=4)
            print(f"Logged predictions to {path}")
