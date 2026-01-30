import evaluate
import os
import json
import torch
import random
from dataclasses import dataclass, field
from typing import Dict, List, Any

from .base import BaseMetric

@dataclass
class RougeMetricConfig:
    """Configuration for the RougeMetric."""
    sample_count: int = 10
    log_predictions: bool = False

class RougeMetric(BaseMetric):
    """
    A class to encapsulate ROUGE metric computation, including pre and post processing.
    The input_ids and attention_masks are precomputed during initialization.
    """
    def __init__(self, config: RougeMetricConfig, tokenizer, eval_dataset: Any, output_dir: str):
        super().__init__(config, tokenizer, eval_dataset, output_dir)
        
        self.precomputed_data = self._preprocess_eval_dataset(eval_dataset)
        print(f"RougeMetric initialized with {len(self.precomputed_data['input_ids'])} precomputed examples.")

    def _preprocess_eval_dataset(self, eval_dataset: Any) -> Dict[str, List[Any]]:
        """
        Preprocesses the evaluation dataset to extract and store necessary columns.
        This includes tokenizing, extracting answers, and questions.
        """
        print("Preprocessing evaluation dataset for RougeMetric...")
        required_cols = ["memory_input_ids", "memory_attention_mask", "answer", "question"]
        if not all(col in eval_dataset.column_names for col in required_cols):
            raise ValueError(f"Required columns {required_cols} not found in dataset: {eval_dataset.column_names}.")

        precomputed = {
            "input_ids": [],
            "attention_mask": [],
            "answer": [],
            "question": []
        }
        
        dataset_size = len(eval_dataset)
        sample_indices = list(range(dataset_size))
        if self.config.sample_count < dataset_size:
            sample_indices = random.sample(sample_indices, self.config.sample_count)
        
        for idx in sample_indices:
            example = eval_dataset[idx]
            precomputed["input_ids"].append(example['memory_input_ids'])
            precomputed["attention_mask"].append(example['memory_attention_mask'])
            precomputed["answer"].append(example["answer"])
            if "question" in example:
                precomputed["question"].append(example["question"])
            else:
                precomputed["question"].append("N/A")

        return precomputed


    def compute_and_log_scores(self, model, state, metrics: Dict):
        """
        Generates predictions, computes ROUGE scores, and logs predictions if configured.
        """
        if model is None or self.tokenizer is None:
            print("Skipping ROUGE evaluation: model or tokenizer not available.")
            return

        print("\nPerforming ROUGE Evaluation...")
        all_preds = []
        all_labels = []
        all_questions_for_log = []

        model.eval()
        for i in range(len(self.precomputed_data["input_ids"])):
            input_ids = torch.tensor([self.precomputed_data['input_ids'][i]]).to(model.device)
            attention_mask = torch.tensor([self.precomputed_data['attention_mask'][i]]).to(model.device)

            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=50,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )

            num_input_tokens = len(input_ids[0])
            pred_ids = generated_ids[0][num_input_tokens:]
            pred_text = self.tokenizer.decode(pred_ids, skip_special_tokens=True).strip()

            all_preds.append(pred_text)
            all_labels.append(self.precomputed_data["answer"][i])
            all_questions_for_log.append(self.precomputed_data["question"][i])

        rouge_metric_evaluator = evaluate.load('rouge')
        rouge_scores = rouge_metric_evaluator.compute(predictions=all_preds, references=all_labels)

        for key, value in rouge_scores.items():
            metrics[f"eval_{key}"] = value

        print(f"Extrinsic ROUGE Scores: {rouge_scores}")

        if state.is_world_process_zero and self.config.log_predictions:
            log_file_path = os.path.join(self.output_dir, f"prediction_log.epoch_{int(state.epoch)}.jsonl")
            print(f"Logging predictions to {log_file_path}")
            with open(log_file_path, "w") as f:
                for i in range(len(all_preds)):
                    log_entry = {
                        "question": all_questions_for_log[i],
                        "ground_truth": all_labels[i],
                        "prediction": all_preds[i]
                    }
                    f.write(json.dumps(log_entry) + "\n")
