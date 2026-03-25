import evaluate
import os
import json
import torch
import random
from dataclasses import dataclass, field
from typing import Dict, List, Any

from .base import BaseMetric

@dataclass
class EvaluateMetricConfig:
    """Configuration for the EvaluateMetric."""
    metric_name: str = "rouge"
    sample_count: int = 10
    log_predictions: bool = False
    extra_kwargs: dict = field(default_factory=lambda: { })
    generation_input_key: str = "eval_memory_input_ids"  # column used as generation prompt (answer-free prefix)

class EvaluateMetric(BaseMetric):
    """
    A class to encapsulate evaluation metric computation using the 'evaluate' library,
    including pre and post processing for prediction generation.
    The input_ids and attention_masks are precomputed during initialization.
    """
    def __init__(self, config: EvaluateMetricConfig, tokenizer, eval_dataset: Any, output_dir: str, prompt_config):
        super().__init__(config, tokenizer, eval_dataset, output_dir)

        self.metric_evaluator = evaluate.load(self.config.metric_name)
        self.precomputed_data = self._preprocess_eval_dataset(eval_dataset)
        print(f"EvaluateMetric({self.config.metric_name}) initialized with {len(self.precomputed_data['input_ids'])} precomputed examples.")

    def _preprocess_eval_dataset(self, eval_dataset: Any) -> Dict[str, List[Any]]:
        """
        Preprocesses the evaluation dataset to extract and store necessary columns.
        This includes tokenizing, extracting answers, and questions.
        """
        print(f"Preprocessing evaluation dataset for EvaluateMetric({self.config.metric_name})...")
        input_key = self.config.generation_input_key
        attn_key = input_key.replace("input_ids", "attention_mask")
        required_cols = [input_key, attn_key, "answer", "question"]
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
            precomputed["input_ids"].append(example[input_key])
            precomputed["attention_mask"].append(example[attn_key])
            precomputed["answer"].append(example["answer"])
            if "question" in example:
                precomputed["question"].append(example["question"])
            else:
                precomputed["question"].append("N/A")

        return precomputed


    def compute_and_log_scores(self, model, state, metrics: Dict):
        """
        Generates oracle and student predictions, computes metric scores on the student,
        and logs both generations if configured.
        """
        if model is None or self.tokenizer is None:
            print(f"Skipping EvaluateMetric({self.config.metric_name}) evaluation: model or tokenizer not available.")
            return
    
        print(f"\nPerforming EvaluateMetric({self.config.metric_name}) Evaluation...")
    
        student_preds = []
        oracle_preds = []
        all_labels = []
        all_questions_for_log = []
    
        model.eval()
    
        for i in range(len(self.precomputed_data["input_ids"])):
            input_ids = torch.tensor([self.precomputed_data["input_ids"][i]]).to(model.device)
            attention_mask = torch.tensor([self.precomputed_data["attention_mask"][i]]).to(model.device)
    
            with torch.no_grad():
                oracle_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=50,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_virtual_tokens=False,
                )
                student_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=50,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_virtual_tokens=True,
                )
    
            num_input_tokens = len(input_ids[0])
    
            oracle_pred_ids = oracle_ids[0][num_input_tokens:]
            student_pred_ids = student_ids[0][num_input_tokens:]
    
            oracle_text = self.tokenizer.decode(oracle_pred_ids, skip_special_tokens=True).strip()
            student_text = self.tokenizer.decode(student_pred_ids, skip_special_tokens=True).strip()
    
            oracle_preds.append(oracle_text)
            student_preds.append(student_text)
            all_labels.append(self.precomputed_data["answer"][i])
            all_questions_for_log.append(self.precomputed_data["question"][i])
    
        # Compute metrics on the student predictions
        metric_scores = self.metric_evaluator.compute(
            predictions=student_preds,
            references=all_labels,
            **self.config.extra_kwargs
        )
    
        for key, value in metric_scores.items():
            metrics[f"eval_{key}"] = value
    
        print(f"Extrinsic EvaluateMetric({self.config.metric_name}) Scores: {metric_scores}")
    
        # ---------- Logging ----------
        if state.is_world_process_zero and self.config.log_predictions:
            log_file_path = os.path.join(self.output_dir, f"prediction_log.epoch_{int(state.epoch)}.jsonl")
            print(f"Logging predictions to {log_file_path}")
    
            with open(log_file_path, "w") as f:
                for i in range(len(student_preds)):
                    log_entry = {
                        "question": all_questions_for_log[i],
                        "ground_truth": all_labels[i],
                        "oracle_prediction": oracle_preds[i],
                        "student_prediction": student_preds[i],
                    }
                    f.write(json.dumps(log_entry) + "\n")
