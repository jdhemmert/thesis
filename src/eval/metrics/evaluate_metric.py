import evaluate
import os
import json
import torch
import torch.nn.functional as F
import random
from dataclasses import dataclass, field
from typing import Dict, List, Any

from .base import BaseMetric
from utils import generation_diagnostics as diagnostic_functions


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
        print(f"EvaluateMetric({self.config.metric_name}) initialized with {len(self.precomputed_data['student_input_ids'])} precomputed examples.")

    def _preprocess_eval_dataset(self, eval_dataset: Any) -> Dict[str, List[Any]]:
        """
        Preprocesses the evaluation dataset to extract and store necessary columns.
        This includes tokenizing, extracting answers, and questions.
        """
        print(f"Preprocessing evaluation dataset for EvaluateMetric({self.config.metric_name})...")
        student_key = self.config.generation_input_key
        student_attn_key = student_key.replace("input_ids", "attention_mask")
        oracle_key = "eval_oracle_input_ids"
        oracle_attn_key = "eval_oracle_attention_mask"
        required_cols = [student_key, student_attn_key, oracle_key, oracle_attn_key, "answer", "question"]
        if not all(col in eval_dataset.column_names for col in required_cols):
            raise ValueError(f"Required columns {required_cols} not found in dataset: {eval_dataset.column_names}.")

        precomputed = {
            "student_input_ids": [],
            "student_attention_mask": [],
            "oracle_input_ids": [],
            "oracle_attention_mask": [],
            "answer": [],
            "question": [],
        }

        dataset_size = len(eval_dataset)
        sample_indices = list(range(dataset_size))
        if self.config.sample_count < dataset_size:
            sample_indices = random.sample(sample_indices, self.config.sample_count)

        for idx in sample_indices:
            example = eval_dataset[idx]
            precomputed["student_input_ids"].append(example[student_key])
            precomputed["student_attention_mask"].append(example[student_attn_key])
            precomputed["oracle_input_ids"].append(example[oracle_key])
            precomputed["oracle_attention_mask"].append(example[oracle_attn_key])
            precomputed["answer"].append(example["answer"])
            precomputed["question"].append(example.get("question", "N/A"))

        return precomputed


    def compute_and_log_scores(self, model, state, metrics: Dict):
        if model is None or self.tokenizer is None:
            print(f"Skipping EvaluateMetric({self.config.metric_name}) evaluation: model or tokenizer not available.")
            return
    
        print(f"\nPerforming EvaluateMetric({self.config.metric_name}) Evaluation...")
    
        student_preds = []
        oracle_preds = []
        all_labels = []
        all_questions_for_log = []
        all_diagnostics = []
    
        model.eval()
    
        vp_active = (
            hasattr(model, "model")
            and getattr(model.model, "virtual_prompt", None) is not None
            and getattr(model.model, "virtual_token_count", 0) > 0
        )
    
        for i in range(len(self.precomputed_data["student_input_ids"])):
            student_input_ids = torch.tensor([self.precomputed_data["student_input_ids"][i]], device=model.device)
            student_attention_mask = torch.tensor([self.precomputed_data["student_attention_mask"][i]], device=model.device)
            oracle_input_ids = torch.tensor([self.precomputed_data["oracle_input_ids"][i]], device=model.device)
            oracle_attention_mask = torch.tensor([self.precomputed_data["oracle_attention_mask"][i]], device=model.device)
    
            student_real_ids = [
                id_ for id_, m in zip(
                    self.precomputed_data["student_input_ids"][i],
                    self.precomputed_data["student_attention_mask"][i]
                )
                if m == 1
            ]
            oracle_real_ids = [
                id_ for id_, m in zip(
                    self.precomputed_data["oracle_input_ids"][i],
                    self.precomputed_data["oracle_attention_mask"][i]
                )
                if m == 1
            ]
    
            gold_answer = self.precomputed_data["answer"][i]
    
            # Correct in-context first answer token, not standalone tokenization
            gold_first_token_id_student = diagnostic_functions.first_generated_token_id_from_prompt_plus_answer(
                self.tokenizer,
                student_real_ids,
                gold_answer,
            )
            gold_first_token_id_oracle = diagnostic_functions.first_generated_token_id_from_prompt_plus_answer(
                self.tokenizer,
                oracle_real_ids,
                gold_answer,
            )
    
            with torch.no_grad():
                oracle_ids = model.generate(
                    input_ids=oracle_input_ids,
                    attention_mask=oracle_attention_mask,
                    max_new_tokens=50,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    **({"use_virtual_tokens": False} if vp_active else {}),
                    do_sample=False,
                )
                student_ids = model.generate(
                    input_ids=student_input_ids,
                    attention_mask=student_attention_mask,
                    max_new_tokens=50,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    **({"use_virtual_tokens": True} if vp_active else {}),
                    do_sample=False,
                )
    
                oracle_diag = diagnostic_functions.next_token_diagnostics(
                    model=model,
                    tokenizer=self.tokenizer,
                    input_ids=oracle_input_ids,
                    attention_mask=oracle_attention_mask,
                    use_virtual_tokens=False,
                    gold_first_token_id=gold_first_token_id_oracle,
                    topk=10,
                )
    
                student_diag = diagnostic_functions.next_token_diagnostics(
                    model=model,
                    tokenizer=self.tokenizer,
                    input_ids=student_input_ids,
                    attention_mask=student_attention_mask,
                    use_virtual_tokens=True if vp_active else False,
                    gold_first_token_id=gold_first_token_id_student,
                    topk=10,
                )
    
                student_no_memory_diag = diagnostic_functions.next_token_diagnostics(
                    model=model,
                    tokenizer=self.tokenizer,
                    input_ids=student_input_ids,
                    attention_mask=student_attention_mask,
                    use_virtual_tokens=False,
                    gold_first_token_id=gold_first_token_id_student,
                    topk=10,
                )
    
                vt_effect_diag = diagnostic_functions.virtual_token_effect_diagnostics(
                    model=model,
                    tokenizer=self.tokenizer,
                    input_ids=student_input_ids,
                    attention_mask=student_attention_mask,
                    gold_first_token_id=gold_first_token_id_student,
                    topk=10,
                )
    
            oracle_pred_ids = oracle_ids[0][len(oracle_input_ids[0]):]
            student_pred_ids = student_ids[0][len(student_input_ids[0]):]
    
            oracle_text = self.tokenizer.decode(oracle_pred_ids, skip_special_tokens=True).strip()
            student_text = self.tokenizer.decode(student_pred_ids, skip_special_tokens=True).strip()
    
            oracle_preds.append(oracle_text)
            student_preds.append(student_text)
            all_labels.append(gold_answer)
            all_questions_for_log.append(self.precomputed_data["question"][i])
    
            all_diagnostics.append({
                "vp_active_for_eval": bool(vp_active),
    
                "gold_first_token_info": {
                    "student_prompt_gold_first_token_id": gold_first_token_id_student,
                    "student_prompt_gold_first_token_text": (
                        diagnostic_functions.safe_decode_token(self.tokenizer, gold_first_token_id_student)
                        if gold_first_token_id_student is not None else None
                    ),
                    "oracle_prompt_gold_first_token_id": gold_first_token_id_oracle,
                    "oracle_prompt_gold_first_token_text": (
                        diagnostic_functions.safe_decode_token(self.tokenizer, gold_first_token_id_oracle)
                        if gold_first_token_id_oracle is not None else None
                    ),
                },
    
                "oracle_next_token_diagnostics": oracle_diag,
                "student_next_token_diagnostics": student_diag,
                "student_no_memory_next_token_diagnostics": student_no_memory_diag,
    
                "student_memory_effect": {
                    "gold_first_token_prob_delta": (
                        None if gold_first_token_id_student is None else
                        student_diag["gold_first_token_prob"] - student_no_memory_diag["gold_first_token_prob"]
                    ),
                    "argmax_changed_vs_no_memory": (
                        student_diag["argmax_token_id"] != student_no_memory_diag["argmax_token_id"]
                    ),
                },
    
                "virtual_token_effect_diagnostics": vt_effect_diag,
    
                "generated_first_token_comparison": {
                    "oracle_generated_first_token_id": int(oracle_pred_ids[0].item()) if oracle_pred_ids.numel() > 0 else None,
                    "oracle_generated_first_token_text": diagnostic_functions.safe_decode_token(self.tokenizer, int(oracle_pred_ids[0].item())) if oracle_pred_ids.numel() > 0 else None,
                    "student_generated_first_token_id": int(student_pred_ids[0].item()) if student_pred_ids.numel() > 0 else None,
                    "student_generated_first_token_text": diagnostic_functions.safe_decode_token(self.tokenizer, int(student_pred_ids[0].item())) if student_pred_ids.numel() > 0 else None,
                    "student_generated_first_token_matches_gold": (
                        None if gold_first_token_id_student is None or student_pred_ids.numel() == 0 else
                        bool(int(student_pred_ids[0].item()) == gold_first_token_id_student)
                    ),
                },
            })
    
        metric_scores = self.metric_evaluator.compute(
            predictions=student_preds,
            references=all_labels,
            **self.config.extra_kwargs
        )
    
        for key, value in metric_scores.items():
            metrics[f"eval_{key}"] = value
    
        print(f"Extrinsic EvaluateMetric({self.config.metric_name}) Scores: {metric_scores}")
    
        if state.is_world_process_zero and self.config.log_predictions:
            log_file_path = os.path.join(self.output_dir, f"prediction_log.epoch_{int(state.epoch)}.jsonl")
            print(f"Logging predictions to {log_file_path}")
    
            with open(log_file_path, "w") as f:
                for i in range(len(student_preds)):
                    oracle_real_ids = [
                        id_ for id_, m in zip(
                            self.precomputed_data["oracle_input_ids"][i],
                            self.precomputed_data["oracle_attention_mask"][i]
                        )
                        if m == 1
                    ]
                    student_real_ids = [
                        id_ for id_, m in zip(
                            self.precomputed_data["student_input_ids"][i],
                            self.precomputed_data["student_attention_mask"][i]
                        )
                        if m == 1
                    ]
    
                    log_entry = {
                        "question": all_questions_for_log[i],
                        "ground_truth": all_labels[i],
                        "oracle_context": self.tokenizer.decode(oracle_real_ids, skip_special_tokens=True),
                        "oracle_prediction": oracle_preds[i],
                        "student_context": self.tokenizer.decode(student_real_ids, skip_special_tokens=True),
                        "student_prediction": student_preds[i],
                        **all_diagnostics[i],
                    }
                    f.write(json.dumps(log_entry) + "\n")