import os
import json
import torch
import evaluate
from transformers import TrainerCallback

class ExtrinsicValidationCallback(TrainerCallback):
    
    def __init__(self, eval_dataset, tokenizer, cfg_task):
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.cfg_task = cfg_task
        self.validation_policy = cfg_task.extrinsic_validation or "always"

    def on_evaluate(self, args, state, control, **kwargs):
        if self.validation_policy == "always":
            self._run_validation(state, **kwargs)

    def on_train_end(self, args, state, control, **kwargs):
        if self.validation_policy == "end" or self.validation_policy == "always":
            self._run_validation(state, **kwargs)

    def _run_validation(self, state, **kwargs):
        model = kwargs.get("model")
        tokenizer = self.tokenizer
        eval_dataset = self.eval_dataset
        metrics = kwargs.get("metrics", {})

        if model is None or tokenizer is None:
            print("Skipping extrinsic validation: model or tokenizer not available.")
            return

        required_cols = ["memory_input_ids", "memory_attention_mask", "answer", "question"]
        if not all(col in eval_dataset.column_names for col in required_cols):
            print(f"Skipping extrinsic validation: Required columns {required_cols} not found in dataset: {eval_dataset.column_names}.")
            return

        print("\nPerforming Extrinsic Validation...")
        all_preds = []
        all_labels = []
        all_questions_for_log = []

        model.eval()
        for example in eval_dataset:
            input_ids = torch.tensor([example['memory_input_ids']]).to(model.device)
            attention_mask = torch.tensor([example['memory_attention_mask']]).to(model.device)

            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=50,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )

            num_input_tokens = len(input_ids[0])
            pred_ids = generated_ids[0][num_input_tokens:]
            pred_text = tokenizer.decode(pred_ids, skip_special_tokens=True).strip()

            all_preds.append(pred_text)
            all_labels.append(example["answer"])
            if "question" in example:
                all_questions_for_log.append(example["question"])

        rouge = evaluate.load('rouge')
        rouge_scores = rouge.compute(predictions=all_preds, references=all_labels)

        for key, value in rouge_scores.items():
            metrics[f"eval_{key}"] = value

        print(f"Extrinsic ROUGE Scores: {rouge_scores}")

        if state.is_world_process_zero and self.cfg_task.log_predictions:
            log_file_path = os.path.join(self.cfg_task.output_dir, f"prediction_log.epoch_{int(state.epoch)}.jsonl")
            print(f"Logging predictions to {log_file_path}")
            with open(log_file_path, "w") as f:
                for i in range(len(all_preds)):
                    log_entry = {
                        "question": all_questions_for_log[i] if i < len(all_questions_for_log) else "N/A",
                        "ground_truth": all_labels[i],
                        "prediction": all_preds[i]
                    }
                    f.write(json.dumps(log_entry) + "\n")
