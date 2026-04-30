"""
Shared OOD evaluation logic for trained memory banks.

Used by both eval_ood.py (post-hoc CLI) and TrainMemoryTask (inline after training).
"""
import json
import math
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Any

import torch
import datasets
import evaluate as hf_evaluate
from datasets import load_dataset
from hydra.utils import instantiate

from src.data.preprocessors import MemoryTaskPreprocessor


def greedy_decode(model, tokenizer, input_ids, attention_mask, max_new_tokens, use_virtual_tokens):
    current_ids = input_ids.clone()
    current_mask = attention_mask.clone()
    generated = []
    for _ in range(max_new_tokens):
        outputs = model(
            input_ids=current_ids,
            attention_mask=current_mask,
            use_cache=False,
            use_virtual_tokens=use_virtual_tokens,
        )
        next_token_id = int(outputs.logits[0, -1, :].argmax())
        if next_token_id == tokenizer.eos_token_id:
            break
        generated.append(next_token_id)
        next_token = torch.tensor([[next_token_id]], dtype=current_ids.dtype, device=current_ids.device)
        current_ids = torch.cat([current_ids, next_token], dim=1)
        current_mask = torch.cat(
            [current_mask, torch.ones(1, 1, dtype=current_mask.dtype, device=current_mask.device)], dim=1
        )
    return generated


def compute_perplexity(model, tokenizer, prompt_text, answer_text, use_virtual_tokens, max_length=1024):
    """
    Compute answer perplexity conditioned on prompt_text.
    AugmentedLlamaForCausalLM handles label padding for virtual tokens internally.
    """
    question_tokenized = tokenizer(prompt_text.rstrip(), return_tensors="pt", add_special_tokens=True)
    full_tokenized = tokenizer(
        prompt_text + answer_text,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
        add_special_tokens=True,
    )
    input_ids = full_tokenized.input_ids.to(model.device)
    attention_mask = full_tokenized.attention_mask.to(model.device)

    labels = input_ids.clone()
    prompt_len = question_tokenized.input_ids.shape[1]
    if prompt_len >= labels.shape[1] or (labels == -100).all():
        return float("nan")
    labels[:, :prompt_len] = -100

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        use_virtual_tokens=use_virtual_tokens,
    )
    loss = outputs.loss
    return math.exp(loss.item()) if not torch.isnan(loss) else float("nan")


def build_bio_dataset(bio_raw, cfg, tokenizer):
    using_new_path = (
        cfg.dataset.get("parser") is not None
        and cfg.dataset.get("biography_task") is not None
    )
    parser = instantiate(cfg.dataset.parser) if using_new_path else None
    biography_task = instantiate(cfg.dataset.biography_task) if using_new_path else None

    preprocessor = MemoryTaskPreprocessor(
        tokenizer=tokenizer,
        max_length=cfg.task.max_length,
        prompts=cfg.prompts,
        dataset_mode=cfg.dataset.get("dataset_mode", "qa"),
        parser=parser,
        biography_task=biography_task,
    )

    bio_hf = datasets.Dataset.from_list([bio_raw])
    return bio_hf.map(preprocessor, batched=True, remove_columns=list(bio_hf.column_names))


def run_ood_eval(
    model,
    tokenizer,
    cfg,
    cwd: str,
    bio_a_meta: Dict[str, Any],
    target_indices: List[int],
    max_new_tokens: int = 50,
    sample_n: Optional[int] = None,
):
    """
    Evaluate a trained memory bank against one or more OOD biographies.

    Results for each target index are written to {cwd}/ood_eval_bio{N}/.
    The model is used as-is (no checkpoint reload).
    """
    import random

    raw_ds = load_dataset("json", data_files=cfg.dataset.path)["train"]
    rouge = hf_evaluate.load("rouge")
    prompts = cfg.prompts

    for target_index in target_indices:
        bio_b_raw = {k: raw_ds[target_index][k] for k in raw_ds.column_names}
        bio_b_name = bio_b_raw.get("name", f"index {target_index}")

        if bio_a_meta.get("biography_index") == target_index:
            print(f"  [ood_eval bio{target_index}] WARNING: matches training bio — in-distribution check.")
        else:
            print(f"  [ood_eval bio{target_index}] OOD: bio_b={bio_b_name}")

        bio_b_dataset = build_bio_dataset(bio_b_raw, cfg, tokenizer)
        print(f"  [ood_eval bio{target_index}] {len(bio_b_dataset)} QA pairs.")

        if sample_n is not None and sample_n < len(bio_b_dataset):
            indices = random.sample(range(len(bio_b_dataset)), sample_n)
            bio_b_dataset = bio_b_dataset.select(indices)
            print(f"  [ood_eval bio{target_index}] Sampled down to {len(bio_b_dataset)} QA pairs.")

        results_per_example = []

        model.eval()
        with torch.no_grad():
            for i in range(len(bio_b_dataset)):
                example = bio_b_dataset[i]
                question  = example.get("question", "")
                answer    = example.get("answer", "")
                biography = example.get("biography", "")

                memory_ids  = torch.tensor([example["eval_memory_input_ids"]], device=model.device)
                memory_mask = torch.tensor([example["eval_memory_attention_mask"]], device=model.device)
                oracle_ids  = torch.tensor([example["eval_oracle_input_ids"]], device=model.device)
                oracle_mask = torch.tensor([example["eval_oracle_attention_mask"]], device=model.device)

                memory_prompt = prompts.direct_qa_generation.format(question=question)
                oracle_prompt = prompts.contextual_qa_generation.format(biography=biography, question=question)

                ood_ids     = greedy_decode(model, tokenizer, memory_ids, memory_mask, max_new_tokens, True)
                nomem_ids   = greedy_decode(model, tokenizer, memory_ids, memory_mask, max_new_tokens, False)
                oracle_ids_ = greedy_decode(model, tokenizer, oracle_ids, oracle_mask, max_new_tokens, False)

                ood_pred    = tokenizer.decode(ood_ids,     skip_special_tokens=True).strip()
                nomem_pred  = tokenizer.decode(nomem_ids,   skip_special_tokens=True).strip()
                oracle_pred = tokenizer.decode(oracle_ids_, skip_special_tokens=True).strip()

                ood_ppl    = compute_perplexity(model, tokenizer, memory_prompt, answer, use_virtual_tokens=True)
                nomem_ppl  = compute_perplexity(model, tokenizer, memory_prompt, answer, use_virtual_tokens=False)
                oracle_ppl = compute_perplexity(model, tokenizer, oracle_prompt, answer, use_virtual_tokens=False)

                results_per_example.append({
                    "question": question,
                    "answer":   answer,
                    "ood_bank":  {"prediction": ood_pred,    "perplexity": ood_ppl},
                    "no_memory": {"prediction": nomem_pred,  "perplexity": nomem_ppl},
                    "oracle":    {"prediction": oracle_pred, "perplexity": oracle_ppl},
                })

                if (i + 1) % 5 == 0 or (i + 1) == len(bio_b_dataset):
                    print(f"  [ood_eval bio{target_index}] [{i+1}/{len(bio_b_dataset)}]")

        answers = [r["answer"] for r in results_per_example]
        rouge_scores = {}
        ppl_scores = {}
        for cond in ["ood_bank", "no_memory", "oracle"]:
            preds = [r[cond]["prediction"] for r in results_per_example]
            rouge_scores[cond] = rouge.compute(predictions=preds, references=answers)
            valid_ppls = [r[cond]["perplexity"] for r in results_per_example if not math.isnan(r[cond]["perplexity"])]
            ppl_scores[cond] = mean(valid_ppls) if valid_ppls else float("nan")

        summary = {
            "bio_a":       bio_a_meta,
            "bio_b_index": target_index,
            "bio_b_name":  bio_b_name,
            "n_examples":  len(results_per_example),
            "rouge":       rouge_scores,
            "perplexity":  ppl_scores,
        }

        print(f"\n  [ood_eval bio{target_index}] Results:")
        for cond in ["ood_bank", "no_memory", "oracle"]:
            r1  = rouge_scores[cond].get("rouge1", float("nan"))
            ppl = ppl_scores[cond]
            print(f"    {cond:12s}  rouge1={r1:.4f}  ppl={ppl:.2f}")

        out_dir = Path(cwd) / f"ood_eval_bio{target_index}"
        out_dir.mkdir(parents=True, exist_ok=True)

        with (out_dir / "ood_eval_results.json").open("w") as f:
            json.dump(summary, f, indent=2)
        with (out_dir / "ood_eval_predictions.jsonl").open("w") as f:
            for row in results_per_example:
                f.write(json.dumps(row) + "\n")

        print(f"  [ood_eval bio{target_index}] Results written to {out_dir}/")
