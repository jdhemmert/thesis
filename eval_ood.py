#!/usr/bin/env python3
"""
Post-hoc OOD evaluation for trained memory banks.

Evaluates a trained soft-prompt checkpoint (bio A) on questions from a different
biography (bio B). Reports ROUGE and perplexity under three conditions:
  - ood_bank:  virtual tokens from A, questions from B (no bio context)
  - no_memory: no virtual tokens, questions from B (no bio context)
  - oracle:    no virtual tokens, bio B context + questions from B

Usage:
    python eval_ood.py \\
        --checkpoint-dir results/experiment/2026-04-21_XX/run/ \\
        --target-index 456 \\
        [--output-dir ./ood_results/] \\
        [--max-new-tokens 50] \\
        [--sample-n 20]
"""
import argparse
import json
import math
import sys
from pathlib import Path
from statistics import mean

import evaluate as hf_evaluate
import torch
import datasets
from datasets import load_dataset
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from hydra.utils import instantiate

sys.path.insert(0, str(Path(__file__).parent))

from src.models.augmented_llama import AugmentedLlamaForCausalLM
from src.data.preprocessors import MemoryTaskPreprocessor


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--target-index", required=True, type=int, help="JSONL row index of bio B")
    p.add_argument("--output-dir", default=None,
                   help="Output location (default: {checkpoint_dir}/ood_eval_bio{N}/)")
    p.add_argument("--max-new-tokens", type=int, default=50)
    p.add_argument("--sample-n", type=int, default=None,
                   help="Max QA pairs from bio B to evaluate (default: all)")
    return p.parse_args()


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
    Mirrors PerplexityMetric logic. AugmentedLlamaForCausalLM handles
    label padding for virtual tokens internally (augmented_llama.py:147-152).
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


def load_model_and_tokenizer(cfg, checkpoint_dir):
    model_path = cfg.model.model_config.model_path
    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    dtype = dtype_map.get(cfg.model.get("precision", "bf16"), torch.bfloat16)

    aug_config = instantiate(cfg.model.model_config)
    model = AugmentedLlamaForCausalLM.from_pretrained(
        model_path,
        config=aug_config,
        dtype=dtype,
        local_files_only=True,
    )

    vp_path = Path(checkpoint_dir) / "virtual_prompt.pt"
    if not vp_path.exists():
        sys.exit(f"No virtual_prompt.pt found at {vp_path}")
    vp = torch.load(vp_path, map_location="cpu")
    model.model.rebuild_virtual_prompt(weights=vp.weight.data)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    return model, tokenizer


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


def main():
    args = parse_args()
    checkpoint_dir = Path(args.checkpoint_dir)

    config_path = checkpoint_dir / ".hydra" / "config.yaml"
    if not config_path.exists():
        sys.exit(f"No .hydra/config.yaml found at {config_path}")
    cfg = OmegaConf.load(config_path)

    meta_path = checkpoint_dir / "bio_metadata.json"
    if not meta_path.exists():
        sys.exit("No bio_metadata.json found — was this checkpoint trained with dataset.biography_index set?")
    with meta_path.open() as f:
        bio_a_meta = json.load(f)
    print(f"Bio A (trained): index={bio_a_meta['biography_index']}, name={bio_a_meta.get('name', '?')}")

    raw_ds = load_dataset("json", data_files=cfg.dataset.path)["train"]
    bio_b_raw = {k: raw_ds[args.target_index][k] for k in raw_ds.column_names}
    bio_b_name = bio_b_raw.get("name", f"index {args.target_index}")
    print(f"Bio B (target):  index={args.target_index}, name={bio_b_name}")

    if bio_a_meta["biography_index"] == args.target_index:
        print("WARNING: target-index matches training bio — this is an in-distribution check, not OOD eval.")

    print("Loading model...")
    model, tokenizer = load_model_and_tokenizer(cfg, checkpoint_dir)

    print("Building bio B eval dataset...")
    bio_b_dataset = build_bio_dataset(bio_b_raw, cfg, tokenizer)
    print(f"Bio B has {len(bio_b_dataset)} QA pairs.")

    if args.sample_n and args.sample_n < len(bio_b_dataset):
        import random
        indices = random.sample(range(len(bio_b_dataset)), args.sample_n)
        bio_b_dataset = bio_b_dataset.select(indices)
        print(f"Sampled down to {len(bio_b_dataset)} QA pairs.")

    prompts = cfg.prompts
    results_per_example = []

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

            ood_ids    = greedy_decode(model, tokenizer, memory_ids, memory_mask, args.max_new_tokens, True)
            nomem_ids  = greedy_decode(model, tokenizer, memory_ids, memory_mask, args.max_new_tokens, False)
            oracle_ids_ = greedy_decode(model, tokenizer, oracle_ids, oracle_mask, args.max_new_tokens, False)

            ood_pred    = tokenizer.decode(ood_ids,    skip_special_tokens=True).strip()
            nomem_pred  = tokenizer.decode(nomem_ids,  skip_special_tokens=True).strip()
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
                print(f"  [{i+1}/{len(bio_b_dataset)}]")

    rouge = hf_evaluate.load("rouge")
    answers = [r["answer"] for r in results_per_example]

    rouge_scores = {}
    ppl_scores   = {}
    for cond in ["ood_bank", "no_memory", "oracle"]:
        preds = [r[cond]["prediction"] for r in results_per_example]
        rouge_scores[cond] = rouge.compute(predictions=preds, references=answers)
        valid_ppls = [r[cond]["perplexity"] for r in results_per_example if not math.isnan(r[cond]["perplexity"])]
        ppl_scores[cond] = mean(valid_ppls) if valid_ppls else float("nan")

    summary = {
        "bio_a":       bio_a_meta,
        "bio_b_index": args.target_index,
        "bio_b_name":  bio_b_name,
        "n_examples":  len(results_per_example),
        "rouge":       rouge_scores,
        "perplexity":  ppl_scores,
    }

    print("\n=== Results ===")
    for cond in ["ood_bank", "no_memory", "oracle"]:
        r1  = rouge_scores[cond].get("rouge1", float("nan"))
        ppl = ppl_scores[cond]
        print(f"  {cond:12s}  rouge1={r1:.4f}  ppl={ppl:.2f}")

    out_dir = Path(args.output_dir) if args.output_dir else checkpoint_dir / f"ood_eval_bio{args.target_index}"
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "ood_eval_results.json").open("w") as f:
        json.dump(summary, f, indent=2)
    with (out_dir / "ood_eval_predictions.jsonl").open("w") as f:
        for row in results_per_example:
            f.write(json.dumps(row) + "\n")

    print(f"\nResults written to {out_dir}/")


if __name__ == "__main__":
    main()
