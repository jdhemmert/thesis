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
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from hydra.utils import instantiate

sys.path.insert(0, str(Path(__file__).parent))

from src.models.augmented_llama import AugmentedLlamaForCausalLM
from src.eval.ood import run_ood_eval


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

    print("Loading model...")
    model, tokenizer = load_model_and_tokenizer(cfg, checkpoint_dir)

    out_dir = args.output_dir  # None → run_ood_eval uses default {cwd}/ood_eval_bio{N}/
    cwd = str(Path(args.output_dir).parent) if out_dir else str(checkpoint_dir)

    run_ood_eval(
        model=model,
        tokenizer=tokenizer,
        cfg=cfg,
        cwd=cwd,
        bio_a_meta=bio_a_meta,
        target_indices=[args.target_index],
        max_new_tokens=args.max_new_tokens,
        sample_n=args.sample_n,
    )


if __name__ == "__main__":
    main()
