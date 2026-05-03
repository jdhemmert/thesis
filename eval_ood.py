#!/usr/bin/env python3
"""
Post-hoc OOD evaluation for trained memory banks.

Evaluates a trained soft-prompt checkpoint (bio A) on questions from a different
biography (bio B). Reports ROUGE and perplexity under three conditions:
  - ood_bank:  virtual tokens from A, questions from B (no bio context)
  - no_memory: no virtual tokens, questions from B (no bio context)
  - oracle:    no virtual tokens, bio B context + questions from B

Usage (single checkpoint):
    python eval_ood.py \\
        --checkpoint-dir results/experiment/run/ \\
        --target-index 456 \\
        [--output-dir ./ood_results/] \\
        [--max-new-tokens 50] \\
        [--sample-n 20]

Usage (sweep — evaluates all checkpoints under a directory):
    python eval_ood.py \\
        --sweep-dir results/ood/distractor_sweep \\
        --target-index 44 \\
        [--overwrite]
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
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint-dir",
                       help="Single checkpoint directory to evaluate")
    group.add_argument("--sweep-dir",
                       help="Root directory containing multiple checkpoints to evaluate")
    p.add_argument("--target-index", required=True, type=int,
                   help="JSONL row index of bio B")
    p.add_argument("--output-dir", default=None,
                   help="Output location; only valid with --checkpoint-dir "
                        "(default: {checkpoint_dir}/ood_eval_bio{N}/)")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-run even if ood_eval_bio{N}/ results already exist")
    p.add_argument("--max-new-tokens", type=int, default=50)
    p.add_argument("--sample-n", type=int, default=None,
                   help="Max QA pairs from bio B to evaluate (default: all)")
    return p.parse_args()


def load_config_and_meta(checkpoint_dir: Path):
    config_path = checkpoint_dir / ".hydra" / "config.yaml"
    if not config_path.exists():
        sys.exit(f"No .hydra/config.yaml found at {config_path}")
    cfg = OmegaConf.load(config_path)

    meta_path = checkpoint_dir / "bio_metadata.json"
    if not meta_path.exists():
        sys.exit(f"No bio_metadata.json found at {meta_path}")
    with meta_path.open() as f:
        bio_a_meta = json.load(f)
    return cfg, bio_a_meta


def load_base_model_and_tokenizer(cfg, checkpoint_dir):
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

    swap_virtual_prompt(model, checkpoint_dir)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    return model, tokenizer


def swap_virtual_prompt(model, checkpoint_dir):
    vp_path = Path(checkpoint_dir) / "virtual_prompt.pt"
    if not vp_path.exists():
        sys.exit(f"No virtual_prompt.pt found at {vp_path}")
    vp = torch.load(vp_path, map_location="cpu")
    model.model.rebuild_virtual_prompt(weights=vp.weight.data)


def find_checkpoints(sweep_dir: Path) -> list:
    found = []
    for vp in sorted(sweep_dir.rglob("virtual_prompt.pt")):
        d = vp.parent
        if (d / ".hydra" / "config.yaml").exists() and (d / "bio_metadata.json").exists():
            found.append(d)
    return found


def main():
    args = parse_args()

    if args.output_dir and args.sweep_dir:
        sys.exit("--output-dir is not valid with --sweep-dir")

    if args.checkpoint_dir:
        checkpoint_dir = Path(args.checkpoint_dir)
        cfg, bio_a_meta = load_config_and_meta(checkpoint_dir)
        print(f"Bio A (trained): index={bio_a_meta['biography_index']}, name={bio_a_meta.get('name', '?')}")
        print("Loading model...")
        model, tokenizer = load_base_model_and_tokenizer(cfg, checkpoint_dir)
        cwd = str(Path(args.output_dir).parent) if args.output_dir else str(checkpoint_dir)
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

    else:
        sweep_dir = Path(args.sweep_dir)
        checkpoints = find_checkpoints(sweep_dir)
        if not checkpoints:
            sys.exit(f"No checkpoints found under {sweep_dir}")
        print(f"Found {len(checkpoints)} checkpoints under {sweep_dir}")

        model = None
        tokenizer = None

        for i, checkpoint_dir in enumerate(checkpoints):
            label = checkpoint_dir.relative_to(sweep_dir)
            out_dir = checkpoint_dir / f"ood_eval_bio{args.target_index}"
            if not args.overwrite and (out_dir / "ood_eval_results.json").exists():
                print(f"[{i+1}/{len(checkpoints)}] Skipping {label} (already done)")
                continue

            cfg, bio_a_meta = load_config_and_meta(checkpoint_dir)

            if model is None:
                print("Loading model...")
                model, tokenizer = load_base_model_and_tokenizer(cfg, checkpoint_dir)
            else:
                swap_virtual_prompt(model, checkpoint_dir)

            print(f"[{i+1}/{len(checkpoints)}] {label}")
            run_ood_eval(
                model=model,
                tokenizer=tokenizer,
                cfg=cfg,
                cwd=str(checkpoint_dir),
                bio_a_meta=bio_a_meta,
                target_indices=[args.target_index],
                max_new_tokens=args.max_new_tokens,
                sample_n=args.sample_n,
            )


if __name__ == "__main__":
    main()
