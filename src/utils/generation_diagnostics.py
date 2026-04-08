"""
Groups together utilities for inspecting and debugging token-level
behavior, memory effects, and inference-time generation dynamics.
"""


import os
import json
from typing import Dict, List, Any, Optional

import torch
import torch.nn.functional as F


def safe_decode_token(tokenizer, token_id: int) -> str:
    try:
        return tokenizer.decode([token_id], skip_special_tokens=False)
    except Exception:
        return f"<decode_error:{token_id}>"


def topk_from_logits(tokenizer, logits: torch.Tensor, k: int = 10) -> List[Dict[str, Any]]:
    probs = F.softmax(logits, dim=-1)
    top_probs, top_ids = torch.topk(probs, k=min(k, probs.shape[-1]), dim=-1)

    out = []
    for tid, prob in zip(top_ids.tolist(), top_probs.tolist()):
        out.append({
            "token_id": int(tid),
            "token_text": safe_decode_token(tokenizer, int(tid)),
            "prob": float(prob),
            "logprob": float(torch.log(torch.tensor(prob)).item()) if prob > 0 else float("-inf"),
        })
    return out


def first_generated_token_id_from_prompt_plus_answer(
    tokenizer,
    prompt_input_ids: List[int],
    answer_text: str,
) -> Optional[int]:
    """
    Computes the true first answer token ID in-context by tokenizing:
      decode(prompt_input_ids_without_pad) + answer_text
    and taking the first token after the prompt boundary.

    This avoids the standalone-tokenization bug where "May" and " May"
    get different token IDs.
    """
    prompt_text = tokenizer.decode(prompt_input_ids, skip_special_tokens=True)
    full_ids = tokenizer(prompt_text + answer_text, add_special_tokens=False)["input_ids"]
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

    if len(full_ids) <= len(prompt_ids):
        return None

    return int(full_ids[len(prompt_ids)])


@torch.no_grad()
def next_token_diagnostics(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    use_virtual_tokens: bool,
    gold_first_token_id: Optional[int],
    topk: int = 10,
) -> Dict[str, Any]:
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_virtual_tokens=use_virtual_tokens,
        use_cache=False,
        return_dict=True,
        output_hidden_states=False,
    )

    next_token_logits = outputs.logits[0, -1, :]
    probs = F.softmax(next_token_logits, dim=-1)

    argmax_id = int(torch.argmax(next_token_logits).item())
    argmax_prob = float(probs[argmax_id].item())

    info = {
        "use_virtual_tokens": bool(use_virtual_tokens),
        "prompt_length": int(attention_mask.sum().item()),
        "argmax_token_id": argmax_id,
        "argmax_token_text": safe_decode_token(tokenizer, argmax_id),
        "argmax_prob": argmax_prob,
        "topk_next_tokens": topk_from_logits(tokenizer, next_token_logits, k=topk),
    }

    if gold_first_token_id is not None:
        gold_prob = float(probs[gold_first_token_id].item())
        gold_rank = int((next_token_logits > next_token_logits[gold_first_token_id]).sum().item()) + 1

        top2_vals, top2_ids = torch.topk(next_token_logits, k=2, dim=-1)
        if int(top2_ids[0].item()) == gold_first_token_id:
            runner_up_logit = float(top2_vals[1].item())
        else:
            runner_up_logit = float(top2_vals[0].item())

        info.update({
            "gold_first_token_id": int(gold_first_token_id),
            "gold_first_token_text": safe_decode_token(tokenizer, int(gold_first_token_id)),
            "gold_first_token_prob": gold_prob,
            "gold_first_token_logprob": float(torch.log(probs[gold_first_token_id]).item()) if gold_prob > 0 else float("-inf"),
            "gold_first_token_rank": gold_rank,
            "gold_first_token_is_argmax": bool(argmax_id == gold_first_token_id),
            "gold_vs_best_logit_margin": float(next_token_logits[gold_first_token_id].item() - runner_up_logit),
        })

    return info


@torch.no_grad()
def virtual_token_effect_diagnostics(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    gold_first_token_id: Optional[int],
    topk: int = 10,
) -> Dict[str, Any]:
    """
    Compares the exact same prompt with and without virtual tokens.
    Logs whether virtual tokens measurably change hidden states or logits.
    """
    out_no = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_virtual_tokens=False,
        use_cache=False,
        return_dict=True,
        output_hidden_states=True,
    )
    out_yes = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_virtual_tokens=True,
        use_cache=False,
        return_dict=True,
        output_hidden_states=True,
    )

    logits_no = out_no.logits[0, -1, :]
    logits_yes = out_yes.logits[0, -1, :]
    probs_no = F.softmax(logits_no, dim=-1)
    probs_yes = F.softmax(logits_yes, dim=-1)

    logit_diff = logits_yes - logits_no
    hidden_no = out_no.hidden_states[-1][0, -1, :]
    hidden_yes = out_yes.hidden_states[-1][0, -1, :]
    hidden_diff = hidden_yes - hidden_no

    argmax_no = int(torch.argmax(logits_no).item())
    argmax_yes = int(torch.argmax(logits_yes).item())

    result = {
        "argmax_no_memory_id": argmax_no,
        "argmax_no_memory_text": safe_decode_token(tokenizer, argmax_no),
        "argmax_with_memory_id": argmax_yes,
        "argmax_with_memory_text": safe_decode_token(tokenizer, argmax_yes),
        "argmax_changed": bool(argmax_no != argmax_yes),
        "max_abs_logit_delta": float(logit_diff.abs().max().item()),
        "mean_abs_logit_delta": float(logit_diff.abs().mean().item()),
        "l2_logit_delta": float(torch.norm(logit_diff, p=2).item()),
        "last_hidden_max_abs_delta": float(hidden_diff.abs().max().item()),
        "last_hidden_mean_abs_delta": float(hidden_diff.abs().mean().item()),
        "last_hidden_l2_delta": float(torch.norm(hidden_diff, p=2).item()),
        "topk_no_memory": topk_from_logits(tokenizer, logits_no, k=topk),
        "topk_with_memory": topk_from_logits(tokenizer, logits_yes, k=topk),
    }

    if gold_first_token_id is not None:
        result.update({
            "gold_first_token_id": int(gold_first_token_id),
            "gold_first_token_text": safe_decode_token(tokenizer, int(gold_first_token_id)),
            "gold_first_token_prob_no_memory": float(probs_no[gold_first_token_id].item()),
            "gold_first_token_prob_with_memory": float(probs_yes[gold_first_token_id].item()),
            "gold_first_token_prob_delta": float(
                probs_yes[gold_first_token_id].item() - probs_no[gold_first_token_id].item()
            ),
            "gold_first_token_logit_delta": float(
                logits_yes[gold_first_token_id].item() - logits_no[gold_first_token_id].item()
            ),
            "gold_first_token_rank_no_memory": int((logits_no > logits_no[gold_first_token_id]).sum().item()) + 1,
            "gold_first_token_rank_with_memory": int((logits_yes > logits_yes[gold_first_token_id]).sum().item()) + 1,
        })

    # Virtual prompt parameter stats
    vp = None
    if hasattr(model, "model"):
        vp = getattr(model.model, "virtual_prompt", None)

    if vp is not None:
        w = vp.weight.detach()
        result["virtual_prompt_stats"] = {
            "exists": True,
            "shape": list(w.shape),
            "requires_grad": bool(vp.weight.requires_grad),
            "norm": float(w.norm().item()),
            "mean_abs": float(w.abs().mean().item()),
            "max_abs": float(w.abs().max().item()),
        }
    else:
        result["virtual_prompt_stats"] = {
            "exists": False,
        }

    return result


def safe_decode_token(tokenizer, token_id: int) -> str:
    try:
        return tokenizer.decode([token_id], skip_special_tokens=False)
    except Exception:
        return f"<decode_error:{token_id}>"


def topk_from_logits(tokenizer, logits: torch.Tensor, k: int = 10) -> List[Dict[str, Any]]:
    probs = F.softmax(logits, dim=-1)
    top_probs, top_ids = torch.topk(probs, k=min(k, probs.shape[-1]), dim=-1)

    out = []
    for tid, prob in zip(top_ids.tolist(), top_probs.tolist()):
        out.append({
            "token_id": int(tid),
            "token_text": safe_decode_token(tokenizer, int(tid)),
            "prob": float(prob),
            "logprob": float(torch.log(torch.tensor(prob)).item()) if prob > 0 else float("-inf"),
        })
    return out


def first_answer_token_info(tokenizer, answer_text: str) -> Dict[str, Any]:
    answer_token_ids = tokenizer.encode(answer_text, add_special_tokens=False)
    if len(answer_token_ids) == 0:
        return {
            "first_answer_token_id": None,
            "first_answer_token_text": None,
            "all_answer_token_ids": [],
        }

    first_id = int(answer_token_ids[0])
    return {
        "first_answer_token_id": first_id,
        "first_answer_token_text": safe_decode_token(tokenizer, first_id),
        "all_answer_token_ids": [int(x) for x in answer_token_ids],
    }


@torch.no_grad()
def next_token_diagnostics(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    use_virtual_tokens: bool,
    gold_first_token_id: int | None,
    topk: int = 10,
) -> Dict[str, Any]:
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_virtual_tokens=use_virtual_tokens,
        use_cache=False,
        return_dict=True,
    )

    # Next-token logits at the final prompt position
    next_token_logits = outputs.logits[0, -1, :]
    probs = F.softmax(next_token_logits, dim=-1)

    argmax_id = int(torch.argmax(next_token_logits).item())
    argmax_prob = float(probs[argmax_id].item())

    info = {
        "use_virtual_tokens": bool(use_virtual_tokens),
        "prompt_length": int(attention_mask.sum().item()),
        "argmax_token_id": argmax_id,
        "argmax_token_text": safe_decode_token(tokenizer, argmax_id),
        "argmax_prob": argmax_prob,
        "topk_next_tokens": topk_from_logits(tokenizer, next_token_logits, k=topk),
    }

    if gold_first_token_id is not None:
        gold_prob = float(probs[gold_first_token_id].item())
        gold_rank = int((next_token_logits > next_token_logits[gold_first_token_id]).sum().item()) + 1

        top2_vals, top2_ids = torch.topk(next_token_logits, k=2, dim=-1)
        if int(top2_ids[0].item()) == gold_first_token_id:
            runner_up_logit = float(top2_vals[1].item())
        else:
            runner_up_logit = float(top2_vals[0].item())

        info.update({
            "gold_first_token_id": int(gold_first_token_id),
            "gold_first_token_text": safe_decode_token(tokenizer, int(gold_first_token_id)),
            "gold_first_token_prob": gold_prob,
            "gold_first_token_logprob": float(torch.log(probs[gold_first_token_id]).item()) if gold_prob > 0 else float("-inf"),
            "gold_first_token_rank": gold_rank,
            "gold_first_token_is_argmax": bool(argmax_id == gold_first_token_id),
            "gold_vs_best_logit_margin": float(next_token_logits[gold_first_token_id].item() - runner_up_logit),
        })

    return info