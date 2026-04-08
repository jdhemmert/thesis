import torch
import torch.nn.functional as F


def _answer_pred_slice(logits, prompt_len, attn_mask):
    """
    Returns the slice of logits positions that predict answer tokens.

    For CausalLM:
      token at position j is predicted by logits at position j-1.
    If answer starts at token index prompt_len, then predictions start at (prompt_len-1).

    logits: [seq, vocab]  where seq may be larger than attn_mask if virtual tokens are prepended
    prompt_len: int  (in original token space, not counting virtual tokens)
    attn_mask: [orig_seq] with 1 for real tokens
    """
    # Virtual tokens prepended to the input shift all logit positions.
    # Infer the offset from the size difference between logits and attn_mask.
    virtual_offset = logits.size(0) - attn_mask.size(0)
    real_len = int(attn_mask.sum().item())
    start = max(virtual_offset + prompt_len - 1, 0)
    end = max(virtual_offset + real_len - 1, 0)
    return logits[start:end, :]


def _count_answer_tokens(prompt_len, attn_mask):
    real_len = int(attn_mask.sum().item())
    return max(real_len - prompt_len, 0)


# TODO: rename
def teacher_only_distill_loss(
    teacher_logits, student_logits,
    oracle_prompt_len, memory_prompt_len,
    oracle_attention_mask, memory_attention_mask,
    temperature: float,
    alpha: float = 1.0,
    use_gt_ce: bool = False,
    memory_labels=None,
):
    bs, seq, vocab = student_logits.shape
    T = float(temperature)

    total_kl = student_logits.new_tensor(0.0)
    total_tok = 0

    for i in range(bs):
        t_slice = _answer_pred_slice(
            teacher_logits[i], int(oracle_prompt_len[i].item()), oracle_attention_mask[i]
        )
        s_slice = _answer_pred_slice(
            student_logits[i], int(memory_prompt_len[i].item()), memory_attention_mask[i]
        )

        L = min(t_slice.size(0), s_slice.size(0))
        if L <= 0:
            continue

        t_logp = F.log_softmax(t_slice[:L, :] / T, dim=-1)
        s_logp = F.log_softmax(s_slice[:L, :] / T, dim=-1)
        t_p = t_logp.exp()

        kl_tok = (t_p * (t_logp - s_logp)).sum(dim=-1)
        total_kl = total_kl + kl_tok.sum() * (T * T)
        total_tok += L

    if total_tok == 0:
        print(
            f"[losses] WARNING: no answer tokens in batch (bs={bs}). "
            f"oracle_prompt_len={oracle_prompt_len.tolist()}, "
            f"oracle_real_lens={oracle_attention_mask.sum(-1).tolist()}, "
            f"memory_prompt_len={memory_prompt_len.tolist()}, "
            f"memory_real_lens={memory_attention_mask.sum(-1).tolist()}"
        )
        return student_logits.sum() * 0.0

    kl_loss = total_kl / total_tok

    if use_gt_ce:
        assert memory_labels is not None, "memory_labels required when use_gt_ce=True"

        seq_diff = student_logits.size(1) - memory_labels.size(1)
        if seq_diff < 0:
            raise ValueError(
                f"student_logits shorter than memory_labels: "
                f"{student_logits.size(1)} vs {memory_labels.size(1)}"
            )
        elif seq_diff > 0:
            pad = torch.full(
                (memory_labels.size(0), seq_diff),
                -100,
                dtype=memory_labels.dtype,
                device=memory_labels.device,
            )
            memory_labels = torch.cat([pad, memory_labels], dim=1)

        assert student_logits.size(1) == memory_labels.size(1)
        shift_logits = student_logits[:, :-1, :].contiguous()
        shift_labels = memory_labels[:, 1:].contiguous()

        ce_loss = F.cross_entropy(
            shift_logits.view(-1, vocab),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="mean",
        )

        loss = alpha * kl_loss + (1 - alpha) * ce_loss
    else:
        loss = kl_loss

    return loss


def __self_distillation_loss(
    oracle_logits: torch.Tensor,
    memory_logits: torch.Tensor,
    temperature: float,
    alpha: float,
    loss_type: str,
):
    """
    Computes the self-distillation loss between an oracle model and a memory-augmented model.

    Args:
        oracle_logits (torch.Tensor): The logits from the oracle model (teacher).
        memory_logits (torch.Tensor): The logits from the memory-augmented model (student).
        temperature (float): The temperature for softening the probability distributions.
        alpha (float): The balancing factor between KL divergence and cross-entropy loss.
        loss_type (str): The type of loss to compute ('balanced', 'kl_divergence', 'cross_entropy').

    Returns:
        torch.Tensor: The computed loss.
    """
    kl_loss_fct = torch.nn.KLDivLoss(reduction="batchmean")
    ce_loss_fct = torch.nn.CrossEntropyLoss()
    log_softmax = torch.nn.LogSoftmax(dim=-1)
    softmax = torch.nn.Softmax(dim=-1)

    # Detach oracle logits to treat them as fixed targets
    oracle_logits = oracle_logits.detach()

    # Soften probabilities with temperature
    soft_oracle_probs = softmax(oracle_logits / temperature)
    soft_memory_log_probs = log_softmax(memory_logits / temperature)

    # KL-Divergence loss: distill "hidden" knowledge
    kl_loss = kl_loss_fct(soft_memory_log_probs, soft_oracle_probs) * (temperature**2)

    # Cross-Entropy loss: emulate behavior
    ce_loss = ce_loss_fct(memory_logits.transpose(1, 2), oracle_logits.argmax(dim=-1))

    if loss_type == "balanced":
        loss = alpha * kl_loss + (1 - alpha) * ce_loss
    elif loss_type == "kl_divergence":
        loss = kl_loss
    elif loss_type == "cross_entropy":
        loss = ce_loss
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    return loss
