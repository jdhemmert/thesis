import torch

def self_distillation_loss(
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

    # KL-Divergence Loss (scaled by T^2 as in Hinton's paper)
    kl_loss = kl_loss_fct(soft_memory_log_probs, soft_oracle_probs) * (temperature**2)

    # Cross-Entropy Loss against the hard labels from the oracle
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
