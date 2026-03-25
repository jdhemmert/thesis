import math
import torch
import torch.nn.functional as F
from transformers import Trainer
from src.training.losses import teacher_only_distill_loss


class SelfDistillationDataCollator:
    """
    Pads and collates oracle/memory streams.
    Also creates answer-only labels for each stream:
      - padding masked to -100
      - prompt portion masked to -100 (answer-only)
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        assert tokenizer.pad_token_id is not None, "Tokenizer must have pad_token_id"

    def __call__(self, features):
        max_len = max(
            max(len(f["oracle_input_ids"]) for f in features),
            max(len(f["memory_input_ids"]) for f in features),
        )

        oracle_batch = self.tokenizer.pad(
            {
                "input_ids": [f["oracle_input_ids"] for f in features],
                "attention_mask": [f["oracle_attention_mask"] for f in features],
            },
            padding="max_length",
            max_length=max_len,
            return_tensors="pt",
        )

        memory_batch = self.tokenizer.pad(
            {
                "input_ids": [f["memory_input_ids"] for f in features],
                "attention_mask": [f["memory_attention_mask"] for f in features],
            },
            padding="max_length",
            max_length=max_len,
            return_tensors="pt",
        )

        oracle_prompt_len = torch.tensor([int(f["oracle_prompt_len"]) for f in features], dtype=torch.long)
        memory_prompt_len = torch.tensor([int(f["memory_prompt_len"]) for f in features], dtype=torch.long)

        # Build answer-only labels
        oracle_labels = oracle_batch["input_ids"].clone()
        memory_labels = memory_batch["input_ids"].clone()

        # Mask padding
        oracle_labels[oracle_batch["attention_mask"] == 0] = -100
        memory_labels[memory_batch["attention_mask"] == 0] = -100

        # Mask prompt part
        for i in range(len(features)):
            oracle_labels[i, : int(oracle_prompt_len[i].item())] = -100
            memory_labels[i, : int(memory_prompt_len[i].item())] = -100

        return {
            "oracle_input_ids": oracle_batch["input_ids"],
            "oracle_attention_mask": oracle_batch["attention_mask"],
            "oracle_prompt_len": oracle_prompt_len,
            "oracle_labels": oracle_labels,

            "memory_input_ids": memory_batch["input_ids"],
            "memory_attention_mask": memory_batch["attention_mask"],
            "memory_prompt_len": memory_prompt_len,
            "memory_labels": memory_labels,
        }

class MemoryBankTrainer(Trainer):
    """Custom Trainer for self-distillation."""
    def __init__(self, loss_type, loss_alpha, temperature, **kwargs):
        self.loss_type = loss_type
        self.alpha = loss_alpha
        self.T = temperature
        super().__init__(**kwargs)

    def evaluate(self, *args, **kwargs):
        metrics = super().evaluate(*args, **kwargs)

        if False and "eval_loss" in metrics:
            print("Calculating perplexity...")
            try:
                metrics["eval_ppl"] = math.exp(metrics["eval_loss"])
            except OverflowError:
                metrics["eval_ppl"] = float("inf")

        return metrics

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        oracle_out = model(
            input_ids=inputs["oracle_input_ids"],
            attention_mask=inputs["oracle_attention_mask"],
            use_virtual_tokens=False,
        )
        student_out = model(
            input_ids=inputs["memory_input_ids"],
            attention_mask=inputs["memory_attention_mask"],
            use_virtual_tokens=True,
        )
    
        loss = teacher_only_distill_loss(
            teacher_logits=oracle_out.logits,
            student_logits=student_out.logits,
            oracle_prompt_len=inputs["oracle_prompt_len"],
            memory_prompt_len=inputs["memory_prompt_len"],
            oracle_attention_mask=inputs["oracle_attention_mask"],
            memory_attention_mask=inputs["memory_attention_mask"],
            temperature=self.T,
            # use_gt_ce=self.use_gt_ce,
            memory_labels=inputs["memory_labels"],
            gt_ce_weight=getattr(self, "gt_ce_weight", 1.0),
        )
        return (loss, student_out) if return_outputs else loss


    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """
        Returns:
          loss: the same teacher-only distillation loss used in compute_loss
          predictions: compact per-example stats
            [student_sum_nll, teacher_sum_nll, kl_sum, L]
          labels: dummy labels so compute_metrics is invoked
        """
        import torch
        import torch.nn.functional as F
    
        def _real_len(attn_mask_1ex: torch.Tensor) -> int:
            return int(attn_mask_1ex.sum().item())
    
        def _answer_pred_slice(logits_1ex: torch.Tensor, prompt_len: int, attn_mask_1ex: torch.Tensor):
            """
            logits_1ex: [seq, vocab]  where seq may exceed attn_mask length if virtual tokens were prepended
            prompt_len: int  (in original token space, not counting virtual tokens)
            """
            virtual_offset = logits_1ex.size(0) - attn_mask_1ex.size(0)
            real_len = _real_len(attn_mask_1ex)
            start = max(virtual_offset + prompt_len - 1, 0)
            end = max(virtual_offset + real_len - 1, 0)
            if end <= start:
                return logits_1ex.new_zeros((0, logits_1ex.size(-1)))
            return logits_1ex[start:end, :]
    
        model.eval()
        inputs = self._prepare_inputs(inputs)
    
        with torch.no_grad():
            oracle_out = model(
                input_ids=inputs["oracle_input_ids"],
                attention_mask=inputs["oracle_attention_mask"],
                use_virtual_tokens=False,
            )
            student_out = model(
                input_ids=inputs["memory_input_ids"],
                attention_mask=inputs["memory_attention_mask"],
                use_virtual_tokens=True,
            )
    
            # Compute the same eval loss as compute_loss
            loss = teacher_only_distill_loss(
                teacher_logits=oracle_out.logits,
                student_logits=student_out.logits,
                oracle_prompt_len=inputs["oracle_prompt_len"],
                memory_prompt_len=inputs["memory_prompt_len"],
                oracle_attention_mask=inputs["oracle_attention_mask"],
                memory_attention_mask=inputs["memory_attention_mask"],
                temperature=self.T,
                memory_labels=inputs["memory_labels"],
                gt_ce_weight=getattr(self, "gt_ce_weight", 1.0),
            )
    
            teacher_logits = oracle_out.logits   # [bs, seq, vocab]
            student_logits = student_out.logits  # [bs, seq, vocab]
    
            bs, seq, vocab = student_logits.shape
            T = float(getattr(self, "T", 1.0))
    
            # [student_sum_nll, teacher_sum_nll, kl_sum, L]
            stats = student_logits.new_zeros((bs, 4), dtype=torch.float32)
    
            for i in range(bs):
                t_prompt = int(inputs["oracle_prompt_len"][i].item())
                s_prompt = int(inputs["memory_prompt_len"][i].item())
    
                # Prediction slices corresponding to answer tokens
                t_slice = _answer_pred_slice(
                    teacher_logits[i], t_prompt, inputs["oracle_attention_mask"][i]
                )  # [Lt, vocab]
                s_slice = _answer_pred_slice(
                    student_logits[i], s_prompt, inputs["memory_attention_mask"][i]
                )  # [Ls, vocab]
    
                L = min(t_slice.size(0), s_slice.size(0))
                if L <= 0:
                    continue
    
                # Targets: the answer tokens themselves (aligned by answer index)
                t_targets = inputs["oracle_input_ids"][i, t_prompt : t_prompt + L]
                s_targets = inputs["memory_input_ids"][i, s_prompt : s_prompt + L]
    
                # Answer-only NLL sums
                teacher_sum_nll = F.cross_entropy(t_slice[:L, :], t_targets, reduction="sum")
                student_sum_nll = F.cross_entropy(s_slice[:L, :], s_targets, reduction="sum")
    
                # KL(teacher || student) over answer predictions
                t_logp = F.log_softmax(t_slice[:L, :] / T, dim=-1)
                s_logp = F.log_softmax(s_slice[:L, :] / T, dim=-1)
                t_p = t_logp.exp()
                kl_tok = (t_p * (t_logp - s_logp)).sum(dim=-1)  # [L]
                kl_sum = kl_tok.sum() * (T * T)
    
                stats[i, 0] = student_sum_nll
                stats[i, 1] = teacher_sum_nll
                stats[i, 2] = kl_sum
                stats[i, 3] = float(L)
    
        dummy_labels = torch.zeros(stats.size(0), dtype=torch.int64, device=stats.device)
        return (loss.detach(), stats.detach(), dummy_labels)



    @classmethod
    def compute_metrics(cls, eval_pred):
        """
        Aggregates stats emitted by prediction_step.
        stats columns:
          0 student_sum_nll
          1 teacher_sum_nll
          2 kl_sum
          3 L (aligned answer-token count used)
        """
        import math
        import numpy as np
    
        stats = eval_pred.predictions
    
        student_sum_nll = stats[:, 0]
        teacher_sum_nll = stats[:, 1]
        kl_sum          = stats[:, 2]
        L_used          = stats[:, 3]
    
        total_tok = float(np.sum(L_used))
        n_examples = float(stats.shape[0])
    
        student_ce = float(np.sum(student_sum_nll)) / max(total_tok, 1.0)
        teacher_ce = float(np.sum(teacher_sum_nll)) / max(total_tok, 1.0)
        kl_per_tok = float(np.sum(kl_sum)) / max(total_tok, 1.0)
    
        return {
            "student_ce_answer": student_ce,
            "teacher_ce_answer": teacher_ce,
            "delta_ce_answer": student_ce - teacher_ce,
            "student_ppl_answer": math.exp(student_ce) if student_ce < 100 else float("inf"),
            "teacher_ppl_answer": math.exp(teacher_ce) if teacher_ce < 100 else float("inf"),
            "kl_per_token_answer": kl_per_tok,
            "n_answer_tokens": total_tok,
            "n_examples": n_examples,
        }
