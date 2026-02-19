from dataclasses import dataclass, field
from typing import Optional, Any

import json
import math
import numpy as np
import evaluate

import torch
import torch.nn.functional as F
import datasets
from omegaconf import DictConfig
from datasets import load_dataset
from transformers import TrainingArguments, Trainer, AutoTokenizer, AutoModelForCausalLM, EvalPrediction

from src.tasks.base import BaseTaskConfig, register_task
# TODO: restructure these imports
from src.training.losses import _count_answer_tokens, _answer_pred_slice, teacher_only_distill_loss
from src.training.callbacks import ExtrinsicValidationFrequency, ExtrinsicValidationConfig, ExtrinsicValidationCallback
from src.models.base import ModelFactory
from src.models.initializers import InitializerName, INITIALIZER_MAP


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
        import torch

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
        )
        student_out = model(
            input_ids=inputs["memory_input_ids"],
            attention_mask=inputs["memory_attention_mask"],
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
        Returns compact per-example stats:
          [student_sum_nll, teacher_sum_nll, kl_sum, L, oracle_answer_len, memory_answer_len]
        where L is the aligned answer-token count actually used for CE/KL.
    
        This patch is focused on diagnosing why eval_n_answer_tokens is small.
        """
        import torch
        import torch.nn.functional as F
    
        def _real_len(attn_mask_1ex: torch.Tensor) -> int:
            return int(attn_mask_1ex.sum().item())
    
        def _answer_lens(prompt_len: int, attn_mask_1ex: torch.Tensor) -> int:
            return max(_real_len(attn_mask_1ex) - prompt_len, 0)
    
        def _answer_pred_slice(logits_1ex: torch.Tensor, prompt_len: int, attn_mask_1ex: torch.Tensor):
            """
            logits_1ex: [seq, vocab]
            Answer token indices: [prompt_len, real_len-1]
            Predicted by logits positions: [prompt_len-1, real_len-2]
            """
            real_len = _real_len(attn_mask_1ex)
            start = max(prompt_len - 1, 0)
            end = max(real_len - 1, 0)  # exclusive
            if end <= start:
                return logits_1ex.new_zeros((0, logits_1ex.size(-1)))
            return logits_1ex[start:end, :]
    
        model.eval()
        inputs = self._prepare_inputs(inputs)
    
        with torch.no_grad():
            oracle_out = model(
                input_ids=inputs["oracle_input_ids"],
                attention_mask=inputs["oracle_attention_mask"],
            )
            student_out = model(
                input_ids=inputs["memory_input_ids"],
                attention_mask=inputs["memory_attention_mask"],
            )
    
            teacher_logits = oracle_out.logits   # [bs, seq, vocab]
            student_logits = student_out.logits  # [bs, seq, vocab]
    
            bs, seq, vocab = student_logits.shape
            T = float(getattr(self, "T", 1.0))
    
            # [student_sum_nll, teacher_sum_nll, kl_sum, L, oracle_answer_len, memory_answer_len]
            stats = student_logits.new_zeros((bs, 6), dtype=torch.float32)
    
            for i in range(bs):
                t_prompt = int(inputs["oracle_prompt_len"][i].item())
                s_prompt = int(inputs["memory_prompt_len"][i].item())
    
                # Pre-alignment answer lengths (diagnostic)
                t_ans_len = _answer_lens(t_prompt, inputs["oracle_attention_mask"][i])
                s_ans_len = _answer_lens(s_prompt, inputs["memory_attention_mask"][i])
    
                stats[i, 4] = float(t_ans_len)
                stats[i, 5] = float(s_ans_len)
    
                # Prediction slices corresponding to answer tokens
                t_slice = _answer_pred_slice(teacher_logits[i], t_prompt, inputs["oracle_attention_mask"][i])  # [Lt, vocab]
                s_slice = _answer_pred_slice(student_logits[i], s_prompt, inputs["memory_attention_mask"][i])  # [Ls, vocab]
    
                L = min(t_slice.size(0), s_slice.size(0))
                if L <= 0:
                    continue
    
                # Targets: the answer tokens themselves (aligned by answer index)
                t_targets = inputs["oracle_input_ids"][i, t_prompt : t_prompt + L]
                s_targets = inputs["memory_input_ids"][i, s_prompt : s_prompt + L]
    
                # Answer-only NLL sums
                teacher_sum_nll = F.cross_entropy(t_slice[:L, :], t_targets, reduction="sum")
                student_sum_nll = F.cross_entropy(s_slice[:L, :], s_targets, reduction="sum")
    
                # KL(teacher || student) over aligned answer predictions
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
        return (None, stats.detach(), dummy_labels)



    @classmethod
    def compute_metrics(cls, eval_pred):
        """
        Aggregates stats emitted by prediction_step.
        stats columns:
          0 student_sum_nll
          1 teacher_sum_nll
          2 kl_sum
          3 L (aligned answer-token count used)
          4 oracle_answer_len (pre-alignment)
          5 memory_answer_len (pre-alignment)
        """
        import math
        import numpy as np
    
        stats = eval_pred.predictions  # numpy [N,6]
    
        student_sum_nll = stats[:, 0]
        teacher_sum_nll = stats[:, 1]
        kl_sum          = stats[:, 2]
        L_used          = stats[:, 3]
        oracle_ans_len  = stats[:, 4]
        memory_ans_len  = stats[:, 5]
    
        total_tok = float(np.sum(L_used))
        n_examples = float(stats.shape[0])
        n_examples_with_answer = float(np.sum(L_used > 0))
    
        # CE/PPL over aligned answer tokens actually used
        student_ce = float(np.sum(student_sum_nll)) / max(total_tok, 1.0)
        teacher_ce = float(np.sum(teacher_sum_nll)) / max(total_tok, 1.0)
        kl_per_tok = float(np.sum(kl_sum)) / max(total_tok, 1.0)
    
        # Diagnostics about answer lengths (pre-alignment)
        frac_oracle_empty = float(np.mean(oracle_ans_len <= 0))
        frac_memory_empty = float(np.mean(memory_ans_len <= 0))
        frac_oracle_shorter = float(np.mean(oracle_ans_len < memory_ans_len))
    
        # Mean/min/max answer lengths over examples where L_used>0
        if n_examples_with_answer > 0:
            mean_answer_len = float(np.sum(L_used)) / n_examples_with_answer
            min_answer_len = float(np.min(L_used[L_used > 0]))
            max_answer_len = float(np.max(L_used[L_used > 0]))
        else:
            mean_answer_len = float("nan")
            min_answer_len = float("nan")
            max_answer_len = float("nan")
    
        return {
            # Core metrics
            "student_ce_answer": student_ce,
            "teacher_ce_answer": teacher_ce,
            "delta_ce_answer": student_ce - teacher_ce,
            "student_ppl_answer": math.exp(student_ce) if student_ce < 100 else float("inf"),
            "teacher_ppl_answer": math.exp(teacher_ce) if teacher_ce < 100 else float("inf"),
            "kl_per_token_answer": kl_per_tok,
            "n_answer_tokens": total_tok,
    
            # Token-count diagnosis
            "n_examples": n_examples,
            "n_examples_with_answer": n_examples_with_answer,
            "mean_answer_len_tokens": mean_answer_len,
            "min_answer_len_tokens": min_answer_len,
            "max_answer_len_tokens": max_answer_len,
    
            # Truncation / masking diagnosis
            "oracle_answer_len_mean": float(np.mean(oracle_ans_len)),
            "memory_answer_len_mean": float(np.mean(memory_ans_len)),
            "frac_oracle_answer_empty": frac_oracle_empty,
            "frac_memory_answer_empty": frac_memory_empty,
            "frac_oracle_shorter_than_memory": frac_oracle_shorter,
        }




@register_task(name="train_memory", group="task")
@dataclass
class TrainMemoryTaskConfig(BaseTaskConfig):
    _target_: str = "src.tasks.train_memory.TrainMemoryTask"
    name: str = "train_memory"
    learning_rate: float = 2e-4
    num_train_epochs: int = 3
    precision: str = "bf16"
    seed: int = 42
    max_length: int = 512
    eval_strategy: str = "epoch"
    eval_steps: int = 100
    sample_n: int = 1000
    sample_strategy: str = "random"
    log_predictions: bool = True
    extrinsic_validation: ExtrinsicValidationConfig = field(default_factory=ExtrinsicValidationConfig)
    save_strategy: str = "steps"

    trainer_config: dict = field(default_factory=lambda: { })
    
    loss_type: str = "balanced"
    alpha: float = 0.2
    temperature: float = 0.5

    initialization_method: InitializerName = InitializerName.RANDOM
    initialization_config: dict = field(default_factory=lambda: { })

    trainable_strategy: str = "all" # all, soft_prompt_only


class TrainMemoryTask:
    def _preprocess(self, examples, tokenizer):
        """
        Builds oracle+memory sequences INCLUDING the answer, and computes prompt lengths
        using the generation templates (no answer) for answer-only masking / slicing.
        """
        max_length = self.config.max_length
    
        bios = examples["biography"]
        qs   = examples["question"]
        ans  = examples["answer"]
    
        # Full sequences (include answer)
        oracle_texts = [
            self.prompts.contextual_qa_training.format(biography=b, question=q, answer=a)
            for b, q, a in zip(bios, qs, ans)
        ]
    
        # You must add this prompt to config:
        # direct_qa_training: "Question: {question}\nAnswer: {answer}"
        memory_texts = [
            self.prompts.direct_qa_training.format(question=q, answer=a)
            for q, a in zip(qs, ans)
        ]
    
        # Prefixes (no answer): used only to compute prompt lengths for masking/slicing
        oracle_prefixes = [
            self.prompts.contextual_qa_generation.format(biography=b, question=q)
            for b, q in zip(bios, qs)
        ]
        memory_prefixes = [
            self.prompts.direct_qa_generation.format(question=q)
            for q in qs
        ]
    
        oracle = tokenizer(oracle_texts, truncation=True, max_length=max_length, add_special_tokens=True)
        memory = tokenizer(memory_texts, truncation=True, max_length=max_length, add_special_tokens=True)
    
        oracle_pref = tokenizer(oracle_prefixes, truncation=True, max_length=max_length, add_special_tokens=True)
        memory_pref = tokenizer(memory_prefixes, truncation=True, max_length=max_length, add_special_tokens=True)
    
        return {
            "oracle_input_ids": oracle.input_ids,
            "oracle_attention_mask": oracle.attention_mask,
            "memory_input_ids": memory.input_ids,
            "memory_attention_mask": memory.attention_mask,
            "oracle_prompt_len": [len(x) for x in oracle_pref.input_ids],
            "memory_prompt_len": [len(x) for x in memory_pref.input_ids],
        }



    def _load_data(self, tokenizer, dataset_config):
        """Loads, samples, and preprocesses the dataset for memory training."""
        dataset = load_dataset("json", data_files=dataset_config.path)["train"]
        original_columns = list(dataset.column_names)

        if self.config.sample_n:
            if self.config.sample_strategy == 'random':
                dataset = dataset.shuffle(seed=self.config.seed).select(range(self.config.sample_n))
            else: # first_n
                dataset = dataset.select(range(self.config.sample_n))

        if self.config.extrinsic_validation.frequency != ExtrinsicValidationFrequency.NEVER:
            remove_columns = None
        else:
            remove_columns = original_columns
        
        tokenized_dataset = dataset.map(
            lambda examples: self._preprocess(examples, tokenizer=tokenizer),
            batched=True,
            remove_columns=remove_columns
        )
        return tokenized_dataset

    def _preprocess_logits_for_metrics(self, logits, labels):
        """
        For CausalLM: return per-example (sum_nll, n_tokens) so compute_metrics
        can aggregate token-weighted loss and perplexity without storing logits.
    
        Returns: tensor [bs, 2] where [:,0]=sum_nll, [:,1]=n_tokens
        """
        if isinstance(logits, (tuple, list)):
            logits = logits[0]  # [bs, seq, vocab]
    
        labels = labels.to(logits.device)
    
        # Ensure labels match logits seq length (pad with -100, don't truncate supervised positions)
        bs, seq_logits, vocab = logits.shape
        seq_labels = labels.shape[1]
        if seq_labels < seq_logits:
            pad = torch.full(
                (bs, seq_logits - seq_labels),
                -100,
                dtype=labels.dtype,
                device=labels.device,
            )
            labels = torch.cat([labels, pad], dim=1)
        elif seq_labels > seq_logits:
            labels = labels[:, :seq_logits]
    
        # Causal shift: predict token t+1 from position t
        shift_logits = logits[:, :-1, :]   # [bs, seq-1, vocab]
        shift_labels = labels[:, 1:]       # [bs, seq-1]
    
        # Per-token CE (no reduction), ignoring -100
        per_tok_nll = F.cross_entropy(
            shift_logits.reshape(-1, vocab),
            shift_labels.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).view(bs, -1)  # [bs, seq-1]
    
        mask = (shift_labels != -100)
        sum_nll = (per_tok_nll * mask).sum(dim=1)                 # [bs]
        n_tok = mask.sum(dim=1).to(dtype=sum_nll.dtype)           # [bs] as float
    
        return torch.stack([sum_nll, n_tok], dim=1).detach()

    def _compute_metrics(self, eval_preds: EvalPrediction):
        stats, _labels = eval_preds  # stats is [N,2] after concatenation over eval set
        sum_nll = stats[:, 0]
        n_tok   = stats[:, 1]
    
        total_nll = float(np.sum(sum_nll))
        total_tok = float(np.sum(n_tok))
    
        mean_loss = total_nll / max(total_tok, 1.0)
        ppl = math.exp(mean_loss) if mean_loss < 100 else float("inf")
    
        return {
            "recomputed_loss": mean_loss,
            "perplexity": ppl,
            "n_tokens": total_tok,
        }

    def __init__(self, **kwargs):
        self.config = TrainMemoryTaskConfig(**kwargs)
        self.metric = evaluate.load("perplexity")

    def main(self, cwd: str, cfg: DictConfig):
        """
        Main entrypoint for the memory training task.
        Incorporates the logic from the 'sequential' training strategy.
        """
        print(f"Running TrainMemoryTask: {self.config.name}")

        self.prompts = cfg.prompts
        
        model, tokenizer = ModelFactory.load(cfg)
        
        # Explicitly initialize the soft prompt if the model supports it
        if hasattr(model, 'initialize_virtual_prompt') and cfg.model.model_config.initialization_method:
            print(f"Performing soft prompt initialization with method: '{self.config.initialization_method.value}'...")
            
            initializer_class = INITIALIZER_MAP.get(self.config.initialization_method.value)
            if not initializer_class:
                raise ValueError(f"Unknown initializer: {self.config.initialization_method.value}")
            
            initializer = initializer_class()
            
            init_kwargs = {
                "main_model": self.model.model,
                "main_tokenizer": tokenizer,
                "virtual_token_count": self.config.virtual_token_count,
                "dataset_path": dataset_path,
                **self.config.initializer_config,
            }
            
            initial_weights = initializer.initialize(**init_kwargs)
            model.model.rebuild_virtual_prompt(weights=initial_weights)
            # model.initialize_virtual_prompt(
            #     tokenizer=tokenizer,
            #     method=cfg.model.model_config.initialization_method,
            #     config=cfg.model.model_config.initializer_config,
            #     dataset_path=cfg.dataset.path
            # )

        # TODO: maybe pass this into initializers instead of path
        tokenized_dataset = self._load_data(tokenizer, cfg.dataset)

        if self.config.trainable_strategy == "soft_prompt_only":
            print("Freezing base model, training soft prompt only.")
            for name, param in model.named_parameters():
                if "virtual_prompt" not in name:
                    param.requires_grad = False
        
        training_args = TrainingArguments(
            output_dir=f"{cwd}/",
            learning_rate=self.config.learning_rate,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=cfg.dataset.physical_batch_size,
            gradient_accumulation_steps=cfg.dataset.accumulation_steps,
            seed=self.config.seed,
            remove_unused_columns=False,
            ddp_find_unused_parameters=False,
            eval_strategy=self.config.eval_strategy,
            eval_steps=self.config.eval_steps,
            save_strategy=self.config.save_strategy,
            prediction_loss_only=False,
        )

        trainer_dataset = tokenized_dataset.remove_columns(
            [c for c in tokenized_dataset.column_names if isinstance(tokenized_dataset.features[c], datasets.Value) and tokenized_dataset.features[c].dtype == 'string']
        )

        callbacks = []
        if self.config.extrinsic_validation.frequency != ExtrinsicValidationFrequency.NEVER:
            callbacks.append(ExtrinsicValidationCallback(self.config.extrinsic_validation, tokenized_dataset, tokenizer, cwd, self.prompts))

        trainer = MemoryBankTrainer(
            model=model,
            loss_type=self.config.loss_type,
            loss_alpha=self.config.alpha,
            temperature=self.config.temperature,
            args=training_args,
            train_dataset=trainer_dataset,
            eval_dataset=trainer_dataset,
            callbacks=callbacks,
            tokenizer=tokenizer,
            data_collator=SelfDistillationDataCollator(tokenizer),
            compute_metrics=MemoryBankTrainer.compute_metrics,
            # preprocess_logits_for_metrics=self._preprocess_logits_for_metrics,
        )

        print("Starting memory training...")
        results = trainer.train()
        trainer.save_model()
        trainer.save_metrics("eval", results.metrics)
        with open(f"{cwd}/eval_log.json", "w") as fout:
            json.dump(trainer.state.log_history, fout)
        print("Memory training complete.")
