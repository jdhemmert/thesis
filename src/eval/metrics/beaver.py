import math
import heapq
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Optional, Type
from abc import ABC, abstractmethod
from enum import Enum, auto

import torch
import torch.nn.functional as F

from .base import BaseMetric


# ============================================================
# Constraint status + KMP helpers
# ============================================================

class ConstraintStatus(Enum):
    IMPOSSIBLE = auto()
    UNDECIDED = auto()
    SATISFIED = auto()


def _kmp_build_pi(pattern: List[int]) -> List[int]:
    """Build KMP prefix function pi for pattern (token ids)."""
    L = len(pattern)
    pi = [0] * L
    j = 0
    for i in range(1, L):
        while j > 0 and pattern[i] != pattern[j]:
            j = pi[j - 1]
        if pattern[i] == pattern[j]:
            j += 1
            pi[i] = j
    return pi


# ============================================================
# Constraints
# ============================================================

class BaseConstraint(ABC):
    """Abstract base class for all semantic constraints."""
    def __init__(self, tokenizer: Any, **kwargs):
        self.tokenizer = tokenizer

    @abstractmethod
    def status(self, *, found: bool, match_len: int, gen_len: int, horizon: int) -> ConstraintStatus:
        """Three-valued status used for pruning/absorbing."""
        raise NotImplementedError

    @abstractmethod
    def step(self, match_len: int, token_id: int) -> Tuple[int, bool]:
        """Update KMP state and return (new_match_len, found_now)."""
        raise NotImplementedError


class ConstraintRegistry:
    _registry: Dict[str, Type[BaseConstraint]] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(constraint_class: Type[BaseConstraint]):
            cls._registry[name] = constraint_class
            return constraint_class
        return decorator

    @classmethod
    def get_constraint(cls, name: str, tokenizer: Any, **kwargs) -> BaseConstraint:
        constraint_class = cls._registry.get(name)
        if constraint_class is None:
            raise ValueError(f"Constraint '{name}' not registered. Available: {list(cls._registry.keys())}")
        return constraint_class(tokenizer=tokenizer, **kwargs)


CONSTRAINT_REGISTRY = ConstraintRegistry()


@CONSTRAINT_REGISTRY.register("answer_in_k")
class AnswerInKConstraint(BaseConstraint):
    """
    Answer occurs as a contiguous substring within the first k generated tokens.

    Implements:
      - SATISFIED absorption once found
      - IMPOSSIBLE pruning when remaining tokens can't complete match (KMP match_len)
    """
    def __init__(self, tokenizer: Any, correct_answer_tokens: List[int], k_tokens: int):
        super().__init__(tokenizer)
        self.ans = list(correct_answer_tokens)
        self.k_tokens = int(k_tokens)
        self.L = len(self.ans)
        self.pi = _kmp_build_pi(self.ans) if self.L > 0 else []

    def status(self, *, found: bool, match_len: int, gen_len: int, horizon: int) -> ConstraintStatus:
        if self.L == 0:
            return ConstraintStatus.SATISFIED
        if found:
            return ConstraintStatus.SATISFIED

        # Effective horizon is provided by caller, but must be <= k_tokens for soundness.
        remaining = max(0, horizon - gen_len)
        need = self.L - match_len

        if remaining < need:
            return ConstraintStatus.IMPOSSIBLE

        return ConstraintStatus.UNDECIDED

    def step(self, match_len: int, token_id: int) -> Tuple[int, bool]:
        if self.L == 0:
            return 0, True

        j = match_len
        while j > 0 and self.ans[j] != token_id:
            j = self.pi[j - 1]
        if self.ans[j] == token_id:
            j += 1
        if j == self.L:
            # Found a full match; allow overlaps by falling back via pi
            found_now = True
            j = self.pi[j - 1] if self.L > 0 else 0
            return j, found_now

        return j, False


# ============================================================
# Config
# ============================================================

@dataclass
class BeaverMetricConfig:
    budget: int = 100
    epsilon: float = 0.01
    selection_strategy: str = "max_mu"

    # Force stop at this many GENERATED tokens
    max_sequence_length: int = 50

    # Dataset sampling
    sample_count: int = 10

    # Constraint config
    constraint_type: str = "answer_in_k"
    correct_answer_dataset_key: str = "answer"
    k_tokens: int = 50

    # Optimizations
    top_k_initial: int = 128
    top_k_max: int = 4096
    top_k_refine_factor: int = 2

    # NEW: priority shaping
    depth_priority_alpha: float = 0.15     # boost deeper nodes slightly
    tail_priority_scale: float = 0.10      # tails get downweighted vs prefixes

    # TRACE / DEBUG LOGGING
    trace: bool = False
    trace_every: int = 1000
    trace_top_items: int = 5
    trace_decode_top_items: bool = False
    trace_decode_max_tokens: int = 30
    trace_jsonl_filename: str = "beaver_trace.jsonl"
    trace_print: bool = True


# ============================================================
# Frontier items
# ============================================================

@dataclass(order=True)
class FrontierItem:
    # min-heap; we want largest "effective score" first => priority = -score
    priority: float
    kind: str = field(compare=False)  # "prefix" or "tail"
    token_ids: Tuple[int, ...] = field(compare=False)

    prefix_log_prob: float = field(compare=False)
    mass: float = field(compare=False)

    # KMP / constraint state for GENERATED suffix
    match_len: int = field(compare=False, default=0)
    found: bool = field(compare=False, default=False)

    # tail bookkeeping
    k_used: int = field(compare=False, default=0)
    tail_factor: float = field(compare=False, default=0.0)


# ============================================================
# Beaver Metric
# ============================================================

class BeaverMetric(BaseMetric):
    def __init__(self, config: BeaverMetricConfig, tokenizer: Any, eval_dataset: Any, output_dir: str):
        super().__init__(config, tokenizer, eval_dataset, output_dir)
        self.output_dir = output_dir
        self.precomputed_prompts = self._preprocess_eval_dataset(eval_dataset)
        print(
            f"BeaverMetric initialized for '{self.config.constraint_type}' constraint with "
            f"{len(self.precomputed_prompts['initial_input_ids'])} examples."
        )

    def _preprocess_eval_dataset(self, eval_dataset: Any) -> Dict[str, List[Any]]:
        print("Preprocessing evaluation dataset for BeaverMetric...")
        required_cols = ["memory_input_ids", "memory_attention_mask", self.config.correct_answer_dataset_key, "question"]
        if not all(col in eval_dataset.column_names for col in required_cols):
            raise ValueError(f"Required columns {required_cols} not found in dataset: {eval_dataset.column_names}.")

        precomputed = {
            "initial_input_ids": [],
            "initial_attention_mask": [],
            "correct_answer_tokens": [],
            "original_question": [],
        }

        for idx in range(self.config.sample_count):
            ex = eval_dataset[idx]
            precomputed["initial_input_ids"].append(ex["memory_input_ids"])
            precomputed["initial_attention_mask"].append(ex["memory_attention_mask"])

            ans = ex[self.config.correct_answer_dataset_key]
            precomputed["correct_answer_tokens"].append(self.tokenizer.encode(ans, add_special_tokens=False))
            precomputed["original_question"].append(ex.get("question", "N/A"))

        return precomputed

    # -------------------------
    # TRACE HELPERS
    # -------------------------

    def _trace_path(self) -> str:
        os.makedirs(self.output_dir, exist_ok=True)
        return os.path.join(self.output_dir, self.config.trace_jsonl_filename)

    def _trace_write(self, fp: Optional[Any], record: Dict[str, Any]):
        if fp is None:
            return
        fp.write(json.dumps(record, ensure_ascii=False) + "\n")
        fp.flush()

    def _decode_suffix(self, token_ids: Tuple[int, ...], prompt_len: int) -> str:
        if not self.config.trace_decode_top_items:
            return ""
        suffix = list(token_ids[prompt_len:prompt_len + self.config.trace_decode_max_tokens])
        try:
            return self.tokenizer.decode(suffix, skip_special_tokens=True)
        except Exception:
            return ""

    def _snapshot_top_items(self, frontier: List[FrontierItem], prompt_len: int) -> List[Dict[str, Any]]:
        M = max(0, int(self.config.trace_top_items))
        if M == 0 or not frontier:
            return []
        top = heapq.nsmallest(min(M, len(frontier)), frontier)
        out = []
        for it in top:
            gen_len = len(it.token_ids) - prompt_len
            rec = {
                "kind": it.kind,
                "mass": it.mass,
                "gen_len": gen_len,
                "match_len": it.match_len,
                "found": it.found,
            }
            if it.kind == "tail":
                rec["k_used"] = it.k_used
                rec["tail_factor"] = it.tail_factor
            if self.config.trace_decode_top_items:
                rec["suffix_preview"] = self._decode_suffix(it.token_ids, prompt_len)
            out.append(rec)
        return out

    # -------------------------
    # MODEL HELPERS
    # -------------------------

    @staticmethod
    def _safe_exp(log_x: float) -> float:
        if log_x < -745.0:
            return 0.0
        return math.exp(log_x)

    def _model_next_logprobs(self, model: Any, device: torch.device, token_ids: Tuple[int, ...]) -> torch.Tensor:
        input_ids = torch.tensor([list(token_ids)], device=device)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
        next_logits = logits[0, -1, :]
        return F.log_softmax(next_logits, dim=-1)

    # -------------------------
    # PRIORITY
    # -------------------------

    def _effective_score(self, mass: float, gen_len: int, kind: str) -> float:
        # Larger score should be popped earlier. We store priority = -score.
        alpha = float(self.config.depth_priority_alpha)
        score = mass * (1.0 + alpha * float(gen_len))
        if kind == "tail":
            score *= float(self.config.tail_priority_scale)
        return score

    def _make_item(
        self,
        *,
        kind: str,
        token_ids: Tuple[int, ...],
        prefix_log_prob: float,
        mass: float,
        prompt_len: int,
        match_len: int,
        found: bool,
        k_used: int = 0,
        tail_factor: float = 0.0,
    ) -> FrontierItem:
        gen_len = len(token_ids) - prompt_len
        score = self._effective_score(mass, gen_len, kind)
        return FrontierItem(
            priority=-score,
            kind=kind,
            token_ids=token_ids,
            prefix_log_prob=prefix_log_prob,
            mass=mass,
            match_len=match_len,
            found=found,
            k_used=k_used,
            tail_factor=tail_factor,
        )

    def _pop_near_horizon_prefix(self, frontier: List[FrontierItem], prompt_len: int, threshold: int) -> Optional[FrontierItem]:
        """
        Find and remove a prefix item with gen_len >= threshold.
        Among those candidates, pick the one with largest mass (or largest priority score).
        Returns None if none exist.
        """
        best_idx = None
        best_mass = -1.0
    
        for idx, it in enumerate(frontier):
            if it.kind != "prefix":
                continue
            gen_len = len(it.token_ids) - prompt_len
            if gen_len < threshold:
                continue
            if it.mass > best_mass:
                best_mass = it.mass
                best_idx = idx
    
        if best_idx is None:
            return None
    
        chosen = frontier[best_idx]
        frontier[best_idx] = frontier[-1]
        frontier.pop()
        heapq.heapify(frontier)
        return chosen


    # -------------------------
    # METRIC ENTRYPOINT
    # -------------------------

    def compute_and_log_scores(self, model: Any, state: Any, metrics: Dict):
        if model is None or self.tokenizer is None:
            print("Skipping BeaverMetric evaluation: model or tokenizer not available.")
            return

        print(f"\nPerforming BeaverMetric({self.config.constraint_type}) Evaluation...")
        model.eval()

        all_plb, all_pub = [], []

        for i in range(len(self.precomputed_prompts["initial_input_ids"])):
            initial_input_ids = self.precomputed_prompts["initial_input_ids"][i]
            initial_attention_mask = self.precomputed_prompts["initial_attention_mask"][i]
            correct_answer_tokens = self.precomputed_prompts["correct_answer_tokens"][i]
            original_question = self.precomputed_prompts["original_question"][i]

            semantic_constraint = CONSTRAINT_REGISTRY.get_constraint(
                self.config.constraint_type,
                tokenizer=self.tokenizer,
                correct_answer_tokens=correct_answer_tokens,
                k_tokens=self.config.k_tokens,
            )

            plb, pub = self._run_beaver_for_prompt(
                prompt_index=i,
                model=model,
                initial_input_ids=initial_input_ids,
                initial_attention_mask=initial_attention_mask,
                semantic_constraint=semantic_constraint,
                eos_token_id=self.tokenizer.eos_token_id,
                max_response_length=self.config.max_sequence_length,
                budget=self.config.budget,
                epsilon=self.config.epsilon,
                top_k_initial=self.config.top_k_initial,
                top_k_max=self.config.top_k_max,
                top_k_refine_factor=self.config.top_k_refine_factor,
            )

            all_plb.append(plb)
            all_pub.append(pub)
            print(f"  Prompt {i+1}: Q: '{original_question}' Bounds: [{plb:.6f}, {pub:.6f}] Gap: {(pub-plb):.6f}")

        avg_plb = sum(all_plb) / len(all_plb) if all_plb else 0.0
        avg_pub = sum(all_pub) / len(all_pub) if all_pub else 0.0
        avg_gap = avg_pub - avg_plb

        metrics["eval_beaver_plb"] = avg_plb
        metrics["eval_beaver_pub"] = avg_pub
        metrics["eval_beaver_gap"] = avg_gap

        print(
            f"\nAggregated BeaverMetric({self.config.constraint_type}) Scores: "
            f"Avg PLB={avg_plb:.6f}, Avg PUB={avg_pub:.6f}, Avg Gap={avg_gap:.6f}"
        )

    # -------------------------
    # CORE BEAVER LOOP
    # -------------------------

    def _run_beaver_for_prompt(
        self,
        prompt_index: int,
        model: Any,
        initial_input_ids: List[int],
        initial_attention_mask: List[int],
        semantic_constraint: BaseConstraint,
        eos_token_id: Optional[int],
        max_response_length: int,
        budget: int,
        epsilon: float,
        top_k_initial: int,
        top_k_max: int,
        top_k_refine_factor: int,
    ) -> Tuple[float, float]:
        device = model.device
        prompt = tuple(initial_input_ids)
        prompt_len = len(prompt)

        # Effective horizon = min(force stop, constraint k) for sound pruning.
        horizon = min(int(max_response_length), int(getattr(semantic_constraint, "k_tokens", max_response_length)))

        # Trace
        fp = None
        if self.config.trace:
            fp = open(self._trace_path(), "a", encoding="utf-8")

        def checkpoint(step: int, reason: str, frontier: List[FrontierItem],
                       plb_mass: float, frontier_mass: float,
                       frontier_prefix_mass: float, frontier_tail_mass: float,
                       counters: Dict[str, int]):
            if not self.config.trace and not self.config.trace_print:
                return
            record = {
                "type": "checkpoint",
                "prompt_index": prompt_index,
                "reason": reason,
                "step": step,
                "budget": budget,
                "epsilon": epsilon,
                "horizon": horizon,
                "plb": plb_mass,
                "pub": plb_mass + frontier_mass,
                "gap": (plb_mass + frontier_mass) - plb_mass,
                "frontier_size": len(frontier),
                "frontier_mass": frontier_mass,
                "frontier_prefix_mass": frontier_prefix_mass,
                "frontier_tail_mass": frontier_tail_mass,
                "counters": counters,
                "top_items": self._snapshot_top_items(frontier, prompt_len),
            }
            if self.config.trace_print:
                print(
                    f"[beaver trace] prompt={prompt_index} step={step} reason={reason} "
                    f"plb={record['plb']:.6g} pub={record['pub']:.6g} gap={record['gap']:.6g} "
                    f"frontier={record['frontier_size']} mass={record['frontier_mass']:.6g} "
                    f"(prefix={record['frontier_prefix_mass']:.6g}, tail={record['frontier_tail_mass']:.6g})"
                )
            self._trace_write(fp, record)

        # Frontier heap + mass bookkeeping
        frontier: List[FrontierItem] = []
        plb_mass = 0.0
        frontier_mass = 0.0
        frontier_prefix_mass = 0.0
        frontier_tail_mass = 0.0

        counters = {
            "expand_prefix": 0,
            "refine_tail": 0,
            "children_pushed_prefix": 0,
            "children_pushed_tail": 0,
            "eos_terminals": 0,
            "force_stop_terminals": 0,
            "pruned_impossible": 0,
            "absorbed_satisfied": 0,
            "tail_saturated_hits": 0,
        }

        # Start item
        start = self._make_item(
            kind="prefix",
            token_ids=prompt,
            prefix_log_prob=0.0,
            mass=1.0,
            prompt_len=prompt_len,
            match_len=0,
            found=False,
        )
        heapq.heappush(frontier, start)
        frontier_mass = 1.0
        frontier_prefix_mass = 1.0

        def pop_item() -> FrontierItem:
            nonlocal frontier_mass, frontier_prefix_mass, frontier_tail_mass
            it = heapq.heappop(frontier)
            frontier_mass -= it.mass
            if it.kind == "prefix":
                frontier_prefix_mass -= it.mass
            else:
                frontier_tail_mass -= it.mass
            return it

        def push_item(it: FrontierItem):
            nonlocal frontier_mass, frontier_prefix_mass, frontier_tail_mass
            heapq.heappush(frontier, it)
            frontier_mass += it.mass
            if it.kind == "prefix":
                frontier_prefix_mass += it.mass
            else:
                frontier_tail_mass += it.mass

        def add_good_mass(m: float):
            nonlocal plb_mass
            plb_mass += m

        checkpoint(0, "start", frontier, plb_mass, frontier_mass, frontier_prefix_mass, frontier_tail_mass, counters)

        current_budget = 0

        while frontier and current_budget < budget:
            pub = plb_mass + frontier_mass
            if (pub - plb_mass) <= epsilon:
                checkpoint(current_budget, "epsilon_reached", frontier, plb_mass, frontier_mass,
                           frontier_prefix_mass, frontier_tail_mass, counters)
                break

            force_depth = True  # could be config-controlled

            forced_it = None
            if force_depth:
                L = int(getattr(semantic_constraint, "L", 0))  # answer token length
                if L > 0:
                    threshold = horizon - L
                    if threshold < 0:
                        threshold = 0
            
                    forced_it = self._pop_near_horizon_prefix(
                        frontier=frontier,
                        prompt_len=prompt_len,
                        threshold=threshold,
                    )
            
            if forced_it is not None:
                it = forced_it
                # IMPORTANT: update frontier_mass split because we removed from heap manually
                frontier_mass -= it.mass
                if it.kind == "prefix":
                    frontier_prefix_mass -= it.mass
                else:
                    frontier_tail_mass -= it.mass
            else:
                it = pop_item()

            gen_len = len(it.token_ids) - prompt_len

            # Constraint status at this prefix
            st = semantic_constraint.status(found=it.found, match_len=it.match_len, gen_len=gen_len, horizon=horizon)

            if st == ConstraintStatus.IMPOSSIBLE:
                counters["pruned_impossible"] += 1
                continue

            if st == ConstraintStatus.SATISFIED:
                counters["absorbed_satisfied"] += 1
                add_good_mass(it.mass)
                continue

            # Force-stop horizon reached => terminal; undecided here means "didn't satisfy"
            if gen_len >= horizon:
                counters["force_stop_terminals"] += 1
                continue

            if it.kind == "prefix":
                counters["expand_prefix"] += 1
                current_budget += 1

                next_log_probs = self._model_next_logprobs(model, device, it.token_ids)
                vocab_size = int(next_log_probs.shape[-1])
                if eos_token_id is not None and eos_token_id >= vocab_size:
                    raise ValueError(f"eos_token_id={eos_token_id} out of range for logits vocab_size={vocab_size}")

                k = min(top_k_initial, vocab_size)
                top_logp, top_ids = torch.topk(next_log_probs, k=k)

                top_probs = torch.exp(top_logp)
                m_S = float(top_probs.sum().item())
                if m_S > 1.0:
                    m_S = 1.0
                m_T = max(0.0, 1.0 - m_S)

                # Concrete children
                for j in range(k):
                    tid = int(top_ids[j].item())
                    cond_logp = float(top_logp[j].item())
                    child_prefix_logp = it.prefix_log_prob + cond_logp
                    child_mass = self._safe_exp(child_prefix_logp)
                    if child_mass == 0.0:
                        continue

                    new_match_len, found_now = semantic_constraint.step(it.match_len, tid)
                    child_found = it.found or found_now
                    child_tok = it.token_ids + (tid,)
                    child_gen_len = gen_len + 1

                    child_st = semantic_constraint.status(
                        found=child_found, match_len=new_match_len, gen_len=child_gen_len, horizon=horizon
                    )
                    if child_st == ConstraintStatus.IMPOSSIBLE:
                        counters["pruned_impossible"] += 1
                        continue
                    if child_st == ConstraintStatus.SATISFIED:
                        counters["absorbed_satisfied"] += 1
                        add_good_mass(child_mass)
                        continue

                    if eos_token_id is not None and tid == eos_token_id:
                        counters["eos_terminals"] += 1
                        continue

                    push_item(self._make_item(
                        kind="prefix",
                        token_ids=child_tok,
                        prefix_log_prob=child_prefix_logp,
                        mass=child_mass,
                        prompt_len=prompt_len,
                        match_len=new_match_len,
                        found=child_found,
                    ))
                    counters["children_pushed_prefix"] += 1

                # Tail bucket
                if m_T > 0.0 and k < vocab_size:
                    tail_mass = it.mass * m_T
                    if tail_mass > 0.0:
                        push_item(self._make_item(
                            kind="tail",
                            token_ids=it.token_ids,
                            prefix_log_prob=it.prefix_log_prob,
                            mass=tail_mass,
                            prompt_len=prompt_len,
                            match_len=it.match_len,
                            found=it.found,
                            k_used=k,
                            tail_factor=m_T,
                        ))
                        counters["children_pushed_tail"] += 1

            elif it.kind == "tail":
                # --- Improvement #1: stop wasting budget on saturated tails ---
                if it.k_used >= top_k_max:
                    counters["tail_saturated_hits"] += 1
                    # Put it back (still unresolved mass contributes to PUB),
                    # but stop the run because we cannot make progress under current top_k_max.
                    push_item(it)
                    checkpoint(current_budget, "tail_saturated_no_progress", frontier, plb_mass, frontier_mass,
                               frontier_prefix_mass, frontier_tail_mass, counters)
                    break

                counters["refine_tail"] += 1
                current_budget += 1

                old_k = it.k_used
                new_k = min(top_k_max, max(old_k * top_k_refine_factor, old_k + 1))
                if new_k <= old_k:
                    # Saturated in practice; same logic as above
                    counters["tail_saturated_hits"] += 1
                    push_item(it)
                    checkpoint(current_budget, "tail_saturated_no_progress", frontier, plb_mass, frontier_mass,
                               frontier_prefix_mass, frontier_tail_mass, counters)
                    break

                next_log_probs = self._model_next_logprobs(model, device, it.token_ids)
                vocab_size = int(next_log_probs.shape[-1])
                if eos_token_id is not None and eos_token_id >= vocab_size:
                    raise ValueError(f"eos_token_id={eos_token_id} out of range for logits vocab_size={vocab_size}")

                new_k = min(new_k, vocab_size)
                top_logp, top_ids = torch.topk(next_log_probs, k=new_k)

                top_probs = torch.exp(top_logp)
                m_S_new = float(top_probs.sum().item())
                if m_S_new > 1.0:
                    m_S_new = 1.0
                m_T_new = max(0.0, 1.0 - m_S_new)

                # Newly revealed children [old_k, new_k)
                for j in range(old_k, new_k):
                    tid = int(top_ids[j].item())
                    cond_logp = float(top_logp[j].item())
                    child_prefix_logp = it.prefix_log_prob + cond_logp
                    child_mass = self._safe_exp(child_prefix_logp)
                    if child_mass == 0.0:
                        continue

                    new_match_len, found_now = semantic_constraint.step(it.match_len, tid)
                    child_found = it.found or found_now
                    child_tok = it.token_ids + (tid,)
                    child_gen_len = gen_len + 1

                    child_st = semantic_constraint.status(
                        found=child_found, match_len=new_match_len, gen_len=child_gen_len, horizon=horizon
                    )
                    if child_st == ConstraintStatus.IMPOSSIBLE:
                        counters["pruned_impossible"] += 1
                        continue
                    if child_st == ConstraintStatus.SATISFIED:
                        counters["absorbed_satisfied"] += 1
                        add_good_mass(child_mass)
                        continue

                    if eos_token_id is not None and tid == eos_token_id:
                        counters["eos_terminals"] += 1
                        continue

                    push_item(self._make_item(
                        kind="prefix",
                        token_ids=child_tok,
                        prefix_log_prob=child_prefix_logp,
                        mass=child_mass,
                        prompt_len=prompt_len,
                        match_len=new_match_len,
                        found=child_found,
                    ))
                    counters["children_pushed_prefix"] += 1

                # Smaller tail bucket
                if m_T_new > 0.0 and new_k < vocab_size:
                    tail_mass_new = self._safe_exp(it.prefix_log_prob) * m_T_new
                    if tail_mass_new > 0.0:
                        push_item(self._make_item(
                            kind="tail",
                            token_ids=it.token_ids,
                            prefix_log_prob=it.prefix_log_prob,
                            mass=tail_mass_new,
                            prompt_len=prompt_len,
                            match_len=it.match_len,
                            found=it.found,
                            k_used=new_k,
                            tail_factor=m_T_new,
                        ))
                        counters["children_pushed_tail"] += 1

            else:
                raise ValueError(f"Unknown frontier item kind: {it.kind}")

            if self.config.trace and self.config.trace_every > 0 and (current_budget % int(self.config.trace_every) == 0):
                checkpoint(current_budget, "periodic", frontier, plb_mass, frontier_mass,
                           frontier_prefix_mass, frontier_tail_mass, counters)

        # Final bounds
        plb = plb_mass
        pub = plb_mass + frontier_mass
        pub = min(1.0, max(plb, pub))
        plb = max(0.0, min(plb, pub))

        if self.config.trace:
            checkpoint(current_budget, "end", frontier, plb_mass, frontier_mass,
                       frontier_prefix_mass, frontier_tail_mass, counters)
            if fp is not None:
                fp.close()

        return plb, pub
