from __future__ import annotations

import json
import math
import re
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple


_EPOCH_RE = re.compile(r"\.epoch_(\d+)")


def load_prediction_rows(prediction_log: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with prediction_log.open("r") as fin:
        for line_no, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{prediction_log}:{line_no}: invalid JSONL") from e
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def extract_prediction_epoch(prediction_log: Path) -> Optional[int]:
    m = _EPOCH_RE.search(prediction_log.name)
    if not m:
        return None
    return int(m.group(1))


def _finite_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    return v


def _bool_as_float(x: Any) -> Optional[float]:
    if isinstance(x, bool):
        return 1.0 if x else 0.0
    return None


def _normalize_answer_text(s: Any) -> str:
    if s is None:
        return ""
    return " ".join(str(s).strip().split()).lower()


def summarize_prediction_rows(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Collapse one prediction log into run-level scalar summaries for sweep heatmaps.
    """
    if not rows:
        return {}

    gold_prob_delta: List[float] = []
    argmax_changed: List[float] = []

    gold_prob_with_mem: List[float] = []
    gold_prob_no_mem: List[float] = []

    gold_rank_with_mem: List[float] = []
    gold_rank_no_mem: List[float] = []

    gold_margin_with_mem: List[float] = []
    gold_margin_no_mem: List[float] = []

    vt_argmax_changed: List[float] = []
    max_abs_logit_delta: List[float] = []
    mean_abs_logit_delta: List[float] = []
    l2_logit_delta: List[float] = []

    first_token_match_gold: List[float] = []
    full_answer_exact_match: List[float] = []

    vp_active: List[float] = []

    for row in rows:
        me = row.get("student_memory_effect") or {}
        sd = row.get("student_next_token_diagnostics") or {}
        snd = row.get("student_no_memory_next_token_diagnostics") or {}
        vtd = row.get("virtual_token_effect_diagnostics") or {}
        gft = row.get("generated_first_token_comparison") or {}

        v = _finite_float(me.get("gold_first_token_prob_delta"))
        if v is not None:
            gold_prob_delta.append(v)

        v = _bool_as_float(me.get("argmax_changed_vs_no_memory"))
        if v is not None:
            argmax_changed.append(v)

        v = _finite_float(sd.get("gold_first_token_prob"))
        if v is not None:
            gold_prob_with_mem.append(v)

        v = _finite_float(snd.get("gold_first_token_prob"))
        if v is not None:
            gold_prob_no_mem.append(v)

        v = _finite_float(sd.get("gold_first_token_rank"))
        if v is not None:
            gold_rank_with_mem.append(v)

        v = _finite_float(snd.get("gold_first_token_rank"))
        if v is not None:
            gold_rank_no_mem.append(v)

        v = _finite_float(sd.get("gold_vs_best_logit_margin"))
        if v is not None:
            gold_margin_with_mem.append(v)

        v = _finite_float(snd.get("gold_vs_best_logit_margin"))
        if v is not None:
            gold_margin_no_mem.append(v)

        v = _bool_as_float(vtd.get("argmax_changed"))
        if v is not None:
            vt_argmax_changed.append(v)

        v = _finite_float(vtd.get("max_abs_logit_delta"))
        if v is not None:
            max_abs_logit_delta.append(v)

        v = _finite_float(vtd.get("mean_abs_logit_delta"))
        if v is not None:
            mean_abs_logit_delta.append(v)

        v = _finite_float(vtd.get("l2_logit_delta"))
        if v is not None:
            l2_logit_delta.append(v)

        v = _bool_as_float(gft.get("student_generated_first_token_matches_gold"))
        if v is not None:
            first_token_match_gold.append(v)

        gt = _normalize_answer_text(row.get("ground_truth"))
        pred = _normalize_answer_text(row.get("student_prediction"))
        if gt:
            full_answer_exact_match.append(1.0 if pred == gt else 0.0)

        if isinstance(row.get("vp_active_for_eval"), bool):
            vp_active.append(1.0 if row["vp_active_for_eval"] else 0.0)

    def maybe_mean(vals: List[float]) -> Optional[float]:
        return mean(vals) if vals else None

    out: Dict[str, float] = {}

    maybe_items = {
        "n_examples": float(len(rows)),
        "mean_gold_first_token_prob_delta": maybe_mean(gold_prob_delta),
        "frac_argmax_changed_vs_no_memory": maybe_mean(argmax_changed),
        "mean_gold_first_token_prob_with_memory": maybe_mean(gold_prob_with_mem),
        "mean_gold_first_token_prob_no_memory": maybe_mean(gold_prob_no_mem),
        "mean_gold_first_token_rank_with_memory": maybe_mean(gold_rank_with_mem),
        "mean_gold_first_token_rank_no_memory": maybe_mean(gold_rank_no_mem),
        "mean_gold_vs_best_logit_margin_with_memory": maybe_mean(gold_margin_with_mem),
        "mean_gold_vs_best_logit_margin_no_memory": maybe_mean(gold_margin_no_mem),
        "frac_virtual_token_argmax_changed": maybe_mean(vt_argmax_changed),
        "mean_max_abs_logit_delta": maybe_mean(max_abs_logit_delta),
        "mean_mean_abs_logit_delta": maybe_mean(mean_abs_logit_delta),
        "mean_l2_logit_delta": maybe_mean(l2_logit_delta),
        "frac_generated_first_token_matches_gold": maybe_mean(first_token_match_gold),
        "frac_full_answer_exact_match": maybe_mean(full_answer_exact_match),
        "frac_vp_active_for_eval": maybe_mean(vp_active),
    }

    for k, v in maybe_items.items():
        if v is not None and math.isfinite(v):
            out[k] = float(v)

    # Useful derived metrics
    if "mean_gold_first_token_prob_with_memory" in out and "mean_gold_first_token_prob_no_memory" in out:
        out["mean_gold_first_token_prob_improvement_ratio"] = (
            out["mean_gold_first_token_prob_with_memory"] /
            max(out["mean_gold_first_token_prob_no_memory"], 1e-12)
        )

    if "mean_gold_first_token_rank_with_memory" in out and "mean_gold_first_token_rank_no_memory" in out:
        out["mean_gold_first_token_rank_improvement"] = (
            out["mean_gold_first_token_rank_no_memory"] -
            out["mean_gold_first_token_rank_with_memory"]
        )

    if "mean_gold_vs_best_logit_margin_with_memory" in out and "mean_gold_vs_best_logit_margin_no_memory" in out:
        out["mean_gold_vs_best_logit_margin_improvement"] = (
            out["mean_gold_vs_best_logit_margin_with_memory"] -
            out["mean_gold_vs_best_logit_margin_no_memory"]
        )

    return out


def discover_prediction_summary_keys(rows: List[Dict[str, Any]]) -> List[str]:
    summary = summarize_prediction_rows(rows)
    return sorted(summary.keys())


def summarize_prediction_log(prediction_log: Path) -> Dict[str, float]:
    rows = load_prediction_rows(prediction_log)
    summary = summarize_prediction_rows(rows)
    epoch = extract_prediction_epoch(prediction_log)
    if epoch is not None:
        summary["log_epoch"] = float(epoch)
    return summary


def format_heatmap_value(v: Optional[float], fmt: str) -> str:
    if v is None or not math.isfinite(v):
        return "NA"
    return format(v, fmt)


def compute_heatmap_limits(
    summaries: List[Dict[str, float]],
    metric: str,
    robust: bool = False,
) -> Tuple[float, float]:
    vals = [s[metric] for s in summaries if metric in s and math.isfinite(s[metric])]
    if not vals:
        return (0.0, 1.0)

    vals = sorted(vals)
    if robust and len(vals) >= 4:
        lo_idx = int(0.05 * (len(vals) - 1))
        hi_idx = int(0.95 * (len(vals) - 1))
        lo = vals[lo_idx]
        hi = vals[hi_idx]
    else:
        lo = vals[0]
        hi = vals[-1]

    if lo == hi:
        pad = 1.0 if lo == 0 else abs(lo) * 0.05
        return (lo - pad, hi + pad)

    return (lo, hi)