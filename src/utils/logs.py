from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ----------------------------
# Logs / metrics
# ----------------------------

def load_eval_rows(eval_log: Path) -> List[Dict[str, Any]]:
    data = json.loads(eval_log.read_text())
    if not isinstance(data, list):
        raise ValueError(f"{eval_log} is not a JSON list.")
    return [r for r in data if isinstance(r, dict)]


def discover_metric_keys(rows: List[Dict[str, Any]]) -> List[str]:
    keys = set()
    for row in rows:
        if "epoch" not in row:
            continue
        for k in row.keys():
            if k.startswith("eval_"):
                keys.add(k)
    return sorted(keys)


def extract_metric_points(rows: List[Dict[str, Any]], metric_key: str) -> List[Tuple[float, float]]:
    pts: List[Tuple[float, float]] = []
    for row in rows:
        if "epoch" not in row or metric_key not in row:
            continue
        try:
            epoch = float(row["epoch"])
            val = float(row[metric_key])
        except (TypeError, ValueError):
            continue
        if not math.isfinite(val):
            val = float("nan")  # gap
        pts.append((epoch, val))
    pts.sort(key=lambda t: t[0])
    return pts


def extract_final_metric(rows: List[Dict[str, Any]], metric_key: str) -> float:
    vals = []
    for row in rows:
        if metric_key not in row:
            continue
        try:
            val = float(row[metric_key])
        except (TypeError, ValueError):
            continue
        if math.isfinite(val):
            vals.append(val)
    if not vals:
        raise ValueError(f"Could not determine final {metric_key}")
    return vals[-1]


def compute_global_limits(
    eval_logs: List[Path],
    metrics: List[str],
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    x_min, x_max = float("inf"), float("-inf")
    y_min, y_max = float("inf"), float("-inf")

    for elog in eval_logs:
        rows = load_eval_rows(elog)
        for m in metrics:
            for x, y in extract_metric_points(rows, m):
                if not (math.isfinite(x) and math.isfinite(y)):
                    continue
                x_min = min(x_min, x)
                x_max = max(x_max, x)
                y_min = min(y_min, y)
                y_max = max(y_max, y)

    if not (math.isfinite(x_min) and math.isfinite(x_max) and math.isfinite(y_min) and math.isfinite(y_max)):
        return (0.0, 1.0), (0.0, 1.0)

    def pad(lo: float, hi: float, frac: float) -> Tuple[float, float]:
        if hi == lo:
            return (lo - 1.0, hi + 1.0)
        d = (hi - lo) * frac
        return (lo - d, hi + d)

    return pad(x_min, x_max, 0.03), pad(y_min, y_max, 0.05)


# ----------------------------
# Titles
# ----------------------------

def format_title(root: Path, base_dir: Optional[Path], explicit_title: Optional[str]) -> str:
    if explicit_title:
        return explicit_title

    r = root.resolve()
    if base_dir is None:
        return str(r)

    b = base_dir.resolve()
    try:
        return str(r.relative_to(b))
    except ValueError:
        return str(r)