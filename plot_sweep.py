#!/usr/bin/env python3
"""
Hydra sweep-aware plotting for eval_log.json, mapping params -> directory via hydra.sweep.subdir.

Requires:
  pip install matplotlib pyyaml

Example:
  python plot_sweep.py \
    --base-dir results \
    --root naive_softprompt/random_init_sweep/2026-02-13_10-31-01 \
    --sweep-config multirun.yaml \
    --x-param task.sample_n \
    --y-param model.model_config.virtual_token_count \
    --pin task.noise=0.1 \
    --metrics eval_student_ce_answer eval_teacher_ce_answer \
    --out grid.png
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import yaml


def canon_param(k: str) -> str:
    k = k.strip()
    return k[1:] if k.startswith("+") else k

# ----------------------------
# Sweep parsing (strings only)
# ----------------------------

def _strip_quotes(s: str) -> str:
    s = s.strip()
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return s[1:-1]
    return s

def split_sweep_values_to_strings(v: Any) -> List[str]:
    """
    Hydra sweeper params are often strings like "0.01, 0.1, 0.5".
    We keep string tokens (trimmed, unquoted) to match directory naming.
    """
    if isinstance(v, list):
        return [_strip_quotes(str(x).strip()) for x in v if str(x).strip() != ""]
    if v is None:
        return []
    raw = str(v)
    parts = [p.strip() for p in raw.split(",") if p.strip() != ""]
    return [_strip_quotes(p) for p in parts]

def parse_sweep_config(path: Path) -> Tuple[Dict[str, List[str]], str]:
    """
    Returns:
      - sweep_params: param_name -> list[str] (YAML order preserved)
      - subdir_template: hydra.sweep.subdir
    """
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("Sweep config must be a YAML mapping at top-level.")

    try:
        params = data["hydra"]["sweeper"]["params"]
    except Exception as e:
        raise KeyError("Expected hydra.sweeper.params in sweep config.") from e

    try:
        subdir = data["hydra"]["sweep"]["subdir"]
    except Exception as e:
        raise KeyError("Expected hydra.sweep.subdir in sweep config.") from e

    if not isinstance(params, dict):
        raise ValueError("hydra.sweeper.params must be a mapping.")
    if not isinstance(subdir, str):
        raise ValueError("hydra.sweep.subdir must be a string.")

    sweep_params: Dict[str, List[str]] = {}
    for k, v in params.items():
        k_raw = str(k)
        k_can = canon_param(k_raw)
    
        vals = split_sweep_values_to_strings(v)
        if not vals:
            raise ValueError(f"Sweep param {k_raw} has no values.")
    
        if k_can in sweep_params:
            raise ValueError(
                f"Ambiguous sweep param keys: both {k_raw!r} and another key map to {k_can!r}. "
                "Remove the duplicate (+-prefixed vs non-prefixed)."
            )
    
        sweep_params[k_can] = vals

    return sweep_params, subdir


# ----------------------------
# Template rendering
# ----------------------------

_PLACEHOLDER_RE = re.compile(r"\$\{([^}]+)\}")

def render_subdir(template: str, assignment: Dict[str, str], sweep_params: Dict[str, List[str]]) -> str:
    """
    Replace ${param.path} with assignment[param.path].

    Strict:
      - placeholder must be a swept param name (present in sweep_params)
      - placeholder must be provided in assignment
      - no resolver syntax supported (e.g. ${now:%Y}) -> error
    """
    def repl(m: re.Match) -> str:
        key = m.group(1).strip()
        # reject resolver-like patterns (":" is a good cheap proxy)
        if ":" in key:
            raise ValueError(
                f"Unsupported placeholder '{m.group(0)}' in hydra.sweep.subdir. "
                "This script supports only simple ${dotted.param} placeholders."
            )
        if key not in sweep_params:
            raise ValueError(
                f"hydra.sweep.subdir references '{key}', but it's not in hydra.sweeper.params."
            )
        if key not in assignment:
            raise ValueError(
                f"Missing value for '{key}' while rendering hydra.sweep.subdir. "
                "This should not happen if axes+pins cover all swept params."
            )
        return assignment[key]

    return _PLACEHOLDER_RE.sub(repl, template)


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


# ----------------------------
# Slice selection
# ----------------------------

def parse_pins(pin_items: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for p in pin_items:
        if "=" not in p:
            raise ValueError(f"Invalid --pin {p!r}; expected key=value.")
        k, v = p.split("=", 1)
        out[k.strip()] = _strip_quotes(v.strip())
    return out

def validate_slice(
    sweep_params: Dict[str, List[str]],
    x_param: Optional[str],
    y_param: Optional[str],
    pins: Dict[str, str],
) -> Tuple[List[Optional[str]], List[Optional[str]], Dict[str, str]]:
    """
    Returns (x_values, y_values, pinned_values) where x_values/y_values are list[str] or [None].
    Enforces:
      - x/y are swept params (if provided)
      - x != y
      - every swept param not on axes is pinned
      - pin value must match one of the swept values EXACTLY
    """
    if x_param is not None and x_param not in sweep_params:
        raise ValueError(f"--x-param {x_param!r} not found in sweep config.")
    if y_param is not None and y_param not in sweep_params:
        raise ValueError(f"--y-param {y_param!r} not found in sweep config.")
    if x_param is not None and y_param is not None and x_param == y_param:
        raise ValueError("--x-param and --y-param must be different.")

    axis = {p for p in [x_param, y_param] if p is not None}
    swept = list(sweep_params.keys())

    required_pins = [p for p in swept if p not in axis]
    missing = [p for p in required_pins if p not in pins]
    if missing:
        raise ValueError(
            "Missing pins for swept params not on axes: "
            + ", ".join(missing)
            + ". Pin them with --pin key=value"
        )

    pinned_norm: Dict[str, str] = {}
    for k, v in pins.items():
        if k not in sweep_params:
            raise ValueError(f"Pin key {k!r} is not a swept param in the sweep config.")
        allowed = sweep_params[k]
        if v not in allowed:
            raise ValueError(
                f"Pin {k}={v!r} did not match any allowed sweep values.\n"
                f"Allowed values for {k} are: {allowed}"
            )
        pinned_norm[k] = v

    x_vals: List[Optional[str]] = sweep_params[x_param] if x_param else [None]
    y_vals: List[Optional[str]] = sweep_params[y_param] if y_param else [None]
    return x_vals, y_vals, pinned_norm


# ----------------------------
# Plotting
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

def plot_sweep_grid(
    *,
    root: Path,
    base_dir: Optional[Path],
    sweep_params: Dict[str, List[str]],
    subdir_template: str,
    x_param: Optional[str],
    y_param: Optional[str],
    pins: Dict[str, str],
    metrics: List[str],
    marker: str,
    title: Optional[str],
    out: Optional[Path],
    show: bool,
) -> None:
    # Determine grid axes and pin validity
    x_vals, y_vals, pinned = validate_slice(sweep_params, x_param, y_param, pins)
    nrows, ncols = len(y_vals), len(x_vals)

    # Build per-cell eval_log paths (and track which exist)
    swept_keys_in_order = list(sweep_params.keys())

    cell_eval_logs: Dict[Tuple[int, int], Optional[Path]] = {}
    existing_eval_logs: List[Path] = []

    for r, yv in enumerate(y_vals):
        for c, xv in enumerate(x_vals):
            assignment: Dict[str, str] = dict(pinned)
            if x_param is not None and xv is not None:
                assignment[x_param] = xv
            if y_param is not None and yv is not None:
                assignment[y_param] = yv

            # Sanity: should cover all swept params
            for k in swept_keys_in_order:
                if k not in assignment:
                    raise RuntimeError(f"Internal error: assignment missing swept param {k}")

            subdir = render_subdir(subdir_template, assignment, sweep_params)
            run_dir = root / subdir
            elog = run_dir / "eval_log.json"

            if elog.exists():
                cell_eval_logs[(r, c)] = elog
                existing_eval_logs.append(elog)
            else:
                cell_eval_logs[(r, c)] = None

    # Global axis limits across existing runs
    (xlo, xhi), (ylo, yhi) = compute_global_limits(existing_eval_logs, metrics)

    # Size scales with grid
    per_ax_w, per_ax_h = 4.2, 3.1
    figsize = (max(1, ncols) * per_ax_w, max(1, nrows) * per_ax_h)

    fig, axes = plt.subplots(
        nrows=max(1, nrows),
        ncols=max(1, ncols),
        squeeze=False,
        figsize=figsize,
        sharex=True,
        sharey=True,
    )

    fig.suptitle(format_title(root, base_dir, title), fontsize=12)

    legend_handles = None
    legend_labels = None

    for r in range(nrows):
        for c in range(ncols):
            ax = axes[r][c]
            elog = cell_eval_logs[(r, c)]

            if elog is None:
                ax.text(0.5, 0.5, "missing", ha="center", va="center", transform=ax.transAxes)
                ax.grid(True, linestyle="--", linewidth=0.5)
            else:
                rows = load_eval_rows(elog)
                for m in metrics:
                    pts = extract_metric_points(rows, m)
                    if not pts:
                        continue
                    xs = [e for e, _ in pts]
                    ys = [v for _, v in pts]
                    ax.plot(xs, ys, marker=marker, label=m)
                ax.grid(True, linestyle="--", linewidth=0.5)

                if legend_handles is None:
                    legend_handles, legend_labels = ax.get_legend_handles_labels()

            ax.set_xlim(xlo, xhi)
            ax.set_ylim(ylo, yhi)

            # Show numbers on all axes (even with sharex/sharey)
            ax.tick_params(axis="both", which="both", labelbottom=True, labelleft=True, labelsize=8)

            # Cell title
            def short_name(p: str) -> str:
                return p.split(".")[-1]
            
            if x_param and y_param:
                ax.set_title(
                    f"{short_name(y_param)}={y_vals[r]} | {short_name(x_param)}={x_vals[c]}",
                    fontsize=9,
                )
            elif x_param:
                ax.set_title(
                    f"{short_name(x_param)}={x_vals[c]}",
                    fontsize=9,
                )
            elif y_param:
                ax.set_title(
                    f"{short_name(y_param)}={y_vals[r]}",
                    fontsize=9,
                )
            else:
                ax.set_title("slice", fontsize=9)

    # Axis labels (outer labels still helpful)
    for ax in axes[-1, :]:
        ax.set_xlabel("Epoch")
    for ax in axes[:, 0]:
        ax.set_ylabel("Value")

    # Global legend
    if legend_handles and legend_labels:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            ncol=min(len(legend_labels), 4),
            frameon=True,
            fontsize=9,
        )

    fig.subplots_adjust(top=0.90, bottom=0.12, wspace=0.25, hspace=0.35)

    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight", dpi=200)
        print(f"Saved: {out}")

    if show or out is None:
        plt.show()


# ----------------------------
# CLI
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--base-dir", type=Path, default=None,
                    help="Base directory; --root is interpreted relative to this.")
    ap.add_argument("--root", type=Path, required=True,
                    help="Sweep root directory (relative to --base-dir if provided).")
    ap.add_argument("--sweep-config", type=Path, required=True,
                    help="Sweep YAML path, interpreted relative to resolved --root.")

    ap.add_argument("--x-param", type=str, default=None, help="Swept param to use for grid columns.")
    ap.add_argument("--y-param", type=str, default=None, help="Swept param to use for grid rows.")
    ap.add_argument("--pin", action="append", default=[],
                    help="Pin swept param not on axes: key=value. Repeatable.")

    ap.add_argument("--metrics", nargs="+", default=None,
                    help="Metric key(s) to plot (e.g., eval_*).")
    ap.add_argument("--marker", type=str, default="",
                    help='Matplotlib marker style (default ""). Example: "o", "x", ".".')
    ap.add_argument("--title", type=str, default=None,
                    help="Override figure title (default: root relative to base-dir).")
    ap.add_argument("--out", type=Path, default=None, help="Save plot to file.")
    ap.add_argument("--show", action="store_true", help="Show interactive window.")

    ap.add_argument("--list-sweep-params", action="store_true", help="Print sweep params and exit.")
    ap.add_argument("--list-metrics", action="store_true",
                    help="Print metric keys from the first eval_log.json found under --root and exit.")

    args = ap.parse_args()

    # Resolve base-dir
    base_dir: Optional[Path]
    if args.base_dir is not None:
        base_dir = args.base_dir.expanduser().resolve()
        if not base_dir.exists():
            raise SystemExit(f"--base-dir does not exist: {base_dir}")
    else:
        base_dir = None

    # Resolve root relative to base-dir
    # root = (base_dir / args.root).expanduser().resolve() if base_dir is not None else args.root.expanduser().resolve()
    root = args.root.expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"--root does not exist: {root}")

    # Resolve sweep-config relative to root
    sweep_cfg = (root / args.sweep_config).expanduser().resolve()
    if not sweep_cfg.exists():
        raise SystemExit(f"--sweep-config does not exist: {sweep_cfg}")

    sweep_params, subdir_template = parse_sweep_config(sweep_cfg)

    if args.list_sweep_params:
        print(f"Swept params from {sweep_cfg}:")
        for k, vals in sweep_params.items():
            print(f"- {k}: {vals}")
        print(f"\nhydra.sweep.subdir: {subdir_template}")
        return

    if args.list_metrics:
        logs = sorted(root.glob("**/eval_log.json"))
        if not logs:
            raise SystemExit(f"No eval_log.json found under {root}")
        rows = load_eval_rows(logs[0])
        keys = discover_metric_keys(rows)
        print(f"Metrics discovered in: {logs[0]}")
        for k in keys:
            print(k)
        return

    if not args.metrics:
        raise SystemExit("You must pass --metrics ... (or use --list-metrics).")

    pins = parse_pins(args.pin)

    plot_sweep_grid(
        root=root,
        base_dir=base_dir,
        sweep_params=sweep_params,
        subdir_template=subdir_template,
        x_param=args.x_param,
        y_param=args.y_param,
        pins=pins,
        metrics=args.metrics,
        marker=args.marker,
        title=args.title,
        out=args.out,
        show=args.show,
    )


if __name__ == "__main__":
    main()