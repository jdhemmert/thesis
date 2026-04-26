#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import math
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from src.utils.sweep import parse_pins, parse_sweep_config, resolve_single_file_per_run
from src.utils.eval_logs import format_title
from src.utils.prediction_logs import (
    compute_heatmap_limits,
    discover_prediction_summary_keys,
    format_heatmap_value,
    load_prediction_rows,
    summarize_prediction_log,
)


DEFAULT_METRIC = "mean_gold_first_token_prob_delta"


def short_name(p: str) -> str:
    return p.split(".")[-1]


def build_metric_grid(
    *,
    root: Path,
    sweep_params: Dict[str, List[str]],
    subdir_template: str,
    x_param: Optional[str],
    y_param: Optional[str],
    pins: Dict[str, str],
    prediction_pattern: str,
    metric: str,
) -> Tuple[List[Optional[str]], List[Optional[str]], np.ndarray, Dict[Tuple[int, int], Optional[Path]], List[Dict[str, float]]]:
    x_vals, y_vals, cell_files, existing_files = resolve_single_file_per_run(
        root=root,
        sweep_params=sweep_params,
        subdir_template=subdir_template,
        x_param=x_param,
        y_param=y_param,
        pins=pins,
        pattern=prediction_pattern,
        require_unique=True,
    )

    nrows, ncols = len(y_vals), len(x_vals)
    grid = np.full((nrows, ncols), np.nan, dtype=float)
    summaries: List[Dict[str, float]] = []

    for r in range(nrows):
        for c in range(ncols):
            p = cell_files[(r, c)]
            if p is None:
                continue
            summary = summarize_prediction_log(p)
            summaries.append(summary)
            if metric in summary and math.isfinite(summary[metric]):
                grid[r, c] = summary[metric]

    return x_vals, y_vals, grid, cell_files, summaries


def plot_prediction_heatmap(
    *,
    root: Path,
    base_dir: Optional[Path],
    sweep_params: Dict[str, List[str]],
    subdir_template: str,
    x_param: Optional[str],
    y_param: Optional[str],
    pins: Dict[str, str],
    metric: str,
    prediction_pattern: str,
    title: Optional[str],
    out: Optional[Path],
    show: bool,
    annotate: bool,
    annotation_fmt: str,
    cmap: str,
    robust_limits: bool,
    center_zero: bool,
) -> None:
    x_vals, y_vals, grid, cell_files, summaries = build_metric_grid(
        root=root,
        sweep_params=sweep_params,
        subdir_template=subdir_template,
        x_param=x_param,
        y_param=y_param,
        pins=pins,
        prediction_pattern=prediction_pattern,
        metric=metric,
    )

    nrows, ncols = grid.shape
    fig_w = max(1, ncols) * 1.8 + 1.5
    fig_h = max(1, nrows) * 1.5 + 1.5

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    vmin, vmax = compute_heatmap_limits(summaries, metric=metric, robust=robust_limits)

    if center_zero:
        bound = max(abs(vmin), abs(vmax))
        vmin, vmax = -bound, bound

    im = ax.imshow(grid, aspect="auto", interpolation="nearest", vmin=vmin, vmax=vmax, cmap=cmap)

    ax.set_title(format_title(root, base_dir, title) + f"\nmetric={metric}", fontsize=11)

    ax.set_xticks(range(ncols))
    ax.set_yticks(range(nrows))

    if x_param:
        ax.set_xticklabels([str(v) for v in x_vals], rotation=45, ha="right")
        ax.set_xlabel(short_name(x_param))
    else:
        ax.set_xticklabels(["slice"])

    if y_param:
        ax.set_yticklabels([str(v) for v in y_vals])
        ax.set_ylabel(short_name(y_param))
    else:
        ax.set_yticklabels(["slice"])

    # Minor gridlines between cells
    ax.set_xticks(np.arange(-0.5, ncols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, nrows, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)

    if annotate:
        for r in range(nrows):
            for c in range(ncols):
                p = cell_files[(r, c)]
                if p is None or not math.isfinite(grid[r, c]):
                    txt = "missing"
                else:
                    txt = format_heatmap_value(grid[r, c], annotation_fmt)
                ax.text(c, r, txt, ha="center", va="center", fontsize=8)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(metric)

    fig.tight_layout()

    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight", dpi=200)
        print(f"Saved: {out}")

    if show or out is None:
        plt.show()


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--base-dir",
        type=Path,
        default=None,
        help="Base directory; used only for title formatting.",
    )
    ap.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Sweep root directory.",
    )
    ap.add_argument(
        "--sweep-config",
        type=Path,
        required=True,
        help="Sweep YAML path, interpreted relative to resolved --root if not absolute.",
    )

    ap.add_argument("--x-param", type=str, default=None, help="Swept param to use for heatmap columns.")
    ap.add_argument("--y-param", type=str, default=None, help="Swept param to use for heatmap rows.")
    ap.add_argument(
        "--pin",
        action="append",
        default=[],
        help="Pin swept param not on axes: key=value. Repeatable.",
    )

    ap.add_argument(
        "--prediction-pattern",
        type=str,
        default="prediction_log.epoch_*.jsonl",
        help="Glob pattern for the per-run prediction log.",
    )
    ap.add_argument(
        "--metric",
        type=str,
        default=DEFAULT_METRIC,
        help="Run-level summary metric to plot.",
    )
    ap.add_argument(
        "--title",
        type=str,
        default=None,
        help="Override figure title (default: root relative to base-dir).",
    )
    ap.add_argument("--out", type=Path, default=None, help="Save plot to file.")
    ap.add_argument("--show", action="store_true", help="Show interactive window.")

    ap.add_argument("--annotate", action="store_true", help="Write cell values into the heatmap.")
    ap.add_argument(
        "--annotation-fmt",
        type=str,
        default=".3g",
        help="Format spec for annotation text, e.g. .2f, .3g.",
    )
    ap.add_argument(
        "--cmap",
        type=str,
        default="viridis",
        help="Matplotlib colormap name.",
    )
    ap.add_argument(
        "--robust-limits",
        action="store_true",
        help="Use approximate 5th/95th percentile limits instead of strict min/max.",
    )
    ap.add_argument(
        "--center-zero",
        action="store_true",
        help="Force a symmetric color scale around zero.",
    )

    ap.add_argument("--list-sweep-params", action="store_true", help="Print sweep params and exit.")
    ap.add_argument(
        "--list-metrics",
        action="store_true",
        help="Print available prediction-summary metrics from the first discovered prediction log and exit.",
    )

    args = ap.parse_args()

    base_dir: Optional[Path]
    if args.base_dir is not None:
        base_dir = args.base_dir.expanduser().resolve()
        if not base_dir.exists():
            raise SystemExit(f"--base-dir does not exist: {base_dir}")
    else:
        base_dir = None

    root = args.root.expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"--root does not exist: {root}")

    if args.sweep_config.is_absolute():
        sweep_cfg = args.sweep_config.expanduser().resolve()
    else:
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
        matches = sorted(root.glob(f"**/{args.prediction_pattern}"))
        if not matches:
            raise SystemExit(f"No files matching {args.prediction_pattern!r} found under {root}")
        rows = load_prediction_rows(matches[0])
        keys = discover_prediction_summary_keys(rows)
        print(f"Prediction-summary metrics discovered from: {matches[0]}")
        for k in keys:
            print(k)
        return

    pins = parse_pins(args.pin)

    plot_prediction_heatmap(
        root=root,
        base_dir=base_dir,
        sweep_params=sweep_params,
        subdir_template=subdir_template,
        x_param=args.x_param,
        y_param=args.y_param,
        pins=pins,
        metric=args.metric,
        prediction_pattern=args.prediction_pattern,
        title=args.title,
        out=args.out,
        show=args.show,
        annotate=args.annotate,
        annotation_fmt=args.annotation_fmt,
        cmap=args.cmap,
        robust_limits=args.robust_limits,
        center_zero=args.center_zero,
    )


if __name__ == "__main__":
    main()