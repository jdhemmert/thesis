#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from typing import Dict, List, Optional

import matplotlib.pyplot as plt

from src.utils.sweep import parse_pins, parse_sweep_config, resolve_eval_logs
from src.utils.logs import (
    compute_global_limits,
    discover_metric_keys,
    extract_metric_points,
    format_title,
    load_eval_rows,
)


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
    x_vals, y_vals, cell_eval_logs, existing_eval_logs = resolve_eval_logs(
        root=root,
        sweep_params=sweep_params,
        subdir_template=subdir_template,
        x_param=x_param,
        y_param=y_param,
        pins=pins,
    )

    nrows, ncols = len(y_vals), len(x_vals)
    (xlo, xhi), (ylo, yhi) = compute_global_limits(existing_eval_logs, metrics)

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

    def short_name(p: str) -> str:
        return p.split(".")[-1]

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
            ax.tick_params(axis="both", which="both", labelbottom=True, labelleft=True, labelsize=8)

            if x_param and y_param:
                ax.set_title(
                    f"{short_name(y_param)}={y_vals[r]} | {short_name(x_param)}={x_vals[c]}",
                    fontsize=9,
                )
            elif x_param:
                ax.set_title(f"{short_name(x_param)}={x_vals[c]}", fontsize=9)
            elif y_param:
                ax.set_title(f"{short_name(y_param)}={y_vals[r]}", fontsize=9)
            else:
                ax.set_title("slice", fontsize=9)

    for ax in axes[-1, :]:
        ax.set_xlabel("Epoch")
    for ax in axes[:, 0]:
        ax.set_ylabel("Value")

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

    ap.add_argument("--x-param", type=str, default=None, help="Swept param to use for grid columns.")
    ap.add_argument("--y-param", type=str, default=None, help="Swept param to use for grid rows.")
    ap.add_argument(
        "--pin",
        action="append",
        default=[],
        help="Pin swept param not on axes: key=value. Repeatable.",
    )

    ap.add_argument(
        "--metrics",
        nargs="+",
        default=None,
        help="Metric key(s) to plot (e.g., eval_*).",
    )
    ap.add_argument(
        "--marker",
        type=str,
        default="",
        help='Matplotlib marker style (default ""). Example: "o", "x", ".".',
    )
    ap.add_argument(
        "--title",
        type=str,
        default=None,
        help="Override figure title (default: root relative to base-dir).",
    )
    ap.add_argument("--out", type=Path, default=None, help="Save plot to file.")
    ap.add_argument("--show", action="store_true", help="Show interactive window.")

    ap.add_argument("--list-sweep-params", action="store_true", help="Print sweep params and exit.")
    ap.add_argument(
        "--list-metrics",
        action="store_true",
        help="Print metric keys from the first eval_log.json found under --root and exit.",
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