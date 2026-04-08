from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
# Run resolution
# ----------------------------

def resolve_eval_logs(
    *,
    root: Path,
    sweep_params: Dict[str, List[str]],
    subdir_template: str,
    x_param: Optional[str],
    y_param: Optional[str],
    pins: Dict[str, str],
) -> Tuple[List[Optional[str]], List[Optional[str]], Dict[Tuple[int, int], Optional[Path]], List[Path]]:
    """
    Resolve the eval_log.json path for each cell in the requested sweep slice.

    Returns:
      - x_vals
      - y_vals
      - cell_eval_logs[(row, col)] -> Path | None
      - existing_eval_logs
    """
    x_vals, y_vals, pinned = validate_slice(sweep_params, x_param, y_param, pins)
    nrows, ncols = len(y_vals), len(x_vals)

    cell_eval_logs: Dict[Tuple[int, int], Optional[Path]] = {}
    existing_eval_logs: List[Path] = []

    swept_keys_in_order = list(sweep_params.keys())

    for r in range(nrows):
        yv = y_vals[r]
        for c in range(ncols):
            xv = x_vals[c]

            assignment: Dict[str, str] = dict(pinned)
            if x_param is not None and xv is not None:
                assignment[x_param] = xv
            if y_param is not None and yv is not None:
                assignment[y_param] = yv

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

    return x_vals, y_vals, cell_eval_logs, existing_eval_logs