#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ============================================================
# 1. User configuration
# ============================================================

SYSTEM = "clang"
SYSTEM_SCALE = "clang_large"

FALLBACKS = [
    "1NN",
    "kNN",
    "kNNw",
    "loess",
    "rf",
    "dal_rf",
    "rf_uncertainty",

]

ALGORITHMS = [
    "smac",
    "bestconfig",
    "flash",
    "ga",
    "hebo",
    "promisetune",
    "random_search",
]

RUN_IDS = list(range(10))
CSV_NAME = "all_results.csv"

OUTPUT_FOLDER_NAME = "motivation"

# Whether to include markers in tex
USE_MARKERS = False

# Sampling interval for exporting coordinates to tex
# e.g., 10 means only export every 10th point, while always keeping the last point
POINT_STEP = 10

# Figure metadata
X_LABEL = "\# measurement"

# Template axis style
AXIS_WIDTH = "5cm"
AXIS_HEIGHT = "5cm"

# If None, compute automatically from data
XMIN = None
XMAX = None
YMIN = None
YMAX = None

# Optional manual ticks
XTICK = None   # e.g. "{1,50,100,150,200,250,300,350}"
YTICK = None   # e.g. "{2500,2800,3100,3400,3700,4000}"

# Legend position copied from your template
LEGEND_AT = "(0.98,0.9)"
LEGEND_ANCHOR = "south east"

# Style map
STYLE_MAP = {
    "bestconfig": {
        "color": "red",
        "fill": "red!70",
        "line_width": "ultra thick",
        "line_style": "dashdotted",
        "marker": "square",
        "marker_options": "thick,solid",
        "legend": r"\texttt{BestConfig}",
    },
    "smac": {
        "color": "teal",
        "fill": "teal!30",
        "line_width": "ultra thick",
        "line_style": "dashed",
        "marker": "o",
        "marker_options": "thick",
        "legend": r"\texttt{SMAC}",
    },
    "flash": {
        "color": "blue",
        "fill": "blue!30",
        "line_width": "ultra thick",
        "line_style": "dotted",
        "marker": "triangle",
        "marker_options": "thick,solid",
        "legend": r"\texttt{FLASH}",
    },
    "promisetune": {
        "color": "orange",
        "fill": "orange!30",
        "line_width": "ultra thick",
        "line_style": "solid",
        "marker": "square",
        "marker_options": "thick,solid",
        "legend": r"\texttt{PromiseTune}",
    },
    "hebo": {
        "color": "purple",
        "fill": "purple!30",
        "line_width": "ultra thick",
        "line_style": "densely dashed",
        "marker": "diamond",
        "marker_options": "thick,solid",
        "legend": r"\texttt{HEBO}",
    },
    "ga": {
        "color": "magenta",
        "fill": "magenta!30",
        "line_width": "ultra thick",
        "line_style": "densely dotted",
        "marker": "asterisk",
        "marker_options": "thick,solid",
        "legend": r"\texttt{GA}",
    },
    "random_search": {
        "color": "black",
        "fill": "black!20",
        "line_width": "ultra thick",
        "line_style": "loosely dashed",
        "marker": "x",
        "marker_options": "thick,solid",
        "legend": r"\texttt{RandomSearch}",
    },
}


# ============================================================
# 2. Paths
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RESULTS_DIR = PROJECT_ROOT / "results_dataset"
OUTPUT_DIR = SCRIPT_DIR / OUTPUT_FOLDER_NAME


# ============================================================
# 3. Helpers
# ============================================================

def is_maximize(system: str) -> bool:
    return system.lower() in {"mysql", "postgresql", "tomcat", "httpd"}


def get_ylabel(system: str) -> str:
    s = system.lower()
    if s in {"mysql", "postgresql"}:
        return "Throughput (tps)"
    if s in {"httpd", "tomcat"}:
        return "Throughput (rps)"
    return "Runtime (s)"


def read_curve(csv_path: Path) -> Optional[np.ndarray]:
    if not csv_path.exists():
        print(f"[WARN] Missing file: {csv_path}")
        return None

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[WARN] Failed to read {csv_path}: {e}")
        return None

    if df.empty:
        print(f"[WARN] Empty file: {csv_path}")
        return None

    perf = pd.to_numeric(df.iloc[:, -1], errors="coerce").dropna()
    vals = perf.to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]

    if len(vals) == 0:
        print(f"[WARN] No finite values in: {csv_path}")
        return None

    return vals


def make_monotonic(values: np.ndarray, maximize: bool) -> np.ndarray:
    if maximize:
        return np.maximum.accumulate(values)
    return np.minimum.accumulate(values)


def collect_curves(fallback: str, algorithm: str) -> List[np.ndarray]:
    curves = []
    maximize = is_maximize(SYSTEM)

    for run_id in RUN_IDS:
        csv_path = (
            RESULTS_DIR
            / f"run{run_id}"
            / fallback
            / algorithm
            / SYSTEM
            / SYSTEM_SCALE
            / CSV_NAME
        )
        vals = read_curve(csv_path)
        if vals is None:
            continue
        curves.append(make_monotonic(vals, maximize=maximize))

    return curves


def pad_with_last(curves: List[np.ndarray]) -> np.ndarray:
    max_len = max(len(c) for c in curves)
    out = []
    for c in curves:
        if len(c) < max_len:
            c = np.concatenate([c, np.full(max_len - len(c), c[-1])])
        out.append(c)
    return np.vstack(out)


def aggregate(curves: List[np.ndarray]) -> Dict[str, np.ndarray]:
    arr = pad_with_last(curves)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    x = np.arange(1, len(mean) + 1, dtype=float)
    return {
        "x": x,
        "mean": mean,
        "upper": mean + std,
        "lower": mean - std,
    }


def downsample_series(x: np.ndarray, y: np.ndarray, step: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Keep every `step`-th point and always keep the last point.
    """
    if step <= 1 or len(x) <= 2:
        return x, y

    idx = list(range(0, len(x), step))
    if idx[-1] != len(x) - 1:
        idx.append(len(x) - 1)

    idx = np.array(idx, dtype=int)
    return x[idx], y[idx]


def fmt_coords(x: np.ndarray, y: np.ndarray, digits: int = 4) -> str:
    return " ".join(f"({xx:.{digits}f}, {yy:.{digits}f})" for xx, yy in zip(x, y))


def compute_limits(all_stats: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, float]:
    xs = []
    ys = []

    for s in all_stats.values():
        xs.extend(s["x"].tolist())
        ys.extend(s["upper"].tolist())
        ys.extend(s["lower"].tolist())

    xmin = min(xs)
    xmax = max(xs)
    ymin = min(ys)
    ymax = max(ys)

    margin = 0.05 * (ymax - ymin) if ymax > ymin else 0.01
    return {
        "xmin": xmin if XMIN is None else XMIN,
        "xmax": xmax if XMAX is None else XMAX,
        "ymin": ymin - margin if YMIN is None else YMIN,
        "ymax": ymax + margin if YMAX is None else YMAX,
    }


def tex_plot_block(
    mean_coords: str,
    upper_coords: str,
    lower_coords: str,
    style: Dict[str, str],
    upper_name: str,
    lower_name: str,
) -> str:
    if USE_MARKERS:
        mean_plot = rf"""
\addplot[
    {style['color']},
    {style['line_style']},
    mark={style['marker']},mark options={{{style['marker_options']}}},
    {style['line_width']}, forget plot
] coordinates {{
{mean_coords}
}}; 
"""
    else:
        mean_plot = rf"""
\addplot[
    {style['color']},
    {style['line_style']},
    {style['line_width']}, forget plot
] coordinates {{
{mean_coords}
}}; 
"""

    return (
        mean_plot
        + rf"""\addlegendentry{{{style['legend']}}}

\addplot[
name path={upper_name},
draw=none, forget plot
] coordinates {{
{upper_coords}
}}; 

\addplot[
name path={lower_name},
draw=none, forget plot
] coordinates {{
{lower_coords}
}}; 

% ========= file shading =========
\addplot[
    {style['fill']},
    fill opacity=0.30, forget plot
] fill between[
    of={upper_name} and {lower_name},
];

"""
    )


def export_one_tex(fallback: str, all_stats: Dict[str, Dict[str, np.ndarray]]) -> None:
    if not all_stats:
        print(f"[WARN] No valid stats for fallback={fallback}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    limits = compute_limits(all_stats)

    out_path = OUTPUT_DIR / f"motivation_{fallback}_{SYSTEM}_{SYSTEM_SCALE}.tex"

    body_blocks = []
    legend_images = []

    for idx, algorithm in enumerate(ALGORITHMS):
        if algorithm not in all_stats:
            continue

        style = STYLE_MAP.get(algorithm)
        if style is None:
            print(f"[WARN] No style defined for {algorithm}, skipped.")
            continue

        stats = all_stats[algorithm]

        x_mean, y_mean = downsample_series(stats["x"], stats["mean"], POINT_STEP)
        x_upper, y_upper = downsample_series(stats["x"], stats["upper"], POINT_STEP)
        x_lower, y_lower = downsample_series(stats["x"], stats["lower"], POINT_STEP)

        mean_coords = fmt_coords(x_mean, y_mean)
        upper_coords = fmt_coords(x_upper, y_upper)
        lower_coords = fmt_coords(x_lower, y_lower)

        upper_name = f"upper{idx}"
        lower_name = f"lower{idx}"

        body_blocks.append(
            tex_plot_block(
                mean_coords=mean_coords,
                upper_coords=upper_coords,
                lower_coords=lower_coords,
                style=style,
                upper_name=upper_name,
                lower_name=lower_name,
            )
        )

        if USE_MARKERS:
            legend_images.append(
                rf"\addlegendimage{{{style['color']}, {style['line_style']}, mark={style['marker']}, mark options={{{style['marker_options']}}}, {style['line_width']}}}"
            )
        else:
            legend_images.append(
                rf"\addlegendimage{{{style['color']}, {style['line_style']}, {style['line_width']}}}"
            )
        legend_images.append(rf"\addlegendentry{{{style['legend']}}}")

    xtick_line = f"      xtick={XTICK}," if XTICK is not None else ""
    ytick_line = f"      ytick={YTICK}," if YTICK is not None else ""

    tex = rf"""\documentclass{{standalone}}
\usepackage{{pgfplots}}
\pgfplotsset{{compat=1.18}}
\usepgfplotslibrary{{fillbetween}}

\begin{{document}}
\begin{{tikzpicture}}
    \begin{{axis}}[
      axis x line  = bottom,
      axis y line  = left,
      width={AXIS_WIDTH}, height={AXIS_HEIGHT}, 
      xlabel={{{X_LABEL}}},
      ylabel={{{get_ylabel(SYSTEM)}}},
      xmin={limits['xmin']:.4f}, xmax={limits['xmax']+20:.4f},
      ymin={limits['ymin']:.4f}, ymax={limits['ymax']:.4f},
      legend cell align=left,
      legend columns=1,
      legend style={{
        draw=none, fill=none,
        at={{{LEGEND_AT}}}, anchor={LEGEND_ANCHOR},
        legend cell align=left
      }}
    ]

{''.join(body_blocks)}
{chr(10).join(legend_images)}

\end{{axis}}
\end{{tikzpicture}}
\end{{document}}
"""

    out_path.write_text(tex, encoding="utf-8")
    print(f"[OK] Saved: {out_path}")


def main() -> None:
    if not RESULTS_DIR.exists():
        raise FileNotFoundError(f"Results directory not found: {RESULTS_DIR}")

    print(f"[INFO] SYSTEM = {SYSTEM}")
    print(f"[INFO] SYSTEM_SCALE = {SYSTEM_SCALE}")
    print(f"[INFO] OBJECTIVE = {'maximize' if is_maximize(SYSTEM) else 'minimize'}")
    print(f"[INFO] Y_LABEL = {get_ylabel(SYSTEM)}")
    print(f"[INFO] FALLBACKS = {FALLBACKS}")
    print(f"[INFO] ALGORITHMS = {ALGORITHMS}")
    print(f"[INFO] RUN_IDS = {RUN_IDS}")
    print(f"[INFO] POINT_STEP = {POINT_STEP}")

    for fallback in FALLBACKS:
        all_stats = {}

        for algorithm in ALGORITHMS:
            curves = collect_curves(fallback, algorithm)
            if len(curves) == 0:
                print(f"[WARN] No valid runs for fallback={fallback}, algorithm={algorithm}")
                continue
            all_stats[algorithm] = aggregate(curves)

        export_one_tex(fallback, all_stats)

    print("[DONE] All tex files have been generated.")


if __name__ == "__main__":
    main()
