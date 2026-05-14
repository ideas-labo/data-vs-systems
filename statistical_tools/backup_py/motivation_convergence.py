#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Flexible convergence plotting script for motivation figures.

Features
--------
- Plot one figure per fallback strategy
- Each figure can contain any number of algorithms
- Per-run trajectories are first converted to monotonic best-so-far curves
- Then aggregated across runs by mean/std
- Save figures to statistical_tools/motivation/

Expected directory structure
----------------------------
results_dataset/
├── run0/
│   ├── 1NN/
│   │   ├── bestconfig/postgresql/postgresql_large/all_results.csv
│   │   ├── smac/postgresql/postgresql_large/all_results.csv
│   │   └── ...
│   ├── rf/
│   │   ├── bestconfig/postgresql/postgresql_large/all_results.csv
│   │   ├── smac/postgresql/postgresql_large/all_results.csv
│   │   └── ...
│   └── ...
├── run1/
...
└── run9/

Notes
-----
- This script assumes the performance column is the LAST column of all_results.csv
- For maximization tasks, convergence is cumulative max
- For minimization tasks, convergence is cumulative min
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. User configuration: modify only this section when needed
# ============================================================

# ---- target system ----
SYSTEM = "gcc"
SYSTEM_SCALE = "gcc_small"

# ---- which fallbacks to plot ----
FALLBACKS = [
    "1NN",
    "kNN",
    "kNNw",
    "loess",
    "rf",
    "dal_rf",
    "rf_uncertainty",
]

# ---- which algorithms to include in each figure ----
ALGORITHMS = [
    "smac",
    "bestconfig",
    "flash",
    "ga",
    "hebo",
    "promisetune",
    "random_search",
]

# ---- which runs to aggregate ----
RUN_IDS = list(range(10))
# 例如只看 run4:
# RUN_IDS = [4]
# 例如看前5次:
# RUN_IDS = list(range(5))

# ---- file name ----
CSV_NAME = "all_results.csv"

# ---- output ----
OUTPUT_FOLDER_NAME = "motivation"

# ---- figure style ----
FIG_WIDTH = 7.2
FIG_HEIGHT = 4.6
DPI = 300

LINE_WIDTH = 2.2
SHADE_ALPHA = 0.15
GRID_ALPHA = 0.25
LEGEND_FRAME = False

# If True, save std-shaded region; otherwise only draw mean curves
SHOW_STD_BAND = True

# If True, save aggregated csv for every (fallback, algorithm)
SAVE_AGGREGATED_CSV = True

# Optional pretty display names
ALGORITHM_DISPLAY_NAMES = {
    "smac": "SMAC",
    "bestconfig": "BestConfig",
    "flash": "FLASH",
    "ga": "GA",
    "hebo": "HEBO",
    "promisetune": "PromiseTune",
    "random_search": "Random Search",
}

FALLBACK_DISPLAY_NAMES = {
    "1NN": "1NN",
    "rf": "RF",
    "kNN": "kNN",
    "kNNw": "kNNw",
    "loess": "LOESS",
    "worst": "Worst",
    "dal_rf": "DAL-RF",
    "rf_uncertainty": "RF-Uncertainty",
}

# Optional line styles per algorithm
ALGORITHM_LINESTYLES = {
    "smac": "-",
    "bestconfig": "--",
    "flash": "-.",
    "ga": ":",
    "hebo": (0, (5, 1)),
    "promisetune": (0, (3, 1, 1, 1)),
    "random_search": (0, (1, 1)),
}

# Optional markers per algorithm (set to None to disable)
ALGORITHM_MARKERS = {
    "smac": None,
    "bestconfig": None,
    "flash": None,
    "ga": None,
    "hebo": None,
    "promisetune": None,
    "random_search": None,
}

# Plot title usage
USE_TITLE = True


# ============================================================
# 2. Paths
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RESULTS_DIR = PROJECT_ROOT / "results_dataset"
OUTPUT_DIR = SCRIPT_DIR / OUTPUT_FOLDER_NAME


# ============================================================
# 3. Utility functions
# ============================================================

def is_maximize(system: str) -> bool:
    """
    Return True if the system is a maximization problem.
    mysql / postgresql / tomcat / httpd are maximization.
    Others are minimization.
    """
    maximize_systems = {
        "mysql",
        "postgresql",
        "tomcat",
        "httpd",
    }
    return system.lower() in maximize_systems


def get_display_name(name: str, mapping: Dict[str, str]) -> str:
    return mapping.get(name, name)


def get_ylabel(system: str) -> str:
    if is_maximize(system):
        return "Best-so-far performance (higher is better)"
    return "Best-so-far performance (lower is better)"


def read_performance_curve(csv_path: Path) -> Optional[np.ndarray]:
    """
    Read the last column of csv as performance values.
    """
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
    if perf.empty:
        print(f"[WARN] No valid numeric performance values: {csv_path}")
        return None

    values = perf.to_numpy(dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        print(f"[WARN] No finite performance values: {csv_path}")
        return None

    return values


def make_monotonic_curve(values: np.ndarray, maximize: bool = True) -> np.ndarray:
    """
    Convert raw performance curve into best-so-far curve.
    """
    if maximize:
        return np.maximum.accumulate(values)
    return np.minimum.accumulate(values)


def collect_curves(
    fallback: str,
    algorithm: str,
    run_ids: List[int],
) -> List[np.ndarray]:
    """
    Collect monotonic curves across runs for one fallback and one algorithm.
    """
    curves = []
    maximize = is_maximize(SYSTEM)

    for run_id in run_ids:
        csv_path = (
            RESULTS_DIR
            / f"run{run_id}"
            / fallback
            / algorithm
            / SYSTEM
            / SYSTEM_SCALE
            / CSV_NAME
        )

        values = read_performance_curve(csv_path)
        if values is None:
            continue

        mono = make_monotonic_curve(values, maximize=maximize)
        curves.append(mono)

    return curves


def pad_curves_with_last_value(curves: List[np.ndarray]) -> np.ndarray:
    """
    Pad all curves to the same length using their last value.
    This preserves the best-so-far semantics.
    """
    if not curves:
        raise ValueError("curves is empty")

    max_len = max(len(curve) for curve in curves)
    padded = []

    for curve in curves:
        if len(curve) < max_len:
            extra = np.full(max_len - len(curve), curve[-1], dtype=float)
            curve = np.concatenate([curve, extra])
        padded.append(curve)

    return np.vstack(padded)


def aggregate_curves(curves: List[np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Aggregate multiple curves into x / mean / std.
    """
    arr = pad_curves_with_last_value(curves)
    x = np.arange(1, arr.shape[1] + 1)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)

    return {
        "x": x,
        "mean": mean,
        "std": std,
        "all": arr,
    }


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def save_aggregated_csv(
    fallback: str,
    algorithm: str,
    stats: Dict[str, np.ndarray],
) -> None:
    if not SAVE_AGGREGATED_CSV:
        return

    out_csv = OUTPUT_DIR / f"aggregation_{fallback}_{algorithm}_{SYSTEM}_{SYSTEM_SCALE}.csv"
    df = pd.DataFrame({
        "evaluation": stats["x"],
        "mean": stats["mean"],
        "std": stats["std"],
    })
    df.to_csv(out_csv, index=False)


def setup_matplotlib_style() -> None:
    plt.rcParams["figure.figsize"] = (FIG_WIDTH, FIG_HEIGHT)
    plt.rcParams["figure.dpi"] = DPI
    plt.rcParams["savefig.dpi"] = DPI
    plt.rcParams["font.size"] = 11
    plt.rcParams["axes.labelsize"] = 11
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["legend.fontsize"] = 10
    plt.rcParams["xtick.labelsize"] = 10
    plt.rcParams["ytick.labelsize"] = 10


def plot_one_fallback(fallback: str) -> None:
    """
    Plot all selected algorithms under one fallback.
    """
    plt.figure()
    plotted_any = False

    for algorithm in ALGORITHMS:
        curves = collect_curves(
            fallback=fallback,
            algorithm=algorithm,
            run_ids=RUN_IDS,
        )

        if len(curves) == 0:
            print(f"[WARN] No valid runs for fallback={fallback}, algorithm={algorithm}")
            continue

        stats = aggregate_curves(curves)
        save_aggregated_csv(fallback, algorithm, stats)

        x = stats["x"]
        mean = stats["mean"]
        std = stats["std"]

        label = f"{get_display_name(algorithm, ALGORITHM_DISPLAY_NAMES)} (n={len(curves)})"
        linestyle = ALGORITHM_LINESTYLES.get(algorithm, "-")
        marker = ALGORITHM_MARKERS.get(algorithm, None)

        plt.plot(
            x,
            mean,
            label=label,
            linewidth=LINE_WIDTH,
            linestyle=linestyle,
            marker=marker,
            markevery=max(1, len(x) // 12) if marker is not None else None,
        )

        if SHOW_STD_BAND:
            plt.fill_between(
                x,
                mean - std,
                mean + std,
                alpha=SHADE_ALPHA,
            )

        plotted_any = True

    if not plotted_any:
        print(f"[WARN] Nothing plotted for fallback={fallback}")
        plt.close()
        return

    plt.xlabel("Evaluation")
    plt.ylabel(get_ylabel(SYSTEM))

    if USE_TITLE:
        fallback_name = get_display_name(fallback, FALLBACK_DISPLAY_NAMES)
        plt.title(f"{SYSTEM_SCALE} under {fallback_name}")

    plt.grid(True, alpha=GRID_ALPHA)
    plt.legend(frameon=LEGEND_FRAME)
    plt.tight_layout()

    png_path = OUTPUT_DIR / f"convergence_{fallback}_{SYSTEM}_{SYSTEM_SCALE}.png"
    pdf_path = OUTPUT_DIR / f"convergence_{fallback}_{SYSTEM}_{SYSTEM_SCALE}.pdf"

    plt.savefig(png_path, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()

    print(f"[OK] Saved: {png_path}")
    print(f"[OK] Saved: {pdf_path}")


def main() -> None:
    setup_matplotlib_style()
    ensure_output_dir()

    print(f"[INFO] PROJECT_ROOT = {PROJECT_ROOT}")
    print(f"[INFO] RESULTS_DIR  = {RESULTS_DIR}")
    print(f"[INFO] OUTPUT_DIR   = {OUTPUT_DIR}")
    print(f"[INFO] SYSTEM       = {SYSTEM}")
    print(f"[INFO] SCALE        = {SYSTEM_SCALE}")
    print(f"[INFO] OBJECTIVE    = {'maximize' if is_maximize(SYSTEM) else 'minimize'}")
    print(f"[INFO] FALLBACKS    = {FALLBACKS}")
    print(f"[INFO] ALGORITHMS   = {ALGORITHMS}")
    print(f"[INFO] RUN_IDS      = {RUN_IDS}")

    if not RESULTS_DIR.exists():
        raise FileNotFoundError(f"Results directory not found: {RESULTS_DIR}")

    for fallback in FALLBACKS:
        print(f"\n[INFO] Plotting fallback: {fallback}")
        plot_one_fallback(fallback)

    print("\n[DONE] Finished plotting all requested fallback figures.")


if __name__ == "__main__":
    main()
