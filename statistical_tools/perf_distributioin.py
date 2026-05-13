#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


# ============================================================
# 1. Paths
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DATASETS_DIR = PROJECT_ROOT / "datasets"
OUTPUT_ROOT = SCRIPT_DIR / "perf_distribution"


# ============================================================
# 2. Config
# ============================================================

DPI = 300
FIG_W = 8
FIG_H = 5

SAVE_PDF = False

# KDE grid size
KDE_POINTS = 500

# Performance column candidates for each system
# Matching is case-insensitive and ignores spaces / underscores / hyphens
TARGET_COLUMN_CANDIDATES = {
    "mysql": ["throughput", "throughtput", "tps"],
    "postgresql": ["throughput", "throughtput", "tps"],
    "httpd": ["rps", "throughput"],
    "tomcat": ["rps", "throughput"],
    "gcc": ["run_time", "runtime", "execution_time", "time"],
    "clang": ["run_time", "runtime", "execution_time", "time"],
    "x264": ["run_time", "runtime", "execution_time", "time"],
    "x265": ["run_time", "runtime", "execution_time", "time"],
}


# ============================================================
# 3. Helpers
# ============================================================

def normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def get_perf_label(system: str) -> str:
    s = system.lower()
    if s in {"mysql", "postgresql"}:
        return "Throughput (tps)"
    if s in {"httpd", "tomcat"}:
        return "Throughput (rps)"
    return "Runtime (s)"


def find_performance_column(df: pd.DataFrame, system: str) -> str:
    """
    Find the performance column according to system-specific rules.
    If no candidate matches, fall back to the last numeric column.
    """
    s = system.lower()
    normalized_to_original = {normalize_name(c): c for c in df.columns}

    # 1) system-specific candidate matching
    candidates = TARGET_COLUMN_CANDIDATES.get(s, [])
    for cand in candidates:
        key = normalize_name(cand)
        if key in normalized_to_original:
            return normalized_to_original[key]

    # 2) global common candidate matching
    global_candidates = [
        "throughput", "throughtput", "tps",
        "rps",
        "run_time", "runtime", "execution_time", "time"
    ]
    for cand in global_candidates:
        key = normalize_name(cand)
        if key in normalized_to_original:
            return normalized_to_original[key]

    # 3) fallback: last numeric column
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        fallback_col = numeric_cols[-1]
        print(
            f"[WARN] Cannot find explicit performance column for system='{system}'. "
            f"Fall back to last numeric column: '{fallback_col}'"
        )
        return fallback_col

    raise ValueError(
        f"No numeric column found in file. Available columns: {list(df.columns)}"
    )


def clean_perf_values(series: pd.Series) -> np.ndarray:
    vals = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    return vals


def extract_scale_from_name(file_stem: str) -> str:
    parts = file_stem.split("_")
    if len(parts) >= 2:
        return parts[-1]
    return "unknown"


def summarize(values: np.ndarray) -> Dict[str, float]:
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=0)),
        "min": float(np.min(values)),
        "q1": float(np.percentile(values, 25)),
        "median": float(np.median(values)),
        "q3": float(np.percentile(values, 75)),
        "max": float(np.max(values)),
    }


def add_summary_text(ax, values: np.ndarray) -> None:
    stats = summarize(values)
    summary_text = (
        f"n={stats['count']}, "
        f"mean={stats['mean']:.4f}, "
        f"median={stats['median']:.4f}, "
        f"std={stats['std']:.4f}"
    )
    ax.text(
        0.98, 0.98,
        summary_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="gray", alpha=0.9)
    )


def plot_kde(
    values: np.ndarray,
    system: str,
    dataset_name: str,
    out_png: Path,
    out_pdf: Optional[Path] = None,
) -> None:
    """
    Plot a standalone KDE figure.
    """
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), constrained_layout=True)

    # To avoid singular covariance when all values are the same
    if np.allclose(values, values[0]):
        x_grid = np.linspace(values[0] - 1e-6, values[0] + 1e-6, KDE_POINTS)
        y_kde = np.zeros_like(x_grid)
        y_kde[len(y_kde) // 2] = 1.0
    else:
        kde = gaussian_kde(values)
        xmin, xmax = np.min(values), np.max(values)
        margin = 0.05 * (xmax - xmin) if xmax > xmin else 1e-6
        x_grid = np.linspace(xmin - margin, xmax + margin, KDE_POINTS)
        y_kde = kde(x_grid)

    ax.plot(x_grid, y_kde, linewidth=2)
    ax.fill_between(x_grid, y_kde, alpha=0.3)

    ax.set_title(f"{dataset_name} KDE", fontsize=13)
    ax.set_xlabel(get_perf_label(system), fontsize=11)
    ax.set_ylabel("Density", fontsize=11)

    add_summary_text(ax, values)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=DPI, bbox_inches="tight")
    if out_pdf is not None:
        fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def plot_violin(
    values: np.ndarray,
    system: str,
    dataset_name: str,
    out_png: Path,
    out_pdf: Optional[Path] = None,
) -> None:
    """
    Plot a standalone violin figure.
    """
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), constrained_layout=True)

    vp = ax.violinplot(
        dataset=[values],
        positions=[1],
        showmeans=True,
        showmedians=True,
        showextrema=True,
        widths=0.8,
    )

    ax.set_title(f"{dataset_name} Violin", fontsize=13)
    ax.set_ylabel(get_perf_label(system), fontsize=11)
    ax.set_xticks([1])
    ax.set_xticklabels([dataset_name], rotation=0)

    add_summary_text(ax, values)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=DPI, bbox_inches="tight")
    if out_pdf is not None:
        fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def process_one_csv(csv_path: Path, system: str) -> Optional[Dict[str, object]]:
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[WARN] Failed to read {csv_path}: {e}")
        return None

    if df.empty:
        print(f"[WARN] Empty file: {csv_path}")
        return None

    try:
        perf_col = find_performance_column(df, system)
    except Exception as e:
        print(f"[WARN] {csv_path}: {e}")
        return None

    values = clean_perf_values(df[perf_col])
    if len(values) == 0:
        print(f"[WARN] No valid performance values in {csv_path}")
        return None

    dataset_name = csv_path.stem
    scale = extract_scale_from_name(dataset_name)

    out_dir = OUTPUT_ROOT / system

    kde_png = out_dir / f"{dataset_name}_kde.png"
    violin_png = out_dir / f"{dataset_name}_violin.png"

    kde_pdf = out_dir / f"{dataset_name}_kde.pdf" if SAVE_PDF else None
    violin_pdf = out_dir / f"{dataset_name}_violin.pdf" if SAVE_PDF else None

    plot_kde(
        values=values,
        system=system,
        dataset_name=dataset_name,
        out_png=kde_png,
        out_pdf=kde_pdf,
    )
    print(f"[OK] Saved KDE: {kde_png}")

    plot_violin(
        values=values,
        system=system,
        dataset_name=dataset_name,
        out_png=violin_png,
        out_pdf=violin_pdf,
    )
    print(f"[OK] Saved Violin: {violin_png}")

    stats = summarize(values)
    row = {
        "system": system,
        "dataset": dataset_name,
        "scale": scale,
        "file": str(csv_path),
        "performance_column": perf_col,
        **stats,
    }
    return row


# ============================================================
# 4. Main
# ============================================================

def main() -> None:
    if not DATASETS_DIR.exists():
        raise FileNotFoundError(f"Datasets directory not found: {DATASETS_DIR}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict[str, object]] = []

    system_dirs = sorted([p for p in DATASETS_DIR.iterdir() if p.is_dir()])
    if not system_dirs:
        print(f"[WARN] No system directories found under {DATASETS_DIR}")
        return

    for system_dir in system_dirs:
        system = system_dir.name
        csv_files = sorted(system_dir.glob("*.csv"))

        if not csv_files:
            print(f"[WARN] No csv files found in {system_dir}")
            continue

        print(f"\n[INFO] Processing system: {system}")

        system_rows = []
        for csv_path in csv_files:
            row = process_one_csv(csv_path, system)
            if row is not None:
                system_rows.append(row)
                all_rows.append(row)

        if system_rows:
            df_sys = pd.DataFrame(system_rows)
            out_summary = OUTPUT_ROOT / system / "summary.csv"
            out_summary.parent.mkdir(parents=True, exist_ok=True)
            df_sys.to_csv(out_summary, index=False)
            print(f"[OK] Saved summary: {out_summary}")

    if all_rows:
        df_all = pd.DataFrame(all_rows)
        out_all = OUTPUT_ROOT / "summary_all_systems.csv"
        df_all.to_csv(out_all, index=False)
        print(f"\n[OK] Saved global summary: {out_all}")

    print("\n[DONE] Performance distribution visualization finished.")


if __name__ == "__main__":
    main()