#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# User configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
ACCURACY_INPUT_DIR = BASE_DIR / "accuracy"
LANDSCAPE_INPUT_DIR = BASE_DIR / "landscape"
OUTPUT_DIR = BASE_DIR / "visualization"

# 只分析 large；如果你后面想分析 medium/small，可以改成 None 或对应列表
TARGET_SCALE = "large"

# 需要可视化的指标
ACCURACY_METRICS = [
    "mre",
    # "median_mre",
    # "p90_mre",
    "top5_overlap",
]

LANDSCAPE_METRICS = [
    "local_optima_ratio_abs_error",
    "local_optima_count_abs_error",
    "auto_correlation_abs_error",
]

# fallback 显示顺序
FALLBACK_ORDER = [
    "worst",
    "1NN",
    "kNN",
    "kNNw",
    "loess",
    "rf",
    "rf_uncertainty",
    "dal_rf",
]

# 是否在柱子上显示数值
SHOW_VALUE_LABELS = True


# ============================================================
# Helpers
# ============================================================

def get_display_name(metric: str) -> str:
    """
    Pretty metric name for figure titles / y-axis.
    """
    mapping = {
        "mre": "MRE",
        "median_mre": "Median MRE",
        "p90_mre": "P90 MRE",
        "mae": "MAE",
        "rmse": "RMSE",
        "bias": "Bias",
        "abs_bias": "Absolute Bias",
        "error_std": "Error Std.",
        "nmae_iqr": "NMAE-IQR",
        "nrmse_iqr": "NRMSE-IQR",
        "top5_overlap": "Top-5% Overlap",
        "top10_overlap": "Top-10% Overlap",
        "ndcg_top5": "NDCG@Top-5%",
        "ndcg_top10": "NDCG@Top-10%",
        "spearman_perf": "Spearman",
        "kendall_perf": "Kendall",
        "pearson_perf": "Pearson",
        "local_optima_ratio_abs_error": "Local Optima Ratio Abs. Error",
        "local_optima_count_abs_error": "Local Optima Count Abs. Error",
        "auto_correlation_abs_error": "Auto-correlation Abs. Error",
    }
    return mapping.get(metric, metric)


def parse_system_scale_from_filename(csv_path: Path):
    """
    Examples:
        clang_large_accuracy_metrics.csv
        clang_large_landscape_metrics.csv
    -> system='clang', scale='large'
    """
    name = csv_path.stem
    parts = name.split("_")

    if len(parts) < 4:
        raise ValueError(f"Unexpected filename format: {csv_path.name}")

    system = parts[0]
    scale = parts[1]
    return system, scale


def load_metric_group(input_dir: Path, suffix: str) -> dict[tuple[str, str], pd.DataFrame]:
    grouped = {}
    if not input_dir.exists():
        print(f"[WARN] Input directory not found: {input_dir}")
        return grouped

    for csv_path in sorted(input_dir.glob(f"*/*_{suffix}.csv")):
        try:
            system, scale = parse_system_scale_from_filename(csv_path)
        except ValueError:
            continue

        if TARGET_SCALE is not None and scale != TARGET_SCALE:
            continue

        df = pd.read_csv(csv_path)
        key = (system, scale)
        grouped.setdefault(key, []).append(df)

    return {
        key: pd.concat(frames, ignore_index=True)
        for key, frames in grouped.items()
        if frames
    }


def add_value_labels(ax, fmt: str = "{:.2f}") -> None:
    """
    Add numeric labels on bars.
    """
    for container in ax.containers:
        labels = []
        for bar in container:
            h = bar.get_height()
            if np.isnan(h):
                labels.append("")
            else:
                labels.append(fmt.format(h))
        ax.bar_label(container, labels=labels, rotation=90, padding=2, fontsize=8)


def summarize_metric(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """
    Group by train_frac and fallback, then average over repeats.

    Output pivot shape:
        index   = fallback
        columns = train_frac
        values  = mean(metric over repeats)
    """
    if metric not in df.columns:
        raise ValueError(f"Metric '{metric}' not found in dataframe columns.")

    summary = (
        df.groupby(["train_frac", "fallback"], as_index=False)[metric]
        .mean()
    )

    pivot_df = summary.pivot(
        index="fallback",
        columns="train_frac",
        values=metric
    )

    # reorder fallback rows
    existing_fallbacks = [fb for fb in FALLBACK_ORDER if fb in pivot_df.index]
    remaining_fallbacks = [fb for fb in pivot_df.index if fb not in existing_fallbacks]
    pivot_df = pivot_df.loc[existing_fallbacks + remaining_fallbacks]

    # sort train_frac columns
    pivot_df = pivot_df.reindex(sorted(pivot_df.columns), axis=1)

    return pivot_df


def plot_metric_for_system(
    df: pd.DataFrame,
    system: str,
    scale: str,
    metric: str,
    output_dir: Path,
) -> None:
    """
    Draw grouped bar chart for one system and one metric.
    """
    pivot_df = summarize_metric(df, metric)

    if pivot_df.empty:
        print(f"[WARN] Empty pivot table for system={system}, metric={metric}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6))

    pivot_df.plot(
        kind="bar",
        ax=ax,
        width=0.8,
    )

    metric_name = get_display_name(metric)

    ax.set_title(f"{system.upper()} ({scale}) - Mean {metric_name} across repeats")
    ax.set_xlabel("Fallback")
    ax.set_ylabel(metric_name)
    ax.set_xticklabels(pivot_df.index, rotation=30, ha="right")
    ax.legend(title="train_frac")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    if SHOW_VALUE_LABELS:
        if metric == "mre":
            add_value_labels(ax, fmt="{:.2f}")
        else:
            add_value_labels(ax, fmt="{:.3f}")

    plt.tight_layout()

    save_path = output_dir / f"{metric}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[SAVE] {save_path}")

    # 可选：顺手把用于画图的均值表也存一下，便于检查
    summary_csv = output_dir / f"{metric}_summary.csv"
    pivot_df.to_csv(summary_csv)
    print(f"[SAVE] {summary_csv}")


# ============================================================
# Main
# ============================================================

def main():
    metric_groups = [
        (
            "accuracy",
            load_metric_group(ACCURACY_INPUT_DIR, "accuracy_metrics"),
            ACCURACY_METRICS,
        ),
        (
            "landscape",
            load_metric_group(LANDSCAPE_INPUT_DIR, "landscape_metrics"),
            LANDSCAPE_METRICS,
        ),
    ]

    for group_name, grouped_data, metrics in metric_groups:
        if not grouped_data:
            print(f"[WARN] No {group_name} metric CSV files found.")
            continue

        for (system, scale), df in grouped_data.items():
            print(f"\n[PROCESS] {group_name}: {system}_{scale}")

            required_cols = {"fallback", "train_frac", "repeat"}
            missing_required = required_cols - set(df.columns)
            if missing_required:
                print(f"[WARN] Missing required columns {missing_required} for {group_name}:{system}_{scale}")
                continue

            system_out_dir = OUTPUT_DIR / group_name / system
            for metric in metrics:
                if metric not in df.columns:
                    print(f"[WARN] Metric '{metric}' not found for {group_name}:{system}_{scale}, skip.")
                    continue

                plot_metric_for_system(
                    df=df,
                    system=system,
                    scale=scale,
                    metric=metric,
                    output_dir=system_out_dir,
                )


if __name__ == "__main__":
    main()
