#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path

import pandas as pd

import accuracy_metrics
from landscape_metrics import compute_landscape_pair_metrics


BASE_DIR = Path(__file__).resolve().parent
PREDICTION_INPUT_DIR = BASE_DIR / "predictions"
ACCURACY_OUTPUT_DIR = BASE_DIR / "accuracy"
LANDSCAPE_OUTPUT_DIR = BASE_DIR / "landscape"

ID_COLS = ["system", "scale", "train_frac", "repeat", "fallback"]
VALUE_COLS = ["true_perf", "pred_perf"]
TOP_FRACS = [0.05, 0.10]
MAXIMIZE_SYSTEMS = {"mysql", "postgresql", "httpd", "tomcat"}

ACCURACY_COLS = [
    "n_samples",
    "mre",
    "median_mre",
    "p90_mre",
    "mae",
    "rmse",
    "bias",
    "abs_bias",
    "error_std",
    "nmae_iqr",
    "nrmse_iqr",
    "spearman_perf",
    "kendall_perf",
    "pearson_perf",
    "top5_overlap",
    "ndcg_top5",
    "top10_overlap",
    "ndcg_top10",
]

LANDSCAPE_ERROR_COLS = [
    "local_optima_ratio_abs_error",
    "local_optima_count_abs_error",
    "auto_correlation_abs_error",
]


def infer_maximize(system: str) -> bool:
    return system in MAXIMIZE_SYSTEMS


def select_existing_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [col for col in columns if col in df.columns]


def get_config_columns(df: pd.DataFrame) -> list[str]:
    blocked = set(ID_COLS + VALUE_COLS)
    return [col for col in df.columns if col not in blocked]


def compute_metrics_from_prediction_group(group_df: pd.DataFrame) -> dict:
    first = group_df.iloc[0]
    system = str(first["system"])

    config_cols = get_config_columns(group_df)
    y_true = group_df["true_perf"].astype(float).to_numpy()
    y_pred = group_df["pred_perf"].astype(float).to_numpy()

    metrics = {
        "system": first["system"],
        "scale": first["scale"],
        "train_frac": first["train_frac"],
        "repeat": first["repeat"],
        "fallback": first["fallback"],
    }
    metrics.update(
        accuracy_metrics.compute_reconstruction_metrics(
            y_true_raw=y_true,
            y_pred_raw=y_pred,
            system=system,
            top_fracs=TOP_FRACS,
        )
    )
    metrics.update(
        compute_landscape_pair_metrics(
            configs=group_df[config_cols],
            true_perfs=y_true,
            pred_perfs=y_pred,
            maximize=infer_maximize(system),
        )
    )
    return metrics


def save_metric_outputs(result_df: pd.DataFrame, system: str, scale: str, fallback: str) -> None:
    accuracy_dir = ACCURACY_OUTPUT_DIR / fallback
    landscape_dir = LANDSCAPE_OUTPUT_DIR / fallback
    accuracy_dir.mkdir(parents=True, exist_ok=True)
    landscape_dir.mkdir(parents=True, exist_ok=True)

    accuracy_cols = select_existing_columns(result_df, ID_COLS + ACCURACY_COLS)
    accuracy_path = accuracy_dir / f"{system}_{scale}_accuracy_metrics.csv"
    result_df[accuracy_cols].to_csv(accuracy_path, index=False)
    print(f"[SAVE] {accuracy_path}")

    landscape_error_cols = select_existing_columns(result_df, ID_COLS + LANDSCAPE_ERROR_COLS)
    landscape_path = landscape_dir / f"{system}_{scale}_landscape_metrics.csv"
    result_df[landscape_error_cols].to_csv(landscape_path, index=False)
    print(f"[SAVE] {landscape_path}")

    landscape_raw_cols = [
        col for col in result_df.columns
        if col.startswith("true_")
        or col.startswith("pred_")
        or col in LANDSCAPE_ERROR_COLS
    ]
    landscape_raw_path = landscape_dir / f"{system}_{scale}_landscape_raw.csv"
    result_df[select_existing_columns(result_df, ID_COLS + landscape_raw_cols)].to_csv(
        landscape_raw_path,
        index=False,
    )
    print(f"[SAVE] {landscape_raw_path}")


def compute_one_prediction_file(csv_path: Path) -> None:
    df = pd.read_csv(csv_path)
    required = set(ID_COLS + VALUE_COLS)
    missing = required - set(df.columns)
    if missing:
        print(f"[WARN] Skip {csv_path}: missing columns {sorted(missing)}")
        return

    rows = []
    for _, group_df in df.groupby(ID_COLS, sort=False):
        rows.append(compute_metrics_from_prediction_group(group_df.reset_index(drop=True)))

    if not rows:
        print(f"[WARN] No metric rows generated from {csv_path}")
        return

    result_df = pd.DataFrame(rows)
    first = result_df.iloc[0]
    save_metric_outputs(
        result_df=result_df,
        system=str(first["system"]),
        scale=str(first["scale"]),
        fallback=str(first["fallback"]),
    )


def main() -> None:
    if not PREDICTION_INPUT_DIR.exists():
        raise FileNotFoundError(f"Prediction directory not found: {PREDICTION_INPUT_DIR}")

    csv_files = sorted(PREDICTION_INPUT_DIR.glob("*/*_predictions.csv"))
    if not csv_files:
        print(f"[WARN] No prediction CSV files found under {PREDICTION_INPUT_DIR}")
        return

    for csv_path in csv_files:
        print(f"[PROCESS] {csv_path}")
        compute_one_prediction_file(csv_path)


if __name__ == "__main__":
    main()
