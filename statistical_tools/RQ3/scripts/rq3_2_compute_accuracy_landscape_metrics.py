#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import accuracy_metrics
from landscape_metrics import compute_landscape_pair_metrics


SCRIPT_DIR = Path(__file__).resolve().parent
RQ3_DIR = SCRIPT_DIR.parent
RESULTS_DIR = RQ3_DIR / "results"
PREDICTION_INPUT_DIR = RESULTS_DIR / "predictions"
ACCURACY_OUTPUT_DIR = RESULTS_DIR / "accuracy"
LANDSCAPE_OUTPUT_DIR = RESULTS_DIR / "landscape"

ID_COLS = ["system", "scale", "train_frac", "repeat", "fallback"]
VALUE_COLS = ["true_perf", "pred_perf"]
TOP_FRACS = [0.05, 0.10]
MAXIMIZE_SYSTEMS = {"mysql", "postgresql", "httpd", "tomcat"}
SYSTEM_ORDER = ["mysql", "postgresql", "httpd", "tomcat", "gcc", "clang", "x264", "x265"]

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


def infer_system_from_prediction_path(csv_path: Path) -> str:
    suffix = "_predictions.csv"
    stem = csv_path.name[:-len(suffix)] if csv_path.name.endswith(suffix) else csv_path.stem
    system, _scale = stem.rsplit("_", 1)
    return system


def find_prediction_csv_files(target_systems: set[str] | None) -> list[Path]:
    csv_files = sorted(PREDICTION_INPUT_DIR.glob("*/*_predictions.csv"))
    if target_systems is None:
        return csv_files

    return [
        csv_path for csv_path in csv_files
        if infer_system_from_prediction_path(csv_path) in target_systems
    ]


def normalize_requested_systems(requested: list[str] | None) -> set[str] | None:
    if not requested:
        return None
    normalized = [system.lower() for system in requested]
    if len(normalized) == 1 and normalized[0] == "all":
        return None
    return set(normalized)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute RQ3 accuracy and landscape metrics from fallback predictions."
    )
    parser.add_argument(
        "--systems",
        nargs="+",
        default=None,
        help=(
            "Systems to process. Defaults to all available prediction CSVs; "
            "use 'all' explicitly for all systems."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_systems = normalize_requested_systems(args.systems)

    if not PREDICTION_INPUT_DIR.exists():
        raise FileNotFoundError(f"Prediction directory not found: {PREDICTION_INPUT_DIR}")

    csv_files = find_prediction_csv_files(target_systems)
    if not csv_files:
        print(f"[WARN] No matching prediction CSV files found under {PREDICTION_INPUT_DIR}")
        return

    if target_systems is not None:
        found_systems = {
            infer_system_from_prediction_path(csv_path)
            for csv_path in csv_files
        }
        missing_systems = sorted(target_systems - found_systems)
        if missing_systems:
            print(f"[WARN] No prediction CSV files found for systems: {missing_systems}")

    for csv_path in csv_files:
        print(f"[PROCESS] {csv_path}")
        compute_one_prediction_file(csv_path)


if __name__ == "__main__":
    main()
