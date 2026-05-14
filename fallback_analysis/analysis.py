#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.neighbors import BallTree

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Adjust this import according to your project structure
from utils.config_manager import ConfigManager
import accuracy_metrics
from landscape_metrics import compute_landscape_pair_metrics


# ============================================================
# User settings
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATASET_ROOT = BASE_DIR.parent / "datasets"
ACCURACY_OUTPUT_DIR = BASE_DIR / "accuracy"
LANDSCAPE_OUTPUT_DIR = BASE_DIR / "landscape"

SYSTEMS = [
    "mysql",
    "postgresql",
    "httpd",
    "tomcat",
    "gcc",
    "clang",
    "x264",
    "x265",
]

SCALES = [
    "large",
]

# Current fallback strategies
FALLBACKS = [
    "worst",
    "1NN",
    "kNN",
    "kNNw",
    "loess",
    "rf",
    "rf_uncertainty",
    "dal_rf",
    # "dt",
    # "hgb",
    # "gp",
    # "tabpfn",
]

# Sparse ratios: percentage of the full tabular data used as observed data
TRAIN_FRACS = [0.05]

REPEATS = 10

# k for kNN / kNNw
KNN_K = 20

# top-region metrics
TOP_FRACS = [0.05, 0.10]

MAXIMIZE_SYSTEMS = {"mysql", "postgresql", "httpd", "tomcat"}

ID_COLS = ["system", "scale", "train_frac", "repeat", "fallback"]
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


# ============================================================
# Basic helpers
# ============================================================

def infer_optimization_goal(system: str) -> str:
    """
    Return the objective direction used by ConfigManager.
    """
    if system in MAXIMIZE_SYSTEMS:
        return "maximum"
    return "minimum"


# ============================================================
# Sparse ConfigManager construction
# ============================================================

def build_sparse_config_manager(
    system: str,
    train_df: pd.DataFrame,
    full_df: pd.DataFrame,
) -> ConfigManager:
    """
    Build a ConfigManager whose observed table is train_df.

    Important:
    - train_df is used for fallback training and exact lookup.
    - full_df is used only to define the configuration domain / encoder categories.
      This does not leak performance values into the fallback model.
    """
    goal = infer_optimization_goal(system)
    cm = ConfigManager(system=system, optimization_goal=goal)

    train_df = train_df.copy().reset_index(drop=True)
    full_df = full_df.copy().reset_index(drop=True)

    # data_df must be train_df because several models, especially DaL, use self.data_df
    cm.data_df = train_df

    config_train = train_df.iloc[:, :-1]
    perf_train = train_df.iloc[:, -1].astype(float)

    config_full = full_df.iloc[:, :-1]

    cm.config_dim = config_full.shape[1]
    cm.config_space = config_full.drop_duplicates().values.tolist()

    cm.value_domains = []
    for col in config_full.columns:
        cm.value_domains.append(config_full[col].drop_duplicates().tolist())

    cm.config_perf_dict = {
        tuple(row[:-1]): float(row[-1])
        for row in train_df.values
    }

    cm.config_idx_dict = {}
    for idx, row in train_df.iterrows():
        key = tuple(row.iloc[:-1])
        cm.config_idx_dict.setdefault(key, []).append(idx)

    # Worst value should be computed only from observed sparse data
    if goal == "minimum":
        cm.worst_perf = float(perf_train.max())
    else:
        cm.worst_perf = float(perf_train.min())

    # Detect feature types and normalization factors using the full domain.
    # This uses only configuration values, not performance values.
    detect_feature_types_from_domain(cm, config_full)

    # Fit encoder using the full configuration domain.
    # This avoids shape mismatch when holdout contains a category absent in train_df.
    fit_encoder_from_domain(cm, config_full)

    cm.reset_surrogates()
    cm.reset_dal()
    cm.init_fallback_registry()

    return cm


def detect_feature_types_from_domain(cm: ConfigManager, config_df: pd.DataFrame) -> None:
    """
    Detect numerical/categorical types and normalization ranges from the full config domain.
    """
    cm.feature_types = []
    cm.norm_factors = []

    for col in config_df.columns:
        values = config_df[col].dropna().tolist()

        try:
            float_vals = list(map(float, values))
            unique_vals = set(float_vals)
        except Exception:
            cm.feature_types.append("categorical")
            cm.norm_factors.append(1.0)
            continue

        if unique_vals <= {0.0, 1.0} and len(unique_vals) <= 2:
            cm.feature_types.append("categorical")
            cm.norm_factors.append(1.0)
        else:
            cm.feature_types.append("numerical")
            max_val = max(float_vals)
            min_val = min(float_vals)
            norm = max_val - min_val
            cm.norm_factors.append(float(norm if norm != 0 else 1.0))


def fit_encoder_from_domain(cm: ConfigManager, config_df: pd.DataFrame) -> None:
    """
    Fit OrdinalEncoder on the full config domain, then build BallTree only for train configs.
    """
    full_configs = config_df.values.tolist()
    full_str_configs = [
        [cm.to_str_clean(v) for v in cfg]
        for cfg in full_configs
    ]

    cm.encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )
    cm.encoder.fit(full_str_configs)

    train_configs = list(cm.config_perf_dict.keys())
    train_str_configs = [
        [cm.to_str_clean(v) for v in cfg]
        for cfg in train_configs
    ]

    if len(train_str_configs) > 0:
        cm.encoded_configs = cm.encoder.transform(train_str_configs)
        cm.balltree = BallTree(cm.encoded_configs, metric="hamming")
    else:
        cm.encoded_configs = None
        cm.balltree = None


def prefit_fallback_if_needed(cm: ConfigManager, fallback: str, seed: int) -> None:
    """
    Pre-fit expensive models once per split, rather than inside every prediction.
    """
    try:
        if fallback in {"rf", "rf_uncertainty"}:
            cm.fit_random_forest(
                n_estimators=300,
                random_state=seed,
                n_jobs=-1,
            )

        elif fallback == "dal_rf":
            cm.fit_dal_rf(
                depth=5,
                min_samples=10,
                seed=seed,
                use_mi_weight=True,
            )

        elif fallback == "dt":
            cm.fit_decision_tree(random_state=seed)

        elif fallback == "hgb":
            cm.fit_hgb(random_state=seed)

        elif fallback == "gp":
            cm.fit_gaussian_process(random_state=seed, max_train=5000)

    except Exception as e:
        print(f"[WARN] Failed to prefit {fallback}: {e}")


# ============================================================
# Split and evaluation
# ============================================================

def sample_sparse_split(
    df: pd.DataFrame,
    train_frac: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Randomly split full tabular data into sparse train and holdout.

    Assumption:
    each row corresponds to one configuration-performance record.
    """
    if not 0 < train_frac < 1:
        raise ValueError(f"train_frac should be in (0, 1), got {train_frac}")

    n = len(df)
    train_size = max(5, int(round(n * train_frac)))
    train_size = min(train_size, n - 1)

    rng = np.random.RandomState(seed)
    train_idx = rng.choice(n, size=train_size, replace=False)

    train_mask = np.zeros(n, dtype=bool)
    train_mask[train_idx] = True

    train_df = df.iloc[train_mask].copy().reset_index(drop=True)
    holdout_df = df.iloc[~train_mask].copy().reset_index(drop=True)

    return train_df, holdout_df


def predict_holdout(
    cm: ConfigManager,
    holdout_df: pd.DataFrame,
    fallback: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict holdout configurations using a given fallback.
    """
    cfgs = holdout_df.iloc[:, :-1].values.tolist()
    y_true = holdout_df.iloc[:, -1].astype(float).values

    y_pred = []

    for cfg in cfgs:
        try:
            # Directly use evaluate() to preserve your original logic:
            # exact lookup if observed, otherwise fallback.
            results, _ = cm.evaluate(
                [cfg],
                fallback=fallback,
                knn_k=KNN_K,
                lam=1.0,
            )
            pred = float(results[0][1])
        except Exception as e:
            print(f"[WARN] fallback={fallback} failed on cfg={cfg}: {e}")
            pred = np.nan

        y_pred.append(pred)

    return y_true, np.asarray(y_pred, dtype=float)


def run_one_dataset(
    system: str,
    scale: str,
    full_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Run sparse-to-full reconstruction analysis for one system-scale dataset.
    """
    rows = []

    for train_frac in TRAIN_FRACS:
        for repeat in range(REPEATS):
            seed = 1000 + repeat

            train_df, holdout_df = sample_sparse_split(
                full_df,
                train_frac=train_frac,
                seed=seed,
            )

            for fallback in FALLBACKS:
                print(
                    f"[RUN] system={system}, scale={scale}, "
                    f"train_frac={train_frac}, repeat={repeat}, fallback={fallback}"
                )

                cm = build_sparse_config_manager(
                    system=system,
                    train_df=train_df,
                    full_df=full_df,
                )

                prefit_fallback_if_needed(cm, fallback, seed)

                y_true, y_pred = predict_holdout(
                    cm=cm,
                    holdout_df=holdout_df,
                    fallback=fallback,
                )

                metric_dict = accuracy_metrics.compute_reconstruction_metrics(
                    y_true_raw=y_true,
                    y_pred_raw=y_pred,
                    system=system,
                    top_fracs=TOP_FRACS,
                )

                if not metric_dict:
                    continue

                maximize = infer_optimization_goal(system) == "maximum"
                landscape_dict = compute_landscape_pair_metrics(
                    configs=holdout_df.iloc[:, :-1],
                    true_perfs=y_true,
                    pred_perfs=y_pred,
                    maximize=maximize,
                )

                row = {
                    "system": system,
                    "scale": scale,
                    "train_frac": train_frac,
                    "repeat": repeat,
                    "fallback": fallback,
                }
                row.update(metric_dict)
                row.update(landscape_dict)
                rows.append(row)

    return pd.DataFrame(rows)


def summarize_results(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate repeated splits.
    """
    group_cols = ["system", "scale", "train_frac", "fallback"]

    metric_cols = [
        c for c in df.columns
        if c not in group_cols + ["repeat"]
    ]

    summary = (
        df.groupby(group_cols)[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )

    # Flatten MultiIndex columns
    summary.columns = [
        "_".join([str(x) for x in col if str(x) != ""])
        if isinstance(col, tuple)
        else col
        for col in summary.columns
    ]

    return summary


def select_existing_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [col for col in columns if col in df.columns]


def save_metric_outputs(result_df: pd.DataFrame, system: str, scale: str) -> None:
    """
    Save accuracy and landscape outputs under fallback-specific folders.
    """
    if result_df.empty:
        return

    landscape_raw_cols = [
        col for col in result_df.columns
        if col.startswith("true_")
        or col.startswith("pred_")
        or col in LANDSCAPE_ERROR_COLS
    ]

    for fallback, fb_df in result_df.groupby("fallback", sort=False):
        accuracy_dir = ACCURACY_OUTPUT_DIR / fallback
        landscape_dir = LANDSCAPE_OUTPUT_DIR / fallback
        accuracy_dir.mkdir(parents=True, exist_ok=True)
        landscape_dir.mkdir(parents=True, exist_ok=True)

        accuracy_cols = select_existing_columns(fb_df, ID_COLS + ACCURACY_COLS)
        accuracy_path = accuracy_dir / f"{system}_{scale}_accuracy_metrics.csv"
        fb_df[accuracy_cols].to_csv(accuracy_path, index=False)
        print(f"[SAVE] {accuracy_path}")

        landscape_error_cols = select_existing_columns(fb_df, ID_COLS + LANDSCAPE_ERROR_COLS)
        landscape_path = landscape_dir / f"{system}_{scale}_landscape_metrics.csv"
        fb_df[landscape_error_cols].to_csv(landscape_path, index=False)
        print(f"[SAVE] {landscape_path}")

        landscape_raw_path = landscape_dir / f"{system}_{scale}_landscape_raw.csv"
        landscape_raw_cols_existing = select_existing_columns(fb_df, ID_COLS + landscape_raw_cols)
        fb_df[landscape_raw_cols_existing].to_csv(landscape_raw_path, index=False)
        print(f"[SAVE] {landscape_raw_path}")


def main():
    random.seed(0)
    np.random.seed(0)

    ACCURACY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LANDSCAPE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for system in SYSTEMS:
        for scale in SCALES:
            csv_path = DATASET_ROOT / system / f"{system}_{scale}.csv"

            if not csv_path.exists():
                print(f"[SKIP] Missing dataset: {csv_path}")
                continue

            print(f"\n==============================")
            print(f"Dataset: {csv_path}")
            print(f"==============================")

            full_df = pd.read_csv(csv_path)

            result_df = run_one_dataset(
                system=system,
                scale=scale,
                full_df=full_df,
            )

            if result_df.empty:
                continue

            save_metric_outputs(result_df, system, scale)


if __name__ == "__main__":
    main()
