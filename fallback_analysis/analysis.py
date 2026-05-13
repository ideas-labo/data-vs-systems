#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau, pearsonr
from sklearn.preprocessing import OrdinalEncoder
from sklearn.neighbors import BallTree

# Adjust this import according to your project structure
from utils.config_manager import ConfigManager


# ============================================================
# User settings
# ============================================================

DATASET_ROOT = Path("../datasets")
OUTPUT_DIR = Path("analysis_results/fallback_reconstruction")

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
TRAIN_FRACS = [0.2]

REPEATS = 3

# k for kNN / kNNw
KNN_K = 20

# top-region metrics
TOP_FRACS = [0.05, 0.10]

MAXIMIZE_SYSTEMS = {"mysql", "postgresql", "httpd", "tomcat"}


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


def objective_to_score(y: np.ndarray, system: str) -> np.ndarray:
    """
    Convert raw performance into a unified score:
      larger score = better.
    """
    y = np.asarray(y, dtype=float)
    if system in MAXIMIZE_SYSTEMS:
        return y
    else:
        return -y


def safe_corr(x: np.ndarray, y: np.ndarray, method: str) -> float:
    """
    Robust correlation calculation.
    Return NaN when correlation is undefined.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 2:
        return float("nan")

    if np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return float("nan")

    try:
        if method == "spearman":
            r, _ = spearmanr(x, y)
        elif method == "kendall":
            r, _ = kendalltau(x, y)
        elif method == "pearson":
            r, _ = pearsonr(x, y)
        else:
            raise ValueError(f"Unknown correlation method: {method}")
        return float(r)
    except Exception:
        return float("nan")


def top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """
    Return indices of top-k items by descending score.
    """
    scores = np.asarray(scores, dtype=float)
    k = max(1, min(k, len(scores)))
    return np.argsort(-scores, kind="mergesort")[:k]


def top_overlap(y_true_score: np.ndarray, y_pred_score: np.ndarray, top_frac: float) -> float:
    """
    Top-region overlap:
      |Top true ∩ Top pred| / |Top true|
    """
    n = len(y_true_score)
    if n == 0:
        return float("nan")

    k = max(1, int(math.ceil(n * top_frac)))

    true_top = set(top_k_indices(y_true_score, k))
    pred_top = set(top_k_indices(y_pred_score, k))

    return float(len(true_top.intersection(pred_top)) / k)


def ndcg_continuous(y_true_score: np.ndarray, y_pred_score: np.ndarray, top_frac: float) -> float:
    """
    NDCG over configurations.

    Relevance is derived from normalized true score.
    Larger true score = better.
    Predicted score determines the ranking.
    """
    y_true_score = np.asarray(y_true_score, dtype=float)
    y_pred_score = np.asarray(y_pred_score, dtype=float)

    n = len(y_true_score)
    if n == 0:
        return float("nan")

    k = max(1, int(math.ceil(n * top_frac)))

    # Normalize true score into [0, 1] as continuous relevance
    ymin = np.nanmin(y_true_score)
    ymax = np.nanmax(y_true_score)
    denom = ymax - ymin

    if denom <= 1e-12:
        return float("nan")

    rel = (y_true_score - ymin) / denom

    pred_order = np.argsort(-y_pred_score, kind="mergesort")[:k]
    ideal_order = np.argsort(-y_true_score, kind="mergesort")[:k]

    discounts = 1.0 / np.log2(np.arange(2, k + 2))

    dcg = np.sum((2.0 ** rel[pred_order] - 1.0) * discounts)
    idcg = np.sum((2.0 ** rel[ideal_order] - 1.0) * discounts)

    if idcg <= 1e-12:
        return float("nan")

    return float(dcg / idcg)


def compute_reconstruction_metrics(
    y_true_raw: np.ndarray,
    y_pred_raw: np.ndarray,
    system: str,
    top_fracs: List[float],
) -> Dict[str, float]:
    """
    Compute sparse-to-full reconstruction metrics.

    Raw performance is used for error metrics.
    Direction-normalized score is used for ranking and top-region metrics.
    """
    y_true_raw = np.asarray(y_true_raw, dtype=float)
    y_pred_raw = np.asarray(y_pred_raw, dtype=float)

    mask = np.isfinite(y_true_raw) & np.isfinite(y_pred_raw)
    y_true_raw = y_true_raw[mask]
    y_pred_raw = y_pred_raw[mask]

    if len(y_true_raw) == 0:
        return {}

    err = y_pred_raw - y_true_raw

    # original error
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))

    # normalizational error
    iqr = float(np.percentile(y_true_raw, 75) - np.percentile(y_true_raw, 25))
    std = float(np.std(y_true_raw))
    value_range = float(np.max(y_true_raw) - np.min(y_true_raw))

    denom_iqr = iqr if iqr > 1e-12 else value_range
    denom_iqr = denom_iqr if denom_iqr > 1e-12 else 1.0

    nmae_iqr = float(mae / denom_iqr)
    nrmse_iqr = float(rmse / denom_iqr)

    y_true_score = objective_to_score(y_true_raw, system)
    y_pred_score = objective_to_score(y_pred_raw, system)

    abs_err = np.abs(y_pred_raw - y_true_raw)
    eps = 1e-12
    denom = np.maximum(np.abs(y_true_raw), eps)
    mre_each = abs_err / denom
    mre = float(np.mean(mre_each) * 100.0)

    metrics = {
        "n_samples": int(len(y_true_raw)),
        "mre": mre,
        "mae": mae,
        "rmse": rmse,
        "nmae_iqr": nmae_iqr,
        "nrmse_iqr": nrmse_iqr,
        "spearman_perf": safe_corr(y_true_score, y_pred_score, "spearman"),
        "kendall_perf": safe_corr(y_true_score, y_pred_score, "kendall"),
        "pearson_perf": safe_corr(y_true_score, y_pred_score, "pearson"),
    }

    for frac in top_fracs:
        tag = int(frac * 100)
        metrics[f"top{tag}_overlap"] = top_overlap(y_true_score, y_pred_score, frac)
        metrics[f"ndcg_top{tag}"] = ndcg_continuous(y_true_score, y_pred_score, frac)

    return metrics


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

                metric_dict = compute_reconstruction_metrics(
                    y_true_raw=y_true,
                    y_pred_raw=y_pred,
                    system=system,
                    top_fracs=TOP_FRACS,
                )

                if not metric_dict:
                    continue

                row = {
                    "system": system,
                    "scale": scale,
                    "train_frac": train_frac,
                    "repeat": repeat,
                    "fallback": fallback,
                }
                row.update(metric_dict)
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


def main():
    random.seed(0)
    np.random.seed(0)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_rows = []

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

            out_path = OUTPUT_DIR / f"{system}_{scale}_reconstruction_raw.csv"
            result_df.to_csv(out_path, index=False)
            print(f"[SAVE] {out_path}")

            all_rows.append(result_df)

    if all_rows:
        all_df = pd.concat(all_rows, ignore_index=True)

        raw_path = OUTPUT_DIR / "fallback_reconstruction_raw_all.csv"
        all_df.to_csv(raw_path, index=False)
        print(f"[SAVE] {raw_path}")

        summary_df = summarize_results(all_df)

        summary_path = OUTPUT_DIR / "fallback_reconstruction_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"[SAVE] {summary_path}")

    else:
        print("[WARN] No results generated.")


if __name__ == "__main__":
    main()