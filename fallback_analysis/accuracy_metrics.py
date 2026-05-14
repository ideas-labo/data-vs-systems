#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import math
from typing import Dict, List

import numpy as np
from scipy.stats import kendalltau, pearsonr, spearmanr


MAXIMIZE_SYSTEMS = {"mysql", "postgresql", "httpd", "tomcat"}


def objective_to_score(y: np.ndarray, system: str) -> np.ndarray:
    """Convert raw performance into a score where larger is better."""
    y = np.asarray(y, dtype=float)
    return y if system in MAXIMIZE_SYSTEMS else -y


def safe_corr(x: np.ndarray, y: np.ndarray, method: str) -> float:
    """Return NaN when correlation is undefined."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 2 or np.nanstd(x) == 0 or np.nanstd(y) == 0:
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
    scores = np.asarray(scores, dtype=float)
    k = max(1, min(k, len(scores)))
    return np.argsort(-scores, kind="mergesort")[:k]


def top_overlap(y_true_score: np.ndarray, y_pred_score: np.ndarray, top_frac: float) -> float:
    n = len(y_true_score)
    if n == 0:
        return float("nan")

    k = max(1, int(math.ceil(n * top_frac)))
    true_top = set(top_k_indices(y_true_score, k))
    pred_top = set(top_k_indices(y_pred_score, k))
    return float(len(true_top.intersection(pred_top)) / k)


def ndcg_continuous(y_true_score: np.ndarray, y_pred_score: np.ndarray, top_frac: float) -> float:
    y_true_score = np.asarray(y_true_score, dtype=float)
    y_pred_score = np.asarray(y_pred_score, dtype=float)

    n = len(y_true_score)
    if n == 0:
        return float("nan")

    k = max(1, int(math.ceil(n * top_frac)))
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
    """Compute pointwise, ranking, and top-region reconstruction metrics."""
    y_true_raw = np.asarray(y_true_raw, dtype=float)
    y_pred_raw = np.asarray(y_pred_raw, dtype=float)

    mask = np.isfinite(y_true_raw) & np.isfinite(y_pred_raw)
    y_true_raw = y_true_raw[mask]
    y_pred_raw = y_pred_raw[mask]

    if len(y_true_raw) == 0:
        return {}

    err = y_pred_raw - y_true_raw
    abs_err = np.abs(err)

    mae = float(np.mean(abs_err))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    bias = float(np.mean(err))
    error_std = float(np.std(err))

    iqr = float(np.percentile(y_true_raw, 75) - np.percentile(y_true_raw, 25))
    value_range = float(np.max(y_true_raw) - np.min(y_true_raw))
    denom_iqr = iqr if iqr > 1e-12 else value_range
    denom_iqr = denom_iqr if denom_iqr > 1e-12 else 1.0

    denom = np.maximum(np.abs(y_true_raw), 1e-12)
    mre_each = abs_err / denom

    y_true_score = objective_to_score(y_true_raw, system)
    y_pred_score = objective_to_score(y_pred_raw, system)

    metrics = {
        "n_samples": int(len(y_true_raw)),
        "mre": float(np.mean(mre_each) * 100.0),
        "median_mre": float(np.median(mre_each) * 100.0),
        "p90_mre": float(np.percentile(mre_each, 90) * 100.0),
        "mae": mae,
        "rmse": rmse,
        "bias": bias,
        "abs_bias": float(abs(bias)),
        "error_std": error_std,
        "nmae_iqr": float(mae / denom_iqr),
        "nrmse_iqr": float(rmse / denom_iqr),
        "spearman_perf": safe_corr(y_true_score, y_pred_score, "spearman"),
        "kendall_perf": safe_corr(y_true_score, y_pred_score, "kendall"),
        "pearson_perf": safe_corr(y_true_score, y_pred_score, "pearson"),
    }

    for frac in top_fracs:
        tag = int(frac * 100)
        metrics[f"top{tag}_overlap"] = top_overlap(y_true_score, y_pred_score, frac)
        metrics[f"ndcg_top{tag}"] = ndcg_continuous(y_true_score, y_pred_score, frac)

    return metrics
