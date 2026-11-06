#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    from scipy.spatial import cKDTree
except ModuleNotFoundError:
    cKDTree = None


def _nan_metrics() -> Dict[str, float]:
    return {
        "radius": float("nan"),
        "avg_neighbor_count": float("nan"),
        "isolated_ratio": float("nan"),
        "local_optima_ratio": float("nan"),
        "local_optima_count": float("nan"),
        "auto_correlation": float("nan"),
    }


def _as_dataframe(configs) -> pd.DataFrame:
    if isinstance(configs, pd.DataFrame):
        return configs.copy().reset_index(drop=True)
    return pd.DataFrame(configs).reset_index(drop=True)


def encode_configs_for_distance(configs) -> np.ndarray:
    """
    Encode mixed configuration values for mixed Manhattan distance.

    Numeric columns are min-max normalized. Binary {0, 1} and non-numeric
    columns are encoded as categorical indicators. With Manhattan distance,
    this matches ConfigManager.mixed_distance without the final division by
    dimension: categorical mismatch contributes 1, numeric dimensions
    contribute normalized absolute difference.
    """
    df = _as_dataframe(configs)
    if df.empty:
        return np.empty((0, 0), dtype=float)

    encoded_cols = []

    for col in df.columns:
        series = df[col]
        numeric = pd.to_numeric(series, errors="coerce")
        non_null = series.notna()
        is_numeric = bool(non_null.sum() == numeric.notna().sum())

        if is_numeric:
            values = numeric.astype(float)
            fill = float(values.median()) if values.notna().any() else 0.0
            arr = values.fillna(fill).to_numpy(dtype=float)

            unique_vals = set(np.unique(arr).tolist())
            if unique_vals <= {0.0, 1.0} and len(unique_vals) <= 2:
                encoded_cols.append(arr.astype(float))
                continue
        else:
            values = series.astype("object").where(series.notna(), "<MISSING>")
            base = values.astype(str).to_numpy()
            categories = sorted(pd.unique(base).tolist())
            if len(categories) <= 1:
                encoded_cols.append(np.zeros((len(df), 0), dtype=float))
                continue

            cat_to_idx = {cat: idx for idx, cat in enumerate(categories)}
            one_hot = np.zeros((len(df), len(categories)), dtype=float)
            for row_idx, value in enumerate(base):
                one_hot[row_idx, cat_to_idx[value]] = 1.0

            # Scale by 0.5 so L1 distance is 1 for any categorical mismatch.
            encoded_cols.append(one_hot * 0.5)
            continue

        min_val = float(np.min(arr)) if len(arr) else 0.0
        max_val = float(np.max(arr)) if len(arr) else 0.0
        denom = max_val - min_val
        if denom > 1e-12:
            arr = (arr - min_val) / denom
        else:
            arr = np.zeros_like(arr, dtype=float)

        encoded_cols.append(arr.reshape(-1, 1))

    non_empty_cols = [
        np.asarray(col, dtype=float).reshape(len(df), -1)
        for col in encoded_cols
        if np.asarray(col).size > 0
    ]
    if not non_empty_cols:
        return np.empty((len(df), 0), dtype=float)

    return np.hstack(non_empty_cols).astype(float)


def _build_tree(encoded_configs: np.ndarray):
    if encoded_configs.shape[0] < 2 or encoded_configs.shape[1] == 0:
        return None
    if cKDTree is None:
        return None
    return cKDTree(encoded_configs)


def _chunk_size(n: int, dim: int, target_elements: int = 20_000_000) -> int:
    denom = max(1, n * max(1, dim))
    return max(1, min(n, target_elements // denom))


def _distance_chunk(chunk: np.ndarray, encoded_configs: np.ndarray) -> np.ndarray:
    return np.abs(chunk[:, None, :] - encoded_configs[None, :, :]).sum(axis=2)


def _average_neighbor_count_bruteforce(encoded_configs: np.ndarray, radius: float) -> float:
    n, dim = encoded_configs.shape
    if n < 2 or dim == 0:
        return float("nan")

    chunk_size = _chunk_size(n, dim)
    counts = []
    for start in range(0, n, chunk_size):
        chunk = encoded_configs[start:start + chunk_size]
        distances = _distance_chunk(chunk, encoded_configs)
        counts.extend((np.sum(distances <= radius, axis=1) - 1).tolist())

    counts = np.asarray(counts, dtype=float)
    return float(np.mean(np.maximum(counts, 0.0))) if len(counts) else float("nan")


def _average_neighbor_count(tree, encoded_configs: np.ndarray, radius: float) -> float:
    if tree is None:
        return _average_neighbor_count_bruteforce(encoded_configs, radius)

    try:
        counts = tree.query_ball_point(
            encoded_configs,
            r=radius,
            p=1,
            return_length=True,
        )
    except TypeError:
        neighbors = tree.query_ball_point(encoded_configs, r=radius, p=1)
        counts = np.asarray([len(row) for row in neighbors], dtype=float)
    counts = np.asarray(counts, dtype=float) - 1.0
    return float(np.mean(np.maximum(counts, 0.0))) if len(counts) else float("nan")


def find_adaptive_radius(
    encoded_configs: np.ndarray,
    target_neighbors: Optional[int] = None,
    tolerance: float = 0.1,
) -> float:
    """
    Choose a Manhattan radius whose average neighbor count is close to target.
    """
    encoded_configs = np.asarray(encoded_configs, dtype=float)
    n, dim = encoded_configs.shape if encoded_configs.ndim == 2 else (0, 0)
    if n < 2 or dim == 0:
        return float("nan")

    if target_neighbors is None:
        target_neighbors = dim
    target_neighbors = max(1, min(int(target_neighbors), n - 1))
    tolerance_count = max(1.0, float(target_neighbors) * tolerance)

    tree = _build_tree(encoded_configs)
    if tree is None:
        return float("nan")

    low = 0.0
    high = float(dim)
    best_radius = high
    best_error = float("inf")

    for _ in range(40):
        mid = (low + high) / 2.0
        avg_count = _average_neighbor_count(tree, encoded_configs, mid)
        error = abs(avg_count - target_neighbors)

        if error < best_error:
            best_error = error
            best_radius = mid

        if error <= tolerance_count:
            best_radius = mid
            break

        if avg_count < target_neighbors:
            low = mid
        else:
            high = mid

    return float(best_radius)


def build_neighbor_graph(encoded_configs: np.ndarray, radius: float) -> List[np.ndarray]:
    encoded_configs = np.asarray(encoded_configs, dtype=float)
    n = encoded_configs.shape[0] if encoded_configs.ndim == 2 else 0
    if n == 0 or not np.isfinite(radius):
        return [np.asarray([], dtype=int) for _ in range(n)]

    tree = _build_tree(encoded_configs)
    if tree is None:
        graph = []
        dim = encoded_configs.shape[1]
        chunk_size = _chunk_size(n, dim)
        for start in range(0, n, chunk_size):
            chunk = encoded_configs[start:start + chunk_size]
            distances = _distance_chunk(chunk, encoded_configs)
            for local_idx in range(distances.shape[0]):
                idx = start + local_idx
                neighbors = np.where(distances[local_idx] <= radius)[0]
                graph.append(np.asarray([j for j in neighbors if j != idx], dtype=int))
        return graph

    raw_neighbors = tree.query_ball_point(encoded_configs, r=radius, p=1)

    graph = []
    for idx, neighbors in enumerate(raw_neighbors):
        arr = np.asarray([j for j in neighbors if j != idx], dtype=int)
        graph.append(arr)
    return graph


def _to_score(perfs, maximize: bool) -> np.ndarray:
    score = np.asarray(perfs, dtype=float)
    return score if maximize else -score


def compute_neighbor_stats(neighbor_graph: List[np.ndarray], radius: float) -> Dict[str, float]:
    neighbor_counts = np.asarray([len(row) for row in neighbor_graph], dtype=float)
    return {
        "radius": float(radius),
        "avg_neighbor_count": (
            float(np.mean(neighbor_counts)) if len(neighbor_counts) else float("nan")
        ),
        "isolated_ratio": (
            float(np.mean(neighbor_counts == 0)) if len(neighbor_counts) else float("nan")
        ),
    }


def compute_local_optima_metrics(
    scores: np.ndarray,
    neighbor_graph: List[np.ndarray],
) -> Dict[str, float]:
    scores = np.asarray(scores, dtype=float)
    finite = np.isfinite(scores)
    local_optima = 0
    denominator = 0

    for i, neighbors in enumerate(neighbor_graph):
        if not finite[i]:
            continue

        valid_neighbors = np.asarray([j for j in neighbors if finite[j]], dtype=int)
        if len(valid_neighbors) == 0:
            continue

        denominator += 1
        if np.all(scores[i] >= scores[valid_neighbors]):
            local_optima += 1

    return {
        "local_optima_ratio": (
            float(local_optima / denominator) if denominator > 0 else float("nan")
        ),
        "local_optima_count": float(local_optima),
    }


def build_random_walk_path(
    neighbor_graph: List[np.ndarray],
    seed: int = 0,
) -> np.ndarray:
    """
    Walk until all points are visited or the path reaches 2 * n.

    Immediate backtracking is blocked, so A->B->A is forbidden. Older nodes
    can still be revisited, so A->B->C->A is allowed.
    """
    n = len(neighbor_graph)
    if n == 0:
        return np.asarray([], dtype=int)

    rng = np.random.RandomState(seed)
    max_len = max(1, 2 * n)
    unvisited = set(range(n))
    path = []
    current = None
    previous = None

    while len(path) < max_len and unvisited:
        if current is None:
            current = int(rng.choice(list(unvisited)))
            previous = None
            path.append(current)
            unvisited.discard(current)
            continue

        candidates = [int(j) for j in neighbor_graph[current] if int(j) != previous]
        if not candidates:
            current = None
            previous = None
            continue

        next_idx = int(rng.choice(candidates))
        path.append(next_idx)
        unvisited.discard(next_idx)
        previous, current = current, next_idx

    return np.asarray(path, dtype=int)


def compute_auto_correlation(scores: np.ndarray, walk_path: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=float)
    walk_path = np.asarray(walk_path, dtype=int)
    if len(walk_path) < 2:
        return float("nan")

    walk_scores = scores[walk_path]
    mask = np.isfinite(walk_scores)
    walk_scores = walk_scores[mask]
    if len(walk_scores) < 2:
        return float("nan")

    mean_score = float(np.mean(walk_scores))
    centered = walk_scores - mean_score
    denominator = float(np.sum(centered ** 2))
    if denominator <= 1e-12:
        return float("nan")

    numerator = float(np.sum(centered[:-1] * centered[1:]))
    return numerator / denominator


def compute_landscape_metrics_from_graph(
    scores: np.ndarray,
    neighbor_graph: List[np.ndarray],
    radius: float,
    walk_path: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    if n == 0:
        return _nan_metrics()

    if walk_path is None:
        walk_path = build_random_walk_path(neighbor_graph)

    metrics = {}
    metrics.update(compute_neighbor_stats(neighbor_graph, radius))
    metrics.update(compute_local_optima_metrics(scores, neighbor_graph))
    metrics["auto_correlation"] = compute_auto_correlation(scores, walk_path)
    return metrics


def compute_landscape_metrics(configs, perfs, maximize: bool = True) -> Dict[str, float]:
    config_df = _as_dataframe(configs)
    encoded = encode_configs_for_distance(config_df)
    radius = find_adaptive_radius(encoded, target_neighbors=config_df.shape[1])
    graph = build_neighbor_graph(encoded, radius)
    scores = _to_score(perfs, maximize)
    return compute_landscape_metrics_from_graph(scores, graph, radius)


def _abs_error(true_value: float, pred_value: float) -> float:
    if not np.isfinite(true_value) or not np.isfinite(pred_value):
        return float("nan")
    return float(abs(pred_value - true_value))


def _rel_error(true_value: float, pred_value: float) -> float:
    if not np.isfinite(true_value) or not np.isfinite(pred_value):
        return float("nan")
    denom = abs(true_value)
    if denom <= 1e-12:
        return float("nan")
    return float(abs(pred_value - true_value) / denom)


def compute_landscape_pair_metrics(
    configs,
    true_perfs,
    pred_perfs,
    maximize: bool = True,
) -> Dict[str, float]:
    """
    Compute true/pred landscape metrics on the same config-derived graph.
    """
    true_perfs = np.asarray(true_perfs, dtype=float)
    pred_perfs = np.asarray(pred_perfs, dtype=float)
    mask = np.isfinite(true_perfs) & np.isfinite(pred_perfs)

    if mask.sum() < 2:
        true_metrics = _nan_metrics()
        pred_metrics = _nan_metrics()
    else:
        config_df = _as_dataframe(configs).iloc[mask].reset_index(drop=True)
        encoded = encode_configs_for_distance(config_df)
        radius = find_adaptive_radius(encoded, target_neighbors=config_df.shape[1])
        graph = build_neighbor_graph(encoded, radius)

        true_scores = _to_score(true_perfs[mask], maximize)
        pred_scores = _to_score(pred_perfs[mask], maximize)
        walk_path = build_random_walk_path(graph)

        true_metrics = compute_landscape_metrics_from_graph(true_scores, graph, radius, walk_path)
        pred_metrics = compute_landscape_metrics_from_graph(pred_scores, graph, radius, walk_path)

    output = {}
    for key, value in true_metrics.items():
        output[f"true_{key}"] = value
    for key, value in pred_metrics.items():
        output[f"pred_{key}"] = value

    output["local_optima_ratio_abs_error"] = _abs_error(
        true_metrics["local_optima_ratio"],
        pred_metrics["local_optima_ratio"],
    )
    output["local_optima_count_abs_error"] = _abs_error(
        true_metrics["local_optima_count"],
        pred_metrics["local_optima_count"],
    )
    output["auto_correlation_abs_error"] = _abs_error(
        true_metrics["auto_correlation"],
        pred_metrics["auto_correlation"],
    )

    return output
