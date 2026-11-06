#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

MPL_CACHE_DIR = Path(tempfile.gettempdir()) / "dataeval_matplotlib_cache"
XDG_CACHE_DIR = Path(tempfile.gettempdir()) / "dataeval_cache"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
XDG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(MPL_CACHE_DIR),
)
os.environ.setdefault(
    "XDG_CACHE_HOME",
    str(XDG_CACHE_DIR),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import KNeighborsRegressor

from landscape_metrics import (
    build_neighbor_graph,
    encode_configs_for_distance,
    find_adaptive_radius,
)


SCRIPT_DIR = Path(__file__).resolve().parent
RQ3_DIR = SCRIPT_DIR.parent
STATISTICAL_TOOLS_DIR = RQ3_DIR.parent
PROJECT_ROOT = STATISTICAL_TOOLS_DIR.parent
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "datasets"
DEFAULT_OUTPUT_DIR = RQ3_DIR / "results" / "local_optima_mds"

SYSTEM_ORDER = ["mysql", "postgresql", "httpd", "tomcat", "gcc", "clang", "x264", "x265"]
MAXIMIZE_SYSTEMS = {"mysql", "postgresql", "httpd", "tomcat"}


def infer_maximize(system: str) -> bool:
    return system in MAXIMIZE_SYSTEMS


def objective_to_score(perfs: np.ndarray, maximize: bool) -> np.ndarray:
    perfs = np.asarray(perfs, dtype=float)
    return perfs if maximize else -perfs


def system_sort_key(system: str) -> tuple[int, str]:
    return (SYSTEM_ORDER.index(system), system) if system in SYSTEM_ORDER else (999, system)


def resolve_systems(dataset_root: Path, requested: list[str] | None) -> list[str]:
    if requested:
        if len(requested) == 1 and requested[0].lower() == "all":
            requested = None
        else:
            return sorted(dict.fromkeys(requested), key=system_sort_key)

    systems = [
        path.name
        for path in dataset_root.iterdir()
        if path.is_dir()
    ]
    known = [system for system in SYSTEM_ORDER if system in systems]
    remaining = sorted(system for system in systems if system not in known)
    return known + remaining


def load_system_dataset(
    dataset_root: Path,
    system: str,
    scale: str,
    performance_col: str | None,
) -> tuple[pd.DataFrame, np.ndarray, str, Path]:
    csv_path = dataset_root / system / f"{system}_{scale}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"Dataset is empty: {csv_path}")

    perf_col = performance_col or str(df.columns[-1])
    if perf_col not in df.columns:
        raise ValueError(f"Performance column '{perf_col}' not found in {csv_path}")

    config_df = df.drop(columns=[perf_col]).reset_index(drop=True)
    perfs = pd.to_numeric(df[perf_col], errors="coerce").to_numpy(dtype=float)
    return config_df, perfs, perf_col, csv_path


def compute_local_optima_flags(
    scores: np.ndarray,
    neighbor_graph: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scores = np.asarray(scores, dtype=float)
    finite = np.isfinite(scores)
    is_local_optimum = np.zeros(len(scores), dtype=bool)
    eligible = np.zeros(len(scores), dtype=bool)
    neighbor_counts = np.asarray([len(neighbors) for neighbors in neighbor_graph], dtype=int)

    for idx, neighbors in enumerate(neighbor_graph):
        if not finite[idx]:
            continue

        valid_neighbors = np.asarray([j for j in neighbors if finite[j]], dtype=int)
        if len(valid_neighbors) == 0:
            continue

        eligible[idx] = True
        is_local_optimum[idx] = bool(np.all(scores[idx] >= scores[valid_neighbors]))

    return is_local_optimum, eligible, neighbor_counts


def make_mds(random_state: int, n_init: int, max_iter: int, n_jobs: int | None) -> MDS:
    kwargs: dict[str, Any] = {
        "n_components": 2,
        "dissimilarity": "precomputed",
        "random_state": random_state,
        "n_init": n_init,
        "max_iter": max_iter,
        "n_jobs": n_jobs,
    }
    try:
        return MDS(**kwargs, normalized_stress="auto")
    except TypeError:
        return MDS(**kwargs)


def run_exact_mds(
    encoded_configs: np.ndarray,
    random_state: int,
    n_init: int,
    max_iter: int,
    n_jobs: int | None,
) -> tuple[np.ndarray, float]:
    distances = pairwise_distances(encoded_configs, metric="manhattan", n_jobs=n_jobs)
    mds = make_mds(
        random_state=random_state,
        n_init=n_init,
        max_iter=max_iter,
        n_jobs=n_jobs,
    )
    coords = mds.fit_transform(distances)
    return np.asarray(coords, dtype=float), float(getattr(mds, "stress_", np.nan))


def select_landmark_indices(n_points: int, n_landmarks: int, random_state: int) -> np.ndarray:
    n_landmarks = max(1, min(int(n_landmarks), int(n_points)))
    if n_landmarks == n_points:
        return np.arange(n_points, dtype=int)

    rng = np.random.RandomState(random_state)
    return np.sort(rng.choice(n_points, size=n_landmarks, replace=False)).astype(int)


def run_landmark_mds(
    encoded_configs: np.ndarray,
    n_landmarks: int,
    landmark_neighbors: int,
    random_state: int,
    n_init: int,
    max_iter: int,
    n_jobs: int | None,
) -> tuple[np.ndarray, float, np.ndarray]:
    n_points = encoded_configs.shape[0]
    landmark_idx = select_landmark_indices(n_points, n_landmarks, random_state)
    landmark_configs = encoded_configs[landmark_idx]

    landmark_distances = pairwise_distances(landmark_configs, metric="manhattan", n_jobs=n_jobs)
    mds = make_mds(
        random_state=random_state,
        n_init=n_init,
        max_iter=max_iter,
        n_jobs=n_jobs,
    )
    landmark_coords = np.asarray(mds.fit_transform(landmark_distances), dtype=float)

    coords = np.empty((n_points, 2), dtype=float)
    coords[landmark_idx] = landmark_coords

    if len(landmark_idx) < n_points:
        k = max(1, min(int(landmark_neighbors), len(landmark_idx)))
        regressor = KNeighborsRegressor(
            n_neighbors=k,
            weights="distance",
            metric="manhattan",
        )
        regressor.fit(landmark_configs, landmark_coords)
        coords = np.asarray(regressor.predict(encoded_configs), dtype=float)
        coords[landmark_idx] = landmark_coords

    return coords, float(getattr(mds, "stress_", np.nan)), landmark_idx


def compute_mds_coordinates(
    encoded_configs: np.ndarray,
    method: str,
    exact_limit: int,
    landmarks: int,
    landmark_neighbors: int,
    random_state: int,
    n_init: int,
    max_iter: int,
    n_jobs: int | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    n_points = int(encoded_configs.shape[0])
    if n_points == 0:
        return np.empty((0, 2), dtype=float), {"mds_method": method, "stress": np.nan}
    if encoded_configs.shape[1] == 0:
        return np.zeros((n_points, 2), dtype=float), {
            "mds_method": "constant",
            "stress": np.nan,
            "landmark_count": 0,
        }

    use_exact = method == "exact" or (method == "auto" and n_points <= exact_limit)
    if use_exact:
        coords, stress = run_exact_mds(
            encoded_configs=encoded_configs,
            random_state=random_state,
            n_init=n_init,
            max_iter=max_iter,
            n_jobs=n_jobs,
        )
        return coords, {
            "mds_method": "exact",
            "stress": stress,
            "landmark_count": n_points,
        }

    coords, stress, landmark_idx = run_landmark_mds(
        encoded_configs=encoded_configs,
        n_landmarks=landmarks,
        landmark_neighbors=landmark_neighbors,
        random_state=random_state,
        n_init=n_init,
        max_iter=max_iter,
        n_jobs=n_jobs,
    )
    return coords, {
        "mds_method": "landmark",
        "stress": stress,
        "landmark_count": int(len(landmark_idx)),
        "landmark_neighbors": int(max(1, min(landmark_neighbors, len(landmark_idx)))),
    }


def build_point_table(
    system: str,
    scale: str,
    config_df: pd.DataFrame,
    performance_col: str,
    perfs: np.ndarray,
    scores: np.ndarray,
    coords: np.ndarray,
    is_local_optimum: np.ndarray,
    eligible: np.ndarray,
    neighbor_counts: np.ndarray,
    include_config_columns: bool,
) -> pd.DataFrame:
    point_df = pd.DataFrame(
        {
            "system": system,
            "scale": scale,
            "row_index": np.arange(len(config_df), dtype=int),
            "mds_x": coords[:, 0],
            "mds_y": coords[:, 1],
            "performance_col": performance_col,
            "performance": perfs,
            "score": scores,
            "is_local_optimum": is_local_optimum,
            "eligible_for_local_optimum": eligible,
            "neighbor_count": neighbor_counts,
        }
    )

    if include_config_columns:
        point_df = pd.concat([point_df, config_df.reset_index(drop=True)], axis=1)

    return point_df


def plot_local_optima_2d(
    point_df: pd.DataFrame,
    system: str,
    scale: str,
    output_path: Path,
    dpi: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    local_df = point_df[point_df["is_local_optimum"].astype(bool)]

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(
        point_df["mds_x"],
        point_df["mds_y"],
        c="black",
        marker=".",
        s=8,
        linewidths=0,
        alpha=0.65,
        label="Configuration",
    )

    if not local_df.empty:
        ax.scatter(
            local_df["mds_x"],
            local_df["mds_y"],
            c="red",
            marker="*",
            s=70,
            linewidths=0.4,
            edgecolors="red",
            alpha=0.95,
            label="Local optimum",
        )

    ax.set_title(f"{system} {scale}: local optima in MDS configuration space")
    ax.set_xlabel("MDS dimension 1")
    ax.set_ylabel("MDS dimension 2")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.25)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_outputs(
    point_df: pd.DataFrame,
    summary_row: dict[str, Any],
    output_dir: Path,
    system: str,
    scale: str,
    make_plot: bool,
    dpi: int,
) -> dict[str, Path]:
    points_dir = output_dir / "points"
    optima_dir = output_dir / "local_optima"
    summary_dir = output_dir / "summary"
    plot_dir = output_dir / "plots"

    points_dir.mkdir(parents=True, exist_ok=True)
    optima_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    points_path = points_dir / f"{system}_{scale}_mds_3d_points.csv"
    optima_path = optima_dir / f"{system}_{scale}_local_optima.csv"
    point_df.to_csv(points_path, index=False)
    point_df[point_df["is_local_optimum"].astype(bool)].to_csv(optima_path, index=False)

    saved = {
        "points": points_path,
        "local_optima": optima_path,
    }

    if make_plot:
        plot_path = plot_dir / f"{system}_{scale}_local_optima_2d.png"
        plot_local_optima_2d(
            point_df=point_df,
            system=system,
            scale=scale,
            output_path=plot_path,
            dpi=dpi,
        )
        saved["plot"] = plot_path

    system_summary_path = summary_dir / f"{system}_{scale}_summary.csv"
    pd.DataFrame([summary_row]).to_csv(system_summary_path, index=False)
    saved["summary"] = system_summary_path

    return saved


def process_system(system: str, args: argparse.Namespace) -> dict[str, Any]:
    config_df, perfs, performance_col, csv_path = load_system_dataset(
        dataset_root=args.dataset_root,
        system=system,
        scale=args.scale,
        performance_col=args.performance_col,
    )

    maximize = infer_maximize(system)
    encoded = encode_configs_for_distance(config_df)
    radius = find_adaptive_radius(
        encoded,
        target_neighbors=args.target_neighbors or config_df.shape[1],
        tolerance=args.radius_tolerance,
    )
    neighbor_graph = build_neighbor_graph(encoded, radius)
    scores = objective_to_score(perfs, maximize=maximize)
    is_local_optimum, eligible, neighbor_counts = compute_local_optima_flags(scores, neighbor_graph)

    coords, mds_info = compute_mds_coordinates(
        encoded_configs=encoded,
        method=args.mds_method,
        exact_limit=args.exact_limit,
        landmarks=args.landmarks,
        landmark_neighbors=args.landmark_neighbors,
        random_state=args.random_state,
        n_init=args.n_init,
        max_iter=args.max_iter,
        n_jobs=args.n_jobs,
    )

    point_df = build_point_table(
        system=system,
        scale=args.scale,
        config_df=config_df,
        performance_col=performance_col,
        perfs=perfs,
        scores=scores,
        coords=coords,
        is_local_optimum=is_local_optimum,
        eligible=eligible,
        neighbor_counts=neighbor_counts,
        include_config_columns=not args.no_config_columns,
    )

    eligible_count = int(eligible.sum())
    local_optima_count = int(is_local_optimum.sum())
    summary_row: dict[str, Any] = {
        "system": system,
        "scale": args.scale,
        "source_path": str(csv_path),
        "n_points": int(len(point_df)),
        "n_config_columns": int(config_df.shape[1]),
        "maximize": bool(maximize),
        "performance_col": performance_col,
        "radius": float(radius) if np.isfinite(radius) else np.nan,
        "avg_neighbor_count": float(np.mean(neighbor_counts)) if len(neighbor_counts) else np.nan,
        "isolated_ratio": float(np.mean(neighbor_counts == 0)) if len(neighbor_counts) else np.nan,
        "eligible_local_optima_points": eligible_count,
        "local_optima_count": local_optima_count,
        "local_optima_ratio": (
            float(local_optima_count / eligible_count) if eligible_count > 0 else np.nan
        ),
    }
    summary_row.update(mds_info)

    saved = save_outputs(
        point_df=point_df,
        summary_row=summary_row,
        output_dir=args.output_dir,
        system=system,
        scale=args.scale,
        make_plot=not args.skip_plots,
        dpi=args.dpi,
    )

    summary_row.update({f"{key}_path": str(path) for key, path in saved.items()})
    return summary_row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate local_optima_mds outputs: point-level local optima, "
            "cached MDS coordinates, summaries, and local-optima plots."
        )
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--scale", default="large")
    parser.add_argument(
        "--systems",
        nargs="+",
        default=None,
        help="Systems to process. Defaults to all dataset directories; use 'all' explicitly for all.",
    )
    parser.add_argument(
        "--performance-col",
        default=None,
        help="Performance column name. Defaults to the last column of each dataset.",
    )
    parser.add_argument(
        "--target-neighbors",
        type=int,
        default=None,
        help="Adaptive-radius target neighbor count. Defaults to the original config column count.",
    )
    parser.add_argument("--radius-tolerance", type=float, default=0.1)
    parser.add_argument(
        "--mds-method",
        choices=["auto", "exact", "landmark"],
        default="auto",
        help="auto uses exact MDS up to --exact-limit rows, then landmark MDS.",
    )
    parser.add_argument("--exact-limit", type=int, default=4000)
    parser.add_argument("--landmarks", type=int, default=1500)
    parser.add_argument("--landmark-neighbors", type=int, default=12)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--n-init", type=int, default=1)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Parallel workers for pairwise distances/MDS. Use 1 by default to avoid joblib process limits.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument(
        "--no-config-columns",
        action="store_true",
        help="Do not append raw configuration columns to the output point CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.dataset_root = args.dataset_root.resolve()
    args.output_dir = args.output_dir.resolve()

    systems = resolve_systems(args.dataset_root, args.systems)
    if not systems:
        raise FileNotFoundError(f"No system directories found under {args.dataset_root}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for system in systems:
        print(f"[PROCESS] system={system}, scale={args.scale}")
        try:
            row = process_system(system, args)
        except FileNotFoundError as exc:
            print(f"[WARN] Skip {system}: {exc}")
            continue
        summary_rows.append(row)
        print(
            f"[DONE] {system}: local_optima={row['local_optima_count']}, "
            f"ratio={row['local_optima_ratio']:.4f}, "
            f"mds={row['mds_method']}"
        )

    if not summary_rows:
        print("[WARN] No systems processed.")
        return

    summary_df = pd.DataFrame(summary_rows)
    summary_path = args.output_dir / "summary" / f"{args.scale}_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_path, index=False)
    print(f"[SAVE] {summary_path}")


if __name__ == "__main__":
    main()
