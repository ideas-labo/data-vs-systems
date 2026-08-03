#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

MPL_CACHE_DIR = Path(tempfile.gettempdir()) / "dataeval_matplotlib_cache"
XDG_CACHE_DIR = Path(tempfile.gettempdir()) / "dataeval_cache"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
XDG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

SCRIPT_DIR = Path(__file__).resolve().parent
DISCUSSION_DIR = SCRIPT_DIR.parent
FALLBACK_ANALYSIS_DIR = DISCUSSION_DIR.parent
RQ3_SCRIPTS_DIR = FALLBACK_ANALYSIS_DIR / "RQ3" / "scripts"
if str(RQ3_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(RQ3_SCRIPTS_DIR))

from rq3_2_compute_accuracy_landscape_metrics import ID_COLS, VALUE_COLS, get_config_columns
from rq3_8_cache_true_landscape_mds import (
    compute_mds_coordinates,
    infer_maximize,
    objective_to_score,
)
from landscape_metrics import encode_configs_for_distance


DEFAULT_PREDICTION_DIR = FALLBACK_ANALYSIS_DIR / "RQ3" / "results" / "predictions"
DEFAULT_OUTPUT_DIR = DISCUSSION_DIR / "results"

DEFAULT_FALLBACKS = ["1NN", "kNN", "kNNw", "loess", "rf", "dal_rf", "rf_uncertainty", "worst"]
FALLBACK_DISPLAY = {
    "1NN": "1NN",
    "kNN": "kNN",
    "kNNw": "wkNN",
    "loess": "LOESS",
    "rf": "RF",
    "dal_rf": "DaL-RF",
    "rf_uncertainty": "U-RF",
    "worst": "Worst",
}


def parse_train_frac(value: str) -> float:
    return float(value)


def output_tag(system: str, scale: str, train_frac: float, repeat: int) -> str:
    frac_tag = str(train_frac).replace(".", "p")
    return f"{system}_{scale}_train{frac_tag}_repeat{repeat}"


def load_prediction_group(
    prediction_dir: Path,
    fallback: str,
    system: str,
    scale: str,
    train_frac: float,
    repeat: int,
) -> pd.DataFrame:
    csv_path = prediction_dir / fallback / f"{system}_{scale}_predictions.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)
    required = set(ID_COLS + VALUE_COLS)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")

    mask = (
        (df["system"].astype(str) == system)
        & (df["scale"].astype(str) == scale)
        & (pd.to_numeric(df["train_frac"], errors="coerce") == train_frac)
        & (pd.to_numeric(df["repeat"], errors="coerce") == repeat)
        & (df["fallback"].astype(str) == fallback)
    )
    group = df.loc[mask].copy().reset_index(drop=True)
    if group.empty:
        raise ValueError(
            f"No rows for fallback={fallback}, system={system}, scale={scale}, "
            f"train_frac={train_frac}, repeat={repeat}"
        )
    return group


def deduplicate_prediction_group(group: pd.DataFrame, config_cols: list[str]) -> pd.DataFrame:
    cols = config_cols + ["true_perf", "pred_perf"]
    out = group[cols].copy()
    if out.duplicated(config_cols).any():
        out = (
            out.groupby(config_cols, as_index=False, dropna=False)
            .agg(true_perf=("true_perf", "mean"), pred_perf=("pred_perf", "mean"))
        )
    return out.reset_index(drop=True)


def build_combined_prediction_table(
    prediction_dir: Path,
    fallbacks: list[str],
    system: str,
    scale: str,
    train_frac: float,
    repeat: int,
) -> tuple[pd.DataFrame, list[str]]:
    base_df = None
    config_cols = None

    for fallback in fallbacks:
        group = load_prediction_group(
            prediction_dir=prediction_dir,
            fallback=fallback,
            system=system,
            scale=scale,
            train_frac=train_frac,
            repeat=repeat,
        )
        if config_cols is None:
            config_cols = get_config_columns(group)

        current = deduplicate_prediction_group(group, config_cols)
        current = current.rename(columns={"pred_perf": f"pred_perf_{fallback}"})

        if base_df is None:
            base_df = current
        else:
            before = len(base_df)
            base_df = base_df.merge(
                current[config_cols + [f"pred_perf_{fallback}"]],
                on=config_cols,
                how="inner",
            )
            if len(base_df) != before:
                print(
                    f"[WARN] Common holdout rows changed after merging {fallback}: "
                    f"{before} -> {len(base_df)}"
                )

    if base_df is None or config_cols is None:
        raise ValueError("No fallback prediction tables were loaded.")
    return base_df.reset_index(drop=True), config_cols


def sample_points(df: pd.DataFrame, sample_size: int | None, random_state: int) -> pd.DataFrame:
    if sample_size is None or sample_size <= 0 or len(df) <= sample_size:
        return df.reset_index(drop=True)
    return df.sample(n=sample_size, random_state=random_state).sort_index().reset_index(drop=True)


def build_discussion_points(
    combined_df: pd.DataFrame,
    config_cols: list[str],
    fallbacks: list[str],
    system: str,
    mds_method: str,
    exact_limit: int,
    landmarks: int,
    landmark_neighbors: int,
    random_state: int,
    n_init: int,
    max_iter: int,
    n_jobs: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    maximize = infer_maximize(system)
    encoded = encode_configs_for_distance(combined_df[config_cols])
    coords, mds_info = compute_mds_coordinates(
        encoded_configs=encoded,
        method=mds_method,
        exact_limit=exact_limit,
        landmarks=landmarks,
        landmark_neighbors=landmark_neighbors,
        random_state=random_state,
        n_init=n_init,
        max_iter=max_iter,
        n_jobs=n_jobs,
    )

    point_df = pd.DataFrame(
        {
            "mds_x": coords[:, 0],
            "mds_y": coords[:, 1],
            "true_perf": pd.to_numeric(combined_df["true_perf"], errors="coerce"),
            "true_score": objective_to_score(combined_df["true_perf"], maximize=maximize),
        }
    )
    for fallback in fallbacks:
        pred_col = f"pred_perf_{fallback}"
        point_df[pred_col] = pd.to_numeric(combined_df[pred_col], errors="coerce")
        point_df[f"pred_score_{fallback}"] = objective_to_score(
            combined_df[pred_col],
            maximize=maximize,
        )

    point_df = pd.concat([point_df, combined_df[config_cols].reset_index(drop=True)], axis=1)
    mds_info["maximize"] = bool(maximize)
    return point_df, mds_info


def finite_values(values: list[np.ndarray]) -> np.ndarray:
    finite_chunks = []
    for arr in values:
        arr = np.asarray(arr, dtype=float)
        finite_chunks.append(arr[np.isfinite(arr)])
    if not finite_chunks:
        return np.asarray([], dtype=float)
    return np.concatenate(finite_chunks)


def color_limits(values: list[np.ndarray], robust_quantile: float) -> tuple[float, float]:
    all_values = finite_values(values)
    if len(all_values) == 0:
        return 0.0, 1.0

    robust_quantile = max(0.0, min(float(robust_quantile), 0.49))
    if robust_quantile > 0:
        low = float(np.quantile(all_values, robust_quantile))
        high = float(np.quantile(all_values, 1.0 - robust_quantile))
    else:
        low = float(np.min(all_values))
        high = float(np.max(all_values))

    if not np.isfinite(low) or not np.isfinite(high):
        return 0.0, 1.0
    if abs(high - low) <= 1e-12:
        pad = max(1.0, abs(low) * 0.05)
        return low - pad, high + pad
    return low, high


def safe_file_stem(name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(name))
    return safe.strip("_") or "plot"


def landscape_value_specs(
    system: str,
    fallbacks: list[str],
) -> tuple[list[tuple[str, str, str]], str]:
    if system.lower() == "mysql":
        specs = [("true_perf", "true", "true")]
        specs.extend(
            (f"pred_perf_{fallback}", FALLBACK_DISPLAY.get(fallback, fallback), fallback)
            for fallback in fallbacks
        )
        return specs, "Throughput"

    specs = [("true_score", "true", "true")]
    specs.extend(
        (f"pred_score_{fallback}", FALLBACK_DISPLAY.get(fallback, fallback), fallback)
        for fallback in fallbacks
    )
    return specs, "Performance score"


def colorbar_ticks(system: str, vmin: float, vmax: float) -> list[float] | None:
    if system.lower() == "mysql":
        return [50, 100, 150, 200, 250, 300]
    return None


def plot_landscape_panel(
    ax,
    point_df: pd.DataFrame,
    value_col: str,
    title: str | None,
    vmin: float,
    vmax: float,
    levels: np.ndarray,
    cmap: str,
    scatter_size: float,
):
    x = pd.to_numeric(point_df["mds_x"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(point_df["mds_y"], errors="coerce").to_numpy(dtype=float)
    z = pd.to_numeric(point_df[value_col], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)

    if mask.sum() >= 3:
        z_plot = np.clip(z[mask], vmin, vmax)
        triangulation = mtri.Triangulation(x[mask], y[mask])
        artist = ax.tricontourf(
            triangulation,
            z_plot,
            levels=levels,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            extend="neither",
        )
        ax.tricontour(
            triangulation,
            z_plot,
            levels=levels[:: max(1, len(levels) // 12)],
            colors="white",
            linewidths=0.35,
            alpha=0.35,
        )
        ax.scatter(
            x[mask],
            y[mask],
            c="black",
            marker=".",
            s=scatter_size,
            linewidths=0,
            alpha=0.35,
        )
    else:
        z_plot = np.clip(z[mask], vmin, vmax)
        artist = ax.scatter(
            x[mask],
            y[mask],
            c=z_plot,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            marker=".",
            s=scatter_size,
        )

    if title:
        ax.set_title(title, fontsize=11)
    ax.set_xlabel("MDS #1")
    ax.set_ylabel("MDS #2")
    ax.set_aspect("equal", adjustable="box")
    return artist


def plot_single_landscape_figures(
    point_df: pd.DataFrame,
    fallbacks: list[str],
    output_dir: Path,
    system: str,
    cmap: str,
    robust_quantile: float,
    contour_levels: int,
    scatter_size: float,
    dpi: int,
) -> list[Path]:
    plot_specs, colorbar_label = landscape_value_specs(system, fallbacks)
    value_arrays = [point_df[col].to_numpy(dtype=float) for col, _file_stem, _key in plot_specs]
    vmin, vmax = color_limits(value_arrays, robust_quantile=robust_quantile)
    ticks = colorbar_ticks(system, vmin, vmax)
    if ticks:
        vmin = min(vmin, min(ticks))
        vmax = max(vmax, max(ticks))
    levels = np.linspace(vmin, vmax, max(3, int(contour_levels)))

    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    for value_col, file_stem, _key in plot_specs:
        fig, ax = plt.subplots(figsize=(4.2, 3.6), constrained_layout=True)
        artist = plot_landscape_panel(
            ax=ax,
            point_df=point_df,
            value_col=value_col,
            title=None,
            vmin=vmin,
            vmax=vmax,
            levels=levels,
            cmap=cmap,
            scatter_size=scatter_size,
        )
        cax = inset_axes(
            ax,
            width="100%",
            height="4%",
            loc="lower left",
            bbox_to_anchor=(0.0, 1.03, 1.0, 1.0),
            bbox_transform=ax.transAxes,
            borderpad=0,
        )
        cbar = fig.colorbar(artist, cax=cax, orientation="horizontal")
        if ticks:
            cbar.set_ticks(ticks)
            cbar.set_ticklabels([f"{int(tick)}" for tick in ticks])
        cbar.ax.xaxis.set_ticks_position("top")
        cbar.ax.xaxis.set_label_position("top")
        cbar.ax.tick_params(axis="x", pad=1)
        cbar.set_label(colorbar_label, labelpad=4)

        output_path = output_dir / f"landscape_{safe_file_stem(file_stem)}.pdf"
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(output_path)

    return saved_paths


def plot_landscape_grid(
    point_df: pd.DataFrame,
    fallbacks: list[str],
    output_path: Path,
    system: str,
    scale: str,
    train_frac: float,
    repeat: int,
    cmap: str,
    robust_quantile: float,
    contour_levels: int,
    scatter_size: float,
    dpi: int,
) -> None:
    plot_specs, colorbar_label = landscape_value_specs(system, fallbacks)
    panel_specs = [(value_col, display_name) for value_col, display_name, _key in plot_specs]
    value_arrays = [point_df[col].to_numpy(dtype=float) for col, _display_name in panel_specs]
    vmin, vmax = color_limits(value_arrays, robust_quantile=robust_quantile)
    levels = np.linspace(vmin, vmax, max(3, int(contour_levels)))

    n_panels = len(panel_specs)
    n_cols = min(4, n_panels)
    n_rows = int(np.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.4 * n_cols, 3.2 * n_rows),
        squeeze=False,
        constrained_layout=True,
    )

    artist = None
    for idx, (value_col, title) in enumerate(panel_specs):
        row = idx // n_cols
        col = idx % n_cols
        artist = plot_landscape_panel(
            ax=axes[row][col],
            point_df=point_df,
            value_col=value_col,
            title=title,
            vmin=vmin,
            vmax=vmax,
            levels=levels,
            cmap=cmap,
            scatter_size=scatter_size,
        )

    for idx in range(n_panels, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].axis("off")

    fig.suptitle(
        f"{system} {scale}: true and fallback-estimated holdout landscapes "
        f"(train={train_frac}, repeat={repeat})",
        fontsize=13,
    )
    if artist is not None:
        cbar = fig.colorbar(artist, ax=axes, location="top", shrink=0.65, pad=0.03)
        cbar.set_label(colorbar_label)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize true and fallback-estimated holdout landscapes for RQ3 discussion."
    )
    parser.add_argument("--prediction-dir", type=Path, default=DEFAULT_PREDICTION_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--system", default="mysql")
    parser.add_argument("--scale", default="large")
    parser.add_argument("--train-frac", type=parse_train_frac, default=0.05)
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--fallbacks", nargs="+", default=DEFAULT_FALLBACKS)
    parser.add_argument("--include-worst", action="store_true")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optionally sample this many holdout points before MDS and plotting.",
    )
    parser.add_argument("--mds-method", choices=["auto", "exact", "landmark"], default="auto")
    parser.add_argument("--exact-limit", type=int, default=5000)
    parser.add_argument("--landmarks", type=int, default=5000)
    parser.add_argument("--landmark-neighbors", type=int, default=20)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--n-init", type=int, default=1)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--cmap", default="RdBu_r")
    parser.add_argument("--robust-quantile", type=float, default=0.0)
    parser.add_argument("--contour-levels", type=int, default=24)
    parser.add_argument("--scatter-size", type=float, default=1.0)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fallbacks = list(args.fallbacks)
    if args.include_worst and "worst" not in fallbacks:
        fallbacks.append("worst")

    combined_df, config_cols = build_combined_prediction_table(
        prediction_dir=args.prediction_dir,
        fallbacks=fallbacks,
        system=args.system,
        scale=args.scale,
        train_frac=args.train_frac,
        repeat=args.repeat,
    )
    combined_df = sample_points(combined_df, args.sample_size, args.random_state)

    print(
        f"[INFO] Loaded {len(combined_df)} common holdout points for "
        f"{args.system}_{args.scale}, repeat={args.repeat}"
    )
    point_df, mds_info = build_discussion_points(
        combined_df=combined_df,
        config_cols=config_cols,
        fallbacks=fallbacks,
        system=args.system,
        mds_method=args.mds_method,
        exact_limit=args.exact_limit,
        landmarks=args.landmarks,
        landmark_neighbors=args.landmark_neighbors,
        random_state=args.random_state,
        n_init=args.n_init,
        max_iter=args.max_iter,
        n_jobs=args.n_jobs,
    )

    tag = output_tag(args.system, args.scale, args.train_frac, args.repeat)
    out_dir = args.output_dir / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    points_path = out_dir / f"{tag}_discussion_points.csv"
    point_df.to_csv(points_path, index=False)
    print(f"[SAVE] {points_path}")

    mds_path = out_dir / f"{tag}_mds_info.csv"
    pd.DataFrame([mds_info]).to_csv(mds_path, index=False)
    print(f"[SAVE] {mds_path}")

    plot_paths = plot_single_landscape_figures(
        point_df=point_df,
        fallbacks=fallbacks,
        system=args.system,
        output_dir=out_dir,
        cmap=args.cmap,
        robust_quantile=args.robust_quantile,
        contour_levels=args.contour_levels,
        scatter_size=args.scatter_size,
        dpi=args.dpi,
    )
    for plot_path in plot_paths:
        print(f"[SAVE] {plot_path}")


if __name__ == "__main__":
    main()
