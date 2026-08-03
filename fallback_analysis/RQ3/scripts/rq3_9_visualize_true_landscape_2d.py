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

if "--show" not in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from scipy.stats import binned_statistic_2d


SCRIPT_DIR = Path(__file__).resolve().parent
RQ3_DIR = SCRIPT_DIR.parent
RESULTS_DIR = RQ3_DIR / "results"
DEFAULT_INPUT_DIR = RESULTS_DIR / "local_optima_mds" / "points"
DEFAULT_OUTPUT_DIR = RESULTS_DIR / "local_optima_mds" / "plots_2d"

SYSTEM_ORDER = ["mysql", "postgresql", "httpd", "tomcat", "gcc", "clang", "x264", "x265"]
MINIMIZE_SYSTEMS = {"gcc", "clang", "x264", "x265"}
REQUIRED_COLS = {"system", "scale", "mds_x", "mds_y", "performance"}
GRID_METHODS = ("binned", "nearest", "linear", "cubic")


def system_sort_key(system: str) -> tuple[int, str]:
    return (SYSTEM_ORDER.index(system), system) if system in SYSTEM_ORDER else (999, system)


def parse_system_scale(csv_path: Path) -> tuple[str, str]:
    suffix = "_mds_3d_points.csv"
    if not csv_path.name.endswith(suffix):
        raise ValueError(f"Unexpected point filename: {csv_path.name}")
    stem = csv_path.name[: -len(suffix)]
    system, scale = stem.rsplit("_", 1)
    return system, scale


def resolve_point_files(
    input_dir: Path,
    systems: list[str] | None,
    scale: str | None,
) -> list[Path]:
    files = sorted(input_dir.glob("*_mds_3d_points.csv"))
    if systems and len(systems) == 1 and systems[0].lower() == "all":
        systems = None

    requested_systems = set(systems or [])
    selected = []
    for csv_path in files:
        try:
            system, file_scale = parse_system_scale(csv_path)
        except ValueError:
            continue

        if requested_systems and system not in requested_systems:
            continue
        if scale is not None and file_scale != scale:
            continue
        selected.append(csv_path)

    return sorted(selected, key=lambda path: system_sort_key(parse_system_scale(path)[0]))


def compute_z_values(
    df: pd.DataFrame,
    system: str,
    z_col: str,
    invert_minimize: bool,
) -> tuple[np.ndarray, str]:
    if z_col not in df.columns:
        raise ValueError(f"Column '{z_col}' not found.")

    z_values = pd.to_numeric(df[z_col], errors="coerce").to_numpy(dtype=float)
    if invert_minimize and system in MINIMIZE_SYSTEMS:
        z_values = -z_values
        label = f"-{z_col} (larger is better)"
    else:
        label = f"{z_col} (larger is better)" if z_col == "score" else z_col
    return z_values, label


def finite_point_mask(df: pd.DataFrame, z_values: np.ndarray) -> np.ndarray:
    x = pd.to_numeric(df["mds_x"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df["mds_y"], errors="coerce").to_numpy(dtype=float)
    return np.isfinite(x) & np.isfinite(y) & np.isfinite(z_values)


def downsample_points(
    df: pd.DataFrame,
    z_values: np.ndarray,
    max_points: int | None,
    random_state: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    if max_points is None or max_points <= 0 or len(df) <= max_points:
        return df, z_values

    sampled_df = df.sample(n=max_points, random_state=random_state).sort_index()
    sampled_z = z_values[sampled_df.index.to_numpy()]
    return sampled_df.reset_index(drop=True), sampled_z


def build_grid_coordinates(
    x: np.ndarray,
    y: np.ndarray,
    grid_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if grid_size < 5:
        raise ValueError("--grid-size must be at least 5.")

    x_min, x_max = float(np.nanmin(x)), float(np.nanmax(x))
    y_min, y_max = float(np.nanmin(y)), float(np.nanmax(y))
    if x_min == x_max or y_min == y_max:
        raise ValueError("Cannot build a grid when MDS coordinates have zero range.")

    x_edges = np.linspace(x_min, x_max, grid_size + 1)
    y_edges = np.linspace(y_min, y_max, grid_size + 1)
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2.0
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2.0
    grid_x, grid_y = np.meshgrid(x_centers, y_centers, indexing="xy")
    return x_edges, y_edges, grid_x, grid_y


def fill_empty_grid_cells(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    grid_z: np.ndarray,
) -> np.ndarray:
    valid_mask = np.isfinite(grid_z)
    if not np.any(valid_mask):
        raise ValueError("Cannot build a contour map because no grid cells contain data.")

    if not np.all(valid_mask):
        filled_z = griddata(
            np.column_stack([grid_x[valid_mask], grid_y[valid_mask]]),
            grid_z[valid_mask],
            (grid_x, grid_y),
            method="nearest",
        )
        grid_z = np.where(valid_mask, grid_z, filled_z)

    return grid_z


def smooth_grid(grid_z: np.ndarray, smooth_sigma: float) -> np.ndarray:
    if smooth_sigma < 0:
        raise ValueError("--smooth-sigma must be non-negative.")
    if smooth_sigma == 0:
        return grid_z

    valid_mask = np.isfinite(grid_z)
    weights = valid_mask.astype(float)
    grid_values = np.where(valid_mask, grid_z, 0.0)
    smoothed_values = gaussian_filter(grid_values, sigma=smooth_sigma)
    smoothed_weights = gaussian_filter(weights, sigma=smooth_sigma)
    with np.errstate(divide="ignore", invalid="ignore"):
        return smoothed_values / smoothed_weights


def build_contour_grid(
    x: np.ndarray,
    y: np.ndarray,
    z_values: np.ndarray,
    grid_size: int,
    grid_method: str,
    smooth_sigma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if grid_method not in GRID_METHODS:
        raise ValueError(f"Unknown grid method: {grid_method}")

    x_edges, y_edges, grid_x, grid_y = build_grid_coordinates(
        x=x,
        y=y,
        grid_size=grid_size,
    )

    if grid_method == "binned":
        grid_z, _, _, _ = binned_statistic_2d(
            x,
            y,
            z_values,
            statistic="mean",
            bins=[x_edges, y_edges],
        )
        grid_z = grid_z.T
    else:
        grid_z = griddata(
            np.column_stack([x, y]),
            z_values,
            (grid_x, grid_y),
            method=grid_method,
        )

    grid_z = smooth_grid(grid_z, smooth_sigma)
    grid_z = fill_empty_grid_cells(grid_x, grid_y, grid_z)
    return grid_x, grid_y, grid_z


def overlay_local_optima(
    ax: plt.Axes,
    df: pd.DataFrame,
    enabled: bool,
    max_local_optima: int | None,
    random_state: int,
) -> int:
    if not enabled or "is_local_optimum" not in df.columns:
        return 0

    local_df = df[df["is_local_optimum"].astype(bool)]
    if local_df.empty:
        return 0

    local_optima_count = int(len(local_df))
    if max_local_optima is not None and max_local_optima > 0 and len(local_df) > max_local_optima:
        local_df = local_df.sample(n=max_local_optima, random_state=random_state)

    ax.scatter(
        local_df["mds_x"],
        local_df["mds_y"],
        c="red",
        marker="*",
        s=42,
        linewidths=0.35,
        edgecolors="red",
        alpha=0.9,
        label="Local optimum",
    )
    return local_optima_count


def plot_mds_2d_contour(
    df: pd.DataFrame,
    z_values: np.ndarray,
    z_label: str,
    system: str,
    scale: str,
    output_path: Path,
    args: argparse.Namespace,
) -> dict[str, float | int]:
    x = pd.to_numeric(df["mds_x"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df["mds_y"], errors="coerce").to_numpy(dtype=float)
    grid_x, grid_y, grid_z = build_contour_grid(
        x=x,
        y=y,
        z_values=z_values,
        grid_size=args.grid_size,
        grid_method=args.grid_method,
        smooth_sigma=args.smooth_sigma,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.8, 6.8), constrained_layout=True)

    filled = ax.contourf(
        grid_x,
        grid_y,
        grid_z,
        levels=args.contours,
        cmap=args.cmap,
    )
    if args.contour_lines > 0:
        ax.contour(
            grid_x,
            grid_y,
            grid_z,
            levels=args.contour_lines,
            colors="black",
            linewidths=0.35,
            alpha=0.45,
        )

    local_optima_count = overlay_local_optima(
        ax=ax,
        df=df,
        enabled=not args.no_local_optima,
        max_local_optima=args.max_local_optima,
        random_state=args.random_state,
    )

    ax.set_title(f"{system} {scale}: MDS contour map")
    ax.set_xlabel("MDS dimension 1")
    ax.set_ylabel("MDS dimension 2")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.25)

    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="upper right", frameon=True, framealpha=0.85)

    cbar = fig.colorbar(filled, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(z_label)

    fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    if args.show:
        plt.show()
    plt.close(fig)

    return {
        "local_optima_count": local_optima_count,
        "z_min": float(np.nanmin(z_values)),
        "z_max": float(np.nanmax(z_values)),
    }


def visualize_one_file(csv_path: Path, args: argparse.Namespace) -> dict[str, object]:
    system, scale = parse_system_scale(csv_path)
    df = pd.read_csv(csv_path)
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")

    z_values, z_label = compute_z_values(
        df=df,
        system=system,
        z_col=args.z_col,
        invert_minimize=not args.no_invert_minimize,
    )
    mask = finite_point_mask(df, z_values)
    df = df.loc[mask].reset_index(drop=True)
    z_values = z_values[mask]

    if df.empty:
        raise ValueError(f"No finite MDS/performance rows in {csv_path}")

    df, z_values = downsample_points(
        df=df,
        z_values=z_values,
        max_points=args.max_points,
        random_state=args.random_state,
    )

    output_path = args.output_dir / f"{system}_{scale}_mds_2d_contour_{args.grid_method}.png"
    stats = plot_mds_2d_contour(
        df=df,
        z_values=z_values,
        z_label=z_label,
        system=system,
        scale=scale,
        output_path=output_path,
        args=args,
    )

    return {
        "system": system,
        "scale": scale,
        "input_path": str(csv_path),
        "output_path": str(output_path),
        "z_col": args.z_col,
        "z_label": z_label,
        "points": int(len(df)),
        "grid_method": args.grid_method,
        "grid_size": int(args.grid_size),
        "smooth_sigma": float(args.smooth_sigma),
        "contours": int(args.contours),
        "contour_lines": int(args.contour_lines),
        **stats,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize cached MDS points as a 2D contour map."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--systems", nargs="+", default=None)
    parser.add_argument("--scale", default="large")
    parser.add_argument(
        "--z-col",
        default="performance",
        help="Column to visualize before optional minimization inversion.",
    )
    parser.add_argument(
        "--no-invert-minimize",
        action="store_true",
        help="Do not negate gcc/clang/x264/x265 values.",
    )
    parser.add_argument(
        "--grid-method",
        choices=GRID_METHODS,
        default="binned",
        help="How to map MDS scatter points onto the contour grid.",
    )
    parser.add_argument("--grid-size", type=int, default=160)
    parser.add_argument(
        "--smooth-sigma",
        type=float,
        default=0.8,
        help="Gaussian smoothing sigma for the contour grid. Use 0 to disable smoothing.",
    )
    parser.add_argument(
        "--contours",
        type=int,
        default=18,
        help="Number of filled contour levels.",
    )
    parser.add_argument(
        "--contour-lines",
        type=int,
        default=12,
        help="Number of line contour levels. Use 0 to hide lines.",
    )
    parser.add_argument("--cmap", default="viridis")
    parser.add_argument(
        "--no-local-optima",
        action="store_true",
        help="Do not overlay local optima as red stars.",
    )
    parser.add_argument(
        "--max-local-optima",
        type=int,
        default=None,
        help="Optionally sample at most this many local-optimum markers for readability.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=None,
        help="Optionally sample points for faster exploratory plots.",
    )
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open an interactive Matplotlib window after saving each figure.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.input_dir = args.input_dir.resolve()
    args.output_dir = args.output_dir.resolve()

    point_files = resolve_point_files(args.input_dir, args.systems, args.scale)
    if not point_files:
        print(f"[WARN] No point CSV files found under {args.input_dir}")
        return

    rows = []
    for csv_path in point_files:
        print(f"[PROCESS] {csv_path}")
        row = visualize_one_file(csv_path, args)
        rows.append(row)
        print(f"[SAVE] {row['output_path']}")

    summary_path = args.output_dir / f"{args.scale}_2d_contour_summary.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    print(f"[SAVE] {summary_path}")


if __name__ == "__main__":
    main()
