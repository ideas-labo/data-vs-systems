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
from scipy.interpolate import RegularGridInterpolator, griddata
from scipy.ndimage import gaussian_filter
from scipy.stats import binned_statistic_2d


SCRIPT_DIR = Path(__file__).resolve().parent
RQ3_DIR = SCRIPT_DIR.parent
RESULTS_DIR = RQ3_DIR / "results"
DEFAULT_INPUT_DIR = RESULTS_DIR / "local_optima_mds" / "points"
DEFAULT_OUTPUT_DIR = RESULTS_DIR / "local_optima_mds" / "plots_3d"

SYSTEM_ORDER = ["mysql", "postgresql", "httpd", "tomcat", "gcc", "clang", "x264", "x265"]
MINIMIZE_SYSTEMS = {"gcc", "clang", "x264", "x265"}
REQUIRED_COLS = {"system", "scale", "mds_x", "mds_y", "performance"}
SURFACE_METHODS = ("binned", "nearest", "linear", "cubic")


def system_sort_key(system: str) -> tuple[int, str]:
    return (SYSTEM_ORDER.index(system), system) if system in SYSTEM_ORDER else (999, system)


def parse_system_scale(csv_path: Path) -> tuple[str, str]:
    suffix = "_mds_3d_points.csv"
    if not csv_path.name.endswith(suffix):
        raise ValueError(f"Unexpected point filename: {csv_path.name}")
    stem = csv_path.name[:-len(suffix)]
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


def downsample_points(df: pd.DataFrame, max_points: int | None, random_state: int) -> pd.DataFrame:
    if max_points is None or max_points <= 0 or len(df) <= max_points:
        return df
    return df.sample(n=max_points, random_state=random_state).sort_index()


def plot_3d_landscape(
    df: pd.DataFrame,
    z_values: np.ndarray,
    z_label: str,
    system: str,
    scale: str,
    output_path: Path,
    dpi: int,
    elev: float,
    azim: float,
    highlight_local_optima: bool,
    color_by_z: bool,
    show: bool,
) -> None:
    x = pd.to_numeric(df["mds_x"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df["mds_y"], errors="coerce").to_numpy(dtype=float)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    if color_by_z:
        scatter = ax.scatter(
            x,
            y,
            z_values,
            c=z_values,
            cmap="viridis",
            marker=".",
            s=8,
            alpha=0.7,
            linewidths=0,
        )
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.65, pad=0.08)
        cbar.set_label(z_label)
    else:
        ax.scatter(
            x,
            y,
            z_values,
            c="black",
            marker=".",
            s=8,
            alpha=0.65,
            linewidths=0,
        )

    if highlight_local_optima and "is_local_optimum" in df.columns:
        local_mask = df["is_local_optimum"].astype(bool).to_numpy()
        if np.any(local_mask):
            ax.scatter(
                x[local_mask],
                y[local_mask],
                z_values[local_mask],
                c="red",
                marker="*",
                s=70,
                alpha=0.95,
                linewidths=0.3,
                edgecolors="red",
                label="Local optimum",
            )
            ax.legend(loc="upper left", frameon=False)

    ax.set_title(f"{system} {scale}: MDS landscape")
    ax.set_xlabel("MDS dimension 1")
    ax.set_ylabel("MDS dimension 2")
    ax.set_zlabel(z_label)
    ax.view_init(elev=elev, azim=azim)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.25)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def build_grid_coordinates(
    x: np.ndarray,
    y: np.ndarray,
    grid_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if grid_size < 5:
        raise ValueError("--surface-grid-size must be at least 5.")

    x_min, x_max = float(np.nanmin(x)), float(np.nanmax(x))
    y_min, y_max = float(np.nanmin(y)), float(np.nanmax(y))
    if x_min == x_max or y_min == y_max:
        raise ValueError("Cannot build a surface when MDS coordinates have zero range.")

    x_edges = np.linspace(x_min, x_max, grid_size + 1)
    y_edges = np.linspace(y_min, y_max, grid_size + 1)
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2.0
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2.0
    grid_x, grid_y = np.meshgrid(x_centers, y_centers, indexing="xy")
    return x_edges, y_edges, grid_x, grid_y


def fill_empty_surface_cells(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    grid_z: np.ndarray,
) -> np.ndarray:
    valid_mask = np.isfinite(grid_z)
    if not np.any(valid_mask):
        raise ValueError("Cannot build a surface because no grid cells contain data.")

    if not np.all(valid_mask):
        filled_z = griddata(
            np.column_stack([grid_x[valid_mask], grid_y[valid_mask]]),
            grid_z[valid_mask],
            (grid_x, grid_y),
            method="nearest",
        )
        grid_z = np.where(valid_mask, grid_z, filled_z)

    return grid_z


def smooth_surface_grid(grid_z: np.ndarray, smooth_sigma: float) -> np.ndarray:
    if smooth_sigma < 0:
        raise ValueError("--surface-smooth-sigma must be non-negative.")
    if smooth_sigma == 0:
        return grid_z

    valid_mask = np.isfinite(grid_z)
    weights = valid_mask.astype(float)
    grid_values = np.where(valid_mask, grid_z, 0.0)
    smoothed_values = gaussian_filter(grid_values, sigma=smooth_sigma)
    smoothed_weights = gaussian_filter(weights, sigma=smooth_sigma)
    with np.errstate(divide="ignore", invalid="ignore"):
        return smoothed_values / smoothed_weights


def build_surface_grid(
    x: np.ndarray,
    y: np.ndarray,
    z_values: np.ndarray,
    grid_size: int,
    smooth_sigma: float,
    surface_method: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if surface_method not in SURFACE_METHODS:
        raise ValueError(f"Unknown surface method: {surface_method}")

    x_edges, y_edges, grid_x, grid_y = build_grid_coordinates(
        x=x,
        y=y,
        grid_size=grid_size,
    )

    if surface_method == "binned":
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
            method=surface_method,
        )

    grid_z = smooth_surface_grid(grid_z, smooth_sigma)
    grid_z = fill_empty_surface_cells(grid_x, grid_y, grid_z)
    return grid_x, grid_y, grid_z


def safe_z_range(z_min: float, z_max: float) -> float:
    z_range = z_max - z_min
    if np.isfinite(z_range) and z_range > 0:
        return float(z_range)
    return max(abs(z_min), abs(z_max), 1.0) * 0.01


def sample_surface_z(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    grid_z: np.ndarray,
    x_points: np.ndarray,
    y_points: np.ndarray,
) -> np.ndarray:
    x_axis = grid_x[0, :]
    y_axis = grid_y[:, 0]
    interpolator = RegularGridInterpolator(
        (y_axis, x_axis),
        grid_z,
        bounds_error=False,
        fill_value=np.nan,
    )
    sampled_z = interpolator(np.column_stack([y_points, x_points]))

    missing_mask = ~np.isfinite(sampled_z)
    if np.any(missing_mask):
        sampled_z[missing_mask] = griddata(
            np.column_stack([grid_x.ravel(), grid_y.ravel()]),
            grid_z.ravel(),
            (x_points[missing_mask], y_points[missing_mask]),
            method="nearest",
        )

    return sampled_z


def plot_3d_surface_landscape(
    df: pd.DataFrame,
    z_values: np.ndarray,
    z_label: str,
    system: str,
    scale: str,
    output_path: Path,
    dpi: int,
    elev: float,
    azim: float,
    highlight_local_optima: bool,
    grid_size: int,
    smooth_sigma: float,
    surface_method: str,
    cmap: str,
    local_optima_z_mode: str,
    show: bool,
) -> None:
    x = pd.to_numeric(df["mds_x"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df["mds_y"], errors="coerce").to_numpy(dtype=float)
    grid_x, grid_y, grid_z = build_surface_grid(
        x=x,
        y=y,
        z_values=z_values,
        grid_size=grid_size,
        smooth_sigma=smooth_sigma,
        surface_method=surface_method,
    )

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    surface = ax.plot_surface(
        grid_x,
        grid_y,
        grid_z,
        cmap=cmap,
        linewidth=0,
        antialiased=True,
        alpha=0.96,
    )

    z_min = float(np.nanmin(grid_z))
    z_max = float(np.nanmax(grid_z))
    z_range = safe_z_range(z_min, z_max)
    contour_offset = z_min - 0.08 * z_range
    ax.contourf(
        grid_x,
        grid_y,
        grid_z,
        zdir="z",
        offset=contour_offset,
        cmap=cmap,
        levels=24,
        alpha=0.75,
    )

    if highlight_local_optima and "is_local_optimum" in df.columns:
        local_mask = df["is_local_optimum"].astype(bool).to_numpy()
        if np.any(local_mask):
            if local_optima_z_mode == "surface":
                local_z = sample_surface_z(
                    grid_x=grid_x,
                    grid_y=grid_y,
                    grid_z=grid_z,
                    x_points=x[local_mask],
                    y_points=y[local_mask],
                )
            else:
                local_z = z_values[local_mask]

            ax.scatter(
                x[local_mask],
                y[local_mask],
                local_z + 0.006 * z_range,
                c="red",
                marker="*",
                s=55,
                alpha=0.95,
                linewidths=0.3,
                edgecolors="red",
                label="Local optimum",
            )
            ax.legend(loc="upper left", frameon=False)

    cbar = fig.colorbar(surface, ax=ax, shrink=0.65, pad=0.08)
    cbar.set_label(z_label)
    ax.set_title(f"{system} {scale}: {surface_method} MDS surface")
    ax.set_xlabel("MDS dimension 1")
    ax.set_ylabel("MDS dimension 2")
    ax.set_zlabel(z_label)
    ax.set_zlim(contour_offset, z_max + 0.04 * z_range)
    ax.view_init(elev=elev, azim=azim)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.25)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


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

    if args.max_points is not None and args.max_points > 0 and len(df) > args.max_points:
        sampled_df = downsample_points(df, args.max_points, args.random_state)
        z_values = z_values[sampled_df.index.to_numpy()]
        df = sampled_df.reset_index(drop=True)

    scatter_output_path = None
    surface_output_path = None

    if args.plot_kind in {"scatter", "both"}:
        scatter_output_path = args.output_dir / f"{system}_{scale}_mds_landscape_3d.png"
        plot_3d_landscape(
            df=df,
            z_values=z_values,
            z_label=z_label,
            system=system,
            scale=scale,
            output_path=scatter_output_path,
            dpi=args.dpi,
            elev=args.elev,
            azim=args.azim,
            highlight_local_optima=not args.no_local_optima,
            color_by_z=not args.no_color_by_z,
            show=args.show,
        )

    if args.plot_kind in {"surface", "both"}:
        if args.surface_method == "binned":
            surface_output_path = args.output_dir / f"{system}_{scale}_mds_landscape_surface_3d.png"
        else:
            surface_output_path = (
                args.output_dir
                / f"{system}_{scale}_mds_landscape_surface_{args.surface_method}_3d.png"
            )
        plot_3d_surface_landscape(
            df=df,
            z_values=z_values,
            z_label=z_label,
            system=system,
            scale=scale,
            output_path=surface_output_path,
            dpi=args.dpi,
            elev=args.elev,
            azim=args.azim,
            highlight_local_optima=not args.no_local_optima,
            grid_size=args.surface_grid_size,
            smooth_sigma=args.surface_smooth_sigma,
            surface_method=args.surface_method,
            cmap=args.surface_cmap,
            local_optima_z_mode=args.local_optima_z_mode,
            show=args.show,
        )

    return {
        "system": system,
        "scale": scale,
        "input_path": str(csv_path),
        "output_path": str(scatter_output_path or surface_output_path),
        "scatter_output_path": str(scatter_output_path or ""),
        "surface_output_path": str(surface_output_path or ""),
        "plot_kind": args.plot_kind,
        "z_col": args.z_col,
        "z_label": z_label,
        "plotted_points": int(len(df)),
        "z_min": float(np.nanmin(z_values)),
        "z_max": float(np.nanmax(z_values)),
        "surface_method": args.surface_method,
        "surface_grid_size": int(args.surface_grid_size),
        "surface_smooth_sigma": float(args.surface_smooth_sigma),
        "local_optima_z_mode": args.local_optima_z_mode,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize cached MDS coordinates as 3D landscapes."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--systems", nargs="+", default=None)
    parser.add_argument("--scale", default="large")
    parser.add_argument(
        "--z-col",
        default="performance",
        help="Column to use as the Z axis before optional minimization inversion.",
    )
    parser.add_argument(
        "--no-invert-minimize",
        action="store_true",
        help="Do not negate gcc/clang/x264/x265 Z values.",
    )
    parser.add_argument(
        "--no-local-optima",
        action="store_true",
        help="Do not overlay local optima as red stars.",
    )
    parser.add_argument(
        "--no-color-by-z",
        action="store_true",
        help="Draw all points in black instead of coloring by Z value.",
    )
    parser.add_argument(
        "--plot-kind",
        choices=["scatter", "surface", "both"],
        default="both",
        help="Choose whether to draw the original scatter plot, the interpolated surface, or both.",
    )
    parser.add_argument(
        "--surface-method",
        choices=SURFACE_METHODS,
        default="binned",
        help=(
            "How to convert MDS scatter points into a surface: binned uses grid-cell "
            "means; nearest/linear/cubic interpolate directly from the original points."
        ),
    )
    parser.add_argument(
        "--surface-grid-size",
        type=int,
        default=120,
        help="Number of grid cells per MDS axis for the interpolated surface.",
    )
    parser.add_argument(
        "--surface-smooth-sigma",
        type=float,
        default=0.8,
        help="Gaussian smoothing sigma for the surface grid. Use 0 to disable smoothing.",
    )
    parser.add_argument(
        "--surface-cmap",
        default="plasma",
        help="Matplotlib colormap for the interpolated surface.",
    )
    parser.add_argument(
        "--local-optima-z-mode",
        choices=["surface", "original"],
        default="surface",
        help="surface projects local-optimum markers onto the smoothed surface; original uses their raw Z values.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=None,
        help="Optionally sample at most this many points per system for faster plotting.",
    )
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--elev", type=float, default=25.0)
    parser.add_argument("--azim", type=float, default=-60.0)
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
        for key in ("scatter_output_path", "surface_output_path"):
            if row[key]:
                print(f"[SAVE] {row[key]}")

    summary_path = args.output_dir / f"{args.scale}_3d_visualization_summary.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    print(f"[SAVE] {summary_path}")


if __name__ == "__main__":
    main()
