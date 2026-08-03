#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot RQ2 true-vs-predicted performance scatter plots.

Input:
    fallback_analysis/RQ3/results/predictions/<fallback>/<system>_<scale>_predictions.csv

Each input row must provide `true_perf` and `pred_perf`. The plot compares
ground-truth performance from the real system with reconstructed performance
from one fallback strategy.

Default output:
    RQ2/results/<scale>/correlation/<system>/<fallback>.pdf
"""

from __future__ import annotations

import argparse
import os
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


SCRIPT_DIR = Path(__file__).resolve().parent
RQ2_DIR = SCRIPT_DIR.parent
STATISTICAL_TOOLS_DIR = RQ2_DIR.parent
REPO_ROOT = STATISTICAL_TOOLS_DIR.parent
DEFAULT_PREDICTION_ROOT = REPO_ROOT / "fallback_analysis" / "RQ3" / "results" / "predictions"
DEFAULT_OUTPUT_ROOT = RQ2_DIR / "results"

SYSTEM_ORDER = ["mysql", "postgresql", "httpd", "tomcat", "gcc", "clang", "x264", "x265"]
FALLBACK_ORDER = ["worst", "1NN", "kNN", "kNNw", "loess", "rf", "dal_rf", "rf_uncertainty"]
FALLBACK_DISPLAY = {
    "worst": "Penalty",
    "1NN": "1NN",
    "kNN": "kNN",
    "kNNw": "wkNN",
    "loess": "LOESS",
    "rf": "RF",
    "dal_rf": "DaL-RF",
    "rf_uncertainty": "U-RF",
}
POINT_COLOR = "purple"


def output_path_for_plot(output_root: Path, scale: str, system: str, fallback: str) -> Path:
    return output_root / scale / "correlation" / system / f"{fallback}.png"


def output_path_for_pdf(output_root: Path, scale: str, system: str, fallback: str) -> Path:
    return output_root / scale / "correlation" / system / f"{fallback}.pdf"


def output_path_for_tex(output_root: Path, scale: str, system: str, fallback: str) -> Path:
    return output_root / scale / "correlation" / system / f"{fallback}.tex"


def load_prediction_points(
    prediction_root: Path,
    fallback: str,
    system: str,
    scale: str,
    train_frac: float | None = None,
    repeat: int | None = None,
) -> pd.DataFrame:
    csv_path = prediction_root / fallback / f"{system}_{scale}_predictions.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)
    required = {"system", "scale", "fallback", "train_frac", "repeat", "true_perf", "pred_perf"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")

    mask = (
        (df["system"].astype(str) == system)
        & (df["scale"].astype(str) == scale)
        & (df["fallback"].astype(str) == fallback)
    )
    if train_frac is not None:
        mask &= pd.to_numeric(df["train_frac"], errors="coerce") == float(train_frac)
    if repeat is not None:
        mask &= pd.to_numeric(df["repeat"], errors="coerce") == int(repeat)

    points = df.loc[mask, ["true_perf", "pred_perf"]].copy()
    points["true_perf"] = pd.to_numeric(points["true_perf"], errors="coerce")
    points["pred_perf"] = pd.to_numeric(points["pred_perf"], errors="coerce")
    points = points.replace([np.inf, -np.inf], np.nan).dropna()

    if points.empty:
        raise ValueError(
            f"No valid prediction rows for fallback={fallback}, system={system}, "
            f"scale={scale}, train_frac={train_frac}, repeat={repeat}"
        )
    return points.reset_index(drop=True)


def sample_points(df: pd.DataFrame, sample_size: int | None, random_state: int) -> pd.DataFrame:
    if sample_size is None or sample_size <= 0 or len(df) <= sample_size:
        return df
    return df.sample(n=sample_size, random_state=random_state).sort_index().reset_index(drop=True)


def axis_limits(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    values = np.concatenate([x.to_numpy(dtype=float), y.to_numpy(dtype=float)])
    low = float(np.min(values))
    high = float(np.max(values))
    if abs(high - low) <= 1e-12:
        pad = max(1.0, abs(low) * 0.05)
    else:
        pad = (high - low) * 0.04
    return low - pad, high + pad


def latex_float(value: float) -> str:
    return f"{float(value):.6e}"


def latex_coordinate_rows(points: pd.DataFrame) -> str:
    return "\n".join(
        f"        ({latex_float(row.true_perf)},{latex_float(row.pred_perf)})"
        for row in points.itertuples(index=False)
    )


def plot_prediction_scatter(
    points: pd.DataFrame,
    output_path: Path,
    system: str,
    fallback: str,
    train_frac: float | None,
    repeat: int | None,
    dpi: int,
    marker_size: float,
    alpha: float,
) -> None:
    x = points["true_perf"]
    y = points["pred_perf"]
    lower, upper = axis_limits(x, y)

    fig, ax = plt.subplots(figsize=(4.6, 4.1), constrained_layout=True)
    ax.scatter(
        x,
        y,
        s=marker_size,
        alpha=alpha,
        c=POINT_COLOR,
        edgecolors="none",
        rasterized=True,
    )
    ax.plot([lower, upper], [lower, upper], color="black", linestyle="--", linewidth=1.0)
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_xlabel("True performance")
    ax.set_ylabel("Predicted performance")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.45)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def write_prediction_scatter_tex(
    points: pd.DataFrame,
    output_path: Path,
    marker_size_pt: float,
) -> None:
    x = points["true_perf"]
    y = points["pred_perf"]
    lower, upper = axis_limits(x, y)
    coordinate_rows = latex_coordinate_rows(points)
    lower_text = latex_float(lower)
    upper_text = latex_float(upper)
    marker_size_text = f"{marker_size_pt:.3f}".rstrip("0").rstrip(".")

    tex = rf"""\documentclass[tikz]{{standalone}}
\usepackage{{pgfplots}}
\pgfplotsset{{compat=newest}}

\begin{{document}}
\begin{{tikzpicture}}
\begin{{axis}}[
    width=3.4in,
    height=3.4in,
    xlabel={{True performance}},
    ylabel={{Predicted performance}},
    xmin={lower_text},
    xmax={upper_text},
    ymin={lower_text},
    ymax={upper_text},
    axis equal image,
    grid=major,
    scaled ticks=false,
    xticklabel style={{/pgf/number format/sci, /pgf/number format/precision=2}},
    yticklabel style={{/pgf/number format/sci, /pgf/number format/precision=2}},
]
\addplot[
    only marks,
    mark=.,
    mark size={marker_size_text}pt,
    draw=blue!45!white,
] coordinates {{
{coordinate_rows}
}};
\addplot[
    black,
    dashed,
    line width=0.7pt,
] coordinates {{
        ({lower_text},{lower_text})
        ({upper_text},{upper_text})
}};
\end{{axis}}
\end{{tikzpicture}}
\end{{document}}
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(tex, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot true-vs-predicted fallback performance for RQ2 visual inspection."
    )
    parser.add_argument("--prediction-root", type=Path, default=DEFAULT_PREDICTION_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--scale", default="large")
    parser.add_argument("--systems", nargs="+", default=SYSTEM_ORDER)
    parser.add_argument("--fallbacks", nargs="+", default=FALLBACK_ORDER)
    parser.add_argument("--train-frac", type=float, default=None)
    parser.add_argument("--repeat", type=int, default=None)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--marker-size", type=float, default=3.0)
    parser.add_argument("--alpha", type=float, default=0.18)
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=["png", "pdf", "tex"],
        default=["pdf"],
        help="Output formats to save.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    saved = 0
    for system in args.systems:
        for fallback in args.fallbacks:
            try:
                points = load_prediction_points(
                    prediction_root=args.prediction_root,
                    fallback=fallback,
                    system=system,
                    scale=args.scale,
                    train_frac=args.train_frac,
                    repeat=args.repeat,
                )
            except (FileNotFoundError, ValueError) as exc:
                print(f"[WARN] Skipping {system}/{fallback}: {exc}")
                continue

            points = sample_points(points, args.sample_size, args.random_state)
            if "png" in args.formats:
                output_path = output_path_for_plot(
                    output_root=args.output_root,
                    scale=args.scale,
                    system=system,
                    fallback=fallback,
                )
                plot_prediction_scatter(
                    points=points,
                    output_path=output_path,
                    system=system,
                    fallback=fallback,
                    train_frac=args.train_frac,
                    repeat=args.repeat,
                    dpi=args.dpi,
                    marker_size=args.marker_size,
                    alpha=args.alpha,
                )
                saved += 1
                print(f"[SAVE] {output_path}")

            if "pdf" in args.formats:
                pdf_path = output_path_for_pdf(
                    output_root=args.output_root,
                    scale=args.scale,
                    system=system,
                    fallback=fallback,
                )
                plot_prediction_scatter(
                    points=points,
                    output_path=pdf_path,
                    system=system,
                    fallback=fallback,
                    train_frac=args.train_frac,
                    repeat=args.repeat,
                    dpi=args.dpi,
                    marker_size=args.marker_size,
                    alpha=args.alpha,
                )
                saved += 1
                print(f"[SAVE] {pdf_path}")

            if "tex" in args.formats:
                tex_path = output_path_for_tex(
                    output_root=args.output_root,
                    scale=args.scale,
                    system=system,
                    fallback=fallback,
                )
                write_prediction_scatter_tex(
                    points=points,
                    output_path=tex_path,
                    marker_size_pt=max(args.marker_size * 0.12, 0.2),
                )
                saved += 1
                print(f"[SAVE] {tex_path}")

    print(f"[DONE] Saved {saved} plot(s).")


if __name__ == "__main__":
    main()
