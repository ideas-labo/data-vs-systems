#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from rq3_metric_report import build_master_table


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_LANDSCAPE_DIR = BASE_DIR / "landscape"
DEFAULT_OUTPUT_DIR = BASE_DIR / "rq3"
DEFAULT_EXCLUDED_FALLBACKS = ["worst"]

ID_COLS = ["system", "scale", "train_frac", "repeat", "fallback"]
CONTEXT_COLS = ["system", "scale", "train_frac", "repeat"]
SYSTEM_ORDER = ["mysql", "postgresql", "httpd", "tomcat", "gcc", "clang", "x264", "x265"]

RAW_COLS = [
    "true_auto_correlation",
    "pred_auto_correlation",
    "auto_correlation_abs_error",
]


def fmt(value: float, digits: int = 3) -> str:
    if value is None or pd.isna(value) or not np.isfinite(value):
        return ""
    return f"{float(value):.{digits}f}"


def fmt_mean_std(mean: float, std: float, digits: int = 3) -> str:
    if mean is None or pd.isna(mean) or not np.isfinite(mean):
        return ""
    if std is None or pd.isna(std) or not np.isfinite(std):
        return fmt(mean, digits)
    return f"{fmt(mean, digits)} +/- {fmt(std, digits)}"


def markdown_table(headers: list[str], rows: list[list[object]], right_align: Iterable[str] = ()) -> str:
    right_align = set(right_align)
    align = ["---:" if header in right_align else "---" for header in headers]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(align) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def load_landscape_raw(input_dir: Path) -> pd.DataFrame:
    frames = []
    for csv_path in sorted(input_dir.glob("*/*_landscape_raw.csv")):
        df = pd.read_csv(csv_path)
        required = set(ID_COLS + RAW_COLS)
        missing = required - set(df.columns)
        if missing:
            print(f"[WARN] Skip {csv_path}: missing columns {sorted(missing)}")
            continue
        frames.append(df[ID_COLS + RAW_COLS])

    if not frames:
        raise FileNotFoundError(f"No landscape raw CSV files found under {input_dir}")

    out = pd.concat(frames, ignore_index=True)
    for col in ["train_frac", "repeat", *RAW_COLS]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def build_true_context_table(raw_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, group in raw_df.groupby(CONTEXT_COLS, sort=False):
        values = group["true_auto_correlation"].replace([np.inf, -np.inf], np.nan).dropna()
        row = dict(zip(CONTEXT_COLS, key))
        row["true_auto_correlation"] = float(values.mean()) if len(values) else float("nan")
        row["true_auto_correlation_std_across_fallbacks"] = float(values.std()) if len(values) > 1 else 0.0
        row["fallback_count"] = int(group["fallback"].nunique())
        rows.append(row)
    return pd.DataFrame(rows)


def build_system_summary(
    context_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    master_df: pd.DataFrame,
    excluded_fallbacks: list[str],
) -> pd.DataFrame:
    pred_df = raw_df[~raw_df["fallback"].astype(str).isin(excluded_fallbacks)].copy()
    rq2_df = master_df[~master_df["fallback"].astype(str).isin(excluded_fallbacks)].copy()

    true_summary = (
        context_df.groupby("system", as_index=False)
        .agg(
            context_count=("true_auto_correlation", "size"),
            true_auto_correlation_mean=("true_auto_correlation", "mean"),
            true_auto_correlation_std=("true_auto_correlation", "std"),
            true_auto_correlation_min=("true_auto_correlation", "min"),
            true_auto_correlation_max=("true_auto_correlation", "max"),
            disagreement_contexts=(
                "true_auto_correlation_std_across_fallbacks",
                lambda values: int((values.fillna(0.0) > 1e-9).sum()),
            ),
        )
    )
    pred_summary = (
        pred_df.groupby("system", as_index=False)
        .agg(
            fallback_count=("fallback", "nunique"),
            pred_auto_correlation_mean=("pred_auto_correlation", "mean"),
            pred_auto_correlation_std=("pred_auto_correlation", "std"),
            auto_correlation_abs_error_mean=("auto_correlation_abs_error", "mean"),
            auto_correlation_abs_error_std=("auto_correlation_abs_error", "std"),
        )
    )
    rq2_summary = (
        rq2_df.groupby("system", as_index=False)
        .agg(
            rq2_spearman_mean=("rq2_spearman", "mean"),
            rq2_spearman_std=("rq2_spearman", "std"),
        )
    )

    summary = true_summary.merge(pred_summary, on="system", how="left").merge(rq2_summary, on="system", how="left")
    summary["_system_order"] = summary["system"].map(
        {system: idx for idx, system in enumerate(SYSTEM_ORDER)}
    ).fillna(999)
    return summary.sort_values(["_system_order", "system"]).drop(columns="_system_order").reset_index(drop=True)


def write_report(output_path: Path, summary: pd.DataFrame, excluded_fallbacks: list[str]) -> None:
    excluded = ", ".join(f"`{fallback}`" for fallback in excluded_fallbacks) if excluded_fallbacks else "`none`"
    headers = [
        "System",
        "RQ2 Spearman",
        "True AutoCorr",
        "True Min",
        "True Max",
        "Pred AutoCorr",
        "AutoCorr Error",
        "Contexts",
        "Fallbacks",
        "Disagree Ctx",
    ]
    rows = []
    for _, row in summary.iterrows():
        rows.append(
            [
                row["system"],
                fmt_mean_std(row["rq2_spearman_mean"], row["rq2_spearman_std"]),
                fmt_mean_std(row["true_auto_correlation_mean"], row["true_auto_correlation_std"]),
                fmt(row["true_auto_correlation_min"]),
                fmt(row["true_auto_correlation_max"]),
                fmt_mean_std(row["pred_auto_correlation_mean"], row["pred_auto_correlation_std"]),
                fmt_mean_std(row["auto_correlation_abs_error_mean"], row["auto_correlation_abs_error_std"]),
                int(row["context_count"]),
                int(row["fallback_count"]) if pd.notna(row["fallback_count"]) else "",
                int(row["disagreement_contexts"]),
            ]
        )

    lines = [
        "# RQ3 Auto-Correlation Report",
        "",
        "This report summarizes true landscape auto-correlation by system.",
        "The true value is de-duplicated by `system, scale, train_frac, repeat`, so repeated fallback files do not overweight the same ground-truth landscape.",
        "",
        "Interpretation: higher positive auto-correlation means neighboring points along the random walk tend to have similar performance; near-zero values indicate a more rugged or weakly structured landscape.",
        "",
        f"Prediction/error columns exclude fallbacks: {excluded}.",
        "",
        markdown_table(
            headers,
            rows,
            right_align=headers[1:],
        ),
        "",
        "## Data Quality Note",
        "",
        "`Disagree Ctx` counts contexts where the same ground-truth context produced different `true_auto_correlation` values across fallback files. A non-zero value usually means the valid-row mask differed across fallback predictions.",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the RQ3 true auto-correlation Markdown report.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_LANDSCAPE_DIR, help="Landscape output directory.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for RQ3 reports.")
    parser.add_argument(
        "--scale",
        action="append",
        dest="scales",
        help="Restrict RQ2 context to one scale. Can be repeated. Defaults to all scales found in metrics.",
    )
    parser.add_argument(
        "--include-worst",
        action="store_true",
        help="Include the 'worst' fallback when summarizing predicted auto-correlation and error.",
    )
    parser.add_argument(
        "--exclude-fallback",
        action="append",
        default=[],
        help="Additional fallback to exclude from predicted auto-correlation/error summaries. Can be repeated.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    excluded = [] if args.include_worst else list(DEFAULT_EXCLUDED_FALLBACKS)
    excluded = sorted(set(excluded + args.exclude_fallback))

    raw_df = load_landscape_raw(args.input_dir)
    context_df = build_true_context_table(raw_df)
    master_df = build_master_table(args.scales)
    summary_df = build_system_summary(context_df, raw_df, master_df, excluded)

    report_path = args.output_dir / "rq3_autocorrelation_report.md"
    write_report(report_path, summary_df, excluded)
    print(f"[SAVE] {report_path}")


if __name__ == "__main__":
    main()
