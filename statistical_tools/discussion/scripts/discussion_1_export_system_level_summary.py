#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DISCUSSION_DIR = SCRIPT_DIR.parent
STATISTICAL_TOOLS_DIR = DISCUSSION_DIR.parent
RQ3_SCRIPTS_DIR = STATISTICAL_TOOLS_DIR / "RQ3" / "scripts"
if str(RQ3_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(RQ3_SCRIPTS_DIR))

from rq3_3_build_metric_report import (
    DEFAULT_EXCLUDED_FALLBACKS,
    LANDSCAPE_INPUT_DIR,
    SYSTEM_ORDER,
    build_master_table,
    build_system_summary as build_metric_system_summary,
    filter_fallbacks,
)


DEFAULT_OUTPUT_DIR = DISCUSSION_DIR / "results" / "latex_tables"
OUTPUT_DIGITS = 3
SYSTEM_SUMMARY_OUTPUT_DIGITS = 2
LATEX_SYSTEM_DISPLAY = {
    "mysql": "\\textsc{MySQL}",
    "postgresql": "\\textsc{PostgreSQL}",
    "httpd": "\\textsc{Httpd}",
    "tomcat": "\\textsc{Tomcat}",
    "gcc": "\\textsc{Gcc}",
    "clang": "\\textsc{Clang}",
    "x264": "\\textsc{x264}",
    "x265": "\\textsc{x265}",
}


def latex_escape(value: object) -> str:
    text = "" if value is None else str(value)
    replacements = {
        "\\": "\\textbackslash{}",
        "&": "\\&",
        "%": "\\%",
        "$": "\\$",
        "#": "\\#",
        "_": "\\_",
        "{": "\\{",
        "}": "\\}",
        "~": "\\textasciitilde{}",
        "^": "\\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def fmt_number(value: float, digits: int = OUTPUT_DIGITS) -> str:
    if value is None or pd.isna(value) or not np.isfinite(value):
        return "--"
    value = float(value)
    if abs(value) < 0.5 * (10 ** -digits):
        value = 0.0
    return f"{value:.{digits}f}"


def fmt_mean_std(mean: float, std: float, digits: int = OUTPUT_DIGITS) -> str:
    if mean is None or pd.isna(mean) or not np.isfinite(mean):
        return "--"
    if std is None or pd.isna(std) or not np.isfinite(std):
        return fmt_number(mean, digits)
    return f"{fmt_number(mean, digits)} ({fmt_number(std, digits)})"


def latex_bold(value: str) -> str:
    return f"\\textbf{{{value}}}"


def latex_tabular(headers: list[str], rows: list[list[str]], align: str) -> str:
    headers = [latex_bold(header) for header in headers]
    lines = [
        f"\\begin{{tabular}}{{{align}}}",
        "\\toprule",
        " & ".join(headers) + " \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(row) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines)


def load_true_auto_correlation_summary(
    input_dir: Path,
    scales: list[str] | None,
) -> pd.DataFrame:
    frames = []
    required_cols = {
        "system",
        "scale",
        "train_frac",
        "repeat",
        "fallback",
        "true_auto_correlation",
    }
    for csv_path in sorted(input_dir.glob("*/*_landscape_raw.csv")):
        df = pd.read_csv(csv_path)
        missing = required_cols - set(df.columns)
        if missing:
            print(f"[WARN] Skip {csv_path}: missing columns {sorted(missing)}")
            continue
        frames.append(df[list(required_cols)])

    if not frames:
        return pd.DataFrame(columns=["system", "true_auto_correlation_mean"])

    raw_df = pd.concat(frames, ignore_index=True)
    if scales:
        raw_df = raw_df[raw_df["scale"].astype(str).isin(scales)].copy()

    raw_df["true_auto_correlation"] = pd.to_numeric(
        raw_df["true_auto_correlation"],
        errors="coerce",
    )
    context_df = (
        raw_df.groupby(["system", "scale", "train_frac", "repeat"], as_index=False)
        .agg(true_auto_correlation=("true_auto_correlation", "mean"))
    )
    summary = (
        context_df.groupby("system", as_index=False)
        .agg(true_auto_correlation_mean=("true_auto_correlation", "mean"))
    )
    return summary


def build_system_summary_table(
    metric_summary: pd.DataFrame,
    auto_correlation_summary: pd.DataFrame,
) -> str:
    summary = metric_summary.merge(auto_correlation_summary, on="system", how="left")
    summary["_system_order"] = summary["system"].astype(str).map(
        {system: idx for idx, system in enumerate(SYSTEM_ORDER)}
    ).fillna(999)
    summary = summary.sort_values(["_system_order", "system"]).drop(columns="_system_order")

    headers = [
        "System",
        "MRE",
        "RMSE",
        "$\\rho_{config}$",
        "Recall@5\\%",
        "ACE",
        "AC",
        "$\\rho_{tuner}$",
    ]
    rows = []
    for _, row in summary.iterrows():
        system = str(row["system"])
        rows.append(
            [
                latex_bold(LATEX_SYSTEM_DISPLAY.get(system, latex_escape(system))),
                fmt_mean_std(
                    row["mre_mean_mean"],
                    row["mre_mean_std"],
                    SYSTEM_SUMMARY_OUTPUT_DIGITS,
                ),
                fmt_mean_std(
                    row["rmse_mean_mean"],
                    row["rmse_mean_std"],
                    SYSTEM_SUMMARY_OUTPUT_DIGITS,
                ),
                fmt_mean_std(
                    row["spearman_perf_mean_mean"],
                    row["spearman_perf_mean_std"],
                    SYSTEM_SUMMARY_OUTPUT_DIGITS,
                ),
                fmt_mean_std(
                    row["top5_overlap_mean_mean"],
                    row["top5_overlap_mean_std"],
                    SYSTEM_SUMMARY_OUTPUT_DIGITS,
                ),
                fmt_mean_std(
                    row["auto_correlation_abs_error_mean_mean"],
                    row["auto_correlation_abs_error_mean_std"],
                    SYSTEM_SUMMARY_OUTPUT_DIGITS,
                ),
                fmt_number(row["true_auto_correlation_mean"], SYSTEM_SUMMARY_OUTPUT_DIGITS),
                fmt_mean_std(
                    row["rq2_spearman_mean"],
                    row["rq2_spearman_std"],
                    SYSTEM_SUMMARY_OUTPUT_DIGITS,
                ),
            ]
        )

    return latex_tabular(headers, rows, "lrrrrrrr")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")
    print(f"[SAVE] {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the discussion system-level summary LaTeX table."
    )
    parser.add_argument(
        "--scale",
        action="append",
        dest="scales",
        help="Restrict output to one scale. Can be repeated. Defaults to all scales found in metrics.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--include-worst",
        action="store_true",
        help="Include the 'worst' fallback in generated discussion table.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    excluded_fallbacks = [] if args.include_worst else list(DEFAULT_EXCLUDED_FALLBACKS)

    master_df = build_master_table(args.scales)
    filtered_df = filter_fallbacks(master_df, excluded_fallbacks)
    system_summary = build_metric_system_summary(filtered_df)
    auto_correlation_summary = load_true_auto_correlation_summary(
        LANDSCAPE_INPUT_DIR,
        args.scales,
    )
    system_table = build_system_summary_table(system_summary, auto_correlation_summary)

    output_dir = args.output_dir.resolve()
    write_text(output_dir / "discussion_system_level_summary.tex", system_table)


if __name__ == "__main__":
    main()
