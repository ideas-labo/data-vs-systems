#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from rq3_3_build_metric_report import (
    DEFAULT_EXCLUDED_FALLBACKS,
    FALLBACK_ORDER,
    METRIC_SPECS,
    build_fallback_summary,
    build_master_table,
    build_system_metric_relation_rows,
    filter_fallbacks,
)


SCRIPT_DIR = Path(__file__).resolve().parent
RQ3_DIR = SCRIPT_DIR.parent
DEFAULT_OUTPUT_DIR = RQ3_DIR / "results" / "latex_tables"
OUTPUT_DIGITS = 3
FALLBACK_SUMMARY_OUTPUT_DIGITS = 2
OMITTED_METRICS = {
    "kendall_perf_mean",
    "top10_overlap_mean",
    "local_optima_ratio_abs_error_mean",
}
OMITTED_RELATION_METRICS = {
    "Config Kendall",
    "Top-10% overlap",
    "Local-optima ratio error",
}
LATEX_FALLBACK_ORDER = ["1NN", "kNN", "kNNw", "loess", "rf", "dal_rf", "rf_uncertainty"]
LATEX_FALLBACK_DISPLAY = {
    "1NN": "1NN",
    "kNN": "kNN",
    "kNNw": "wkNN",
    "loess": "LOESS",
    "rf": "RF",
    "dal_rf": "DaL-RF",
    "rf_uncertainty": "U-RF",
}
SUMMARY_METRICS = [
    metric
    for metric in METRIC_SPECS
    if metric[0] != "rq2_spearman" and metric[0] not in OMITTED_METRICS
] + [
    next(metric for metric in METRIC_SPECS if metric[0] == "rq2_spearman")
]

SUMMARY_LABEL_OVERRIDES = {
    "rq2_spearman": "$\\rho_{tuner}$",
    "spearman_perf_mean": "$\\rho_{config}$",
    "kendall_perf_mean": "Config $\\tau$",
    "top5_overlap_mean": "Recall@5\\%",
    "top10_overlap_mean": "Top-10\\%",
    "auto_correlation_abs_error_mean": "ACE",
    "local_optima_ratio_abs_error_mean": "Local-opt. err.",
}

RELATION_LABEL_OVERRIDES = {
    "Config Spearman": "$\\rho_{config}$",
    "Config Kendall": "Config $\\tau$",
    "Top-5% overlap": "Recall@5\\%",
    "Top-10% overlap": "Top-10\\%",
    "Auto-corr error": "ACE",
    "Local-optima ratio error": "Local-opt. err.",
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


def fallback_sort_key(fallback: str) -> tuple[int, str]:
    if fallback in LATEX_FALLBACK_ORDER:
        return (LATEX_FALLBACK_ORDER.index(fallback), fallback)
    return (999, fallback)


def metric_label(metric: str, default_label: str) -> str:
    return SUMMARY_LABEL_OVERRIDES.get(metric, default_label)


def latex_tabular(
    headers: list[str],
    rows: list[list[str]],
    align: str,
) -> str:
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


def build_fallback_summary_table(summary: pd.DataFrame) -> str:
    summary = summary.copy()
    summary["_fallback_order"] = summary["fallback"].astype(str).map(fallback_sort_key)
    summary = summary.sort_values("_fallback_order").drop(columns="_fallback_order")

    headers = ["Fallback"] + [
        metric_label(metric, label)
        for metric, label, _direction, _digits in SUMMARY_METRICS
    ]

    rows = []
    for _, row in summary.iterrows():
        fallback = str(row["fallback"])
        rendered = [
            latex_bold(LATEX_FALLBACK_DISPLAY.get(fallback, latex_escape(fallback)))
        ]
        for metric, _label, _direction, _digits in SUMMARY_METRICS:
            rendered.append(
                fmt_mean_std(
                    row[f"{metric}_mean"],
                    row[f"{metric}_std"],
                    FALLBACK_SUMMARY_OUTPUT_DIGITS,
                )
            )
        rows.append(rendered)

    align = "l" + "r" * (len(headers) - 1)
    return latex_tabular(headers, rows, align)


def build_relation_summary(relation_rows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    relation_rows = relation_rows.copy()
    relation_rows = relation_rows[
        ~relation_rows["Metric"].astype(str).isin(OMITTED_RELATION_METRICS)
    ].reset_index(drop=True)
    relation_rows["Metric"] = relation_rows["Metric"].replace(RELATION_LABEL_OVERRIDES)
    relation_rows.loc[
        relation_rows["Better"].astype(str) == "lower",
        "Metric",
    ] = "$-$" + relation_rows.loc[
        relation_rows["Better"].astype(str) == "lower",
        "Metric",
    ].astype(str)

    for (metric, better), group in relation_rows.groupby(["Metric", "Better"], sort=False):
        spearman = group["Spearman with RQ2"].replace([np.inf, -np.inf], np.nan)
        pearson = group["Pearson with RQ2"].replace([np.inf, -np.inf], np.nan)
        rows.append(
            {
                "Metric": metric,
                "Better": better,
                "metric_order": int(group["metric_order"].min()),
                "Systems": int(spearman.notna().sum()),
                "Positive Spearman Systems": int((spearman > 0).sum()),
                "Spearman Mean": float(spearman.mean(skipna=True)),
                "Spearman Std": float(spearman.std(skipna=True)),
                "Pearson Mean": float(pearson.mean(skipna=True)),
                "Pearson Std": float(pearson.std(skipna=True)),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(["metric_order"])
        .drop(columns="metric_order")
        .reset_index(drop=True)
    )


def build_relation_bar_chart(relation_summary: pd.DataFrame) -> str:
    coordinates = []
    metric_labels = []

    for idx, (_row_idx, row) in enumerate(relation_summary.iterrows(), start=1):
        metric = str(row["Metric"])
        mean = row["Spearman Mean"]
        std = row["Spearman Std"]
        coordinates.append(f"({idx},{fmt_number(mean)}) +- (0,{fmt_number(std)})")
        metric_labels.append(metric)

    xticks = ", ".join(str(idx) for idx in range(1, len(metric_labels) + 1))
    xticklabels = ", ".join(f"{{{label}}}" for label in metric_labels)
    coord_text = "\n        ".join(coordinates)

    return "\n".join(
        [
            "\\begin{tikzpicture}",
            "\\begin{axis}[",
            "    ybar,",
            "    width=\\linewidth,",
            "    height=0.42\\linewidth,",
            "    ymin=0,",
            "    ymax=1.0,",
            "    ylabel={Mean Spearman correlation with RQ2},",
            "    xlabel={Metric quality (errors multiplied by $-1$)},",
            "    xtick={" + xticks + "},",
            "    xticklabels={" + xticklabels + "},",
            "    x tick label style={rotate=30, anchor=east},",
            "    bar width=14pt,",
            "    enlarge x limits=0.16,",
            "    grid=major,",
            "]",
            "\\addplot+[",
            "    draw=black,",
            "    fill=gray!55,",
            "    error bars/y dir=both,",
            "    error bars/y explicit,",
            "] coordinates {",
            f"        {coord_text}",
            "    };",
            "\\end{axis}",
            "\\end{tikzpicture}",
        ]
    )


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")
    print(f"[SAVE] {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate RQ3 LaTeX tables.")
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
        help="Include the 'worst' fallback in generated tables.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    excluded_fallbacks = [] if args.include_worst else list(DEFAULT_EXCLUDED_FALLBACKS)

    master_df = build_master_table(args.scales)
    filtered_df = filter_fallbacks(master_df, excluded_fallbacks)

    fallback_summary = build_fallback_summary(filtered_df)
    fallback_table = build_fallback_summary_table(fallback_summary)
    relation_rows = build_system_metric_relation_rows(filtered_df)
    relation_summary = build_relation_summary(relation_rows)
    relation_chart = build_relation_bar_chart(relation_summary)

    output_dir = args.output_dir.resolve()
    write_text(output_dir / "rq3_fallback_level_summary.tex", fallback_table)
    write_text(output_dir / "rq3_metric_rq2_bar_chart.tex", relation_chart)


if __name__ == "__main__":
    main()
