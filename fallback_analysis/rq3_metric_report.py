#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
STATISTICAL_TOOLS_DIR = PROJECT_ROOT / "statistical_tools"

ACCURACY_INPUT_DIR = BASE_DIR / "accuracy"
LANDSCAPE_INPUT_DIR = BASE_DIR / "landscape"
OUTPUT_DIR = BASE_DIR / "rq3"

GROUND_TRUTH_PATH = STATISTICAL_TOOLS_DIR / "ground_truth.csv"
RQ1_ROOT = STATISTICAL_TOOLS_DIR / "RQ1"

ID_COLS = ["system", "scale", "train_frac", "repeat", "fallback"]
DEFAULT_EXCLUDED_FALLBACKS = ["worst"]

METRIC_SPECS = [
    ("rq2_spearman", "RQ2 Spearman", "higher", 3),
    ("mre_mean", "MRE", "lower", 2),
    ("rmse_mean", "RMSE", "lower", 3),
    ("spearman_perf_mean", "Config Spearman", "higher", 3),
    ("kendall_perf_mean", "Config Kendall", "higher", 3),
    ("top5_overlap_mean", "Top-5% overlap", "higher", 3),
    ("top10_overlap_mean", "Top-10% overlap", "higher", 3),
    ("auto_correlation_abs_error_mean", "Auto-corr error", "lower", 3),
    ("local_optima_ratio_abs_error_mean", "Local-optima ratio error", "lower", 3),
]

FALLBACK_ORDER = [
    "worst",
    "1NN",
    "kNN",
    "kNNw",
    "loess",
    "rf",
    "rf_uncertainty",
    "dal_rf",
]

SYSTEM_ORDER = [
    "mysql",
    "postgresql",
    "httpd",
    "tomcat",
    "gcc",
    "clang",
    "x264",
    "x265",
]

ALGO_NAME_MAP = {
    "promise": "promisetune",
    "promisetune": "promisetune",
    "promiseTune": "promisetune",
    "PromiseTune": "promisetune",
    "rs": "random_search",
    "random_search": "random_search",
    "randomsearch": "random_search",
    "RandomSearch": "random_search",
    "flash": "flash",
    "ga": "ga",
    "smac": "smac",
    "hebo": "hebo",
    "bestconfig": "bestconfig",
}


def normalize_algorithm_name(name: str) -> str:
    if pd.isna(name):
        return name
    raw = str(name).strip()
    key = raw.replace("-", "_").replace(" ", "_")
    return ALGO_NAME_MAP.get(raw, ALGO_NAME_MAP.get(key, key.lower()))


def system_sort_key(system: str) -> tuple[int, str]:
    return (SYSTEM_ORDER.index(system), system) if system in SYSTEM_ORDER else (999, system)


def fallback_sort_key(fallback: str) -> tuple[int, str]:
    return (FALLBACK_ORDER.index(fallback), fallback) if fallback in FALLBACK_ORDER else (999, fallback)


def safe_spearman(x: pd.Series, y: pd.Series) -> float:
    common = x.dropna().index.intersection(y.dropna().index)
    if len(common) < 2:
        return float("nan")

    x_rank = x.loc[common].astype(float).rank(method="average")
    y_rank = y.loc[common].astype(float).rank(method="average")
    if x_rank.std() == 0 or y_rank.std() == 0:
        return float("nan")
    return float(x_rank.corr(y_rank))


def safe_pearson(x: pd.Series, y: pd.Series) -> float:
    mask = x.notna() & y.notna()
    if mask.sum() < 2:
        return float("nan")
    x_vals = x[mask].astype(float)
    y_vals = y[mask].astype(float)
    if x_vals.std() == 0 or y_vals.std() == 0:
        return float("nan")
    return float(x_vals.corr(y_vals))


def fmt(value: float, digits: int = 3) -> str:
    if value is None or pd.isna(value) or not np.isfinite(value):
        return ""
    return f"{float(value):.{digits}f}"


def fmt_mean_std(mean: float, std: float, digits: int) -> str:
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


def read_wide_rank_table(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, header=[0, 1], index_col=0)
    df.index = df.index.astype(str).map(normalize_algorithm_name)
    if df.index.duplicated().any():
        df = df.groupby(level=0).first()
    return df


def extract_rank_series(df_wide: pd.DataFrame, system: str) -> pd.Series:
    col = (system, "r")
    if col not in df_wide.columns:
        return pd.Series(dtype=float)

    series = df_wide[col].copy()
    series = series.apply(
        lambda value: (
            np.nan
            if pd.isna(value) or str(value).strip() == ""
            else str(value).strip().strip("[]")
        )
    )
    series = pd.to_numeric(series, errors="coerce")
    series.name = system
    return series


def load_metric_files(input_dir: Path, suffix: str, metric_cols: list[str]) -> pd.DataFrame:
    frames = []
    for csv_path in sorted(input_dir.glob(f"*/*_{suffix}.csv")):
        df = pd.read_csv(csv_path)
        required = set(ID_COLS + metric_cols)
        missing = required - set(df.columns)
        if missing:
            print(f"[WARN] Skip {csv_path}: missing columns {sorted(missing)}")
            continue
        frames.append(df[ID_COLS + metric_cols])

    if not frames:
        return pd.DataFrame(columns=ID_COLS + metric_cols)
    return pd.concat(frames, ignore_index=True)


def summarize_reconstruction_metrics() -> pd.DataFrame:
    accuracy_df = load_metric_files(
        ACCURACY_INPUT_DIR,
        "accuracy_metrics",
        ["mre", "rmse", "spearman_perf", "kendall_perf", "top5_overlap", "top10_overlap"],
    )
    landscape_df = load_metric_files(
        LANDSCAPE_INPUT_DIR,
        "landscape_metrics",
        ["auto_correlation_abs_error", "local_optima_ratio_abs_error"],
    )

    if accuracy_df.empty:
        raise FileNotFoundError(f"No accuracy metric CSV files found under {ACCURACY_INPUT_DIR}")
    if landscape_df.empty:
        raise FileNotFoundError(f"No landscape metric CSV files found under {LANDSCAPE_INPUT_DIR}")

    merged = accuracy_df.merge(landscape_df, on=ID_COLS, how="inner")
    if merged.empty:
        raise ValueError("Accuracy and landscape metric files have no matching rows.")

    return (
        merged.groupby(["system", "scale", "fallback"], as_index=False)
        .agg(
            train_frac_count=("train_frac", "nunique"),
            repeat_count=("repeat", "nunique"),
            row_count=("fallback", "size"),
            mre_mean=("mre", "mean"),
            mre_std=("mre", "std"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            spearman_perf_mean=("spearman_perf", "mean"),
            spearman_perf_std=("spearman_perf", "std"),
            kendall_perf_mean=("kendall_perf", "mean"),
            kendall_perf_std=("kendall_perf", "std"),
            top5_overlap_mean=("top5_overlap", "mean"),
            top5_overlap_std=("top5_overlap", "std"),
            top10_overlap_mean=("top10_overlap", "mean"),
            top10_overlap_std=("top10_overlap", "std"),
            auto_correlation_abs_error_mean=("auto_correlation_abs_error", "mean"),
            auto_correlation_abs_error_std=("auto_correlation_abs_error", "std"),
            local_optima_ratio_abs_error_mean=("local_optima_ratio_abs_error", "mean"),
            local_optima_ratio_abs_error_std=("local_optima_ratio_abs_error", "std"),
        )
        .reset_index(drop=True)
    )


def collect_rq2_spearman(
    systems: Iterable[str],
    scales: Iterable[str],
    fallbacks: Iterable[str],
) -> pd.DataFrame:
    if not GROUND_TRUTH_PATH.exists():
        raise FileNotFoundError(GROUND_TRUTH_PATH)

    gt_df = read_wide_rank_table(GROUND_TRUTH_PATH)
    systems = sorted(set(systems), key=system_sort_key)
    rows = []

    for scale in sorted(set(scales)):
        for fallback in sorted(set(fallbacks), key=fallback_sort_key):
            est_path = RQ1_ROOT / scale / fallback / "scott_knott.csv"
            if not est_path.exists():
                for system in systems:
                    rows.append(
                        {
                            "system": system,
                            "scale": scale,
                            "fallback": fallback,
                            "rq2_spearman": np.nan,
                            "rq2_common_tuners": 0,
                        }
                    )
                continue

            est_df = read_wide_rank_table(est_path)
            for system in systems:
                gt_ranks = extract_rank_series(gt_df, system)
                est_ranks = extract_rank_series(est_df, system)
                common = gt_ranks.dropna().index.intersection(est_ranks.dropna().index)
                rows.append(
                    {
                        "system": system,
                        "scale": scale,
                        "fallback": fallback,
                        "rq2_spearman": safe_spearman(gt_ranks, est_ranks),
                        "rq2_common_tuners": int(len(common)),
                    }
                )

    return pd.DataFrame(rows)


def sort_report_table(df: pd.DataFrame) -> pd.DataFrame:
    sorted_df = df.copy()
    sorted_df["_system_order"] = sorted_df["system"].map(
        {system: idx for idx, system in enumerate(SYSTEM_ORDER)}
    ).fillna(999)
    sorted_df["_fallback_order"] = sorted_df["fallback"].map(
        {fallback: idx for idx, fallback in enumerate(FALLBACK_ORDER)}
    ).fillna(999)
    sorted_df = sorted_df.sort_values(
        ["scale", "_system_order", "system", "_fallback_order", "fallback"]
    )
    return sorted_df.drop(columns=["_system_order", "_fallback_order"]).reset_index(drop=True)


def build_master_table(scales: list[str] | None = None) -> pd.DataFrame:
    metric_summary = summarize_reconstruction_metrics()
    if scales:
        metric_summary = metric_summary[metric_summary["scale"].isin(scales)].reset_index(drop=True)
        if metric_summary.empty:
            raise ValueError(f"No metric rows found for scales: {scales}")

    rq2_df = collect_rq2_spearman(
        systems=metric_summary["system"].unique(),
        scales=metric_summary["scale"].unique(),
        fallbacks=metric_summary["fallback"].unique(),
    )
    return sort_report_table(
        metric_summary.merge(rq2_df, on=["system", "scale", "fallback"], how="left")
    )


def filter_fallbacks(df: pd.DataFrame, excluded_fallbacks: list[str]) -> pd.DataFrame:
    if not excluded_fallbacks:
        return df.copy()
    return df[~df["fallback"].astype(str).isin(excluded_fallbacks)].copy()


def build_system_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for system, group in df.groupby("system", sort=False):
        row = {
            "system": system,
            "scale_count": int(group["scale"].nunique()),
            "fallback_count": int(group["fallback"].nunique()),
        }
        for metric, _label, _direction, _digits in METRIC_SPECS:
            values = group[metric].replace([np.inf, -np.inf], np.nan)
            row[f"{metric}_mean"] = float(values.mean(skipna=True))
            row[f"{metric}_std"] = float(values.std(skipna=True))
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary["_system_order"] = summary["system"].map(
        {system: idx for idx, system in enumerate(SYSTEM_ORDER)}
    ).fillna(999)
    return summary.sort_values(["_system_order", "system"]).drop(columns="_system_order").reset_index(drop=True)


def build_rank_summary(summary: pd.DataFrame) -> pd.DataFrame:
    rank_df = summary[["system"]].copy()
    for metric, label, direction, _digits in METRIC_SPECS:
        ascending = direction == "lower"
        rank_df[label] = summary[f"{metric}_mean"].rank(
            method="min",
            ascending=ascending,
            na_option="bottom",
        ).astype("Int64")
    return rank_df


def build_metric_relation_summary(master_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    rq2 = master_df["rq2_spearman"]
    for metric, label, direction, _digits in METRIC_SPECS:
        if metric == "rq2_spearman":
            continue
        values = master_df[metric]
        quality_values = -values if direction == "lower" else values
        rows.append(
            {
                "Metric": label,
                "Better": direction,
                "N": int((quality_values.notna() & rq2.notna()).sum()),
                "Spearman with RQ2": safe_spearman(quality_values, rq2),
                "Pearson with RQ2": safe_pearson(quality_values, rq2),
            }
        )
    return pd.DataFrame(rows)


def fallback_tables_markdown(master_df: pd.DataFrame) -> str:
    lines = [
        "## Per-System Fallback Metrics",
        "",
        "Each section is sorted by RQ2 Spearman, high to low.",
        "",
    ]

    for (scale, system), group in master_df.groupby(["scale", "system"], sort=False):
        table_df = group.sort_values(
            ["rq2_spearman", "fallback"],
            ascending=[False, True],
            na_position="last",
        )
        headers = [
            "Fallback",
            "RQ2 Spearman",
            "MRE",
            "RMSE",
            "Config Spearman",
            "Config Kendall",
            "Top-5%",
            "Top-10%",
            "AutoCorr Err",
            "LocalOpt Err",
        ]
        rows = []
        for _, row in table_df.iterrows():
            rows.append(
                [
                    row["fallback"],
                    fmt(row["rq2_spearman"]),
                    fmt(row["mre_mean"], 2),
                    fmt(row["rmse_mean"]),
                    fmt(row["spearman_perf_mean"]),
                    fmt(row["kendall_perf_mean"]),
                    fmt(row["top5_overlap_mean"]),
                    fmt(row["top10_overlap_mean"]),
                    fmt(row["auto_correlation_abs_error_mean"]),
                    fmt(row["local_optima_ratio_abs_error_mean"]),
                ]
            )
        lines.extend([f"### {system} / {scale}", "", markdown_table(headers, rows, right_align=headers[1:]), ""])

    return "\n".join(lines)


def system_summary_markdown(summary: pd.DataFrame) -> str:
    headers = ["System", "Fallbacks"] + [label for _metric, label, _direction, _digits in METRIC_SPECS]
    rows = []
    for _, row in summary.iterrows():
        rendered = [row["system"], int(row["fallback_count"])]
        for metric, _label, _direction, digits in METRIC_SPECS:
            rendered.append(fmt_mean_std(row[f"{metric}_mean"], row[f"{metric}_std"], digits))
        rows.append(rendered)
    return markdown_table(headers, rows, right_align=headers[1:])


def rank_summary_markdown(rank_df: pd.DataFrame) -> str:
    headers = list(rank_df.columns)
    rows = rank_df.astype(str).values.tolist()
    return markdown_table(headers, rows, right_align=headers[1:])


def relation_markdown(relation_df: pd.DataFrame) -> str:
    rows = []
    for _, row in relation_df.iterrows():
        rows.append(
            [
                row["Metric"],
                row["Better"],
                int(row["N"]),
                fmt(row["Spearman with RQ2"]),
                fmt(row["Pearson with RQ2"]),
            ]
        )
    return markdown_table(
        ["Metric", "Better", "N", "Spearman with RQ2", "Pearson with RQ2"],
        rows,
        right_align=["N", "Spearman with RQ2", "Pearson with RQ2"],
    )


def write_metric_report(
    output_path: Path,
    master_df: pd.DataFrame,
    excluded_fallbacks: list[str],
) -> None:
    filtered_df = filter_fallbacks(master_df, excluded_fallbacks)
    system_summary = build_system_summary(filtered_df)
    system_ranks = build_rank_summary(system_summary)
    relation_summary = build_metric_relation_summary(filtered_df)
    excluded = ", ".join(f"`{fallback}`" for fallback in excluded_fallbacks) if excluded_fallbacks else "`none`"

    lines = [
        "# RQ3 Fallback Metric Report",
        "",
        "This report links fallback reconstruction metrics with RQ2 tuner-ranking consistency.",
        "Metric directions: RQ2 Spearman, configuration Spearman/Kendall, and Top-k overlaps are higher-is-better; MRE, RMSE, auto-correlation error, and local-optima ratio error are lower-is-better.",
        "",
        fallback_tables_markdown(master_df),
        "",
        "## System-Level Summary",
        "",
        f"Excluded fallbacks for this summary: {excluded}.",
        "Each cell is the mean +/- std across fallback strategies after each fallback has already been averaged over train fractions and repeats.",
        "",
        system_summary_markdown(system_summary),
        "",
        "## System-Level Ranks",
        "",
        "Rank 1 is best for each metric after applying the metric direction above.",
        "",
        rank_summary_markdown(system_ranks),
        "",
        "## Overall Metric-RQ2 Relation",
        "",
        "This table summarizes whether each metric aligns with RQ2 Spearman across all system/fallback rows.",
        "",
        relation_markdown(relation_summary),
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the RQ3 fallback metric Markdown report.")
    parser.add_argument(
        "--scale",
        action="append",
        dest="scales",
        help="Restrict output to one scale. Can be repeated. Defaults to all scales found in metrics.",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Directory for RQ3 reports.")
    parser.add_argument(
        "--include-worst-summary",
        action="store_true",
        help="Include the 'worst' fallback in system-level summary tables.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    excluded = [] if args.include_worst_summary else list(DEFAULT_EXCLUDED_FALLBACKS)

    master_df = build_master_table(args.scales)
    report_path = args.output_dir / "rq3_metric_report.md"
    write_metric_report(report_path, master_df, excluded)
    print(f"[SAVE] {report_path}")


if __name__ == "__main__":
    main()
