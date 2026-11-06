#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


SCRIPT_DIR = Path(__file__).resolve().parent
GUIDELINE_DIR = SCRIPT_DIR.parent
STATISTICAL_TOOLS_DIR = GUIDELINE_DIR.parent
RQ3_SCRIPTS_DIR = STATISTICAL_TOOLS_DIR / "RQ3" / "scripts"
if str(RQ3_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(RQ3_SCRIPTS_DIR))

from rq3_3_build_metric_report import build_master_table


DEFAULT_OUTPUT_DIR = GUIDELINE_DIR / "results" / "reports"
DEFAULT_EXCLUDED_FALLBACKS = ["worst"]

RAW_FEATURES = [
    ("mre_mean", "q_mre", "MRE quality", "lower"),
    ("rmse_mean", "q_rmse", "RMSE quality", "lower"),
    ("spearman_perf_mean", "q_spearman", "Config-Spearman quality", "higher"),
    ("kendall_perf_mean", "q_kendall", "Config-Kendall quality", "higher"),
    ("top5_overlap_mean", "q_top5", "Top-5 quality", "higher"),
    ("top10_overlap_mean", "q_top10", "Top-10 quality", "higher"),
    ("auto_correlation_abs_error_mean", "q_auto_corr", "Auto-corr quality", "lower"),
    (
        "local_optima_ratio_abs_error_mean",
        "q_local_optima",
        "Local-optima quality",
        "lower",
    ),
]
QUALITY_COLS = [item[1] for item in RAW_FEATURES]
EQUAL_WEIGHTS = np.ones(len(QUALITY_COLS), dtype=float) / float(len(QUALITY_COLS))

SELECTORS = {
    "MRE-only": {
        "description": "Selects the fallback with the lowest MRE, i.e., the largest `q_mre`.",
        "quality_col": "q_mre",
    },
    "RMSE-only": {
        "description": "Selects the fallback with the lowest RMSE, i.e., the largest `q_rmse`.",
        "quality_col": "q_rmse",
    },
    "Spearman-only": {
        "description": "Selects the fallback with the highest configuration-level Spearman, i.e., the largest `q_spearman`.",
        "quality_col": "q_spearman",
    },
    "Kendall-only": {
        "description": "Selects the fallback with the highest configuration-level Kendall tau, i.e., the largest `q_kendall`.",
        "quality_col": "q_kendall",
    },
    "Top5-only": {
        "description": "Selects the fallback with the highest Top-5% overlap, i.e., the largest `q_top5`.",
        "quality_col": "q_top5",
    },
    "Top10-only": {
        "description": "Selects the fallback with the highest Top-10% overlap, i.e., the largest `q_top10`.",
        "quality_col": "q_top10",
    },
    "AutoCorr-only": {
        "description": "Selects the fallback with the lowest auto-correlation error, i.e., the largest `q_auto_corr`.",
        "quality_col": "q_auto_corr",
    },
    "LocalOptima-only": {
        "description": "Selects the fallback with the lowest local-optima ratio error, i.e., the largest `q_local_optima`.",
        "quality_col": "q_local_optima",
    },
    "Equal-weight": {
        "description": "Averages all eight quality features with equal weights.",
        "weights": EQUAL_WEIGHTS,
    },
    "Pairwise-random-forest": {
        "description": (
            "Trains a random-forest pairwise ranker. For fallback A and B, the model receives "
            "`quality(A) - quality(B)` and learns whether A has higher RQ2 Spearman than B."
        ),
        "weights": None,
    },
    "LambdaRank": {
        "description": (
            "Learns a linear scoring function with LambdaRank-style pairwise gradients. "
            "Pairs with larger potential NDCG impact receive larger training weight."
        ),
        "weights": None,
    },
}

SYSTEM_ORDER = ["mysql", "postgresql", "httpd", "tomcat", "gcc", "clang", "x264", "x265"]


@dataclass
class SelectionResult:
    selector: str
    system: str
    scale: str
    selected_fallback: str
    selected_rq2: float
    selected_rank: float
    oracle_fallback: str
    oracle_rq2: float
    normalized_regret: float


@dataclass
class LambdaRankModel:
    weights: np.ndarray

    def predict(self, features: np.ndarray) -> np.ndarray:
        return features @ self.weights


def system_sort_key(system: str) -> tuple[int, str]:
    return (SYSTEM_ORDER.index(system), system) if system in SYSTEM_ORDER else (999, system)


def fmt(value: float, digits: int = 3) -> str:
    if value is None or pd.isna(value) or not np.isfinite(value):
        return "-"
    return f"{float(value):.{digits}f}"


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


def console_table(headers: list[str], rows: list[list[object]], right_align: Iterable[str] = ()) -> str:
    right_align = set(right_align)
    rendered = [[str(cell) for cell in row] for row in rows]
    widths = [
        max(len(header), *(len(row[idx]) for row in rendered))
        for idx, header in enumerate(headers)
    ]

    def line(values: list[str]) -> str:
        cells = []
        for idx, value in enumerate(values):
            cells.append(value.rjust(widths[idx]) if headers[idx] in right_align else value.ljust(widths[idx]))
        return "  ".join(cells)

    return "\n".join([line(headers), "  ".join("-" * width for width in widths), *(line(row) for row in rendered)])


def load_master_table(excluded_fallbacks: list[str], scales: list[str] | None = None) -> pd.DataFrame:
    required = {
        "system",
        "scale",
        "fallback",
        "rq2_spearman",
        *(raw_col for raw_col, _quality_col, _label, _direction in RAW_FEATURES),
    }
    df = build_master_table(scales)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Guideline selector table is missing columns: {sorted(missing)}")

    df = df[~df["fallback"].astype(str).isin(excluded_fallbacks)].copy()
    for raw_col, _quality_col, _label, _direction in RAW_FEATURES:
        df[raw_col] = pd.to_numeric(df[raw_col], errors="coerce")
    df["rq2_spearman"] = pd.to_numeric(df["rq2_spearman"], errors="coerce")
    return df.reset_index(drop=True)


def add_quality_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize each raw metric within the same system/scale.

    Every produced quality feature is in [0, 1], and larger is better.
    """
    out = df.copy()
    for raw_col, quality_col, _label, direction in RAW_FEATURES:
        pieces = []
        for _key, group in out.groupby(["system", "scale"], sort=False):
            values = group[raw_col].astype(float)
            low = values.replace([np.inf, -np.inf], np.nan).min(skipna=True)
            high = values.replace([np.inf, -np.inf], np.nan).max(skipna=True)

            if pd.isna(low) or pd.isna(high) or abs(high - low) <= 1e-12:
                quality = pd.Series(0.5, index=group.index)
            elif direction == "lower":
                quality = (high - values) / (high - low)
            else:
                quality = (values - low) / (high - low)
            pieces.append(quality.clip(0.0, 1.0))

        out[quality_col] = pd.concat(pieces).sort_index()
    return out


def supervised_rows(df: pd.DataFrame) -> pd.DataFrame:
    needed = ["rq2_spearman", *QUALITY_COLS]
    return (
        df.dropna(subset=needed)
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=needed)
        .sort_values(["system", "scale", "fallback"])
        .reset_index(drop=True)
    )


def build_pairwise_examples(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    x_rows = []
    y_rows = []

    for _key, group in df.groupby(["system", "scale"], sort=False):
        group = group.reset_index(drop=True)
        features = group[QUALITY_COLS].to_numpy(dtype=float)
        rq2 = group["rq2_spearman"].to_numpy(dtype=float)

        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                delta = rq2[i] - rq2[j]
                if abs(delta) <= 1e-12:
                    continue

                diff = features[i] - features[j]
                label = 1 if delta > 0 else 0
                x_rows.append(diff)
                y_rows.append(label)
                x_rows.append(-diff)
                y_rows.append(1 - label)

    return np.asarray(x_rows, dtype=float), np.asarray(y_rows, dtype=int)


def train_pairwise_random_forest(train_df: pd.DataFrame) -> RandomForestClassifier:
    x_train, y_train = build_pairwise_examples(train_df)
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=3,
        min_samples_leaf=2,
        random_state=0,
    )
    model.fit(x_train, y_train)
    return model


def normalized_relevance(group: pd.DataFrame) -> np.ndarray:
    rq2 = group["rq2_spearman"].to_numpy(dtype=float)
    low = float(np.min(rq2))
    high = float(np.max(rq2))
    if high - low <= 1e-12:
        return np.zeros_like(rq2)
    return (rq2 - low) / (high - low)


def dcg(relevance: np.ndarray) -> float:
    discounts = 1.0 / np.log2(np.arange(len(relevance), dtype=float) + 2.0)
    return float(np.sum(relevance * discounts))


def train_lambdarank(
    train_df: pd.DataFrame,
    epochs: int = 600,
    learning_rate: float = 0.08,
    l2: float = 0.01,
) -> LambdaRankModel:
    groups = []
    for _key, group in train_df.groupby(["system", "scale"], sort=False):
        relevance = normalized_relevance(group)
        ideal = dcg(np.sort(relevance)[::-1])
        if ideal <= 1e-12:
            continue
        features = group[QUALITY_COLS].to_numpy(dtype=float)
        groups.append((features, relevance, ideal))

    weights = np.zeros(len(QUALITY_COLS), dtype=float)
    for _epoch in range(epochs):
        gradient = np.zeros_like(weights)
        pair_count = 0

        for features, relevance, ideal in groups:
            scores = features @ weights
            order = np.argsort(-scores, kind="mergesort")
            ranks = np.empty(len(scores), dtype=int)
            ranks[order] = np.arange(len(scores))
            discounts = 1.0 / np.log2(np.arange(len(scores), dtype=float) + 2.0)

            for i in range(len(features)):
                for j in range(len(features)):
                    if relevance[i] <= relevance[j]:
                        continue

                    delta_ndcg = (
                        abs(relevance[i] - relevance[j])
                        * abs(discounts[ranks[i]] - discounts[ranks[j]])
                        / ideal
                    )
                    score_diff = float(np.clip(scores[i] - scores[j], -50.0, 50.0))
                    lambda_weight = delta_ndcg / (1.0 + np.exp(score_diff))
                    gradient += lambda_weight * (features[i] - features[j])
                    pair_count += 1

        if pair_count == 0:
            break
        gradient = gradient / pair_count - l2 * weights
        weights += learning_rate * gradient

    return LambdaRankModel(weights=weights)


def oracle_for(group: pd.DataFrame) -> pd.Series:
    return group.sort_values(["rq2_spearman", "fallback"], ascending=[False, True]).iloc[0]


def selected_rank(group: pd.DataFrame, selected_rq2: float) -> float:
    return float(1 + np.sum(group["rq2_spearman"].to_numpy(dtype=float) > selected_rq2))


def normalized_regret(group: pd.DataFrame, selected_rq2: float) -> float:
    best = float(group["rq2_spearman"].max())
    worst = float(group["rq2_spearman"].min())
    denom = best - worst
    return 0.0 if denom <= 1e-12 else float((best - selected_rq2) / denom)


def make_result(selector: str, group: pd.DataFrame, selected: pd.Series) -> SelectionResult:
    oracle = oracle_for(group)
    selected_rq2 = float(selected["rq2_spearman"])
    return SelectionResult(
        selector=selector,
        system=str(selected["system"]),
        scale=str(selected["scale"]),
        selected_fallback=str(selected["fallback"]),
        selected_rq2=selected_rq2,
        selected_rank=selected_rank(group, selected_rq2),
        oracle_fallback=str(oracle["fallback"]),
        oracle_rq2=float(oracle["rq2_spearman"]),
        normalized_regret=normalized_regret(group, selected_rq2),
    )


def select_by_weight(group: pd.DataFrame, selector: str, weights: np.ndarray) -> SelectionResult:
    scored = group.copy()
    scored["selector_score"] = scored[QUALITY_COLS].to_numpy(dtype=float) @ weights
    selected = scored.sort_values(["selector_score", "fallback"], ascending=[False, True]).iloc[0]
    return make_result(selector, group, selected)


def select_by_quality_col(group: pd.DataFrame, selector: str, quality_col: str) -> SelectionResult:
    scored = group.copy()
    scored["selector_score"] = scored[quality_col].astype(float)
    selected = scored.sort_values(["selector_score", "fallback"], ascending=[False, True]).iloc[0]
    return make_result(selector, group, selected)


def select_by_random_forest(group: pd.DataFrame, model: RandomForestClassifier) -> SelectionResult:
    group = group.copy().reset_index(drop=True)
    features = group[QUALITY_COLS].to_numpy(dtype=float)
    scores = []

    for i in range(len(group)):
        probabilities = []
        for j in range(len(group)):
            if i == j:
                continue
            diff = (features[i] - features[j]).reshape(1, -1)
            prob = model.predict_proba(diff)[0, 1]
            probabilities.append(prob)
        scores.append(float(np.mean(probabilities)))

    scored = group.copy()
    scored["selector_score"] = scores
    selected = scored.sort_values(["selector_score", "fallback"], ascending=[False, True]).iloc[0]
    return make_result("Pairwise-random-forest", group, selected)


def select_by_lambdarank(group: pd.DataFrame, model: LambdaRankModel) -> SelectionResult:
    scored = group.copy()
    features = scored[QUALITY_COLS].to_numpy(dtype=float)
    scored["selector_score"] = model.predict(features)
    selected = scored.sort_values(["selector_score", "fallback"], ascending=[False, True]).iloc[0]
    return make_result("LambdaRank", group, selected)


def evaluate_loso(
    df: pd.DataFrame,
) -> tuple[dict[str, list[SelectionResult]], list[np.ndarray], list[tuple[str, np.ndarray]]]:
    results = {selector: [] for selector in SELECTORS}
    rf_importances = []
    lambdarank_weights = []

    for test_system in sorted(df["system"].unique(), key=system_sort_key):
        train_df = df[df["system"] != test_system].copy()
        test_df = df[df["system"] == test_system].copy()
        rf_model = train_pairwise_random_forest(train_df)
        lambdarank_model = train_lambdarank(train_df)
        rf_importances.append(rf_model.feature_importances_)
        lambdarank_weights.append((test_system, lambdarank_model.weights))

        for _key, group in test_df.groupby(["system", "scale"], sort=False):
            for selector, spec in SELECTORS.items():
                if selector == "Pairwise-random-forest":
                    results[selector].append(select_by_random_forest(group, rf_model))
                elif selector == "LambdaRank":
                    results[selector].append(select_by_lambdarank(group, lambdarank_model))
                elif "quality_col" in spec:
                    results[selector].append(select_by_quality_col(group, selector, spec["quality_col"]))
                else:
                    results[selector].append(select_by_weight(group, selector, spec["weights"]))

    return results, rf_importances, lambdarank_weights


def summarize(results: list[SelectionResult]) -> dict[str, float]:
    ranks = np.asarray([row.selected_rank for row in results], dtype=float)
    regrets = np.asarray([row.normalized_regret for row in results], dtype=float)
    selected_rq2 = np.asarray([row.selected_rq2 for row in results], dtype=float)
    return {
        "systems": len(results),
        "top1_rate": float(np.mean(ranks <= 1)),
        "top2_rate": float(np.mean(ranks <= 2)),
        "mean_regret": float(np.mean(regrets)),
        "mean_selected_rq2": float(np.mean(selected_rq2)),
    }


def summary_rows(results_by_selector: dict[str, list[SelectionResult]]) -> list[dict[str, object]]:
    rows = []
    for selector, results in results_by_selector.items():
        rows.append({"selector": selector, **summarize(results)})
    return rows


def result_markdown_table(results: list[SelectionResult]) -> str:
    rows = [
        [
            row.system,
            row.selected_fallback,
            fmt(row.selected_rq2),
            fmt(row.selected_rank, digits=0),
            row.oracle_fallback,
            fmt(row.oracle_rq2),
            fmt(row.normalized_regret),
        ]
        for row in sorted(results, key=lambda item: system_sort_key(item.system))
    ]
    return markdown_table(
        ["System", "Selected", "Selected RQ2", "RQ2 rank", "Oracle", "Oracle RQ2", "Regret"],
        rows,
        right_align=["Selected RQ2", "RQ2 rank", "Oracle RQ2", "Regret"],
    )


def write_report(
    output_path: Path,
    raw_df: pd.DataFrame,
    df: pd.DataFrame,
    results_by_selector: dict[str, list[SelectionResult]],
    rf_importances: list[np.ndarray],
    lambdarank_weights: list[tuple[str, np.ndarray]],
    excluded_fallbacks: list[str],
) -> None:
    excluded = ", ".join(f"`{fallback}`" for fallback in excluded_fallbacks) if excluded_fallbacks else "`none`"
    lines = [
        "# Guideline Fallback Selector Experiment",
        "",
        "This report evaluates fallback recommendation strategies using leave-one-system-out validation.",
        "RQ2 Spearman is used only as validation signal, not as an input feature.",
        "",
        "## Data",
        "",
        f"- Input rows after exclusion: `{len(raw_df)}`",
        f"- Supervised rows: `{len(df)}`",
        f"- Systems: `{df['system'].nunique()}`",
        f"- Candidate fallbacks per system: `{int(df.groupby('system')['fallback'].nunique().median())}`",
        f"- Excluded fallbacks before normalization: {excluded}",
        "",
        "Quality features are normalized within each system and larger is always better:",
        "",
        "- `q_mre`: inverse normalized MRE",
        "- `q_rmse`: inverse normalized RMSE",
        "- `q_spearman`: normalized configuration-level Spearman",
        "- `q_kendall`: normalized configuration-level Kendall tau",
        "- `q_top5`: normalized Top-5% overlap",
        "- `q_top10`: normalized Top-10% overlap",
        "- `q_auto_corr`: inverse normalized auto-correlation absolute error",
        "- `q_local_optima`: inverse normalized local-optima ratio absolute error",
        "",
        "## Strategies",
        "",
    ]

    for selector, spec in SELECTORS.items():
        lines.extend([f"### {selector}", "", spec["description"], ""])
        if selector == "Pairwise-random-forest" and rf_importances:
            avg_importance = np.vstack(rf_importances).mean(axis=0)
            rows = [
                [label, fmt(value)]
                for (_raw, _quality, label, _direction), value in zip(RAW_FEATURES, avg_importance)
            ]
            lines.extend([
                "Average RF feature importance across LOSO folds:",
                "",
                markdown_table(["Feature", "Importance"], rows, right_align=["Importance"]),
                "",
            ])
        if selector == "LambdaRank" and lambdarank_weights:
            avg_weights = np.vstack([weights for _system, weights in lambdarank_weights]).mean(axis=0)
            rows = [
                [label, fmt(value)]
                for (_raw, _quality, label, _direction), value in zip(RAW_FEATURES, avg_weights)
            ]
            lines.extend([
                "Average learned feature weight across LOSO folds:",
                "",
                markdown_table(["Feature", "Weight"], rows, right_align=["Weight"]),
                "",
            ])
        lines.extend(["Held-out system results:", "", result_markdown_table(results_by_selector[selector]), ""])

    comparison_rows = []
    for row in summary_rows(results_by_selector):
        comparison_rows.append(
            [
                row["selector"],
                row["systems"],
                fmt(row["top1_rate"]),
                fmt(row["top2_rate"]),
                fmt(row["mean_regret"]),
                fmt(row["mean_selected_rq2"]),
            ]
        )

    lines.extend([
        "## Overall Comparison",
        "",
        "Top-1 and Top-2 measure whether the selected fallback is in the oracle best one or two fallbacks by RQ2 Spearman.",
        "Mean regret is normalized within each system, where lower is better.",
        "",
        markdown_table(
            ["Selector", "N", "Top-1", "Top-2", "Mean regret", "Mean selected RQ2"],
            comparison_rows,
            right_align=["N", "Top-1", "Top-2", "Mean regret", "Mean selected RQ2"],
        ),
        "",
    ])

    output_path.write_text("\n".join(lines), encoding="utf-8")


def save_report(
    output_path: Path,
    raw_df: pd.DataFrame,
    df: pd.DataFrame,
    results_by_selector: dict[str, list[SelectionResult]],
    rf_importances: list[np.ndarray],
    lambdarank_weights: list[tuple[str, np.ndarray]],
    excluded_fallbacks: list[str],
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_report(output_path, raw_df, df, results_by_selector, rf_importances, lambdarank_weights, excluded_fallbacks)
    return output_path


def print_summary(df: pd.DataFrame, results_by_selector: dict[str, list[SelectionResult]], report_path: Path) -> None:
    rows = []
    for row in summary_rows(results_by_selector):
        rows.append(
            [
                row["selector"],
                row["systems"],
                fmt(row["top1_rate"]),
                fmt(row["top2_rate"]),
                fmt(row["mean_regret"]),
                fmt(row["mean_selected_rq2"]),
            ]
        )

    print("\n=== Guideline Fallback Selector Experiment ===")
    print("Input: generated from current accuracy/landscape/RQ1 data")
    print(f"Systems: {df['system'].nunique()}, supervised rows: {len(df)}")
    print(
        console_table(
            ["Selector", "N", "Top-1", "Top-2", "Mean regret", "Mean selected RQ2"],
            rows,
            right_align=["N", "Top-1", "Top-2", "Mean regret", "Mean selected RQ2"],
        )
    )
    print(f"\nSaved report: {report_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run guideline fallback selector experiments.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    parser.add_argument(
        "--scale",
        action="append",
        dest="scales",
        help="Restrict output to one scale. Can be repeated. Defaults to all scales found in metrics.",
    )
    parser.add_argument(
        "--include-worst",
        action="store_true",
        help="Keep the 'worst' fallback. By default it is excluded before normalization.",
    )
    parser.add_argument(
        "--exclude-fallback",
        action="append",
        default=[],
        help="Additional fallback to exclude before normalization. Can be repeated.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    excluded = [] if args.include_worst else list(DEFAULT_EXCLUDED_FALLBACKS)
    excluded = sorted(set(excluded + args.exclude_fallback))

    raw_df = load_master_table(excluded, args.scales)
    feature_df = add_quality_features(raw_df)
    df = supervised_rows(feature_df)
    if df["system"].nunique() < 2:
        raise ValueError("Need at least two systems with RQ2 Spearman for leave-one-system-out validation.")

    results_by_selector, rf_importances, lambdarank_weights = evaluate_loso(df)
    report_path = save_report(
        args.output_dir / "guideline_selector_experiment.md",
        raw_df,
        df,
        results_by_selector,
        rf_importances,
        lambdarank_weights,
        excluded,
    )
    print_summary(df, results_by_selector, report_path)


if __name__ == "__main__":
    main()
