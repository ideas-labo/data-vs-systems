import os
from typing import List, Dict, Tuple

import pandas as pd
import numpy as np
from scipy.stats import spearmanr, kendalltau, pearsonr


# ============================================================
# Basic correlation helpers
# ============================================================

def pearson_on_system(gt_r: pd.Series, est_r: pd.Series) -> Tuple[float, int]:
    common_algos = gt_r.dropna().index.intersection(est_r.dropna().index)
    if len(common_algos) < 2:
        return float("nan"), int(len(common_algos))

    x = gt_r.loc[common_algos].values
    y = est_r.loc[common_algos].values
    r, _ = pearsonr(x, y)
    return float(r), int(len(common_algos))


def kendall_on_system(gt_r: pd.Series, est_r: pd.Series) -> Tuple[float, int]:
    common = gt_r.dropna().index.intersection(est_r.dropna().index)
    if len(common) < 2:
        return float("nan"), int(len(common))

    x = gt_r.loc[common].values
    y = est_r.loc[common].values
    tau, _ = kendalltau(x, y)
    return float(tau), int(len(common))


def spearman_on_system(gt_r: pd.Series, est_r: pd.Series) -> Tuple[float, int]:
    common_algos = gt_r.dropna().index.intersection(est_r.dropna().index)
    if len(common_algos) < 2:
        return float("nan"), int(len(common_algos))

    x = gt_r.loc[common_algos].values
    y = est_r.loc[common_algos].values
    rho, _ = spearmanr(x, y)
    return float(rho), int(len(common_algos))


# ============================================================
# CSV readers
# ============================================================

def read_wide_rank_table(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)
    df = pd.read_csv(csv_path, header=[0, 1], index_col=0)
    df.index = df.index.astype(str).str.strip()
    return df


def extract_rank_series(df_wide: pd.DataFrame, system: str) -> pd.Series:
    col = (system, "r")
    if col not in df_wide.columns:
        return pd.Series(dtype=float)

    s = df_wide[col].copy()
    s = s.apply(lambda v: np.nan if (pd.isna(v) or str(v).strip() == "") else v)
    s = s.apply(lambda v: str(v).strip().strip("[]"))
    s = pd.to_numeric(s, errors="coerce")
    s.name = system
    return s


# ============================================================
# Display configuration
# ============================================================

DISPLAY_SYSTEMS: Dict[str, str] = {
    "mysql": r"\textsc{MySQL}",
    "postgresql": r"\textsc{PostgreSQL}",
    "httpd": r"\textsc{Httpd}",
    "tomcat": r"\textsc{Tomcat}",
    "gcc": r"\textsc{Gcc}",
    "clang": r"\textsc{Clang}",
    "x264": r"\textsc{X264}",
    "x265": r"\textsc{X265}",
}

DISPLAY_FALLBACKS: Dict[str, str] = {
    "worst": "Penalty",
    "1NN": "NN",
    "kNN": "kNN",
    "kNNw": "wkNN",
    "loess": "LOESS",
    "rf": "RF",
    "dal_rf": "DaL-RF",
    "rf_uncertainty": "U-RF",
}

DISPLAY_METRICS: Dict[str, str] = {
    "pearson": "Pearson's $r$",
    "spearman": "Spearman's $\\rho$",
    "kendall": "Kendall's $\\tau$",
}

METRIC_ORDER = ["pearson", "spearman", "kendall"]
FALLBACK_ORDER = ["worst", "1NN", "kNN", "kNNw", "loess", "rf", "dal_rf", "rf_uncertainty"]


# ============================================================
# Formatting helpers
# ============================================================

def format_numeric_cell(value: float) -> str:
    if not np.isfinite(value):
        return r"\textcolor{red}{\ding{55}}"

    text = f"{value:.3f}"
    if value > 0.69:
        return rf"\cellcolor{{blue!15}} {text}"
    elif value > 0.39:
        return rf"\cellcolor{{teal!15}} {text}"
    else:
        return text


# ============================================================
# Core collection
# ============================================================

def collect_scale_rows(
    rq1_root: str,
    scale: str,
    estimations: List[str],
    systems: List[str],
    gt_path: str,
) -> pd.DataFrame:
    gt_df = read_wide_rank_table(gt_path)
    gt_ranks = {s: extract_rank_series(gt_df, s) for s in systems}

    metric_functions = {
        "pearson": pearson_on_system,
        "spearman": spearman_on_system,
        "kendall": kendall_on_system,
    }

    rows = []

    for metric_key in METRIC_ORDER:
        metric_func = metric_functions[metric_key]

        for est in FALLBACK_ORDER:
            if est not in estimations:
                continue

            est_path = os.path.join(rq1_root, scale, est, "scott_knott.csv")
            if not os.path.exists(est_path):
                continue

            est_df = read_wide_rank_table(est_path)
            est_ranks = {s: extract_rank_series(est_df, s) for s in systems}

            row = {
                "metric_key": metric_key,
                "metric": DISPLAY_METRICS[metric_key],
                "fallback_key": est,
                "fallback": DISPLAY_FALLBACKS[est],
            }

            values = []
            for sys in systems:
                score, _ = metric_func(gt_ranks[sys], est_ranks[sys])
                row[sys] = score if np.isfinite(score) else np.nan
                if np.isfinite(score):
                    values.append(score)

            row["mean"] = float(np.mean(values)) if values else np.nan
            rows.append(row)

    df = pd.DataFrame(rows)

    if not df.empty:
        df["metric_order"] = df["metric_key"].map({m: i for i, m in enumerate(METRIC_ORDER)})
        df["fallback_order"] = df["fallback_key"].map({f: i for i, f in enumerate(FALLBACK_ORDER)})
        df = df.sort_values(["metric_order", "fallback_order"]).reset_index(drop=True)
        df = df.drop(columns=["metric_order", "fallback_order"])

    return df


# ============================================================
# LaTeX export: tabular only
# ============================================================

def dataframe_to_latex_tabular(df: pd.DataFrame, systems: List[str]) -> str:
    n_sys = len(systems)
    col_spec = "l|l|" + "c|" * n_sys + "c"

    latex_lines = []
    latex_lines.append(r"\begin{tabular}{" + col_spec + r"}")
    latex_lines.append(r"\hline")

    header_cells = [
        r"\textbf{Metric}",
        r"\textbf{Fallback}",
    ]
    header_cells.extend([rf"\textbf{{{DISPLAY_SYSTEMS.get(s, s)}}}" for s in systems])
    header_cells.append(r"\textbf{Mean}")

    latex_lines.append(" & ".join(header_cells) + r" \\")
    latex_lines.append(r"\hline")

    grouped = list(df.groupby("metric_key", sort=False))

    for g_idx, (metric_key, group_df) in enumerate(grouped):
        group_df = group_df.reset_index(drop=True)
        n_rows = len(group_df)

        for i, row in group_df.iterrows():
            row_cells = []

            if i == 0:
                row_cells.append(rf"\multirow{{{n_rows}}}{{*}}{{{row['metric']}}}")
            else:
                row_cells.append("")

            row_cells.append(str(row["fallback"]))

            for sys in systems:
                row_cells.append(format_numeric_cell(row[sys]))

            row_cells.append(format_numeric_cell(row["mean"]))
            latex_lines.append(" & ".join(row_cells) + r" \\")

        latex_lines.append(r"\hline")

    latex_lines.append(r"\end{tabular}")
    return "\n".join(latex_lines)


def save_scale_tabular(
    df: pd.DataFrame,
    scale: str,
    systems: List[str],
    output_root: str = "RQ2",
) -> str:
    out_dir = os.path.join(output_root, scale)
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, "rank_correlation_table.tex")

    latex = dataframe_to_latex_tabular(df=df, systems=systems)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(latex)

    return out_path


# ============================================================
# Main
# ============================================================

def main():
    rq1_root = "RQ1"
    output_root = "RQ2"

    scales = ["large", "medium", "small"]
    estimations = ["worst", "1NN", "kNN", "kNNw", "loess", "rf", "dal_rf", "rf_uncertainty"]
    systems = ["mysql", "postgresql", "httpd", "tomcat", "gcc", "clang", "x264", "x265"]
    gt_path = "ground_truth.csv"

    for scale in scales:
        df_scale = collect_scale_rows(
            rq1_root=rq1_root,
            scale=scale,
            estimations=estimations,
            systems=systems,
            gt_path=gt_path,
        )

        out_path = save_scale_tabular(
            df=df_scale,
            scale=scale,
            systems=systems,
            output_root=output_root,
        )
        print(f"[Saved] {out_path}")


if __name__ == "__main__":
    main()