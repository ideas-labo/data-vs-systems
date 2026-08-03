import os
from typing import List, Dict, Tuple

import pandas as pd
import numpy as np
from scipy.stats import spearmanr, kendalltau, pearsonr

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RQ2_DIR = os.path.dirname(SCRIPT_DIR)
STATISTICAL_TOOLS_DIR = os.path.dirname(RQ2_DIR)
PROJECT_ROOT = os.path.dirname(STATISTICAL_TOOLS_DIR)
RESULTS_SYSTEM_DIR = os.path.join(PROJECT_ROOT, "results_system")
RQ2_RESULTS_DIR = os.path.join(RQ2_DIR, "results")

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
    "ezr": "ezr",
    "EZR": "ezr",
    "line": "line",
    "LINE": "line",
}

EXPECTED_ALGORITHMS = [
    "flash",
    "ga",
    "smac",
    "hebo",
    "bestconfig",
    "promisetune",
    "random_search",
    "ezr",
    "line",
]


def normalize_algorithm_name(name: str) -> str:
    """
    Normalize algorithm names across ground-truth and fallback results.
    """
    if pd.isna(name):
        return name

    raw = str(name).strip()
    key = raw.replace("-", "_").replace(" ", "_")

    return ALGO_NAME_MAP.get(
        raw,
        ALGO_NAME_MAP.get(key, ALGO_NAME_MAP.get(key.lower(), key.lower())),
    )


def rank_overlap_details(gt_r: pd.Series, est_r: pd.Series) -> Tuple[List[str], List[str]]:
    """
    Return the overlapping algorithms used by a rank correlation and the
    expected algorithms that are missing from that overlap.
    """
    common = gt_r.dropna().index.intersection(est_r.dropna().index)
    common_list = sorted(common.astype(str).tolist())
    common_set = set(common_list)
    missing_expected = [algo for algo in EXPECTED_ALGORITHMS if algo not in common_set]
    return common_list, missing_expected
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
    if len(pd.unique(x)) < 2 or len(pd.unique(y)) < 2:
        return float("nan"), int(len(common_algos))

    rho, _ = spearmanr(x, y)
    return float(rho), int(len(common_algos))


# ============================================================
# CSV readers
# ============================================================

def read_wide_rank_table(csv_path: str) -> pd.DataFrame:
    """
    Read wide table like:
      columns = MultiIndex (system, metric) where metric in {'r','mean',...}
      index   = Algorithm

    Also normalize algorithm names so that:
      promise -> promisetune
      rs      -> random_search
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path, header=[0, 1], index_col=0)

    # normalize algorithm names
    df.index = df.index.astype(str).map(normalize_algorithm_name)

    # if duplicated names appear after normalization, keep the first valid row
    if df.index.duplicated().any():
        df = df.groupby(level=0).first()

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

METRIC_ORDER = ["spearman"]
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
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    gt_df = read_wide_rank_table(gt_path)
    gt_ranks = {s: extract_rank_series(gt_df, s) for s in systems}

    metric_functions = {
        "pearson": pearson_on_system,
        "spearman": spearman_on_system,
        "kendall": kendall_on_system,
    }

    rows = []
    overlap_rows = []

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

                if metric_key == METRIC_ORDER[0]:
                    common_algos, missing_expected = rank_overlap_details(
                        gt_ranks[sys],
                        est_ranks[sys],
                    )
                    overlap_rows.append({
                        "scale": scale,
                        "fallback_key": est,
                        "fallback": DISPLAY_FALLBACKS[est],
                        "system": sys,
                        "n_algorithms": len(common_algos),
                        "expected_algorithms": len(EXPECTED_ALGORITHMS),
                        "algorithms_used": ",".join(common_algos),
                        "missing_expected_algorithms": ",".join(missing_expected),
                    })

                    if missing_expected:
                        print(
                            f"[WARN] {scale}/{est}/{sys}: rank correlation uses "
                            f"{len(common_algos)}/{len(EXPECTED_ALGORITHMS)} expected algorithms; "
                            f"missing: {', '.join(missing_expected)}"
                        )

            row["mean"] = float(np.mean(values)) if values else np.nan
            rows.append(row)

    df = pd.DataFrame(rows)

    if not df.empty:
        df["metric_order"] = df["metric_key"].map({m: i for i, m in enumerate(METRIC_ORDER)})
        df["fallback_order"] = df["fallback_key"].map({f: i for i, f in enumerate(FALLBACK_ORDER)})
        df = df.sort_values(["metric_order", "fallback_order"]).reset_index(drop=True)
        df = df.drop(columns=["metric_order", "fallback_order"])

    return df, pd.DataFrame(overlap_rows)


# ============================================================
# LaTeX export: tabular only
# ============================================================

def dataframe_to_latex_tabular(df: pd.DataFrame, systems: List[str]) -> str:
    df = df[df["metric_key"] == "spearman"].copy()
    n_sys = len(systems)
    col_spec = "l|" + "c|" * n_sys + "c"

    latex_lines = []
    latex_lines.append(r"\begin{tabular}{" + col_spec + r"}")
    latex_lines.append(r"\hline")

    header_cells = [
        r"\textbf{Fallback}",
    ]
    header_cells.extend([rf"\textbf{{{DISPLAY_SYSTEMS.get(s, s)}}}" for s in systems])
    header_cells.append(r"\textbf{Mean}")

    latex_lines.append(" & ".join(header_cells) + r" \\")
    latex_lines.append(r"\hline")

    for _, row in df.iterrows():
        row_cells = [str(row["fallback"])]
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
    output_root: str = RQ2_RESULTS_DIR,
) -> str:
    out_dir = os.path.join(output_root, scale)
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, "rank_correlation_table.tex")

    latex = dataframe_to_latex_tabular(df=df, systems=systems)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(latex)

    return out_path


def save_scale_algorithm_overlap(
    overlap_df: pd.DataFrame,
    scale: str,
    output_root: str = RQ2_RESULTS_DIR,
) -> str:
    out_dir = os.path.join(output_root, scale)
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, "rank_correlation_algorithm_overlap.csv")
    overlap_df.to_csv(out_path, index=False)
    return out_path


# ============================================================
# Main
# ============================================================

def main():
    rq1_root = os.path.join(STATISTICAL_TOOLS_DIR, "RQ1", "results")
    output_root = RQ2_RESULTS_DIR

    scales = ["large", "medium", "small"]
    estimations = ["worst", "1NN", "kNN", "kNNw", "loess", "rf", "dal_rf", "rf_uncertainty"]
    systems = ["mysql", "postgresql", "httpd", "tomcat", "gcc", "clang", "x264", "x265"]
    gt_path = os.path.join(RESULTS_SYSTEM_DIR, "ground_truth.csv")

    for scale in scales:
        df_scale, overlap_df = collect_scale_rows(
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

        overlap_path = save_scale_algorithm_overlap(
            overlap_df=overlap_df,
            scale=scale,
            output_root=output_root,
        )
        print(f"[Saved] {overlap_path}")


if __name__ == "__main__":
    main()
