import os
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, kendalltau
from typing import Tuple



def kendall_on_system(gt_r: pd.Series, est_r: pd.Series):
    common = gt_r.dropna().index.intersection(est_r.dropna().index)
    if len(common) < 2:
        return float("nan"), int(len(common))

    x = gt_r.loc[common].values
    y = est_r.loc[common].values
    tau, _ = kendalltau(x, y)
    return float(tau), int(len(common))

def top1_hit(gt_r: pd.Series, est_r: pd.Series) -> bool:
    """
    Return True if best algorithm (min rank) matches.
    """
    gt = gt_r.dropna()
    est = est_r.dropna()

    if len(gt) == 0 or len(est) == 0:
        return False

    gt_best = gt.idxmin()
    est_best = est.idxmin()
    return gt_best == est_best


def read_wide_rank_table(csv_path: str) -> pd.DataFrame:
    """
    Read wide table like:
      columns = MultiIndex (system, metric) where metric in {'r','mean',...}
      index   = Algorithm
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)
    df = pd.read_csv(csv_path, header=[0, 1], index_col=0)
    # normalize algorithm index
    df.index = df.index.astype(str).str.strip()
    return df


def extract_rank_series(df_wide: pd.DataFrame, system: str) -> pd.Series:
    """
    Return rank series for one system: index=Algorithm, values=float ranks.
    """
    col = (system, 'r')
    if col not in df_wide.columns:
        return pd.Series(dtype=float)

    s = df_wide[col].copy()
    # convert to numeric robustly (handles '[1]' or '1' or '')
    s = s.apply(lambda v: np.nan if (pd.isna(v) or str(v).strip() == "") else v)
    s = s.apply(lambda v: str(v).strip().strip('[]'))
    s = pd.to_numeric(s, errors='coerce')
    s.name = system
    return s


def spearman_on_system(gt_r: pd.Series, est_r: pd.Series):

    """
    Compute Spearman between two rank vectors for overlapping algorithms.
    Return (rho, n_pairs). If insufficient pairs -> (nan, n)
    """
    common_algos = gt_r.dropna().index.intersection(est_r.dropna().index)
    if len(common_algos) < 2:
        return float("nan"), int(len(common_algos))

    x = gt_r.loc[common_algos].values
    y = est_r.loc[common_algos].values
    rho, _p = spearmanr(x, y)
    return float(rho), int(len(common_algos))


def analyze_scale_consistency(
    rq1_root: str,
    scale: str,
    estimations: list,
    systems: list,
    gt_path: str,
):
    gt_df = read_wide_rank_table(gt_path)
    gt_ranks = {s: extract_rank_series(gt_df, s) for s in systems}

    print(f"\n================ Scale: {scale} ================")

    for metric in ["spearman", "kendall", "top1"]:
        print(f"\n---- {metric.upper()} ----")

        rows = []

        for est in estimations:
            est_path = os.path.join(rq1_root, scale, est, "scott_knott.csv")
            if not os.path.exists(est_path):
                continue

            est_df = read_wide_rank_table(est_path)
            est_ranks = {s: extract_rank_series(est_df, s) for s in systems}

            values = []
            hits = []

            row = {"estimation": est}

            for sys in systems:
                if metric == "spearman":
                    rho, n = spearman_on_system(gt_ranks[sys], est_ranks[sys])
                    row[sys] = f"{rho:.3f}" if np.isfinite(rho) else "—"
                    if np.isfinite(rho):
                        values.append(rho)

                elif metric == "kendall":
                    tau, n = kendall_on_system(gt_ranks[sys], est_ranks[sys])
                    row[sys] = f"{tau:.3f}" if np.isfinite(tau) else "—"
                    if np.isfinite(tau):
                        values.append(tau)

                else:  # top1
                    hit = top1_hit(gt_ranks[sys], est_ranks[sys])
                    row[sys] = "✓" if hit else "✗"
                    hits.append(hit)

            if metric in ["spearman", "kendall"]:
                row["mean"] = f"{np.mean(values):.3f}" if values else "—"
            else:
                row["hit_rate"] = f"{sum(hits)}/{len(hits)}"

            rows.append(row)

        pretty_print(rows, systems, metric)

def pretty_print(rows, systems, metric):
    if metric in ["spearman", "kendall"]:
        cols = ["estimation"] + systems + ["mean"]
    else:
        cols = ["estimation"] + systems + ["hit_rate"]

    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in cols}

    header = " | ".join(c.ljust(widths[c]) for c in cols)
    sep = "-+-".join("-" * widths[c] for c in cols)

    print(header)
    print(sep)

    for r in rows:
        print(" | ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols))


def main():

    rq1_root = "RQ1"
    scales = ["large", "medium", "small"]
    estimations = ['1NN', 'kNN', 'kNNw', 'loess', 'rf', 'rf_uncertainty', 'dal_rf']
    systems = ['mysql', 'postgresql', 'httpd', 'tomcat', 'x264', 'x265']

    gt_path = os.path.join("ground_truth.csv")

    for scale in scales:
        analyze_scale_consistency(
            rq1_root=rq1_root,
            scale=scale,
            estimations=estimations,
            systems=systems,
            gt_path=gt_path,
        )


if __name__ == "__main__":
    main()
