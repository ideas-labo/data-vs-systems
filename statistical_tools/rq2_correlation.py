import os
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, kendalltau, pearsonr
from typing import Tuple, Optional, List


def pearson_on_system(gt_r: pd.Series, est_r: pd.Series):
    """
    Compute Pearson correlation between two rank vectors for overlapping algorithms.
    Return (r, n_pairs). If insufficient pairs -> (nan, n)
    """
    common_algos = gt_r.dropna().index.intersection(est_r.dropna().index)
    if len(common_algos) < 2:
        return float("nan"), int(len(common_algos))

    x = gt_r.loc[common_algos].values
    y = est_r.loc[common_algos].values
    r, _ = pearsonr(x, y)
    return float(r), int(len(common_algos))

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
    Tie-aware top-group hit:
    Return True if the best Scott-Knott group on the estimated ranking
    overlaps with the best Scott-Knott group on the ground-truth ranking.
    """
    gt = gt_r.dropna()
    est = est_r.dropna()

    if len(gt) == 0 or len(est) == 0:
        return False

    common = gt.index.intersection(est.index)
    if len(common) == 0:
        return False

    gt = gt.loc[common]
    est = est.loc[common]

    gt_best_rank = gt.min()
    est_best_rank = est.min()

    gt_best_group = set(gt[gt == gt_best_rank].index)
    est_best_group = set(est[est == est_best_rank].index)

    return len(gt_best_group.intersection(est_best_group)) > 0


def read_wide_rank_table(csv_path: str) -> pd.DataFrame:
    """
    Read wide table like:
      columns = MultiIndex (system, metric) where metric in {'r','mean',...}
      index   = Algorithm
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)
    df = pd.read_csv(csv_path, header=[0, 1], index_col=0)
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
    rho, _ = spearmanr(x, y)
    return float(rho), int(len(common_algos))


# =========================
# NDCG helpers (tie-aware)
# =========================

def rank_series_to_relevance(rank_s: pd.Series) -> pd.Series:
    """
    Convert Scott-Knott ranks to dense relevance levels.
    Smaller rank = better group = higher relevance.

    Example:
        ranks: [1, 1, 3, 4, 4]
        unique sorted groups: [1, 3, 4]
        relevance mapping: {1: 3, 3: 2, 4: 1}
        output: [3, 3, 2, 1, 1]
    """
    s = rank_s.dropna().copy()
    if len(s) == 0:
        return pd.Series(dtype=float)

    unique_groups = sorted(pd.unique(s))
    n_groups = len(unique_groups)
    mapping = {g: n_groups - i for i, g in enumerate(unique_groups)}
    rel = s.map(mapping).astype(float)
    return rel


def dcg_from_ordered_relevances(rels: np.ndarray, k: Optional[int] = None) -> float:
    """
    Standard DCG when the item order is already fixed.
    """
    if rels is None or len(rels) == 0:
        return float("nan")

    if k is None:
        k = len(rels)
    k = min(k, len(rels))
    if k <= 0:
        return float("nan")

    rels = np.asarray(rels[:k], dtype=float)
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    gains = (2.0 ** rels - 1.0)
    return float(np.sum(gains * discounts))


def tie_aware_dcg(relevances: np.ndarray, pred_scores: np.ndarray, k: Optional[int] = None) -> float:
    """
    Tie-aware DCG.

    If multiple items share the same predicted score (e.g., same Scott-Knott group),
    we average their contribution within that tied block instead of arbitrarily
    permuting them.

    Parameters
    ----------
    relevances : array-like
        Ground-truth relevance values (higher is better).
    pred_scores : array-like
        Predicted scores used for ranking (higher is better).
        Here we can use -estimated_rank, because smaller rank means better.
    k : int or None
        Cutoff for NDCG@K.

    Returns
    -------
    float
        Tie-aware DCG value.
    """
    relevances = np.asarray(relevances, dtype=float)
    pred_scores = np.asarray(pred_scores, dtype=float)

    if len(relevances) == 0:
        return float("nan")

    if k is None:
        k = len(relevances)
    k = min(k, len(relevances))
    if k <= 0:
        return float("nan")

    discounts = 1.0 / np.log2(np.arange(2, k + 2))

    # Sort by predicted score descending; ties handled blockwise
    order = np.argsort(-pred_scores, kind="mergesort")
    sorted_scores = pred_scores[order]
    sorted_rels = relevances[order]

    dcg = 0.0
    pos = 0
    n = len(sorted_scores)

    while pos < n and pos < k:
        end = pos + 1
        while end < n and sorted_scores[end] == sorted_scores[pos]:
            end += 1

        block_rels = sorted_rels[pos:end]
        block_len = end - pos
        used_len = min(block_len, k - pos)

        # Expected contribution under arbitrary tie permutation
        avg_gain = np.mean(2.0 ** block_rels - 1.0)
        dcg += avg_gain * np.sum(discounts[pos:pos + used_len])

        pos = end

    return float(dcg)


def ndcg_on_system(gt_r: pd.Series, est_r: pd.Series, k: Optional[int] = None):
    """
    Compute NDCG@K using:
      - ground-truth Scott-Knott ranks -> dense relevance levels
      - estimated Scott-Knott ranks -> predicted scores
      - tie-aware DCG for estimated ties

    Returns
    -------
    (ndcg, n_common)
    """
    common = gt_r.dropna().index.intersection(est_r.dropna().index)
    if len(common) < 1:
        return float("nan"), int(len(common))

    gt_common = gt_r.loc[common]
    est_common = est_r.loc[common]

    # Ground-truth relevance: better Scott-Knott group => higher relevance
    rel_s = rank_series_to_relevance(gt_common)
    if len(rel_s) == 0:
        return float("nan"), int(len(common))

    # Align to same algorithm order
    common = rel_s.index.intersection(est_common.index)
    rel_s = rel_s.loc[common]
    est_common = est_common.loc[common]

    # predicted score: smaller estimated rank = better => larger score
    pred_scores = -est_common.values.astype(float)
    relevances = rel_s.values.astype(float)

    dcg = tie_aware_dcg(relevances, pred_scores, k=k)

    # Ideal DCG from ground-truth relevance
    ideal_rels = np.sort(relevances)[::-1]
    idcg = dcg_from_ordered_relevances(ideal_rels, k=k)

    if not np.isfinite(dcg) or not np.isfinite(idcg) or idcg <= 0:
        return float("nan"), int(len(common))

    return float(dcg / idcg), int(len(common))


def analyze_scale_consistency(
    rq1_root: str,
    scale: str,
    estimations: list,
    systems: list,
    gt_path: str,
    ndcg_ks: Optional[List[int]] = None,
):
    gt_df = read_wide_rank_table(gt_path)
    gt_ranks = {s: extract_rank_series(gt_df, s) for s in systems}

    print(f"\n================ Scale: {scale} ================")

    metrics = ["spearman", "kendall", "pearson"]
    if ndcg_ks:
        metrics.extend([f"ndcg@{k}" for k in ndcg_ks])
    metrics.append("top1")

    for metric in metrics:
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

                elif metric == "pearson":
                    r, n = pearson_on_system(gt_ranks[sys], est_ranks[sys])
                    row[sys] = f"{r:.3f}" if np.isfinite(r) else "—"
                    if np.isfinite(r):
                        values.append(r)

                elif metric.startswith("ndcg@"):
                    k = int(metric.split("@")[1])
                    ndcg, n = ndcg_on_system(gt_ranks[sys], est_ranks[sys], k=k)
                    row[sys] = f"{ndcg:.3f}" if np.isfinite(ndcg) else "—"
                    if np.isfinite(ndcg):
                        values.append(ndcg)

                else:  # top1
                    hit = top1_hit(gt_ranks[sys], est_ranks[sys])
                    row[sys] = "✓" if hit else "✗"
                    hits.append(hit)

            if metric in ["spearman", "kendall", "pearson"] or metric.startswith("ndcg@"):
                row["mean"] = f"{np.mean(values):.3f}" if values else "—"
            else:
                row["hit_rate"] = f"{sum(hits)}/{len(hits)}"

            rows.append(row)

        pretty_print(rows, systems, metric)


def pretty_print(rows, systems, metric):
    if metric in ["spearman", "kendall", "pearson"] or metric.startswith("ndcg@"):
        cols = ["estimation"] + systems + ["mean"]
    else:
        cols = ["estimation"] + systems + ["hit_rate"]

    if not rows:
        print("(no available results)")
        return

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
    estimations = ['worst', '1NN', 'kNN', 'kNNw', 'loess', 'rf', 'rf_uncertainty', 'dal_rf']
    systems = ['mysql', 'postgresql', 'httpd', 'tomcat', 'gcc', 'clang', 'x264', 'x265']

    gt_path = os.path.join("ground_truth.csv")

    # change K here freely
    ndcg_ks = [1, 2]

    for scale in scales:
        analyze_scale_consistency(
            rq1_root=rq1_root,
            scale=scale,
            estimations=estimations,
            systems=systems,
            gt_path=gt_path,
            ndcg_ks=ndcg_ks,
        )


if __name__ == "__main__":
    main()

