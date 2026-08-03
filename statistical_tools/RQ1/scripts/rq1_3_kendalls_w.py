#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute RQ1 ranking agreement across fallback strategies.

The input is one RQ1 scale directory, for example `RQ1/results/large`.
Each fallback strategy under that directory must contain a `scott_knott.csv`
file produced by `rq1_1_scott_knott.py`.

For each system, the script builds a rank matrix:
    rows    = tuners / algorithms
    columns = fallback strategies
    values  = Scott-Knott rank groups

Kendall's W treats fallback strategies as "judges" that rank the same set of
algorithms. W is close to 1 when fallback strategies induce similar tuner
rankings, and close to 0 when their rankings disagree. Scott-Knott ranks contain
ties, so each strategy column is converted from dense rank groups to mid-ranks
before computing the tie-corrected W statistic.

Outputs are written to `RQ1/results/_kendalls_w_results` by default:
    - kendalls_w_by_system.csv
    - ranking_matrix_dense_<system>.csv
    - ranking_matrix_midranks_<system>.csv
    - kendalls_w_barplot.png
    - kendalls_w_barplot.tex
    - summary.txt
"""

from __future__ import annotations

import argparse
import os
import re
import tempfile
from pathlib import Path
from typing import Dict, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
RQ1_DIR = SCRIPT_DIR.parent
RESULTS_DIR = RQ1_DIR / "results"

MPL_CACHE_DIR = Path(tempfile.gettempdir()) / "dataeval_matplotlib_cache"
XDG_CACHE_DIR = Path(tempfile.gettempdir()) / "dataeval_cache"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
XDG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE_DIR))
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd

try:
    from scipy.stats import chi2
except ModuleNotFoundError:
    chi2 = None

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None


DEFAULT_STRATEGY_ORDER = [
    "worst",
    "1NN",
    "kNN",
    "kNNw",
    "loess",
    "rf",
    "dal_rf",
    "rf_uncertainty",
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


def sort_by_preferred_order(values, preferred):
    value_set = set(values)
    ordered = [v for v in preferred if v in value_set]
    ordered += sorted(v for v in values if v not in set(ordered))
    return ordered


def clean_text(x) -> str:
    """Normalize strings read from CSV headers/index."""
    if pd.isna(x):
        return ""
    return str(x).strip()


def normalize_algorithm_name(x: str) -> str:
    """Normalize algorithm names for alignment across strategies."""
    return clean_text(x).lower()


def normalize_system_name(x: str) -> str:
    """Normalize system names."""
    return clean_text(x).lower()


def load_scott_knott_ranks(csv_path: Path) -> pd.DataFrame:
    """
    Load one scott_knott.csv and return a DataFrame:
        index   = algorithms
        columns = systems
        values  = Scott-Knott rank/group labels (dense ranks)

    It first tries a 2-row header format:
        system names on row 1
        'r' / 'mean' on row 2

    If that fails, it falls back to flat-column parsing such as:
        mysql_r, mysql.rank, mysql r, ...
    """
    # Try MultiIndex header first
    try:
        df = pd.read_csv(csv_path, header=[0, 1], index_col=0)
        if isinstance(df.columns, pd.MultiIndex):
            rank_cols = {}
            for col0, col1 in df.columns:
                system = normalize_system_name(col0)
                metric = clean_text(col1).lower()

                if metric in {"r", "rank"}:
                    series = pd.to_numeric(df[(col0, col1)], errors="coerce")
                    if system in rank_cols:
                        raise ValueError(
                            f"Duplicate rank column detected for system '{system}' in {csv_path}"
                        )
                    rank_cols[system] = series

            if rank_cols:
                out = pd.DataFrame(rank_cols)
                out.index = [normalize_algorithm_name(x) for x in out.index]
                return out
    except Exception:
        pass

    # Fallback: flat header
    df = pd.read_csv(csv_path, index_col=0)
    df.index = [normalize_algorithm_name(x) for x in df.index]

    rank_cols = {}
    for col in df.columns:
        col_clean = clean_text(col)
        col_low = col_clean.lower()

        matched = re.match(r"^(.*?)[\s._-]*(r|rank)$", col_low)
        if matched:
            system = normalize_system_name(matched.group(1))
            rank_cols[system] = pd.to_numeric(df[col], errors="coerce")

    if not rank_cols:
        raise ValueError(
            f"Could not find any rank columns in {csv_path}. "
            f"Please check the CSV format."
        )

    return pd.DataFrame(rank_cols)


def dense_groups_to_midranks(rank_series: pd.Series) -> pd.Series:
    """
    Convert dense Scott-Knott rank groups to mid-ranks.

    Example:
        dense ranks: [1, 1, 2, 3, 4, 5, 6]
        mid-ranks:   [1.5, 1.5, 3, 4, 5, 6, 7]

    This assumes:
    - smaller rank = better
    - equal values belong to the same tie group
    """
    s = pd.to_numeric(rank_series, errors="coerce")
    if s.isna().any():
        raise ValueError("Rank series contains NaN values after numeric conversion.")

    value_counts = s.value_counts().sort_index()
    mapping = {}
    current_pos = 1

    for dense_rank, count in value_counts.items():
        start = current_pos
        end = current_pos + count - 1
        mid = (start + end) / 2.0
        mapping[dense_rank] = mid
        current_pos = end + 1

    return s.map(mapping).astype(float)


def kendalls_w_tie_corrected(rank_df: pd.DataFrame) -> Tuple[float, float, pd.DataFrame]:
    """
    Compute tie-corrected Kendall's W.

    In this script:
    - items are algorithms
    - judges are fallback strategies
    - each judge assigns Scott-Knott rank groups to the algorithms

    Parameters
    ----------
    rank_df : DataFrame
        Rows = algorithms/items
        Columns = strategies/judges
        Values = dense/group ranks from Scott-Knott

    Returns
    -------
    W : float
        Kendall's coefficient of concordance
    p_value : float
        Chi-square approximation p-value
    midrank_df : DataFrame
        Rank matrix after converting dense/group ranks to mid-ranks
    """
    # Kendall's W requires every judge to rank the same items.
    rank_df = rank_df.dropna(axis=0, how="any").copy()

    n_items, n_judges = rank_df.shape
    if n_items < 2:
        raise ValueError("Need at least 2 algorithms/items to compute Kendall's W.")
    if n_judges < 2:
        raise ValueError("Need at least 2 strategies/judges to compute Kendall's W.")

    # Convert each strategy's dense tie groups to mid-ranks before computing W.
    midrank_df = rank_df.apply(dense_groups_to_midranks, axis=0)

    # Sum of ranks per item
    rank_sums = midrank_df.sum(axis=1)

    # Mean rank sum under null
    mean_rank_sum = n_judges * (n_items + 1) / 2.0

    # Dispersion of rank sums
    S = ((rank_sums - mean_rank_sum) ** 2).sum()

    # Tie correction:
    # T_j = sum_g (t_g^3 - t_g), where t_g is the size of tie group g
    # within judge/strategy j.
    T_total = 0.0
    for col in midrank_df.columns:
        counts = midrank_df[col].value_counts()
        T_j = ((counts ** 3) - counts).sum()
        T_total += T_j

    denominator = (n_judges ** 2) * (n_items ** 3 - n_items) - n_judges * T_total
    if denominator <= 0:
        raise ValueError("Invalid denominator in Kendall's W calculation.")

    W = 12.0 * S / denominator

    # Chi-square approximation for testing whether agreement is stronger than
    # expected by chance.
    chi_square = n_judges * (n_items - 1) * W
    p_value = chi2.sf(chi_square, df=n_items - 1) if chi2 is not None else np.nan

    return float(W), float(p_value), midrank_df


def interpret_w(w: float) -> str:
    """A simple qualitative interpretation."""
    if w < 0.1:
        return "very weak"
    if w < 0.4:
        return "weak"
    if w < 0.7:
        return "moderate"
    if w < 0.9:
        return "strong"
    if w < 1:
        return "very strong"
    return "near-perfect"


def collect_strategy_tables(root_dir: Path) -> Dict[str, pd.DataFrame]:
    """
    Read all strategy/scott_knott.csv files under root_dir.
    Returns:
        strategy_name -> DataFrame(index=algorithms, columns=systems, values=ranks)
    """
    strategy_tables = {}

    for child in sorted(root_dir.iterdir()):
        if not child.is_dir():
            continue
        if 'pool' in child.name.lower():
            print(f"[Skip] Ignoring folder with 'pool' in name: {child.name}")
            continue

        csv_path = child / "scott_knott.csv"
        if not csv_path.exists():
            print(f"[Skip] {csv_path} not found.")
            continue

        try:
            df = load_scott_knott_ranks(csv_path)
            strategy_tables[child.name] = df
            print(f"[OK] Loaded {csv_path}")
        except Exception as e:
            print(f"[Error] Failed to load {csv_path}: {e}")

    if not strategy_tables:
        raise RuntimeError(f"No valid strategy folders found under: {root_dir}")

    return strategy_tables


def analyze_kendalls_w(root_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    strategy_tables = collect_strategy_tables(root_dir)
    strategies = sort_by_preferred_order(strategy_tables.keys(), DEFAULT_STRATEGY_ORDER)

    if chi2 is None:
        print("[Skip] scipy is not installed; p_value will be NaN.")

    # Use only systems that are present in every strategy table.
    common_systems = None
    for strategy, df in strategy_tables.items():
        systems = set(df.columns)
        common_systems = systems if common_systems is None else (common_systems & systems)

    if not common_systems:
        raise RuntimeError("No common systems found across all strategies.")

    common_systems = sorted(common_systems)

    results = []

    for system in common_systems:
        # Build one vector of algorithm ranks per strategy.
        per_strategy_series = []
        for strategy in strategies:
            s = strategy_tables[strategy][system].copy()
            s.name = strategy
            per_strategy_series.append(s)

        # Keep only algorithms that appear in every strategy for this system.
        rank_matrix = pd.concat(per_strategy_series, axis=1, join="inner")
        rank_matrix = rank_matrix.sort_index()

        if rank_matrix.empty:
            print(f"[Warn] No common algorithms for system: {system}")
            continue

        try:
            W, p_value, midrank_matrix = kendalls_w_tie_corrected(rank_matrix)

            results.append({
                "system": system,
                "kendalls_w": W,
                "p_value": p_value,
                "n_strategies": rank_matrix.shape[1],
                "n_algorithms": rank_matrix.shape[0],
                "interpretation": interpret_w(W),
            })

            # Save matrices so the W calculation can be audited later.
            rank_matrix.to_csv(out_dir / f"ranking_matrix_dense_{system}.csv")
            midrank_matrix.to_csv(out_dir / f"ranking_matrix_midranks_{system}.csv")

            print(
                f"[Done] {system:12s} | W = {W:.4f} | p = {p_value:.4g} "
                f"| strategies = {rank_matrix.shape[1]} | algorithms = {rank_matrix.shape[0]}"
            )

        except Exception as e:
            print(f"[Error] Failed on system '{system}': {e}")

    if not results:
        raise RuntimeError("No Kendall's W results were successfully computed.")

    results_df = pd.DataFrame(results)
    existing = set(results_df["system"].astype(str))
    ordered_systems = [s for s in SYSTEM_ORDER if s in existing]
    remaining_systems = [
        s for s in results_df["system"].astype(str).tolist()
        if s not in ordered_systems
    ]
    final_order = ordered_systems + remaining_systems

    results_df["system"] = pd.Categorical(
        results_df["system"].astype(str),
        categories=final_order,
        ordered=True,
    )
    results_df = results_df.sort_values("system").reset_index(drop=True)
    results_df.to_csv(out_dir / "kendalls_w_by_system.csv", index=False)

    # Write numeric summary files.
    mean_w = results_df["kendalls_w"].mean()
    std_w = results_df["kendalls_w"].std(ddof=1) if len(results_df) > 1 else 0.0
    min_w = results_df["kendalls_w"].min()
    max_w = results_df["kendalls_w"].max()

    with open(out_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write(f"Root directory: {root_dir}\n")
        f.write(f"Strategies used ({len(strategies)}): {', '.join(strategies)}\n")
        f.write(f"Systems analyzed ({len(results_df)}): {', '.join(results_df['system'].tolist())}\n\n")
        f.write(f"Mean Kendall's W: {mean_w:.4f}\n")
        f.write(f"Std  Kendall's W: {std_w:.4f}\n")
        f.write(f"Min  Kendall's W: {min_w:.4f}\n")
        f.write(f"Max  Kendall's W: {max_w:.4f}\n")

    # Write a quick visual summary when matplotlib is available.
    if plt is None:
        print("[Skip] matplotlib is not installed; skipping kendalls_w_barplot.png.")
    else:
        plt.figure(figsize=(10, 5))
        bars = plt.bar(results_df["system"].astype(str), results_df["kendalls_w"])
        plt.ylim(0, 1.0)
        plt.ylabel("Kendall's W")
        plt.xlabel("System")
        plt.title("Ranking Agreement Across Strategies")
        plt.grid(axis="y", linestyle="--", alpha=0.4)

        for bar, w in zip(bars, results_df["kendalls_w"]):
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                min(w + 0.02, 0.98),
                f"{w:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        plt.tight_layout()
        plt.savefig(out_dir / "kendalls_w_barplot.png", dpi=300)
        plt.close()

    print("\n=== Summary ===")
    print(results_df.to_string(index=False))
    print(f"\nMean Kendall's W: {mean_w:.4f}")
    print(f"Results saved to: {out_dir}")

    save_kendalls_w_barplot_tex(results_df, out_dir)




def latex_escape(text: str) -> str:
    """Escape LaTeX special characters."""
    if pd.isna(text):
        return ""
    text = str(text)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def save_kendalls_w_barplot_tex(results_df: pd.DataFrame, out_dir: Path,
                                filename: str = "kendalls_w_barplot.tex") -> Path:
    """
    Export Kendall's W bar plot to a standalone PGFPlots .tex file.

    Expected columns in results_df:
        - 'system'
        - 'kendalls_w'
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = results_df.copy()
    df["system"] = df["system"].astype(str)
    df["kendalls_w"] = df["kendalls_w"].astype(float)

    raw_systems = [str(s) for s in df["system"].tolist()]

    display_name_map = {
        "mysql": r"\textsc{MySQL}",
        "postgresql": r"\textsc{PostgreSQL}",
        "httpd": r"\textsc{Httpd}",
        "tomcat": r"\textsc{Tomcat}",
        "x264": r"\textsc{x264}",
        "x265": r"\textsc{x265}",
        "gcc": r"\textsc{Gcc}",
        "clang": r"\textsc{Clang}",
    }

    display_systems = [
        display_name_map.get(s.lower(), r"\textsc{" + latex_escape(s) + "}")
        for s in raw_systems
    ]

    coords = "\n".join(
        f"({latex_escape(sys)},{w:.4f})"
        for sys, w in zip(raw_systems, df["kendalls_w"])
    )

    tex = rf"""
    \documentclass[tikz]{{standalone}}
    \usepackage{{pgfplots}}
    \pgfplotsset{{compat=newest}}

    \begin{{document}}

    \begin{{tikzpicture}}
    \begin{{axis}}[
        height=6cm,
        width=9cm,
        grid style={{dashed}},
        ymajorgrids=true,
        axis x line=bottom,
        axis y line=left,
        ymin=0,
        ymax=1.0,
        enlarge x limits=0.08,
        ybar=0pt,
        bar width=16pt,
        ylabel={{Kendall's W}},
        xlabel={{System}},
        symbolic x coords={{{",".join(latex_escape(s) for s in raw_systems)}}},
        xtick=data,
        xticklabels={{{",".join(display_systems)}}},
        xticklabel style={{rotate=25, anchor=east, font=\footnotesize}},
        y axis line style={{black}},
        yticklabel style={{black}},
        nodes near coords,
        every node near coord/.append style={{font=\small, yshift=2pt}},
        nodes near coords align={{vertical}},
    ]

\addplot[
    fill=blue!30,
    draw=black,
    opacity=1
]
coordinates {{
{coords}
}};

\end{{axis}}
\end{{tikzpicture}}

\end{{document}}
""".strip() + "\n"

    tex_path = out_dir / filename
    tex_path.write_text(tex, encoding="utf-8")
    return tex_path
def main():
    parser = argparse.ArgumentParser(
        description="Compute Kendall's W across strategies from Scott-Knott ranking CSVs."
    )
    parser.add_argument(
        "--root",
        type=str,
        default=str(RESULTS_DIR / "large"),
        help="Root directory containing strategy subfolders.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(RESULTS_DIR / "_kendalls_w_results"),
        help="Output directory.",
    )
    args = parser.parse_args()

    root_dir = Path(args.root)
    out_dir = Path(args.out)

    if not root_dir.exists():
        raise FileNotFoundError(f"Root directory does not exist: {root_dir}")

    analyze_kendalls_w(root_dir, out_dir)


if __name__ == "__main__":
    main()
