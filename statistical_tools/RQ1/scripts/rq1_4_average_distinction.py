#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Summarize RQ1 Scott-Knott ranks with average-rank views.

Input:
    RQ1/results/<scale>/<fallback>/scott_knott.csv

For each fallback strategy, this script extracts the Scott-Knott rank columns
and averages each algorithm's rank across systems. Smaller values mean the
algorithm is usually placed in a better Scott-Knott group.

The output is descriptive rather than a statistical test. It complements
`rq1_3_kendalls_w.py`: Kendall's W measures agreement among fallback
strategies, while this script shows which algorithms tend to rank higher under
each fallback strategy.

Default outputs:
    - average_ranks.csv
    - average_ranks.tex
    - average_ranks_sorted.csv
    - overall_order_by_strategy.csv
    - winner_matrix.csv
    - average_rank_heatmap.png
"""

from __future__ import annotations

import argparse
import os
import re
import tempfile
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

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

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None


ALGORITHM_ORDER = [
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

STRATEGY_ORDER = [
    "worst",
    "1NN",
    "kNN",
    "kNNw",
    "loess",
    "rf",
    "dal_rf",
    "rf_uncertainty",
]

ALGORITHM_DISPLAY_NAMES = {
    "flash": "FLASH",
    "ga": "GA",
    "smac": "SMAC",
    "hebo": "HEBO",
    "bestconfig": "BestConfig",
    "promisetune": "PromiseTune",
    "random_search": "RS",
    "ezr": "EZR",
    "line": "LINE",
}

STRATEGY_DISPLAY_NAMES = {
    "worst": "Penalty",
    "1NN": "NN",
    "kNN": "kNN",
    "kNNw": "wkNN",
    "loess": "LOESS",
    "rf": "RF",
    "dal_rf": "DaL-RF",
    "rf_uncertainty": "U-RF",
}

LATEX_ALGORITHM_ORDER = [ALGORITHM_DISPLAY_NAMES[a] for a in ALGORITHM_ORDER]
LATEX_STRATEGY_ORDER = [STRATEGY_DISPLAY_NAMES[s] for s in STRATEGY_ORDER]
INVALID_RANK_SYMBOL = r"\textcolor{red}{\ding{55}}"
INVALID_AVERAGE_RANK_COLUMNS = {"Penalty"}

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


def clean_text(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def norm_name(x: str) -> str:
    return clean_text(x).lower()


def load_scott_knott_ranks(csv_path: Path) -> pd.DataFrame:
    """
    Load one scott_knott.csv and return:
        index   = algorithms
        columns = systems
        values  = Scott-Knott rank/group labels (r)
    """
    # Try 2-row header first
    try:
        df = pd.read_csv(csv_path, header=[0, 1], index_col=0)
        if isinstance(df.columns, pd.MultiIndex):
            rank_cols = {}
            for col0, col1 in df.columns:
                system = norm_name(col0)
                metric = clean_text(col1).lower()
                if metric in {"r", "rank"}:
                    rank_cols[system] = pd.to_numeric(df[(col0, col1)], errors="coerce")

            if rank_cols:
                out = pd.DataFrame(rank_cols)
                out.index = [norm_name(i) for i in out.index]
                return out
    except Exception:
        pass

    # Fallback: flat header like mysql_r / mysql-r / mysql rank
    df = pd.read_csv(csv_path, index_col=0)
    df.index = [norm_name(i) for i in df.index]

    rank_cols = {}
    for col in df.columns:
        col_low = clean_text(col).lower()
        matched = re.match(r"^(.*?)[\s._-]*(r|rank)$", col_low)
        if matched:
            system = norm_name(matched.group(1))
            rank_cols[system] = pd.to_numeric(df[col], errors="coerce")

    if not rank_cols:
        raise ValueError(f"Could not find any rank columns in {csv_path}")

    return pd.DataFrame(rank_cols)


def collect_strategy_tables(root_dir: Path) -> Dict[str, pd.DataFrame]:
    """
    Return:
        strategy_name -> DataFrame(index=algorithms, columns=systems, values=ranks)
    """
    strategy_tables = {}

    # Load known fallback strategies in the paper/report order.
    for strategy in STRATEGY_ORDER:
        child = root_dir / strategy
        if not child.is_dir():
            print(f"[Skip] {child} not found.")
            continue

        csv_path = child / "scott_knott.csv"
        if not csv_path.exists():
            print(f"[Skip] {csv_path} not found.")
            continue

        try:
            df = load_scott_knott_ranks(csv_path)
            strategy_tables[strategy] = df
            print(f"[OK] Loaded {csv_path}")
        except Exception as e:
            print(f"[Error] Failed to load {csv_path}: {e}")

    if not strategy_tables:
        raise RuntimeError(f"No valid strategy folders found under: {root_dir}")

    return strategy_tables


def get_common_systems(strategy_tables: Dict[str, pd.DataFrame]) -> List[str]:
    common_systems = None
    for df in strategy_tables.values():
        systems = set(df.columns)
        common_systems = systems if common_systems is None else (common_systems & systems)

    if not common_systems:
        raise RuntimeError("No common systems found across all strategies.")

    ordered = [s for s in SYSTEM_ORDER if s in common_systems]
    extras = sorted(common_systems - set(SYSTEM_ORDER))
    return ordered + extras


def get_common_algorithms(strategy_tables: Dict[str, pd.DataFrame]) -> List[str]:
    common_algorithms = None
    for df in strategy_tables.values():
        algs = set(df.index)
        common_algorithms = algs if common_algorithms is None else (common_algorithms & algs)

    if not common_algorithms:
        raise RuntimeError("No common algorithms found across all strategies.")

    ordered = [alg for alg in ALGORITHM_ORDER if alg in common_algorithms]
    extras = sorted(common_algorithms - set(ALGORITHM_ORDER))
    return ordered + extras


def get_ordered_strategies(strategy_tables: Dict[str, pd.DataFrame]) -> List[str]:
    ordered = [s for s in STRATEGY_ORDER if s in strategy_tables]
    extras = sorted(set(strategy_tables.keys()) - set(STRATEGY_ORDER))
    return ordered + extras


def compute_average_ranks(
    strategy_tables: Dict[str, pd.DataFrame],
    systems: List[str],
    algorithms: List[str],
) -> pd.DataFrame:
    """
    Return:
        rows = algorithms
        cols = strategies
        values = average rank across systems

    Scott-Knott ranks are already ordinal groups, so the average is only a
    compact descriptive summary. It should not be interpreted as another
    significance test.
    """
    result = {}
    ordered_strategies = get_ordered_strategies(strategy_tables)

    for strategy in ordered_strategies:
        df = strategy_tables[strategy]
        sub = df.loc[algorithms, systems].copy()
        avg_rank = sub.mean(axis=1, skipna=True)
        result[strategy] = avg_rank

    out = pd.DataFrame(result)
    out = out.reindex(index=algorithms, columns=ordered_strategies)
    out.index.name = "algorithm"
    return out


def compute_overall_order(avg_rank_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each strategy, sort algorithms by average rank (smaller is better).
    Tied average ranks are kept in the same position group.
    """
    alg_pos = {alg: i for i, alg in enumerate(avg_rank_df.index.tolist())}
    rows = []

    for strategy in avg_rank_df.columns:
        s = avg_rank_df[strategy]
        valid = s.dropna()

        row = {"strategy": strategy}
        unique_vals = sorted(valid.unique())
        for i, val in enumerate(unique_vals, start=1):
            tied_algs = sorted(
                valid.index[valid == val].tolist(),
                key=lambda alg: alg_pos.get(alg, 10**9)
            )
            row[f"pos_{i}"] = "; ".join(tied_algs)
        rows.append(row)

    return pd.DataFrame(rows)


def compute_winner_matrix(
    strategy_tables: Dict[str, pd.DataFrame],
    systems: List[str],
    algorithms: List[str],
) -> pd.DataFrame:
    """
    For each system × strategy, record winner(s) according to the best Scott-Knott rank.
    """
    ordered_strategies = get_ordered_strategies(strategy_tables)
    winner_matrix = pd.DataFrame(index=systems, columns=ordered_strategies)

    for strategy in ordered_strategies:
        df = strategy_tables[strategy]
        sub = df.loc[algorithms, systems]

        for system in systems:
            col = sub[system].dropna()
            if col.empty:
                winner_matrix.loc[system, strategy] = ""
                continue

            best_rank = col.min()
            winners = [alg for alg in algorithms if alg in col[col == best_rank].index]
            winner_matrix.loc[system, strategy] = "; ".join(winners)

    winner_matrix.index.name = "system"
    return winner_matrix


def save_sorted_average_rank_table(avg_rank_df: pd.DataFrame, out_path: Path) -> None:
    """
    Save a human-friendly table where each strategy column is sorted by average rank.
    """
    rows = []
    alg_pos = {alg: i for i, alg in enumerate(avg_rank_df.index.tolist())}

    for strategy in avg_rank_df.columns:
        s = avg_rank_df[strategy]
        valid = s.dropna()

        ordered_algs = sorted(
            valid.index.tolist(),
            key=lambda alg: (valid.loc[alg], alg_pos.get(alg, 10**9))
        )
        ordered_vals = [valid.loc[alg] for alg in ordered_algs]

        row_block = pd.DataFrame({
            f"{strategy}_algorithm": ordered_algs,
            f"{strategy}_avg_rank": [f"{v:.2f}" for v in ordered_vals],
        })
        rows.append(row_block.reset_index(drop=True))

    merged = pd.concat(rows, axis=1)
    merged.to_csv(out_path, index=False)


def plot_heatmap(avg_rank_df: pd.DataFrame, out_path: Path) -> None:
    """
    Save a heatmap of average ranks.

    Rows are algorithms, columns are fallback strategies, and smaller values
    are better. The existing row/column order is preserved.
    """
    if plt is None:
        print("[Skip] matplotlib is not installed; skipping average_rank_heatmap.png.")
        return

    plot_df = avg_rank_df.copy()
    data = plot_df.values

    fig, ax = plt.subplots(
        figsize=(1.2 * len(plot_df.columns) + 3, 0.55 * len(plot_df.index) + 2)
    )
    im = ax.imshow(data, aspect="auto")

    ax.set_xticks(np.arange(len(plot_df.columns)))
    ax.set_yticks(np.arange(len(plot_df.index)))
    ax.set_xticklabels(plot_df.columns, rotation=45, ha="right")
    ax.set_yticklabels(plot_df.index)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            text = "" if pd.isna(val) else f"{val:.2f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=9)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Average rank (smaller is better)")

    ax.set_xlabel("Handling strategy")
    ax.set_ylabel("Algorithm")
    ax.set_title("Average Scott-Knott ranks across systems")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def analyze(root_dir: Path, out_dir: Path, make_heatmap: bool = True) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    strategy_tables = collect_strategy_tables(root_dir)
    systems = get_common_systems(strategy_tables)
    algorithms = get_common_algorithms(strategy_tables)
    ordered_strategies = get_ordered_strategies(strategy_tables)

    print(f"\n[Info] Common systems ({len(systems)}): {systems}")
    print(f"[Info] Common algorithms ({len(algorithms)}): {algorithms}")
    print(f"[Info] Strategy order used ({len(ordered_strategies)}): {ordered_strategies}")

    avg_rank_df = compute_average_ranks(strategy_tables, systems, algorithms)
    overall_order_df = compute_overall_order(avg_rank_df)
    winner_matrix_df = compute_winner_matrix(strategy_tables, systems, algorithms)

    # Raw table for downstream analysis or manual inspection.
    avg_rank_df.to_csv(out_dir / "average_ranks.csv", float_format="%.2f")

    avg_rank_df_tex = avg_rank_df.copy()
    avg_rank_df_tex = avg_rank_df_tex.rename(
        index=ALGORITHM_DISPLAY_NAMES,
        columns=STRATEGY_DISPLAY_NAMES,
    )

    save_average_rank_tex(
        avg_rank_df_tex,
        out_dir / "average_ranks.tex",
        algorithm_order=LATEX_ALGORITHM_ORDER,
        strategy_order=LATEX_STRATEGY_ORDER,
        best_color="blue!15",
        second_color="red!15",
        highlight_best=True,
        highlight_second=True,
        bold_best=False,
        bold_second=False,
        use_texttt_for_algorithms=True,
        invalid_columns=INVALID_AVERAGE_RANK_COLUMNS,
    )


    # One sorted block per strategy; useful for quickly reading each fallback.
    save_sorted_average_rank_table(avg_rank_df, out_dir / "average_ranks_sorted.csv")

    overall_order_df.to_csv(out_dir / "overall_order_by_strategy.csv", index=False)
    winner_matrix_df.to_csv(out_dir / "winner_matrix.csv")
    if make_heatmap:
        plot_heatmap(avg_rank_df, out_dir / "average_rank_heatmap.png")

    print("\n=== Average ranks ===")
    print(avg_rank_df.round(3).to_string())

    print("\n=== Overall order by strategy ===")
    print(overall_order_df.to_string(index=False))

    print(f"\nResults saved to: {out_dir}")




def latex_escape(text: str) -> str:
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


def save_average_rank_tex(
    avg_rank_df: pd.DataFrame,
    out_path: Path,
    algorithm_order: list[str] | None = None,
    strategy_order: list[str] | None = None,
    best_color: str = "blue!15",
    second_color: str = "teal!15",
    highlight_best: bool = True,
    highlight_second: bool = True,
    bold_best: bool = False,
    bold_second: bool = False,
    use_texttt_for_algorithms: bool = True,
    invalid_columns: set[str] | None = None,
) -> None:
    """
    Convert avg_rank_df directly into a LaTeX table.

    Rows: algorithms
    Columns: strategies
    Values: average ranks (smaller is better)

    Highlighting rule:
    - best: smallest value in each column
    - second: smallest value strictly larger than the best
    Ties are all highlighted.
    """
    df = avg_rank_df.copy()
    invalid_columns = set(invalid_columns or [])

    if algorithm_order is not None:
        existing_rows = [r for r in algorithm_order if r in df.index]
        remaining_rows = [r for r in df.index if r not in existing_rows]
        df = df.loc[existing_rows + remaining_rows]

    if strategy_order is not None:
        existing_cols = [c for c in strategy_order if c in df.columns]
        remaining_cols = [c for c in df.columns if c not in existing_cols]
        df = df[existing_cols + remaining_cols]

    # find best and second-best per column
    best_vals = {}
    second_vals = {}

    for col in df.columns:
        if col in invalid_columns:
            best_vals[col] = None
            second_vals[col] = None
            continue

        valid = df[col].dropna().astype(float)
        if valid.empty:
            best_vals[col] = None
            second_vals[col] = None
            continue

        unique_sorted = sorted(valid.unique())
        best_vals[col] = unique_sorted[0]
        second_vals[col] = unique_sorted[1] if len(unique_sorted) > 1 else None

    col_spec = "l" + "c" * len(df.columns)
    lines = []
    lines.append(r"% Requires: \usepackage[table]{xcolor}")
    lines.append(r"% Requires: \usepackage{pifont}")
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")

    header = [r"\textbf{Tuner}"] + [rf"\textbf{{{latex_escape(c)}}}" for c in df.columns]
    lines.append(" & ".join(header) + r" \\")
    lines.append(r"\midrule")

    for alg in df.index:
        first_cell = rf"\texttt{{{latex_escape(alg)}}}" if use_texttt_for_algorithms else latex_escape(alg)
        row_cells = [first_cell]

        for col in df.columns:
            val = df.loc[alg, col]

            if col in invalid_columns:
                cell = INVALID_RANK_SYMBOL
            elif pd.isna(val):
                cell = "--"
            else:
                val_str = f"{val:.2f}"
                is_best = best_vals[col] is not None and np.isclose(val, best_vals[col], rtol=0, atol=1e-12)
                is_second = second_vals[col] is not None and np.isclose(val, second_vals[col], rtol=0, atol=1e-12)

                if highlight_best and is_best:
                    cell = (
                        rf"\cellcolor{{{best_color}}}\textbf{{{val_str}}}"
                        if bold_best else
                        rf"\cellcolor{{{best_color}}}{val_str}"
                    )
                elif highlight_second and is_second:
                    cell = (
                        rf"\cellcolor{{{second_color}}}\textbf{{{val_str}}}"
                        if bold_second else
                        rf"\cellcolor{{{second_color}}}{val_str}"
                    )
                else:
                    cell = val_str

            row_cells.append(cell)

        lines.append(" & ".join(row_cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    out_path = Path(out_path)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def main():
    parser = argparse.ArgumentParser(
        description="Compute average algorithm ranks across strategies."
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
        default=str(RESULTS_DIR / "_avg_rank_results"),
        help="Output directory.",
    )
    parser.add_argument(
        "--skip-heatmap",
        action="store_true",
        help="Skip average_rank_heatmap.png generation.",
    )
    args = parser.parse_args()

    root_dir = Path(args.root)
    out_dir = Path(args.out)

    if not root_dir.exists():
        raise FileNotFoundError(f"Root directory does not exist: {root_dir}")

    analyze(root_dir, out_dir, make_heatmap=not args.skip_heatmap)


if __name__ == "__main__":
    main()
