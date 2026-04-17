#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

LATEX_NL = "\\\\"

# Canonical internal order
DEFAULT_ALGO_ORDER = [
    "flash",
    "ga",
    "smac",
    "hebo",
    "bestconfig",
    "promisetune",
    "rs",
]

DISPLAY_SYSTEMS = {
    "mysql": "\\textbf{\\textsc{MySQL}}",
    "postgresql": "\\textbf{\\textsc{PostgreSQL}}",
    "httpd": "\\textbf{\\textsc{Httpd}}",
    "tomcat": "\\textbf{\\textsc{Tomcat}}",
    "gcc": "\\textbf{\\textsc{GCC}}",
    "clang": "\\textbf{\\textsc{Clang}}",
    "x264": "\\textbf{\\textsc{X264}}",
    "x265": "\\textbf{\\textsc{X265}}",
}

DISPLAY_ALGOS = {
    "flash": "\\texttt{FLASH}",
    "ga": "\\texttt{GA}",
    "smac": "\\texttt{SMAC}",
    "hebo": "\\texttt{HEBO}",
    "bestconfig": "\\texttt{BestConfig}",
    "promisetune": "\\texttt{PromiseTune}",
    "rs": "\\texttt{RS}",
}

# True = maximize, False = minimize
SYSTEM_OBJECTIVE_MAX = {
    "mysql": True,
    "postgresql": True,
    "httpd": True,
    "tomcat": True,
    "gcc": False,
    "clang": False,
    "x264": False,
    "x265": False,
}


def clean_text(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def norm_name(x) -> str:
    return clean_text(x).lower()


def canonical_algo_name(name: str) -> str:
    """
    Normalize algorithm aliases into canonical internal names.
    """
    n = norm_name(name)

    alias_map = {
        "flash": "flash",
        "ga": "ga",
        "smac": "smac",
        "hebo": "hebo",

        "bestconfig": "bestconfig",
        "best_config": "bestconfig",

        "promisetune": "promisetune",
        "promise_tune": "promisetune",
        "promise": "promisetune",

        "rs": "rs",
        "randomsearch": "rs",
        "random_search": "rs",
        "random-search": "rs",
    }
    return alias_map.get(n, n)


def latex_escape(text: str) -> str:
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
    out = str(text)
    for old, new in replacements.items():
        out = out.replace(old, new)
    return out


def format_number(x) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "--"
    if isinstance(x, str):
        x = x.strip()
        return latex_escape(x) if x else "--"
    try:
        val = float(x)
    except Exception:
        return latex_escape(str(x))
    return f"{val:.2f}".rstrip("0").rstrip(".")


def try_parse_float(x) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        if isinstance(x, float) and math.isnan(x):
            return None
        return float(x)
    try:
        return float(str(x).strip())
    except Exception:
        return None


def try_parse_mean_numeric(x) -> Optional[float]:
    """
    Parse the numeric mean from either:
    - 123.45
    - '123.45'
    - '123.45 (6.78)'
    """
    if x is None:
        return None
    if isinstance(x, (int, float)):
        if isinstance(x, float) and math.isnan(x):
            return None
        return float(x)

    s = str(x).strip()
    if not s:
        return None

    m = re.match(r"^\s*([+-]?\d+(?:\.\d+)?)", s)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return None
    return None


def load_ground_truth_raw(csv_path: Path) -> pd.DataFrame:
    """
    Load CSV. Support either:
    - multi-header columns: (system, metric)
    - flat header columns: mysql_r, mysql_mean, ...
    Normalize algorithm names immediately.
    """
    try:
        df = pd.read_csv(csv_path, header=[0, 1], index_col=0)
        if isinstance(df.columns, pd.MultiIndex):
            df.index = [canonical_algo_name(i) for i in df.index]
            return df
    except Exception:
        pass

    df = pd.read_csv(csv_path, index_col=0)
    df.index = [canonical_algo_name(i) for i in df.index]
    return df


def extract_system_metric_table(df: pd.DataFrame) -> Dict[str, Dict[str, pd.Series]]:
    """
    Return:
        {
            system: {
                metric: series(index=algorithms)
            }
        }
    """
    result: Dict[str, Dict[str, pd.Series]] = {}

    if isinstance(df.columns, pd.MultiIndex):
        for c0, c1 in df.columns:
            system = norm_name(c0)
            metric = norm_name(c1)
            result.setdefault(system, {})[metric] = df[(c0, c1)]
        return result

    for col in df.columns:
        col_low = norm_name(col)
        m = re.match(r"^(.*?)[\s._-]*(r|rank|mean|std|sd|sigma)$", col_low)
        if not m:
            continue
        system = norm_name(m.group(1))
        metric = norm_name(m.group(2))
        result.setdefault(system, {})[metric] = df[col]

    return result


def sort_by_preferred_order(values: List[str], preferred: List[str]) -> List[str]:
    """
    Stable sort by preferred order first, then append extras alphabetically.
    Values are assumed already canonicalized if needed.
    """
    values = [norm_name(v) for v in values]

    seen = set()
    unique_values = []
    for v in values:
        if v not in seen:
            seen.add(v)
            unique_values.append(v)

    preferred_norm = [norm_name(v) for v in preferred]
    ordered = [v for v in preferred_norm if v in seen]
    ordered += sorted(v for v in unique_values if v not in set(ordered))
    return ordered


def display_algo(name: str) -> str:
    key = canonical_algo_name(name)
    return DISPLAY_ALGOS.get(key, f"\\texttt{{{latex_escape(clean_text(name))}}}")


def display_system(name: str) -> str:
    return DISPLAY_SYSTEMS.get(
        norm_name(name),
        f"\\textbf{{\\textsc{{{latex_escape(clean_text(name))}}}}}"
    )


def get_rank_value(metrics: Dict[str, pd.Series], algo: str) -> Optional[float]:
    for key in ["r", "rank"]:
        if key in metrics:
            return try_parse_float(metrics[key].get(algo, None))
    return None


def get_mean_value(metrics: Dict[str, pd.Series], algo: str) -> Optional[float]:
    if "mean" in metrics:
        return try_parse_mean_numeric(metrics["mean"].get(algo, None))
    return None


def get_min_rank(metrics: Dict[str, pd.Series], algos: List[str]) -> Optional[float]:
    vals = [get_rank_value(metrics, algo) for algo in algos]
    vals = [v for v in vals if v is not None]
    return min(vals) if vals else None


def get_best_rank_group_algos(metrics: Dict[str, pd.Series], algos: List[str]) -> List[str]:
    min_rank = get_min_rank(metrics, algos)
    if min_rank is None:
        return []

    out = []
    for algo in algos:
        r = get_rank_value(metrics, algo)
        if r is not None and abs(r - min_rank) < 1e-12:
            out.append(algo)
    return out


def get_best_mean_algos_in_best_rank_group(
    system: str,
    metrics: Dict[str, pd.Series],
    algos: List[str],
) -> List[str]:
    best_group = get_best_rank_group_algos(metrics, algos)
    if not best_group:
        return []

    pairs = []
    for algo in best_group:
        v = get_mean_value(metrics, algo)
        if v is not None:
            pairs.append((algo, v))

    if not pairs:
        return []

    maximize = SYSTEM_OBJECTIVE_MAX.get(norm_name(system), False)
    target = max(v for _, v in pairs) if maximize else min(v for _, v in pairs)

    return [algo for algo, val in pairs if abs(val - target) < 1e-12]


def build_cell_text(metrics: Dict[str, pd.Series], algo: str) -> Tuple[str, str, Optional[float]]:
    rank_val = None
    mean_val = None
    std_val = None

    for key in ["r", "rank"]:
        if key in metrics:
            rank_val = metrics[key].get(algo, None)
            break

    if "mean" in metrics:
        mean_val = metrics["mean"].get(algo, None)

    for key in ["std", "sd", "sigma"]:
        if key in metrics:
            std_val = metrics[key].get(algo, None)
            break

    rank_text = format_number(rank_val)

    if isinstance(mean_val, str) and "(" in mean_val and ")" in mean_val:
        mean_text = latex_escape(mean_val.strip())
    elif mean_val is not None and not (isinstance(mean_val, float) and math.isnan(mean_val)):
        mean_text = format_number(mean_val)
        if std_val is not None and not (isinstance(std_val, float) and math.isnan(std_val)):
            mean_text = f"{mean_text} ({format_number(std_val)})"
    else:
        mean_text = "--"

    return rank_text, mean_text, try_parse_float(rank_val)


def style_rank_cell(rank_text: str, in_best_rank_group: bool, color: str) -> str:
    if rank_text == "--":
        return rank_text
    return f"\\cellcolor{{{color}}}{rank_text}" if in_best_rank_group else rank_text


def style_mean_cell(mean_text: str, in_best_rank_group: bool, is_best_mean: bool, color: str) -> str:
    if mean_text == "--":
        return mean_text

    text = f"\\textbf{{{mean_text}}}" if is_best_mean else mean_text
    return f"\\cellcolor{{{color}}}{text}" if in_best_rank_group else text


def generate_ground_truth_latex(
    csv_path: Path,
    out_path: Path,
    label: str = "Ground Truth",
    highlight_color: str = "yellow!25",
) -> None:
    raw_df = load_ground_truth_raw(csv_path)
    table = extract_system_metric_table(raw_df)
    if not table:
        raise RuntimeError(f"Could not parse system/metric columns from {csv_path}")

    systems = sort_by_preferred_order(list(table.keys()), list(DISPLAY_SYSTEMS.keys()))
    algos = sort_by_preferred_order(list(raw_df.index.tolist()), DEFAULT_ALGO_ORDER)

    n_systems = len(systems)
    n_algos = len(algos)
    colspec = "cl" + "ll" * n_systems

    best_rank_by_system = {}
    best_mean_winners_by_system = {}

    for system in systems:
        metrics = table[system]
        best_rank_by_system[system] = get_min_rank(metrics, algos)
        best_mean_winners_by_system[system] = set(
            get_best_mean_algos_in_best_rank_group(system, metrics, algos)
        )

    lines: List[str] = []
    lines.append("% Requires: \\usepackage{multirow}")
    lines.append("% Requires: \\usepackage[table]{xcolor}")
    lines.append(f"% Highlight color used for best-rank-group cells: {highlight_color}")
    lines.append("\\begin{tabular}{" + colspec + "}")
    lines.append("\\hline")

    header1 = [
        "\\multicolumn{1}{c|}{\\multirow{2}{*}{\\textbf{Fallback}}}",
        "\\multicolumn{1}{c|}{\\multirow{2}{*}{\\textbf{Tuner}}}",
    ]
    for i, system in enumerate(systems):
        tail = "|" if i < len(systems) - 1 else ""
        header1.append(f"\\multicolumn{{2}}{{c{tail}}}{{{display_system(system)}}}")
    lines.append(" &\n  ".join(header1) + f" {LATEX_NL}")

    last_col = 2 + 2 * n_systems
    lines.append(f"\\cline{{3-{last_col}}}")

    header2 = ["\\multicolumn{1}{c|}{}", "\\multicolumn{1}{c|}{}"]
    for i in range(n_systems):
        if i < n_systems - 1:
            header2.extend(["\\textbf{$r$}", "\\multicolumn{1}{l|}{\\textbf{Mean (Std)}}"])
        else:
            header2.extend(["\\textbf{$r$}", "\\textbf{Mean (Std)}"])
    lines.append(" &\n  ".join(header2) + f" {LATEX_NL} \\hline")

    for a_idx, algo in enumerate(algos):
        row = []

        if a_idx == 0:
            row.append(
                f"\\multicolumn{{1}}{{c|}}{{\\multirow{{{n_algos}}}{{*}}{{\\textbf{{{latex_escape(label)}}}}}}}"
            )
        else:
            row.append("\\multicolumn{1}{c|}{}")

        row.append(f"\\multicolumn{{1}}{{l|}}{{{display_algo(algo)}}}")

        for sys_idx, system in enumerate(systems):
            metrics = table[system]
            rank_text, mean_text, rank_value = build_cell_text(metrics, algo)

            in_best_rank_group = (
                rank_value is not None
                and best_rank_by_system[system] is not None
                and abs(rank_value - best_rank_by_system[system]) < 1e-12
            )
            is_best_mean = algo in best_mean_winners_by_system[system]

            rank_cell = style_rank_cell(rank_text, in_best_rank_group, highlight_color)
            mean_cell = style_mean_cell(mean_text, in_best_rank_group, is_best_mean, highlight_color)

            if sys_idx < n_systems - 1:
                row.extend([rank_cell, f"\\multicolumn{{1}}{{l|}}{{{mean_cell}}}"])
            else:
                row.extend([rank_cell, mean_cell])

        lines.append(" &\n  ".join(row) + f" {LATEX_NL}")

    lines.append("\\hline")
    lines.append("\\end{tabular}")

    out_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"[Done] LaTeX table written to: {out_path}")
    print(f"[Info] Systems: {systems}")
    print(f"[Info] Algorithms: {algos}")


def main():
    parser = argparse.ArgumentParser(description="Generate a LaTeX table from ground_truth.csv")
    parser.add_argument("--csv", type=str, default="ground_truth.csv", help="Path to ground_truth.csv")
    parser.add_argument("--out", type=str, default=None, help="Output .tex file path")
    parser.add_argument("--label", type=str, default="Ground Truth", help="Text shown in the Estimation column")
    parser.add_argument(
        "--highlight-color",
        type=str,
        default="yellow!25",
        help="Cell color for best-rank-group cells"
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file does not exist: {csv_path}")

    out_path = Path(args.out) if args.out else csv_path.with_suffix(".tex")

    generate_ground_truth_latex(
        csv_path=csv_path,
        out_path=out_path,
        label=args.label,
        highlight_color=args.highlight_color,
    )


if __name__ == "__main__":
    main()