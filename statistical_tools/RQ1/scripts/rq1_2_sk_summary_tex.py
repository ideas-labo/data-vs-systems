
"""
Generate a LaTeX summary table from RQ1 Scott-Knott CSV outputs.

Input:
    RQ1/results/<scale>/<fallback>/scott_knott.csv

Output:
    RQ1/results/<scale>/summary_table.tex by default

The table shows each fallback strategy, tuner, system-level Scott-Knott rank,
and mean performance. Cells in the best Scott-Knott group are highlighted;
within that highlighted group, the best mean value is bolded.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
RQ1_DIR = SCRIPT_DIR.parent
RESULTS_DIR = RQ1_DIR / "results"

LATEX_NL = "\\\\"
INVALID_RANK_SYMBOL = "\\textcolor{red}{\\ding{55}}"
INVALID_RANK_STRATEGIES = {"worst"}

# Preferred fallback strategy order in the generated table.
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

DEFAULT_ALGO_ORDER = [
    "flash",
    "ga",
    "smac",
    "hebo",
    "bestconfig",
    "promisetune",
    "rs",
    "random_search",
    "ezr",
    "line",
]

DISPLAY_SYSTEMS = {
    "mysql": "\\textbf{\\textsc{MySQL}}",
    "postgresql": "\\textbf{\\textsc{PostgreSQL}}",
    "httpd": "\\textbf{\\textsc{Httpd}}",
    "tomcat": "\\textbf{\\textsc{Tomcat}}",
    "gcc": "\\textbf{\\textsc{Gcc}}",
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
    "randomsearch": "\\texttt{RS}",
    "random_search": "\\texttt{RS}",
    "ezr": "\\texttt{LITE}",
    "line": "\\texttt{LINE}",
}

DISPLAY_STRATEGIES = {
    "worst": "Penalty",
    "1nn": "NN",
    "knn": "kNN",
    "knnw": "wkNN",
    "loess": "LOESS",
    "rf": "RF",
    "dal_rf": "DaL-RF",
    "rf_uncertainty": "U-RF",
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


def norm_name(x: str) -> str:
    return clean_text(x).lower()


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
    Parse the numeric mean value.
    Supports:
    - 273.602
    - '273.602 (4.95)'
    - '9.194 (0.263)'
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

    # match the leading numeric value
    m = re.match(r"^\s*([+-]?\d+(?:\.\d+)?)", s)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return None
    return None


def load_scott_knott_raw(csv_path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(csv_path, header=[0, 1], index_col=0)
        if isinstance(df.columns, pd.MultiIndex):
            df.index = [norm_name(i) for i in df.index]
            return df
    except Exception:
        pass

    df = pd.read_csv(csv_path, index_col=0)
    df.index = [norm_name(i) for i in df.index]
    return df


def extract_system_metric_table(df: pd.DataFrame) -> Dict[str, Dict[str, pd.Series]]:
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
    preferred_norm = [v for v in preferred]
    value_set = set(values)
    ordered = [v for v in preferred_norm if v in value_set]
    ordered += sorted(v for v in values if v not in set(ordered))
    return ordered


def display_strategy(name: str) -> str:
    return DISPLAY_STRATEGIES.get(norm_name(name), clean_text(name).upper())


def display_algo(name: str) -> str:
    return DISPLAY_ALGOS.get(norm_name(name), f"\\texttt{{{latex_escape(clean_text(name))}}}")


def display_system(name: str) -> str:
    return DISPLAY_SYSTEMS.get(norm_name(name), f"\\textbf{{\\textsc{{{latex_escape(clean_text(name))}}}}}")


def is_invalid_rank_strategy(strategy: str) -> bool:
    return norm_name(strategy) in INVALID_RANK_STRATEGIES


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
    best_group = []
    for algo in algos:
        r = get_rank_value(metrics, algo)
        if r is not None and abs(r - min_rank) < 1e-12:
            best_group.append(algo)
    return best_group


def get_best_mean_algos_in_best_rank_group(
    system: str,
    metrics: Dict[str, pd.Series],
    algos: List[str],
    bold_tied_best_mean: bool = True
) -> List[str]:
    """
    Among the algorithms in the best-rank group, find those with the best mean value.
    - maximize for mysql/postgresql/httpd/tomcat
    - minimize for gcc/clang/x264/x265
    """
    best_group = get_best_rank_group_algos(metrics, algos)
    if not best_group:
        return []

    if not bold_tied_best_mean and len(best_group) == len(algos):
        return []

    pairs = []
    for algo in best_group:
        mean_val = get_mean_value(metrics, algo)
        if mean_val is not None:
            pairs.append((algo, mean_val))

    if not pairs:
        return []

    maximize = SYSTEM_OBJECTIVE_MAX.get(norm_name(system), False)

    if maximize:
        target = max(v for _, v in pairs)
    else:
        target = min(v for _, v in pairs)

    winners = [algo for algo, val in pairs if abs(val - target) < 1e-12]
    return winners


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


def style_rank_cell(rank_text: str, is_best_rank_group: bool, color: str) -> str:
    """
    rank cell:
    - highlight if in best-rank group
    - do NOT bold based on performance
    """
    if rank_text == "--":
        return rank_text
    return f"\\cellcolor{{{color}}}{rank_text}" if is_best_rank_group else rank_text


def style_mean_cell(mean_text: str, in_best_rank_group: bool, is_best_mean: bool, color: str) -> str:
    """
    mean cell:
    - highlight if in best-rank group
    - bold only if it is the best mean within the best-rank group
    """
    if mean_text == "--":
        return mean_text

    text = f"\\textbf{{{mean_text}}}" if is_best_mean else mean_text
    return f"\\cellcolor{{{color}}}{text}" if in_best_rank_group else text


def generate_latex_table(root_dir: Path, out_path: Path, highlight_color: str = "yellow!25") -> None:
    strategy_dirs = [
        d for d in sorted(root_dir.iterdir())
        if d.is_dir()
           and (d / "scott_knott.csv").exists()
           and "pool" not in d.name.lower()
    ]

    if not strategy_dirs:
        raise RuntimeError(f"No strategy folders with scott_knott.csv found under {root_dir}")

    tables: Dict[str, Dict[str, Dict[str, pd.Series]]] = {}
    all_systems = None
    all_algos = None

    for d in strategy_dirs:
        strategy = d.name
        raw_df = load_scott_knott_raw(d / "scott_knott.csv")
        table = extract_system_metric_table(raw_df)
        if not table:
            raise RuntimeError(f"Could not parse system/metric columns from {d / 'scott_knott.csv'}")

        tables[strategy] = table
        systems_here = set(table.keys())
        algos_here = set(raw_df.index.tolist())
        all_systems = systems_here if all_systems is None else (all_systems & systems_here)
        all_algos = algos_here if all_algos is None else (all_algos & algos_here)

    if not all_systems or not all_algos:
        raise RuntimeError("No common systems or algorithms found across strategies.")

    strategies = sort_by_preferred_order([d.name for d in strategy_dirs], DEFAULT_STRATEGY_ORDER)
    systems = sort_by_preferred_order(list(all_systems), list(DISPLAY_SYSTEMS.keys()))
    algos = sort_by_preferred_order(list(all_algos), DEFAULT_ALGO_ORDER)

    n_systems = len(systems)
    n_algos = len(algos)
    colspec = "cl" + "ll" * n_systems

    lines: List[str] = []
    lines.append("% Requires: \\usepackage{multirow}")
    lines.append("% Requires: \\usepackage[table]{xcolor}")
    lines.append("% Requires: \\usepackage{pifont}")
    lines.append(f"% Highlight color used for best-rank-group cells: {highlight_color}")
    lines.append("\\begin{tabular}{" + colspec + "}")
    lines.append("\\hline")

    header1 = [
        "\\multicolumn{1}{c|}{\\multirow{2}{*}{\\textbf{Estimation}}}",
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
            header2.extend(["\\textbf{$r$}", "\\multicolumn{1}{l|}{\\textbf{mean}}"])
        else:
            header2.extend(["\\textbf{$r$}", "\\textbf{mean}"])
    lines.append(" &\n  ".join(header2) + f" {LATEX_NL} \\hline")

    for s_idx, strategy in enumerate(strategies):
        strategy_table = tables[strategy]
        disp_strategy = display_strategy(strategy)
        invalid_rank_strategy = is_invalid_rank_strategy(strategy)

        best_rank_by_system = {}
        best_mean_winners_by_system = {}

        for system in systems:
            metrics = strategy_table[system]
            if invalid_rank_strategy:
                best_rank_by_system[system] = None
                best_mean_winners_by_system[system] = set()
            else:
                best_rank_by_system[system] = get_min_rank(metrics, algos)
                best_mean_winners_by_system[system] = set(
                    get_best_mean_algos_in_best_rank_group(
                        system,
                        metrics,
                        algos,
                    )
                )

        for a_idx, algo in enumerate(algos):
            row = []
            if a_idx == 0:
                row.append(f"\\multicolumn{{1}}{{c|}}{{\\multirow{{{n_algos}}}{{*}}{{\\textbf{{{disp_strategy}}}}}}}")
            else:
                row.append("\\multicolumn{1}{c|}{}")

            row.append(f"\\multicolumn{{1}}{{l|}}{{{display_algo(algo)}}}")

            for sys_idx, system in enumerate(systems):
                metrics = strategy_table[system]
                rank_text, mean_text, rank_value = build_cell_text(metrics, algo)

                in_best_rank_group = (
                    rank_value is not None
                    and best_rank_by_system[system] is not None
                    and abs(rank_value - best_rank_by_system[system]) < 1e-12
                )

                is_best_mean = algo in best_mean_winners_by_system[system]

                if invalid_rank_strategy:
                    rank_cell = INVALID_RANK_SYMBOL
                    mean_cell = mean_text
                else:
                    rank_cell = style_rank_cell(rank_text, in_best_rank_group, highlight_color)
                    mean_cell = style_mean_cell(mean_text, in_best_rank_group, is_best_mean, highlight_color)

                if sys_idx < n_systems - 1:
                    row.extend([rank_cell, f"\\multicolumn{{1}}{{l|}}{{{mean_cell}}}"])
                else:
                    row.extend([rank_cell, mean_cell])

            lines.append(" &\n  ".join(row) + f" {LATEX_NL}")

        lines.append("\\hline")
        if s_idx < len(strategies) - 1:
            spacer = ["\\multicolumn{1}{l}{\\textbf{}}", ""] + [""] * (2 * n_systems)
            lines.append(" &\n   ".join(spacer) + f" {LATEX_NL} \\hline")

    lines.append("\\end{tabular}")
    out_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"[Done] LaTeX table written to: {out_path}")
    print(f"[Info] Strategies: {strategies}")
    print(f"[Info] Systems: {systems}")
    print(f"[Info] Algorithms: {algos}")


def main():
    parser = argparse.ArgumentParser(description="Generate a LaTeX summary table from strategy/scott_knott.csv files.")
    parser.add_argument(
        "--root",
        type=str,
        default=str(RESULTS_DIR / "large"),
        help="Root directory containing strategy subfolders, e.g., RQ1/results/large",
    )
    parser.add_argument("--out", type=str, default=None, help="Output .tex file path")
    parser.add_argument("--highlight-color", type=str, default="blue!15", help="Cell color for best-rank-group cells")
    args = parser.parse_args()

    root_dir = Path(args.root)
    if not root_dir.exists():
        raise FileNotFoundError(f"Root directory does not exist: {root_dir}")

    out_path = Path(args.out) if args.out else (root_dir / "summary_table.tex")
    generate_latex_table(root_dir, out_path, highlight_color=args.highlight_color)


if __name__ == "__main__":
    main()
