"""
Generate RQ1 Scott-Knott rankings from fallback experiment outputs.

For each fallback strategy, scale group, and system, this script:
1. Reads each algorithm's best observed performance from every run.
2. Applies Scott-Knott ESD to rank algorithms by those run-level best values.
3. Writes per-system CSV files and a wide `scott_knott.csv` table under
   RQ1/results/.

The wide `scott_knott.csv` files are consumed by the downstream RQ1/RQ2 scripts.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr


pandas2ri.activate()
SCOTT_KNOTT_ESD = importr("ScottKnottESD")


SCRIPT_DIR = Path(__file__).resolve().parent
RQ1_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = RQ1_DIR.parent.parent

MAXIMIZE_SYSTEMS = {"h2", "mysql", "postgresql", "tomcat", "httpd"}
SCALE_GROUPS = ("small", "medium", "large")

SYSTEMS = ["mysql", "postgresql", "httpd", "tomcat", "gcc", "clang", "x264", "x265"]
ALGORITHMS = [
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
FALLBACK_STRATEGIES = [
    "worst",
    "1NN",
    "kNN",
    "kNNw",
    "loess",
    "rf",
    "rf_uncertainty",
    "dal_rf",
]

RUNS = 10
BASE_RESULTS_DIR = PROJECT_ROOT / "results_dataset"
OUTPUT_ROOT = RQ1_DIR / "results"


def calculate_iqr(data):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    return q3 - q1


def is_maximize(system: str) -> bool:
    return system in MAXIMIZE_SYSTEMS


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def has_workload_dir(base_path, estimation, algorithms, system, system_scale, probe_run=0):
    """Return True when at least one algorithm has data for this workload."""
    for algorithm in algorithms:
        path = os.path.join(
            base_path,
            f"run{probe_run}",
            estimation,
            algorithm,
            system,
            system_scale,
        )
        if os.path.isdir(path):
            return True
    return False


def compute_sk_ranks(algo_df, maximize=True, sk_pkg=None, tie_tolerance=0.0):
    """
    Compute Scott-Knott ranks for one system/fallback/scale table.

    Parameters
    ----------
    algo_df : pandas.DataFrame
        Columns are algorithms and rows are run-level best values.
    maximize : bool
        True when larger performance values are better.
    sk_pkg
        Imported R package object for ScottKnottESD.
    tie_tolerance : float
        Optional tolerance for treating near-identical algorithm means as one group.

    Returns
    -------
    dict
        Mapping from algorithm name to Scott-Knott rank. Smaller rank is better.
    """
    if sk_pkg is None:
        sk_pkg = SCOTT_KNOTT_ESD

    algo_df = algo_df.apply(pd.to_numeric, errors="coerce")
    algo_df = algo_df.dropna(axis=1, how="all").dropna(axis=0, how="all")

    if algo_df.shape[1] == 0:
        return {}

    if algo_df.shape[1] == 1:
        return {algo_df.columns[0]: 1}

    if is_degenerate_ranking_input(algo_df, tie_tolerance):
        return {algorithm: 1 for algorithm in algo_df.columns}

    try:
        r_sk = sk_pkg.sk_esd(algo_df, version="p")
        order = [i - 1 for i in r_sk[3]]
        if not maximize:
            order = order[::-1]

        ranks = list(r_sk[1])
        ordered_algorithms = [algo_df.columns[i] for i in order]
        return {algorithm: int(rank) for algorithm, rank in zip(ordered_algorithms, ranks)}

    except Exception as error:
        print("[WARN] Scott-Knott failed, fallback to mean-based tie ranking.")
        print(f"       reason: {error}")
        return compute_mean_based_ranks(algo_df, maximize=maximize)


def is_degenerate_ranking_input(algo_df, tie_tolerance=0.0) -> bool:
    """
    Detect inputs where Scott-Knott should not split algorithms.

    The `worst` fallback often creates a penalty-dominated table where nearly every
    value is identical. In that case, ranking all algorithms as group 1 is more
    faithful than forcing artificial groups from numerical noise.
    """
    nunique_per_col = algo_df.nunique(dropna=True)

    flat_values = pd.unique(algo_df.values.ravel())
    flat_values = [value for value in flat_values if pd.notna(value)]
    total_unique = len(flat_values)

    if (nunique_per_col <= 1).all():
        return True

    if total_unique <= 2:
        return True

    value_frequencies = pd.Series(algo_df.values.ravel()).dropna().value_counts(normalize=True)
    if not value_frequencies.empty and value_frequencies.iloc[0] >= 0.8:
        return True

    means = algo_df.mean(axis=0, skipna=True)
    if means.nunique(dropna=True) <= 1:
        return True

    if tie_tolerance > 0:
        mean_spread = means.max(skipna=True) - means.min(skipna=True)
        if pd.notna(mean_spread) and mean_spread <= tie_tolerance:
            return True

    return False


def compute_mean_based_ranks(algo_df, maximize=True):
    """Fallback ranking when the R Scott-Knott call fails."""
    means = algo_df.mean(axis=0, skipna=True)
    sorted_algorithms = means.sort_values(ascending=not maximize, kind="mergesort")

    ranks = {}
    current_rank = 1
    previous_value = None

    for position, (algorithm, value) in enumerate(sorted_algorithms.items()):
        if previous_value is None or np.isclose(value, previous_value, rtol=1e-10, atol=1e-12):
            ranks[algorithm] = current_rank
        else:
            current_rank = position + 1
            ranks[algorithm] = current_rank
        previous_value = value

    return ranks


def collect_by_scale_group(
    systems,
    algorithms,
    runs,
    base_path,
    estimation,
    out_root,
    sk_pkg,
):
    """
    Write RQ1/results/<scale_group>/<estimation>/{system}.csv and scott_knott.csv.

    Each per-system CSV stores the rank, mean/std, and IQR for one scale group.
    The wide scott_knott.csv stores all systems for one scale/fallback and is the
    main input for downstream RQ1/RQ2 scripts.
    """
    for scale_group in SCALE_GROUPS:
        out_dir = os.path.join(out_root, scale_group, estimation)
        ensure_dir(out_dir)

        wide_rows = {algorithm: {} for algorithm in algorithms}

        for system in systems:
            system_scale = f"{system}_{scale_group}"
            if not has_workload_dir(base_path, estimation, algorithms, system, system_scale):
                continue

            algo_data = collect_algorithm_best_values(
                algorithms=algorithms,
                runs=runs,
                base_path=base_path,
                estimation=estimation,
                system=system,
                system_scale=system_scale,
            )

            algo_df = pd.DataFrame({
                algorithm: pd.Series(values)
                for algorithm, values in algo_data.items()
            })
            tie_tolerance = 1e-2 if estimation == "worst" else 0.0
            ranks = compute_sk_ranks(
                algo_df,
                maximize=is_maximize(system),
                sk_pkg=sk_pkg,
                tie_tolerance=tie_tolerance,
            )

            rows = build_system_rows(
                scale_group=scale_group,
                system=system,
                system_scale=system_scale,
                algorithms=algorithms,
                algo_data=algo_data,
                ranks=ranks,
                wide_rows=wide_rows,
            )
            pd.DataFrame(rows).to_csv(os.path.join(out_dir, f"{system}.csv"), index=False)

        write_wide_scott_knott_table(wide_rows, systems, out_dir)


def collect_algorithm_best_values(algorithms, runs, base_path, estimation, system, system_scale):
    """Collect one best performance value per run for each algorithm."""
    algo_data = {}
    for algorithm in algorithms:
        values = []
        for run in range(runs):
            result_path = os.path.join(
                base_path,
                f"run{run}",
                estimation,
                algorithm,
                system,
                system_scale,
                "all_results.csv",
            )
            if not os.path.exists(result_path):
                continue

            df = pd.read_csv(result_path)
            performance = df.iloc[:, -1]
            performance = performance[performance >= 0]
            if performance.empty:
                continue

            best_value = performance.max() if is_maximize(system) else performance.min()
            values.append(best_value)

        algo_data[algorithm] = values

    return algo_data


def build_system_rows(scale_group, system, system_scale, algorithms, algo_data, ranks, wide_rows):
    """Build the per-system CSV rows and update the wide output table."""
    rows = []
    for algorithm in algorithms:
        values = algo_data.get(algorithm, [])
        mean_std, iqr = format_summary_stats(system, values)
        rank = ranks.get(algorithm, "")

        rows.append({
            "Scale": scale_group,
            "Workload": system_scale,
            "Algorithm": algorithm,
            "r": rank,
            "Mean (Std)": mean_std,
            "IQR": iqr,
        })

        wide_rows[algorithm][(system, "r")] = rank
        wide_rows[algorithm][(system, "mean")] = mean_std

    return rows


def format_summary_stats(system, values):
    if not values:
        return "", ""

    scale_factor = 1000 if system in {"clang", "gcc"} else 1
    mean = float(np.mean(values)) * scale_factor
    std = float(np.std(values)) * scale_factor
    iqr = float(calculate_iqr(values)) * scale_factor
    return f"{mean:.2f} ({std:.2f})", f"{iqr:.2f}"


def write_wide_scott_knott_table(wide_rows, systems, out_dir):
    wide_df = pd.DataFrame.from_dict(wide_rows, orient="index")
    wide_df.index.name = "Algorithm"
    wide_df.columns = pd.MultiIndex.from_tuples(wide_df.columns)

    ordered_columns = []
    for system in systems:
        if (system, "r") in wide_df.columns:
            ordered_columns.append((system, "r"))
        if (system, "mean") in wide_df.columns:
            ordered_columns.append((system, "mean"))

    wide_df = wide_df.reindex(columns=pd.MultiIndex.from_tuples(ordered_columns))
    wide_df.to_csv(os.path.join(out_dir, "scott_knott.csv"))


def main():
    for estimation in FALLBACK_STRATEGIES:
        print(f"\n===== Estimation Method: {estimation} =====")
        collect_by_scale_group(
            systems=SYSTEMS,
            algorithms=ALGORITHMS,
            runs=RUNS,
            base_path=BASE_RESULTS_DIR,
            estimation=estimation,
            out_root=OUTPUT_ROOT,
            sk_pkg=SCOTT_KNOTT_ESD,
        )


if __name__ == "__main__":
    main()
