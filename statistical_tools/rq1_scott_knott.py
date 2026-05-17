import os
import numpy as np
from rpy2.robjects import r, pandas2ri
import pandas as pd
from rpy2.robjects.packages import importr
from collections import Counter

pandas2ri.activate()
sk = importr('ScottKnottESD')


MAXIMIZE_SYSTEMS = {'h2', 'mysql', 'postgresql', 'tomcat', 'httpd'}
SCALE_SUFFIXES = ("_small", "_medium", "_large")


def parse_system_and_scale(system_scale: str):
    """
    system_scale: e.g., 'httpd_large', 'postgresql_small'
    return: (system, scale_group) where scale_group in {'small','medium','large'}
    """
    for suf in SCALE_SUFFIXES:
        if system_scale.endswith(suf):
            scale_group = suf[1:]  # remove leading '_'
            system = system_scale[: -len(suf)]
            return system, scale_group
    return None, None


def calculate_iqr(data):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    return q3 - q1


def is_maximize(system: str) -> bool:
    return system in MAXIMIZE_SYSTEMS


def safe_mkdir(p: str):
    os.makedirs(p, exist_ok=True)


def list_workloads(base_path, estimation, algo, any_system, probe_run=0):
    """
    返回该 system 下的 workload 目录名列表（如 httpd_large 等）
    """
    p = os.path.join(base_path, f'run{probe_run}', estimation, algo, any_system)
    if not os.path.isdir(p):
        return []
    return sorted([d for d in os.listdir(p) if os.path.isdir(os.path.join(p, d))])


# def compute_sk_ranks(algo_df, maximize=True, sk=None):
#     """
#     algo_df: DataFrame, col=algo, row=runs best
#     returns: dict algo->rank
#     """
#     algo_df = algo_df.dropna(axis=1, how='all')
#     if algo_df.shape[1] == 0:
#         return {}
#
#     if (algo_df.nunique(dropna=True) <= 1).any():
#         return {a: 1 for a in algo_df.columns}
#
#     r_sk = sk.sk_esd(algo_df, version='p')
#     order = [i - 1 for i in r_sk[3]]
#     if not maximize:
#         order = order[::-1]
#
#     ranks = list(r_sk[1])
#     ordered_algos = [algo_df.columns[i] for i in order]
#     return {a: int(r) for a, r in zip(ordered_algos, ranks)}


def compute_sk_ranks(algo_df, maximize=True, sk=None, tie_tolerance=0.0):
    """
    algo_df: DataFrame, col=algo, row=runs best
    returns: dict algo->rank
    """
    # 保证是数值
    algo_df = algo_df.apply(pd.to_numeric, errors="coerce")

    # 去掉全空列 / 全空行
    algo_df = algo_df.dropna(axis=1, how='all').dropna(axis=0, how='all')

    if algo_df.shape[1] == 0:
        return {}

    # 只有一个算法，直接给 rank 1
    if algo_df.shape[1] == 1:
        return {algo_df.columns[0]: 1}

    # -----------------------------
    # 1) 明显退化：整张表几乎没有区分度
    # -----------------------------

    # 每列唯一值个数
    nunique_per_col = algo_df.nunique(dropna=True)

    # 全表唯一值个数
    flat_vals = pd.unique(algo_df.values.ravel())
    flat_vals = [x for x in flat_vals if pd.notna(x)]
    total_unique = len(flat_vals)

    # 条件A：所有列都几乎是常数
    if (nunique_per_col <= 1).all():
        return {a: 1 for a in algo_df.columns}

    # 条件B：全表只有极少数不同值（worst 情况很常见）
    if total_unique <= 2:
        return {a: 1 for a in algo_df.columns}

    # 条件C：某个值占比过高，说明大量 worst penalty
    value_counts = pd.Series(algo_df.values.ravel()).dropna().value_counts(normalize=True)
    if not value_counts.empty and value_counts.iloc[0] >= 0.8:
        return {a: 1 for a in algo_df.columns}

    # 条件D：所有算法均值几乎完全一样
    means = algo_df.mean(axis=0, skipna=True)
    if means.nunique(dropna=True) <= 1:
        return {a: 1 for a in algo_df.columns}

    # 条件E：worst penalty 会加入很小扰动，均值只在扰动量级内波动时不应分组
    if tie_tolerance > 0:
        mean_spread = means.max(skipna=True) - means.min(skipna=True)
        if pd.notna(mean_spread) and mean_spread <= tie_tolerance:
            return {a: 1 for a in algo_df.columns}

    # -----------------------------
    # 2) 正常尝试 Scott-Knott
    # -----------------------------
    try:
        r_sk = sk.sk_esd(algo_df, version='p')
        order = [i - 1 for i in r_sk[3]]
        if not maximize:
            order = order[::-1]

        ranks = list(r_sk[1])
        ordered_algos = [algo_df.columns[i] for i in order]
        return {a: int(r) for a, r in zip(ordered_algos, ranks)}

    except Exception as e:
        print("[WARN] Scott-Knott failed, fallback to mean-based tie ranking.")
        print(f"       reason: {e}")

        # fallback：按均值排序；若均值相同则同 rank
        means = algo_df.mean(axis=0, skipna=True)

        if maximize:
            sorted_algos = means.sort_values(ascending=False, kind="mergesort")
        else:
            sorted_algos = means.sort_values(ascending=True, kind="mergesort")

        ranks = {}
        current_rank = 1
        prev_val = None

        for i, (algo, val) in enumerate(sorted_algos.items()):
            if prev_val is None:
                ranks[algo] = current_rank
            else:
                if np.isclose(val, prev_val, rtol=1e-10, atol=1e-12):
                    ranks[algo] = current_rank
                else:
                    current_rank = i + 1
                    ranks[algo] = current_rank
            prev_val = val

        return ranks

def collect_by_scale_group(
    systems,
    algorithms,
    runs,
    base_path,
    estimation,
    out_root,
    sk_pkg
):
    """
    输出到：RQ1/<scale_group>/<estimation>/{system}.csv 以及 scott_knott.csv
    其中 scale_group ∈ {small, medium, large}
    """

    for g in ['small', 'medium', 'large']:
        out_dir = os.path.join(out_root, g, estimation)
        safe_mkdir(out_dir)

        wide_rows = {algo: {} for algo in algorithms}

        for system in systems:
            system_scale = f"{system}_{g}"

            # 如果这个 workload 目录根本不存在，就跳过该 system
            any_exist = False
            for algo in algorithms[:1]:
                p = os.path.join(base_path, f'run0', estimation, algo, system, system_scale)
                if os.path.isdir(p):
                    any_exist = True
                    break
            if not any_exist:
                # 也可以 print 提示一下
                # print(f"[WARN] Missing workload dir: {system}/{system_scale} under {estimation}")
                continue

            algo_data = {}
            for algo in algorithms:
                vals = []
                for run in range(runs):
                    f = os.path.join(base_path, f'run{run}', estimation, algo, system, system_scale, 'all_results.csv')
                    if not os.path.exists(f):
                        continue

                    df = pd.read_csv(f)
                    perf = df.iloc[:, -1]

                    # 去掉负数
                    perf = perf[perf >= 0]

                    # 如果这一轮全是无效值，直接跳过
                    if perf.empty:
                        continue

                    best = perf.max() if is_maximize(system) else perf.min()
                    vals.append(best)

                algo_data[algo] = vals

            algo_df = pd.DataFrame({a: pd.Series(v) for a, v in algo_data.items()})
            tie_tolerance = 1e-2 if estimation == 'worst' else 0.0
            ranks = compute_sk_ranks(
                algo_df,
                maximize=is_maximize(system),
                sk=sk_pkg,
                tie_tolerance=tie_tolerance
            )

            rows = []
            for algo in algorithms:
                vals = algo_data.get(algo, [])
                if len(vals) == 0:
                    mean_std, iqr = '', ''
                else:

                    if system in ['clang', 'gcc']:
                        mean = float(np.mean(vals)) * 1000
                        std = float(np.std(vals)) * 1000
                        iqr = float(calculate_iqr(vals)) * 1000
                        mean_std = f'{mean:.2f} ({std:.2f})'
                        iqr = f'{iqr:.2f}'
                    else:
                        mean = float(np.mean(vals))
                        std = float(np.std(vals))
                        iqr = float(calculate_iqr(vals))
                        mean_std = f'{mean:.2f} ({std:.2f})'
                        iqr = f'{iqr:.2f}'


                rnk = ranks.get(algo, '')
                rows.append({
                    'Scale': g,
                    'Workload': system_scale,
                    'Algorithm': algo,
                    'r': rnk,
                    'Mean (Std)': mean_std,
                    'IQR': iqr
                })

                wide_rows[algo][(system, 'r')] = rnk
                wide_rows[algo][(system, 'mean')] = mean_std

            pd.DataFrame(rows).to_csv(os.path.join(out_dir, f'{system}.csv'), index=False)

        wide_df = pd.DataFrame.from_dict(wide_rows, orient='index')
        wide_df.index.name = 'Algorithm'
        wide_df.columns = pd.MultiIndex.from_tuples(wide_df.columns)

        ordered_cols = []
        for s in systems:
            if (s, 'r') in wide_df.columns:
                ordered_cols.append((s, 'r'))
            if (s, 'mean') in wide_df.columns:
                ordered_cols.append((s, 'mean'))
        wide_df = wide_df.reindex(columns=pd.MultiIndex.from_tuples(ordered_cols))

        wide_df.to_csv(os.path.join(out_dir, 'scott_knott.csv'))


def main():

    systems = ['mysql', 'postgresql', 'httpd', 'tomcat', 'gcc', 'clang', 'x264', 'x265']
    compared_algorithms = ['flash', 'ga', 'smac', 'hebo', 'bestconfig', 'promisetune', 'random_search']
    # compared_algorithms = ['ezr_pool', 'flash_pool', 'line_pool', 'random_search_pool']
    runs = 10
    estimations = ['worst', '1NN', 'kNN', 'kNNw', 'loess', 'rf', 'rf_uncertainty', 'dal_rf']
    # estimations = ['pool']

    for estimation in estimations:
        print(f"\n===== Estimation Method: {estimation} =====")

        collect_by_scale_group(
            systems=systems,
            algorithms=compared_algorithms,
            runs=runs,
            base_path='../results_dataset',
            estimation=estimation,
            out_root='RQ1',
            sk_pkg=sk
        )




if __name__ == '__main__':
    main()
