import os
import numpy as np
from rpy2.robjects import r, pandas2ri
import pandas as pd
from rpy2.robjects.packages import importr
from collections import Counter

pandas2ri.activate()
sk = importr('ScottKnottESD')


def count_rank_ones(results_df):
    rank_ones_df = results_df[results_df['r'] == 1]
    return Counter(rank_ones_df['Algorithm'])

def calculate_average_rank(results_df):
    avg_ranks = results_df.groupby('Algorithm')['r'].mean().sort_values()
    print("\nAverage Rank of Each Algorithm:")
    for algo, avg_rank in avg_ranks.items():
        print(f"{algo}: {avg_rank:.2f}")

def calculate_iqr(data):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    return q3 - q1


def collect_data(systems, algorithms, runs, base_path='../results_dataset', estimation='worst'):
    os.makedirs(f"RQ1/{estimation}", exist_ok=True)
    all_result_list = []

    for system in systems:
        results_list = []
        env_path = os.path.join(base_path, f'run8', estimation, algorithms[0], system)

        scales = sorted([
            f for f in os.listdir(env_path)
            if os.path.isdir(os.path.join(env_path, f))
        ])

        for scale in scales:
            algo_data = {}

            for algo in algorithms:
                algo_best_vals = []

                for run in range(runs):
                    result_file = os.path.join(base_path, f'run{run}', estimation, algo, system, scale, 'all_results.csv')
                    if os.path.exists(result_file):
                        df = pd.read_csv(result_file)
                        perf_values = df.iloc[:, -1]
                        best_val = perf_values.max() if system in ['h2', 'mysql', 'postgresql', 'tomcat', 'httpd'] else perf_values.min()
                        algo_best_vals.append(best_val)

                algo_data[algo] = algo_best_vals



            # algo_df = pd.DataFrame(algo_data)
            # r_sk = sk.sk_esd(algo_df, version='p')
            # column_order = [i - 1 for i in r_sk[3]]
            #
            #
            #
            # if system not in ['h2', 'mysql', 'postgresql', 'tomcat', 'httpd']:
            #     column_order = [i - 1 for i in r_sk[3]][::-1]  # Reverse the rankings and change to minimize
            #
            #
            # ranking_df = pd.DataFrame({
            #     'Algorithms': [algo_df.columns[i] for i in column_order],
            #     'rankings': r_sk[1].astype("int"),
            # }).set_index("Algorithms")

            # -----------------------------------------------------------------------
            algo_df = pd.DataFrame(algo_data)

            if (algo_df.nunique() <= 1).any():
                print(f"there is an algo that obtains the same results on {scale}, skipping")

                sk_ranks = [1] * algo_df.shape[1]
                column_order = list(range(algo_df.shape[1]))
            else:
                try:
                    r_sk = sk.sk_esd(algo_df, version='p')
                except Exception as e:
                    a = 6

                if system not in ['h2', 'mysql', 'postgresql', 'tomcat', 'httpd']:
                    column_order = [i - 1 for i in r_sk[3]][::-1]
                else:
                    column_order = [i - 1 for i in r_sk[3]]

                sk_ranks = list(r_sk[1])

            ranking_df = pd.DataFrame({
                'Algorithms': [algo_df.columns[i] for i in column_order],
                'rankings': sk_ranks,
            }).set_index("Algorithms")
            # -----------------------------------------------------------------------

            for algo in algorithms:
                vals = algo_data[algo]
                mean = round(np.mean(vals), 3)
                std = round(np.std(vals), 3)
                iqr = round(calculate_iqr(vals), 3)
                rank = ranking_df.loc[algo, 'rankings']

                result = {
                    'Workload': scale,
                    'Algorithm': algo,
                    'r': rank,
                    'Mean (Std)': f'{mean} ({std})',
                    'IQR': iqr
                }
                results_list.append(result)

                all_result_list.append(result)

        results_df = pd.DataFrame(results_list)
        results_df.to_csv(f'RQ1/{estimation}/{system}.csv', index=False)

    all_df = pd.DataFrame(all_result_list)
    all_df.to_csv(f'RQ1/{estimation}/scott_knott.csv', index=False)


def main():

    systems = ['httpd', 'tomcat', 'postgresql']
    compared_algorithms = ['flash', 'ga', 'smac', 'hebo', 'bestconfig']

    runs = 9
    estimations = ['1NN', 'kNN', 'kNNw']
    # estimations = ['loess']

    for estimation in estimations:
        print(f"\n===== Estimation Method: {estimation} =====")

        collect_data(systems, compared_algorithms, runs, estimation=estimation)

        df = pd.read_csv(f'RQ1/{estimation}/scott_knott.csv')
        counter = count_rank_ones(df)
        for algo, count in counter.items():
            print(f"{algo} ranked first {count} times.")

        calculate_average_rank(df)

        # -------------------------------
        print("\nRank 1 Count per Algorithm per System:")
        for system in systems:
            system_df = pd.read_csv(f'RQ1/{estimation}/{system}.csv')
            counter = count_rank_ones(system_df)
            print(f"\nSystem: {system}")
            for algo in compared_algorithms:
                print(f"{algo}: {counter.get(algo, 0)}")
        # -------------------------------


if __name__ == '__main__':
    main()
