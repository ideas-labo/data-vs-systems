#!/usr/bin/env python3
"""Plot all logged run counts without exclusions and compare fixed paper budgets.

The percentile column is the empirical CDF: 100 * count(N <= budget) / runs.
Budget positions are descriptive, not a test of budget equivalence.
"""
from __future__ import annotations

import argparse
import csv
import os
import tempfile
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'tabeval-matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from count_system_evaluations import collect_counts, DEFAULT_RESULTS_DIR, SYSTEM_ORDER

BUDGETS = dict(mysql=350, postgresql=350, httpd=250, tomcat=250,
               gcc=350, clang=350, x264=200, x265=200)
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / 'results' / 'evaluation_distribution'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--fidelity', choices=('all', 'single', 'multi'), default='all')
    args = parser.parse_args()
    counts = collect_counts(args.results_dir, args.fidelity)
    unknown = set(counts) - set(BUDGETS)
    if unknown:
        parser.error(f'No paper budgets defined for {sorted(unknown)}')
    systems = [s for s in SYSTEM_ORDER if s in counts]
    algorithms = sorted({a for records in counts.values() for a, _, _ in records})
    colors = {a: plt.get_cmap('tab20')(i % 20) for i, a in enumerate(algorithms)}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 4, figsize=(14, 8), constrained_layout=True)
    fig2, axes2 = plt.subplots(2, 4, figsize=(16, 9), constrained_layout=True)
    rng = np.random.default_rng(42)
    summary, raw = [], []
    print('System       Budget   Min     Q1  Median     Q3   Max  Runs<=budget')
    for ax, detail, system in zip(axes.flat, axes2.flat, systems):
        records = counts[system]
        values = np.array([n for _, _, n in records])
        budget = BUDGETS[system]
        q1, med, q3 = np.percentile(values, [25, 50, 75])
        ecdf = 100 * np.mean(values <= budget)
        summary.append(dict(system=system, budget=budget, runs=len(values),
                            minimum=int(values.min()), q1=q1, median=med, q3=q3,
                            maximum=int(values.max()), percent_at_or_below_budget=ecdf))
        print(f'{system:<12} {budget:>6} {values.min():>5} {q1:>6.1f} {med:>7.1f} {q3:>6.1f} {values.max():>5} {ecdf:>12.1f}%')
        if np.ptp(values) > 0:
            violin = ax.violinplot(values, positions=[0], widths=.65,
                                   showextrema=False, bw_method='scott')
            for body in violin['bodies']:
                body.set_facecolor('#9baebd')
                body.set_alpha(.3)
        ax.boxplot([values], positions=[0], widths=.14, showfliers=False,
                   medianprops={'color': '#111111', 'linewidth': 2})
        present = [a for a in algorithms if any(r[0] == a for r in records)]
        for i, algorithm in enumerate(present):
            ns = [n for a, _, n in records if a == algorithm]
            # Horizontal placement has no quantitative meaning.
            offset = (i - (len(present)-1)/2) * .045
            ax.scatter(offset + rng.uniform(-.012, .012, len(ns)), ns,
                       s=16, color=colors[algorithm], alpha=.8, zorder=3)
            detail.scatter(i + rng.uniform(-.15, .15, len(ns)), ns,
                           s=15, color=colors[algorithm], alpha=.8)
            detail.plot([i-.22, i+.22], [np.median(ns)]*2, color='black', linewidth=1.6)
        for a in (ax, detail):
            a.axhline(budget, color='#c43c39', linestyle='--', linewidth=1.5)
            a.set_ylim(bottom=0, top=max(values.max(), budget)*1.13)
            a.set_ylabel('Logged evaluations per run')
            a.grid(axis='y', alpha=.18)
        ax.set_xlim(-.5, .5)
        ax.set_xticks([])
        ax.set_title(f'{system} | budget={budget}\n{ecdf:.1f}% of runs at or below budget')
        detail.set_title(system)
        detail.set_xticks(range(len(present)), present, rotation=55, ha='right')
        for algorithm, run, n in records:
            raw.append(dict(system=system, algorithm=algorithm, run=run, evaluations=n))
    for a in list(axes.flat)[len(systems):] + list(axes2.flat)[len(systems):]:
        a.set_visible(False)
    legend = [Line2D([], [], marker='o', linestyle='', color=colors[a], label=a) for a in algorithms]
    legend += [Line2D([], [], color='#c43c39', linestyle='--', label='Tabular budget')]
    fig.legend(handles=legend, loc='outside lower center', ncol=10, frameon=False)
    fig.suptitle('Real-system evaluation counts and tabular budgets\nEach point is one run; box: median and IQR; whiskers: 1.5 IQR; all points retained')
    fig2.suptitle('Evaluation counts by tuner (black bars: per-tuner medians; dashed red: tabular budget)')
    for figure, name in ((fig, 'system_evaluation_distribution'), (fig2, 'evaluation_counts_by_algorithm')):
        for extension in ('png', 'pdf'):
            figure.savefig(args.output_dir / f'{name}.{extension}', dpi=200)
        plt.close(figure)
    for name, rows in [('budget_distribution_summary', summary), ('run_evaluation_counts', raw)]:
        with (args.output_dir / f'{name}.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f'Output: {args.output_dir.resolve()}')


if __name__ == '__main__':
    main()
