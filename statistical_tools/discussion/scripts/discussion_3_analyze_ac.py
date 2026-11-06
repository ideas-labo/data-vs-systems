#!/usr/bin/env python3
"""Full measured datasets: lag-1 autocorrelation only."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import tempfile
os.environ.setdefault('XDG_CACHE_HOME', str(Path(tempfile.gettempdir())/'dataeval_cache'))
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'dataeval_matplotlib_cache'))
import pandas as pd
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'RQ3/scripts'))
from rq3_6_generate_and_visualize_local_optima_mds import SYSTEM_ORDER, infer_maximize, load_system_dataset
from landscape_metrics import build_neighbor_graph, build_random_walk_path, compute_auto_correlation, encode_configs_for_distance

ROOT = Path(__file__).resolve().parents[3]


def process_system(system, args):
    configs, performance, _, _ = load_system_dataset(args.dataset_root, system, args.scale, None)
    maximize = infer_maximize(system)
    score = performance if maximize else -performance
    # Retain the existing RQ3 radius to preserve the AC neighborhood definition.
    radius_path = args.cache_dir/'summary'/f'{system}_{args.scale}_summary.csv'
    radius = float(pd.read_csv(radius_path).radius.iloc[0])
    graph = build_neighbor_graph(encode_configs_for_distance(configs), radius)
    path = build_random_walk_path(graph, seed=args.seed)
    ac = compute_auto_correlation(score, path)
    print(f'[DONE] {system}: AC={ac:.4f}', flush=True)
    return {'System': system, 'AC': ac}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--systems', nargs='+', choices=SYSTEM_ORDER, default=SYSTEM_ORDER)
    parser.add_argument('--scale', default='large')
    parser.add_argument('--dataset-root', type=Path, default=ROOT/'datasets')
    parser.add_argument('--cache-dir', type=Path, default=ROOT/'statistical_tools/RQ3/results/local_optima_mds',
                        help='Read only the cached per-system radius from summary/; no MDS points required.')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'statistical_tools/discussion/results/ac')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    output = args.output_dir/args.scale
    output.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame([process_system(system, args) for system in args.systems])
    summary.to_csv(output/'ac_summary.csv', index=False)
    print(f'[SAVE] {output}', flush=True)


if __name__ == '__main__':
    main()
