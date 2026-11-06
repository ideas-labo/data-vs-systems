#!/usr/bin/env python3
"""Maximum percentile-scale marginal eta-squared across configuration options."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from parameter_effects import encode_parameters, effect_profile
from discussion_3_analyze_ac import SYSTEM_ORDER, infer_maximize, ROOT


def analyze_frame(data, system, maximize):
    score = data.iloc[:, -1].to_numpy(float) * (1 if maximize else -1)
    if not len(score) or not np.isfinite(score).all():
        raise ValueError(f'{system}: expected nonempty, finite performance')
    quality = pd.Series(score).rank(method='average', pct=True).to_numpy()
    codes, _ = encode_parameters(data.iloc[:, :-1], max_bins=6)
    effects = {name: effect_profile(code, quality)[0] for name, code in codes.items()}
    valid = {name: value for name, value in effects.items() if np.isfinite(value)}
    strongest = max(valid, key=valid.get) if valid else ''
    return {'system': system, 'strongest_parameter': strongest,
            'max_percentile_eta_squared': valid[strongest] if strongest else np.nan}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--systems', nargs='+', choices=SYSTEM_ORDER, default=SYSTEM_ORDER)
    parser.add_argument('--scale', default='large')
    parser.add_argument('--dataset-root', type=Path, default=ROOT/'datasets')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'statistical_tools/discussion/results/parameter_effects')
    args = parser.parse_args()
    rows = []
    for system in args.systems:
        data = pd.read_csv(args.dataset_root/system/f'{system}_{args.scale}.csv')
        rows.append(analyze_frame(data, system, infer_maximize(system)))
    output = args.output_dir/args.scale
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output/'system_summary.csv', index=False)
    print(f'[SAVE] {output}/system_summary.csv')


if __name__ == '__main__':
    main()
