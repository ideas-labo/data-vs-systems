"""Descriptive single-option associations in true datasets.

Effects are marginal group-mean patterns under the observed sampling design,
not causal effects or independent contributions that can be summed.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def encode_parameters(configs: pd.DataFrame, max_bins=6):
    codes, metadata = {}, []
    for name in configs:
        values = configs[name]
        method = 'original_levels'
        if pd.api.types.is_numeric_dtype(values) and values.nunique() > max_bins:
            values = pd.qcut(values, q=max_bins, duplicates='drop')
            method = 'quantile'
        encoded, labels = pd.factorize(values, sort=True)
        if np.any(encoded < 0):
            raise ValueError(f'Missing configuration values in {name}')
        codes[name] = encoded
        metadata.append({'parameter': name, 'original_levels': configs[name].nunique(),
                         'levels': len(labels), 'binning': method,
                         'labels': [str(label) for label in labels]})
    return codes, metadata


def effect_profile(a, y):
    """Return one-way ANOVA eta², centered group means, and group counts.

    eta² = sum_g n_g (mean_g - grand_mean)^2 / total sum of squares.
    This measures marginal association without controlling other options.
    Constant performance has undefined eta² and returns NaN.
    """
    a, y = np.asarray(a, int), np.asarray(y, float)
    size = int(a.max()) + 1
    counts = np.bincount(a, minlength=size).astype(float)
    sums = np.bincount(a, weights=y, minlength=size)
    means = np.divide(sums, counts, out=np.full(size, np.nan), where=counts > 0)
    profile = means - y.mean()
    valid = counts > 0
    sst = np.sum((y-y.mean())**2)
    eta = float(np.sum(counts[valid]*profile[valid]**2)/sst) if sst > 0 else np.nan
    return eta, profile, counts
