# Discussion Analysis

This folder contains discussion-focused scripts and outputs that reuse RQ3 predictions and true-landscape utilities.

The discussion analysis is intended for case studies: pick one system, scale, training fraction, and repeat, then compare the true holdout landscape with the landscapes reconstructed by each fallback strategy.

## Directory Layout

- `scripts/`: runnable discussion scripts.
- `results/`: generated discussion figures and case-study outputs.

## Scripts

### 1. Visualize True Vs Fallback Landscape

Command:

`python fallback_analysis/discussion/scripts/discussion_1_visualize_true_vs_fallback_landscape.py`

Purpose:

- Loads one matched holdout split across fallback strategies.
- Computes a shared MDS embedding from configuration distances.
- Plots the true performance landscape and each fallback-predicted landscape on the same 2D coordinate system.
- Supports sub-sampling through `--sample-size` for faster exploratory case studies.

Default inputs:

- `fallback_analysis/RQ3/results/predictions/{fallback}/{system}_{scale}_predictions.csv`

Default output folder:

- `fallback_analysis/discussion/results/{system}_{scale}_train{train_frac}_repeat{repeat}/`

Default output files:

- `{system}_{scale}_train{train_frac}_repeat{repeat}_discussion_points.csv`
- `{system}_{scale}_train{train_frac}_repeat{repeat}_mds_info.csv`
- `landscape_true.pdf`
- `landscape_{fallback_display_name}.pdf`

Example focused run:

`python fallback_analysis/discussion/scripts/discussion_1_visualize_true_vs_fallback_landscape.py --system mysql --scale large --train-frac 0.05 --repeat 0 --fallbacks 1NN rf --sample-size 200`

The script reads predictions from `fallback_analysis/RQ3/results/predictions` by default and writes figures under `fallback_analysis/discussion/results`.
