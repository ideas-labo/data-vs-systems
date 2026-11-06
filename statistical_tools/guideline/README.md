# Guideline Fallback Selection

This folder contains guideline-oriented analysis: how to choose a fallback strategy using measurable reconstruction and landscape metrics.

Guideline analysis is separate from RQ3. RQ3 explains why fallback strategies differ; Guideline evaluates whether those explanatory metrics can support a practical fallback-selection rule.

## Directory Layout

- `scripts/`: runnable guideline scripts.
- `results/reports/`: generated guideline reports.

## Scripts

### 1. Run Selector Experiment

Command:

`python statistical_tools/guideline/scripts/guideline_1_run_selector_experiment.py`

Purpose:

- Tests whether fallback quality metrics can select the fallback that best matches RQ2 real-system tuner ranking.
- Compares single-metric selectors, equal-weight scoring, pairwise random forest, and LambdaRank-style scoring.
- Uses leave-one-system-out validation across systems.

Default inputs:

- The script builds an RQ3 master table from:
- `statistical_tools/RQ3/results/accuracy/{fallback}/*_accuracy_metrics.csv`
- `statistical_tools/RQ3/results/landscape/{fallback}/*_landscape_metrics.csv`
- `results_system/ground_truth.csv`
- `statistical_tools/RQ1/results/**`

Default output:

- `statistical_tools/guideline/results/reports/guideline_selector_experiment.md`
