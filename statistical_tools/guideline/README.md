# Guideline: Fallback Selection

This experiment evaluates whether reconstruction and landscape metrics can select a fallback strategy whose tuner ranking agrees with real-system results.

Run from the repository root after the RQ1, RQ2, and RQ3 analyses:

```sh
python statistical_tools/guideline/scripts/guideline_1_run_selector_experiment.py
```

The experiment compares single-metric selection, equal-weight scoring, pairwise random forest, and LambdaRank-style scoring using leave-one-system-out validation. It reads the RQ3 quality metrics, RQ1 ranking results, and `results_system/ground_truth.csv`, then saves a Markdown report under `results/reports/`.
