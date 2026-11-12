# RQ3: Fallback Quality

This analysis examines how prediction accuracy, configuration ordering, and landscape preservation relate to agreement with real-system tuner rankings.

## Main workflow

Run from the repository root:

```sh
python statistical_tools/RQ3/scripts/rq3_1_generate_predictions.py
python statistical_tools/RQ3/scripts/rq3_2_compute_accuracy_landscape_metrics.py
python statistical_tools/RQ3/scripts/rq3_3_build_metric_report.py
python statistical_tools/RQ3/scripts/rq3_4_export_latex_tables.py
```

1. Generate fallback predictions on held-out configurations. Systems, training fractions, strategies, and repeats are configured in the prediction script.
2. Compute prediction errors, rank correlations, top-performing configuration overlap, and landscape-preservation errors.
3. Combine these metrics with RQ1 rankings and RQ2 real-system comparisons.
4. Export LaTeX tables and chart data.

The workflow uses benchmark datasets, RQ1 results, and `results_system/ground_truth.csv`. Predictions, metrics, reports, and figures are saved under `results/`.

To compute metrics for selected systems:

```sh
python statistical_tools/RQ3/scripts/rq3_2_compute_accuracy_landscape_metrics.py --systems mysql x264
```

## Figures

| Script | Purpose |
| --- | --- |
| `rq3_5_visualize_metric_summary.py` | Compare fallback quality metrics |
| `rq3_6_generate_and_visualize_local_optima_mds.py` | Compute true-landscape local optima and cache MDS coordinates |
| `rq3_7_visualize_true_landscape_2d.py` | Plot cached landscapes in 2D |
| `rq3_8_visualize_true_landscape_3d.py` | Plot cached landscapes in 3D |

Run script 6 before scripts 7 and 8. Shared calculations are implemented in `accuracy_metrics.py` and `landscape_metrics.py`.
