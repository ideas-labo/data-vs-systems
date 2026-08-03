# RQ3 Fallback Mechanism Analysis

This folder keeps RQ3-specific scripts and generated outputs under `fallback_analysis` while the workflow is being cleaned up.

RQ3 focuses on why fallback strategies differ. The main flow is:

1. Generate sparse-observation fallback predictions.
2. Compute pointwise accuracy and landscape-preservation metrics.
3. Summarize which metrics explain agreement with real-system rankings.
4. Export paper-ready tables and optional visual diagnostics.

## Directory Layout

- `scripts/`: runnable RQ3 scripts and shared metric helpers.
- `results/predictions/`: fallback-predicted performance generated from sparse observed data.
- `results/accuracy/`: pointwise prediction accuracy metrics.
- `results/landscape/`: landscape-structure preservation metrics.
- `results/reports/`: Markdown summaries and selector-analysis reports.
- `results/latex_tables/`: paper-ready LaTeX tables.
- `results/visualization/`: metric summary figures.
- `results/local_optima_mds/`: cached true-landscape MDS points and landscape plots.

## Main Pipeline

Run these scripts in order for the core RQ3 analysis.

### 1. Generate Predictions

Command:

`python fallback_analysis/RQ3/scripts/rq3_1_generate_predictions.py`

Purpose:

- Builds sparse training tables from each full benchmark dataset.
- Applies each fallback strategy to reconstruct holdout configuration performance.
- Produces the prediction data consumed by all downstream RQ3 metric scripts.

Default inputs:

- `datasets/{system}/{system}_{scale}.csv`

Default outputs:

- `fallback_analysis/RQ3/results/predictions/{fallback}/{system}_{scale}_predictions.csv`

Each prediction CSV contains identifiers, configuration columns, `true_perf`, and `pred_perf`.

### 2. Compute Accuracy And Landscape Metrics

Command:

`python fallback_analysis/RQ3/scripts/rq3_2_compute_accuracy_landscape_metrics.py`

Purpose:

- Reads fallback prediction CSVs.
- Computes pointwise prediction accuracy metrics such as MRE, RMSE, rank correlation, top-k overlap, and NDCG.
- Computes landscape-preservation errors such as local-optima error and auto-correlation error.

Default inputs:

- `fallback_analysis/RQ3/results/predictions/{fallback}/{system}_{scale}_predictions.csv`

Default outputs:

- `fallback_analysis/RQ3/results/accuracy/{fallback}/{system}_{scale}_accuracy_metrics.csv`
- `fallback_analysis/RQ3/results/landscape/{fallback}/{system}_{scale}_landscape_metrics.csv`
- `fallback_analysis/RQ3/results/landscape/{fallback}/{system}_{scale}_landscape_raw.csv`

By default this processes all available prediction CSVs. For a focused run:

`python fallback_analysis/RQ3/scripts/rq3_2_compute_accuracy_landscape_metrics.py --systems mysql x264`

This script can be slow because landscape metrics build neighbor graphs over holdout configurations.

### 3. Build Metric Report

Command:

`python fallback_analysis/RQ3/scripts/rq3_3_build_metric_report.py`

Purpose:

- Merges RQ3 accuracy metrics, landscape metrics, RQ1 fallback ranking results, and RQ2 real-system correlation context.
- Builds a Markdown report explaining which fallback metrics align with real-system ranking agreement.

Default inputs:

- `fallback_analysis/RQ3/results/accuracy/{fallback}/*_accuracy_metrics.csv`
- `fallback_analysis/RQ3/results/landscape/{fallback}/*_landscape_metrics.csv`
- `results_system/ground_truth.csv`
- `statistical_tools/RQ1/results/**`

Default output:

- `fallback_analysis/RQ3/results/reports/rq3_metric_report.md`

### 4. Export LaTeX Tables

Command:

`python fallback_analysis/RQ3/scripts/rq3_4_export_latex_tables.py`

Purpose:

- Converts the RQ3 metric summaries into paper-ready LaTeX tables and a PGFPlots-style chart snippet.

Default inputs:

- Same metric inputs as `rq3_3_build_metric_report.py`
- `fallback_analysis/RQ3/results/landscape/{fallback}/*_landscape_raw.csv` for true auto-correlation summaries

Default outputs:

- `fallback_analysis/RQ3/results/latex_tables/rq3_fallback_level_summary.tex`
- `fallback_analysis/RQ3/results/latex_tables/rq3_system_level_summary.tex`
- `fallback_analysis/RQ3/results/latex_tables/rq3_metric_rq2_bar_chart.tex`
- `fallback_analysis/RQ3/results/latex_tables/rq3_tables.tex`

## Optional Analyses

These scripts are not required before the core report, but they support interpretation and figures.

### 5. Build Auto-Correlation Report

Command:

`python fallback_analysis/RQ3/scripts/rq3_5_build_autocorrelation_report.py`

Purpose:

- Summarizes true landscape auto-correlation and fallback-predicted auto-correlation.
- Separates system-level true ruggedness from fallback-induced auto-correlation error.

Default inputs:

- `fallback_analysis/RQ3/results/landscape/{fallback}/{system}_{scale}_landscape_raw.csv`
- RQ3 master table built from `rq3_3_build_metric_report.py` helpers

Default output:

- `fallback_analysis/RQ3/results/reports/rq3_autocorrelation_report.md`

### 6. Run Selector Experiment

Command:

`python fallback_analysis/RQ3/scripts/rq3_6_run_selector_experiment.py`

Purpose:

- Tests whether RQ3 metrics can select the fallback that best matches RQ2 real-system ranking.
- Compares simple metric-only selectors, equal-weight scoring, pairwise random forest, and LambdaRank-style scoring.

Default inputs:

- RQ3 master table built from accuracy, landscape, RQ1, and RQ2 context.

Default output:

- `fallback_analysis/RQ3/results/reports/rq3_selector_experiment.md`

### 7. Visualize Metric Summary

Command:

`python fallback_analysis/RQ3/scripts/rq3_7_visualize_metric_summary.py`

Purpose:

- Creates per-system bar charts for selected accuracy and landscape metrics.
- Writes the summarized plotting data beside each figure.

Default inputs:

- `fallback_analysis/RQ3/results/accuracy/{fallback}/{system}_{scale}_accuracy_metrics.csv`
- `fallback_analysis/RQ3/results/landscape/{fallback}/{system}_{scale}_landscape_metrics.csv`

Default outputs:

- `fallback_analysis/RQ3/results/visualization/accuracy/{system}/{metric}.png`
- `fallback_analysis/RQ3/results/visualization/accuracy/{system}/{metric}_summary.csv`
- `fallback_analysis/RQ3/results/visualization/landscape/{system}/{metric}.png`
- `fallback_analysis/RQ3/results/visualization/landscape/{system}/{metric}_summary.csv`

### 8. Cache True Landscape MDS

Command:

`python fallback_analysis/RQ3/scripts/rq3_8_cache_true_landscape_mds.py`

Purpose:

- Computes true-landscape local optima from full benchmark datasets.
- Computes and caches MDS coordinates so expensive landscape visualizations can be reused.
- Optionally creates local-optima scatter plots.

Default inputs:

- `datasets/{system}/{system}_{scale}.csv`

Default outputs:

- `fallback_analysis/RQ3/results/local_optima_mds/points/{system}_{scale}_mds_3d_points.csv`
- `fallback_analysis/RQ3/results/local_optima_mds/local_optima/{system}_{scale}_local_optima.csv`
- `fallback_analysis/RQ3/results/local_optima_mds/plots/{system}_{scale}_local_optima_2d.png`
- `fallback_analysis/RQ3/results/local_optima_mds/summary/{system}_{scale}_summary.csv`
- `fallback_analysis/RQ3/results/local_optima_mds/summary/{scale}_summary.csv`

### 9. Visualize True Landscape In 2D

Command:

`python fallback_analysis/RQ3/scripts/rq3_9_visualize_true_landscape_2d.py`

Purpose:

- Uses cached MDS points to draw 2D contour maps of true performance landscapes.
- Can overlay local optima and optionally sample points for faster exploratory plots.

Default inputs:

- `fallback_analysis/RQ3/results/local_optima_mds/points/{system}_{scale}_mds_3d_points.csv`

Default outputs:

- `fallback_analysis/RQ3/results/local_optima_mds/plots_2d/{system}_{scale}_mds_2d_contour_{grid_method}.png`
- `fallback_analysis/RQ3/results/local_optima_mds/plots_2d/{scale}_2d_contour_summary.csv`

### 10. Visualize True Landscape In 3D

Command:

`python fallback_analysis/RQ3/scripts/rq3_10_visualize_true_landscape_3d.py`

Purpose:

- Uses cached MDS points to draw 3D scatter or interpolated surface plots of true performance landscapes.
- Supports local-optima overlays and point sampling for faster plotting.

Default inputs:

- `fallback_analysis/RQ3/results/local_optima_mds/points/{system}_{scale}_mds_3d_points.csv`

Default outputs:

- `fallback_analysis/RQ3/results/local_optima_mds/plots_3d/{system}_{scale}_mds_landscape_3d.png`
- `fallback_analysis/RQ3/results/local_optima_mds/plots_3d/{system}_{scale}_mds_landscape_surface_3d.png`
- `fallback_analysis/RQ3/results/local_optima_mds/plots_3d/{system}_{scale}_mds_landscape_surface_{surface_method}_3d.png`
- `fallback_analysis/RQ3/results/local_optima_mds/plots_3d/{scale}_3d_visualization_summary.csv`

## Shared Helper Scripts

- `accuracy_metrics.py`: pure metric helpers for pointwise prediction accuracy; it does not read or write files directly.
- `landscape_metrics.py`: pure metric helpers for configuration-distance encoding, neighbor graph construction, local optima, and auto-correlation; it does not read or write files directly.
