# RQ3 Fallback Mechanism Analysis

This folder keeps RQ3-specific scripts and generated outputs under `statistical_tools`.

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
- `results/reports/`: Markdown metric summaries.
- `results/latex_tables/`: paper-ready LaTeX tables.
- `results/visualization/`: metric summary figures.
- `results/local_optima_mds/`: cached true-landscape MDS points and landscape plots.

## Main Pipeline

Run these scripts in order for the core RQ3 analysis.

### 1. Generate Predictions

Command:

`python statistical_tools/RQ3/scripts/rq3_1_generate_predictions.py`

Purpose:

- Builds sparse training tables from each full benchmark dataset.
- Applies each fallback strategy to reconstruct holdout configuration performance.
- Produces the prediction data consumed by all downstream RQ3 metric scripts.

Default inputs:

- `datasets/{system}/{system}_{scale}.csv`

Default outputs:

- `statistical_tools/RQ3/results/predictions/{fallback}/{system}_{scale}_predictions.csv`

Each prediction CSV contains identifiers, configuration columns, `true_perf`, and `pred_perf`.

### 2. Compute Accuracy And Landscape Metrics

Command:

`python statistical_tools/RQ3/scripts/rq3_2_compute_accuracy_landscape_metrics.py`

Purpose:

- Reads fallback prediction CSVs.
- Computes pointwise prediction accuracy metrics such as MRE, RMSE, rank correlation, top-k overlap, and NDCG.
- Computes landscape-preservation errors such as local-optima error and auto-correlation error.

Default inputs:

- `statistical_tools/RQ3/results/predictions/{fallback}/{system}_{scale}_predictions.csv`

Default outputs:

- `statistical_tools/RQ3/results/accuracy/{fallback}/{system}_{scale}_accuracy_metrics.csv`
- `statistical_tools/RQ3/results/landscape/{fallback}/{system}_{scale}_landscape_metrics.csv`
- `statistical_tools/RQ3/results/landscape/{fallback}/{system}_{scale}_landscape_raw.csv`

By default this processes all available prediction CSVs. For a focused run:

`python statistical_tools/RQ3/scripts/rq3_2_compute_accuracy_landscape_metrics.py --systems mysql x264`

This script can be slow because landscape metrics build neighbor graphs over holdout configurations.

### 3. Build Metric Report

Command:

`python statistical_tools/RQ3/scripts/rq3_3_build_metric_report.py`

Purpose:

- Merges RQ3 accuracy metrics, landscape metrics, RQ1 fallback ranking results, and RQ2 real-system correlation context.
- Builds a Markdown report explaining which fallback metrics align with real-system ranking agreement.

Default inputs:

- `statistical_tools/RQ3/results/accuracy/{fallback}/*_accuracy_metrics.csv`
- `statistical_tools/RQ3/results/landscape/{fallback}/*_landscape_metrics.csv`
- `results_system/ground_truth.csv`
- `statistical_tools/RQ1/results/**`

Default output:

- `statistical_tools/RQ3/results/reports/rq3_metric_report.md`

### 4. Export LaTeX Tables

Command:

`python statistical_tools/RQ3/scripts/rq3_4_export_latex_tables.py`

Purpose:

- Converts the RQ3 metric summaries into paper-ready LaTeX tables and a PGFPlots-style chart snippet.

Default inputs:

- Same metric inputs as `rq3_3_build_metric_report.py`

Default outputs:

- `statistical_tools/RQ3/results/latex_tables/rq3_fallback_level_summary.tex`
- `statistical_tools/RQ3/results/latex_tables/rq3_metric_rq2_bar_chart.tex`

## Optional Analyses

These scripts are not required before the core report, but they support interpretation and figures.

### 5. Visualize Metric Summary

Command:

`python statistical_tools/RQ3/scripts/rq3_5_visualize_metric_summary.py`

Purpose:

- Creates per-system bar charts for selected accuracy and landscape metrics.
- Writes the summarized plotting data beside each figure.

Default inputs:

- `statistical_tools/RQ3/results/accuracy/{fallback}/{system}_{scale}_accuracy_metrics.csv`
- `statistical_tools/RQ3/results/landscape/{fallback}/{system}_{scale}_landscape_metrics.csv`

Default outputs:

- `statistical_tools/RQ3/results/visualization/accuracy/{system}/{metric}.png`
- `statistical_tools/RQ3/results/visualization/accuracy/{system}/{metric}_summary.csv`
- `statistical_tools/RQ3/results/visualization/landscape/{system}/{metric}.png`
- `statistical_tools/RQ3/results/visualization/landscape/{system}/{metric}_summary.csv`

### 6. Generate And Visualize Local Optima MDS

Command:

`python statistical_tools/RQ3/scripts/rq3_6_generate_and_visualize_local_optima_mds.py`

Purpose:

- Generates the `results/local_optima_mds/` result set from full benchmark datasets.
- Computes true-landscape local optima and caches MDS coordinates for reuse.
- Creates local-optima visualization plots unless `--skip-plots` is used.

Default inputs:

- `datasets/{system}/{system}_{scale}.csv`

Default outputs:

- `statistical_tools/RQ3/results/local_optima_mds/points/{system}_{scale}_mds_3d_points.csv`
- `statistical_tools/RQ3/results/local_optima_mds/local_optima/{system}_{scale}_local_optima.csv`
- `statistical_tools/RQ3/results/local_optima_mds/plots/{system}_{scale}_local_optima_2d.png`
- `statistical_tools/RQ3/results/local_optima_mds/summary/{system}_{scale}_summary.csv`
- `statistical_tools/RQ3/results/local_optima_mds/summary/{scale}_summary.csv`

### 7. Visualize True Landscape In 2D

Command:

`python statistical_tools/RQ3/scripts/rq3_7_visualize_true_landscape_2d.py`

Purpose:

- Uses cached MDS points to draw 2D contour maps of true performance landscapes.
- Can overlay local optima and optionally sample points for faster exploratory plots.

Default inputs:

- `statistical_tools/RQ3/results/local_optima_mds/points/{system}_{scale}_mds_3d_points.csv`

Default outputs:

- `statistical_tools/RQ3/results/local_optima_mds/plots_2d/{system}_{scale}_mds_2d_contour_{grid_method}.png`
- `statistical_tools/RQ3/results/local_optima_mds/plots_2d/{scale}_2d_contour_summary.csv`

### 8. Visualize True Landscape In 3D

Command:

`python statistical_tools/RQ3/scripts/rq3_8_visualize_true_landscape_3d.py`

Purpose:

- Uses cached MDS points to draw 3D scatter or interpolated surface plots of true performance landscapes.
- Supports local-optima overlays and point sampling for faster plotting.

Default inputs:

- `statistical_tools/RQ3/results/local_optima_mds/points/{system}_{scale}_mds_3d_points.csv`

Default outputs:

- `statistical_tools/RQ3/results/local_optima_mds/plots_3d/{system}_{scale}_mds_landscape_3d.png`
- `statistical_tools/RQ3/results/local_optima_mds/plots_3d/{system}_{scale}_mds_landscape_surface_3d.png`
- `statistical_tools/RQ3/results/local_optima_mds/plots_3d/{system}_{scale}_mds_landscape_surface_{surface_method}_3d.png`
- `statistical_tools/RQ3/results/local_optima_mds/plots_3d/{scale}_3d_visualization_summary.csv`

## Shared Helper Scripts

- `accuracy_metrics.py`: pure metric helpers for pointwise prediction accuracy; it does not read or write files directly.
- `landscape_metrics.py`: pure metric helpers for configuration-distance encoding, neighbor graph construction, local optima, and auto-correlation; it does not read or write files directly.
