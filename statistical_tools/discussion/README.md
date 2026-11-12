# Discussion

These scripts characterize measured configuration landscapes and compare them with fallback reconstructions.

## System characteristics

Run from the repository root:

```sh
python statistical_tools/discussion/scripts/discussion_3_analyze_ac.py
python statistical_tools/discussion/scripts/discussion_4_analyze_parameter_effects.py
python statistical_tools/discussion/scripts/discussion_5_export_ac_parameter_tikz.py
```

The analysis uses full benchmark datasets. Autocorrelation uses the cached neighborhood radius from RQ3 script 6 and a random-walk seed of 0. Parameter effects are measured by one-way ANOVA eta-squared on performance percentiles; numeric options with more than six levels are grouped into up to six quantile bins. The largest effect and its option are retained for each system.

The final script combines autocorrelation and maximum parameter effects into a two-panel TeX figure. Outputs are stored under `results/`.

Parameter effects describe marginal associations under the observed sampling design. They do not establish causal effects or show which patterns a fallback model uses.

## Other analyses

- `discussion_1_export_system_level_summary.py`: export system-level summary tables.
- `discussion_2_visualize_true_vs_fallback_landscape.py`: compare measured and reconstructed landscapes for a selected case.
