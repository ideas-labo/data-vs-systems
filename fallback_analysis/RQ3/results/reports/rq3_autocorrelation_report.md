# RQ3 Auto-Correlation Report

This report summarizes true landscape auto-correlation by system.
The true value is de-duplicated by `system, scale, train_frac, repeat`, so repeated fallback files do not overweight the same ground-truth landscape.

Interpretation: higher positive auto-correlation means neighboring points along the random walk tend to have similar performance; near-zero values indicate a more rugged or weakly structured landscape.

Prediction/error columns exclude fallbacks: `worst`.

| System | RQ2 Spearman | True AutoCorr | True Min | True Max | Pred AutoCorr | AutoCorr Error | Contexts | Fallbacks | Disagree Ctx |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mysql | 0.435 +/- 0.242 | 0.189 +/- 0.008 | 0.176 | 0.205 | 0.329 +/- 0.106 | 0.161 +/- 0.071 | 10 | 7 | 0 |
| postgresql | 0.528 +/- 0.410 | 0.029 +/- 0.005 | 0.020 | 0.037 | 0.244 +/- 0.103 | 0.215 +/- 0.103 | 10 | 7 | 0 |
| httpd | 0.407 +/- 0.209 | 0.830 +/- 0.005 | 0.820 | 0.838 | 0.791 +/- 0.084 | 0.042 +/- 0.082 | 10 | 7 | 0 |
| tomcat | 0.719 +/- 0.213 | 0.067 +/- 0.008 | 0.056 | 0.079 | 0.232 +/- 0.168 | 0.165 +/- 0.167 | 10 | 7 | 0 |
| gcc | -0.203 +/- 0.266 | -0.001 +/- 0.006 | -0.012 | 0.012 | 0.322 +/- 0.113 | 0.323 +/- 0.113 | 10 | 7 | 0 |
| clang | -0.134 +/- 0.150 | 0.002 +/- 0.005 | -0.005 | 0.010 | 0.323 +/- 0.110 | 0.322 +/- 0.111 | 10 | 7 | 0 |
| x264 | 0.515 +/- 0.281 | 0.346 +/- 0.005 | 0.339 | 0.354 | 0.361 +/- 0.099 | 0.078 +/- 0.062 | 10 | 7 | 0 |
| x265 | 0.715 +/- 0.263 | 0.437 +/- 0.016 | 0.410 | 0.460 | 0.457 +/- 0.115 | 0.094 +/- 0.070 | 10 | 7 | 0 |

## Data Quality Note

`Disagree Ctx` counts contexts where the same ground-truth context produced different `true_auto_correlation` values across fallback files. A non-zero value usually means the valid-row mask differed across fallback predictions.
