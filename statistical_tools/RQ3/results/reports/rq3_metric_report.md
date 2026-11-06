# RQ3 Fallback Metric Report

This report links fallback reconstruction metrics with RQ2 tuner-ranking consistency.
Metric directions: RQ2 Spearman, configuration Spearman/Kendall, and Top-k overlaps are higher-is-better; MRE, RMSE, auto-correlation error, and local-optima ratio error are lower-is-better.

## Per-System Fallback Metrics

Each section is sorted by RQ2 Spearman, high to low.

### mysql / large

| Fallback | RQ2 Spearman | MRE | RMSE | Config Spearman | Config Kendall | Top-5% | Top-10% | AutoCorr Err | LocalOpt Err |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dal_rf | 0.615 | 3.99 | 13.787 | 0.753 | 0.582 | 0.778 | 0.742 | 0.131 | 0.009 |
| rf | 0.588 | 3.84 | 13.229 | 0.771 | 0.596 | 0.793 | 0.762 | 0.150 | 0.011 |
| rf_uncertainty | 0.586 | 7.16 | 19.762 | 0.657 | 0.488 | 0.779 | 0.727 | 0.084 | 0.014 |
| loess | 0.522 | 13.72 | 33.177 | 0.138 | 0.092 | 0.127 | 0.208 | 0.252 | 0.011 |
| kNN | 0.406 | 4.96 | 16.039 | 0.558 | 0.395 | 0.278 | 0.417 | 0.213 | 0.005 |
| kNNw | 0.406 | 4.94 | 16.010 | 0.564 | 0.399 | 0.283 | 0.423 | 0.223 | 0.006 |
| 1NN | -0.079 | 6.61 | 20.897 | 0.351 | 0.243 | 0.181 | 0.291 | 0.073 | 0.041 |
| worst |  | 59.25 | 121.836 | 0.002 | 0.001 | 0.049 | 0.102 | 0.189 | 0.024 |

### postgresql / large

| Fallback | RQ2 Spearman | MRE | RMSE | Config Spearman | Config Kendall | Top-5% | Top-10% | AutoCorr Err | LocalOpt Err |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| kNN | 0.774 | 15.00 | 57.060 | 0.087 | 0.058 | 0.075 | 0.129 | 0.252 | 0.016 |
| dal_rf | 0.762 | 14.19 | 55.690 | 0.147 | 0.099 | 0.084 | 0.134 | 0.167 | 0.034 |
| kNNw | 0.741 | 15.00 | 57.054 | 0.088 | 0.059 | 0.076 | 0.130 | 0.263 | 0.018 |
| loess | 0.627 | 24.55 | 94.822 | 0.028 | 0.019 | 0.063 | 0.110 | 0.388 | 0.003 |
| rf_uncertainty | 0.618 | 17.98 | 77.149 | 0.156 | 0.105 | 0.083 | 0.139 | 0.189 | 0.046 |
| rf | 0.560 | 13.81 | 54.096 | 0.168 | 0.113 | 0.087 | 0.142 | 0.208 | 0.038 |
| 1NN | -0.384 | 19.69 | 78.023 | 0.030 | 0.020 | 0.074 | 0.115 | 0.041 | 0.021 |
| worst |  | 66.46 | 247.565 | -0.001 | -0.000 | 0.047 | 0.097 | 0.026 | 0.007 |

### httpd / large

| Fallback | RQ2 Spearman | MRE | RMSE | Config Spearman | Config Kendall | Top-5% | Top-10% | AutoCorr Err | LocalOpt Err |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| rf_uncertainty | 0.674 | 3.78 | 95.043 | 0.979 | 0.885 | 0.249 | 0.466 | 0.005 | 0.012 |
| rf | 0.584 | 2.76 | 81.292 | 0.980 | 0.888 | 0.240 | 0.457 | 0.004 | 0.004 |
| dal_rf | 0.498 | 3.02 | 89.583 | 0.979 | 0.884 | 0.235 | 0.449 | 0.003 | 0.004 |
| loess | 0.475 | 19.68 | 206.281 | 0.923 | 0.752 | 0.215 | 0.412 | 0.010 | 0.003 |
| kNN | 0.331 | 30.50 | 340.872 | 0.908 | 0.729 | 0.202 | 0.400 | 0.015 | 0.004 |
| kNNw | 0.144 | 29.97 | 335.576 | 0.909 | 0.731 | 0.202 | 0.400 | 0.013 | 0.004 |
| 1NN | 0.141 | 24.73 | 528.826 | 0.801 | 0.636 | 0.180 | 0.352 | 0.241 | 0.029 |
| worst |  | 57.48 | 1226.558 | -0.001 | -0.001 | 0.050 | 0.096 | 0.831 | 0.032 |

### tomcat / large

| Fallback | RQ2 Spearman | MRE | RMSE | Config Spearman | Config Kendall | Top-5% | Top-10% | AutoCorr Err | LocalOpt Err |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| rf | 0.923 | 1.81 | 68.207 | 0.419 | 0.290 | 0.170 | 0.273 | 0.027 | 0.021 |
| rf_uncertainty | 0.901 | 2.52 | 93.827 | 0.410 | 0.284 | 0.161 | 0.272 | 0.013 | 0.036 |
| kNNw | 0.759 | 2.37 | 112.279 | 0.220 | 0.148 | 0.118 | 0.200 | 0.323 | 0.025 |
| dal_rf | 0.739 | 1.85 | 72.878 | 0.418 | 0.289 | 0.173 | 0.272 | 0.040 | 0.021 |
| kNN | 0.737 | 2.37 | 112.310 | 0.218 | 0.147 | 0.119 | 0.199 | 0.308 | 0.023 |
| loess | 0.691 | 17.96 | 548.713 | -0.034 | -0.023 | 0.044 | 0.091 | 0.418 | 0.007 |
| 1NN | 0.279 | 3.14 | 144.766 | 0.101 | 0.068 | 0.081 | 0.147 | 0.029 | 0.028 |
| worst |  | 52.08 | 1374.714 | 0.000 | 0.000 | 0.053 | 0.105 | 0.067 | 0.006 |

### gcc / large

| Fallback | RQ2 Spearman | MRE | RMSE | Config Spearman | Config Kendall | Top-5% | Top-10% | AutoCorr Err | LocalOpt Err |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| kNN | 0.190 | 7.08 | 0.008 | 0.004 | 0.002 | 0.050 | 0.099 | 0.262 | 0.016 |
| kNNw | 0.089 | 7.09 | 0.008 | 0.004 | 0.002 | 0.051 | 0.099 | 0.303 | 0.020 |
| 1NN | -0.108 | 9.74 | 0.011 | -0.000 | -0.000 | 0.050 | 0.098 | 0.107 | 0.017 |
| loess | -0.342 | 45.43 | 0.043 | 0.001 | 0.001 | 0.048 | 0.096 | 0.445 | 0.034 |
| rf | -0.345 | 7.15 | 0.008 | 0.001 | 0.001 | 0.049 | 0.101 | 0.430 | 0.031 |
| rf_uncertainty | -0.375 | 11.39 | 0.012 | 0.005 | 0.003 | 0.049 | 0.101 | 0.416 | 0.029 |
| dal_rf | -0.531 | 7.83 | 0.009 | 0.006 | 0.004 | 0.048 | 0.100 | 0.295 | 0.030 |
| worst |  | 35.60 | 0.032 | 0.004 | 0.003 | 0.053 | 0.105 | 0.008 | 0.001 |

### clang / large

| Fallback | RQ2 Spearman | MRE | RMSE | Config Spearman | Config Kendall | Top-5% | Top-10% | AutoCorr Err | LocalOpt Err |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dal_rf | 0.048 | 7.74 | 0.009 | -0.001 | -0.001 | 0.050 | 0.101 | 0.299 | 0.031 |
| 1NN | 0.045 | 9.68 | 0.011 | -0.004 | -0.002 | 0.047 | 0.099 | 0.106 | 0.017 |
| kNNw | -0.107 | 7.02 | 0.008 | -0.002 | -0.002 | 0.053 | 0.102 | 0.309 | 0.022 |
| kNN | -0.135 | 7.01 | 0.008 | -0.002 | -0.001 | 0.054 | 0.103 | 0.268 | 0.017 |
| loess | -0.166 | 45.87 | 0.043 | -0.007 | -0.005 | 0.046 | 0.094 | 0.433 | 0.034 |
| rf | -0.258 | 7.08 | 0.008 | -0.003 | -0.002 | 0.052 | 0.099 | 0.428 | 0.031 |
| rf_uncertainty | -0.363 | 11.01 | 0.011 | -0.003 | -0.002 | 0.051 | 0.102 | 0.409 | 0.029 |
| worst |  | 33.36 | 0.030 | 0.005 | 0.003 | 0.049 | 0.098 | 0.005 | 0.000 |

### x264 / large

| Fallback | RQ2 Spearman | MRE | RMSE | Config Spearman | Config Kendall | Top-5% | Top-10% | AutoCorr Err | LocalOpt Err |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| rf | 0.814 | 27.28 | 17.853 | 0.908 | 0.741 | 0.582 | 0.641 | 0.033 | 0.016 |
| rf_uncertainty | 0.723 | 67.30 | 30.114 | 0.876 | 0.697 | 0.537 | 0.602 | 0.058 | 0.018 |
| dal_rf | 0.706 | 29.16 | 20.633 | 0.888 | 0.714 | 0.579 | 0.628 | 0.011 | 0.013 |
| kNN | 0.585 | 72.60 | 30.884 | 0.675 | 0.484 | 0.243 | 0.352 | 0.066 | 0.009 |
| kNNw | 0.504 | 72.08 | 30.734 | 0.680 | 0.489 | 0.245 | 0.353 | 0.074 | 0.010 |
| loess | 0.222 | 74.87 | 27.937 | 0.679 | 0.491 | 0.226 | 0.324 | 0.086 | 0.008 |
| 1NN | 0.052 | 81.61 | 42.628 | 0.390 | 0.269 | 0.158 | 0.234 | 0.215 | 0.033 |
| worst |  | 1176.17 | 245.615 | -0.003 | -0.002 | 0.049 | 0.101 | 0.349 | 0.023 |

### x265 / large

| Fallback | RQ2 Spearman | MRE | RMSE | Config Spearman | Config Kendall | Top-5% | Top-10% | AutoCorr Err | LocalOpt Err |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| rf | 0.946 | 34.56 | 183.560 | 0.918 | 0.752 | 0.569 | 0.677 | 0.041 | 0.005 |
| rf_uncertainty | 0.916 | 78.46 | 302.655 | 0.898 | 0.726 | 0.552 | 0.673 | 0.049 | 0.006 |
| kNN | 0.895 | 85.49 | 235.256 | 0.731 | 0.539 | 0.375 | 0.480 | 0.105 | 0.006 |
| dal_rf | 0.854 | 33.54 | 192.630 | 0.911 | 0.740 | 0.549 | 0.670 | 0.040 | 0.003 |
| kNNw | 0.698 | 84.28 | 233.272 | 0.738 | 0.545 | 0.378 | 0.483 | 0.115 | 0.005 |
| 1NN | 0.363 | 85.64 | 313.228 | 0.529 | 0.374 | 0.270 | 0.358 | 0.229 | 0.051 |
| loess | 0.332 | 90.61 | 210.951 | 0.694 | 0.508 | 0.153 | 0.296 | 0.082 | 0.016 |
| worst |  | 4104.94 | 3373.057 | -0.001 | -0.001 | 0.055 | 0.099 | 0.434 | 0.041 |


## System-Level Summary

Excluded fallbacks for summary tables: `worst`.
Each cell is the mean across fallback strategies after each fallback has already been averaged over train fractions and repeats.

| System | Fallbacks | RQ2 Spearman | MRE | RMSE | Config Spearman | Config Kendall | Top-5% overlap | Top-10% overlap | Auto-corr error | Local-optima ratio error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mysql | 7 | 0.435 | 6.46 | 18.986 | 0.542 | 0.399 | 0.460 | 0.510 | 0.161 | 0.014 |
| postgresql | 7 | 0.528 | 17.17 | 67.699 | 0.101 | 0.068 | 0.077 | 0.129 | 0.215 | 0.025 |
| httpd | 7 | 0.407 | 16.35 | 239.639 | 0.926 | 0.786 | 0.218 | 0.419 | 0.042 | 0.008 |
| tomcat | 7 | 0.719 | 4.57 | 164.711 | 0.250 | 0.172 | 0.124 | 0.207 | 0.165 | 0.023 |
| gcc | 7 | -0.203 | 13.67 | 0.014 | 0.003 | 0.002 | 0.049 | 0.099 | 0.323 | 0.025 |
| clang | 7 | -0.134 | 13.63 | 0.014 | -0.003 | -0.002 | 0.050 | 0.100 | 0.322 | 0.026 |
| x264 | 7 | 0.515 | 60.70 | 28.683 | 0.728 | 0.555 | 0.367 | 0.448 | 0.078 | 0.015 |
| x265 | 7 | 0.715 | 70.37 | 238.793 | 0.774 | 0.598 | 0.407 | 0.519 | 0.094 | 0.013 |

## System-Level Ranks

Rank 1 is best for each metric after applying the metric direction above.

| system | RQ2 Spearman | MRE | RMSE | Config Spearman | Config Kendall | Top-5% overlap | Top-10% overlap | Auto-corr error | Local-optima ratio error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mysql | 5 | 2 | 3 | 4 | 4 | 1 | 2 | 4 | 3 |
| postgresql | 3 | 6 | 5 | 6 | 6 | 6 | 6 | 6 | 6 |
| httpd | 6 | 5 | 8 | 1 | 1 | 4 | 4 | 1 | 1 |
| tomcat | 1 | 1 | 6 | 5 | 5 | 5 | 5 | 5 | 5 |
| gcc | 8 | 4 | 2 | 7 | 7 | 8 | 8 | 8 | 7 |
| clang | 7 | 3 | 1 | 8 | 8 | 7 | 7 | 7 | 8 |
| x264 | 4 | 7 | 4 | 3 | 3 | 3 | 3 | 2 | 4 |
| x265 | 2 | 8 | 7 | 2 | 2 | 2 | 1 | 3 | 2 |

## Fallback-Level Summary

Excluded fallbacks for summary tables: `worst`.
Each cell is the mean across systems after each system/fallback row has already been averaged over train fractions and repeats.

| Fallback | Systems | Scales | RQ2 Spearman | MRE | RMSE | Config Spearman | Config Kendall | Top-5% overlap | Top-10% overlap | Auto-corr error | Local-optima ratio error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1NN | 8 | 1 | 0.039 | 30.10 | 141.049 | 0.275 | 0.201 | 0.130 | 0.212 | 0.130 | 0.030 |
| kNN | 8 | 1 | 0.473 | 28.13 | 99.055 | 0.397 | 0.294 | 0.175 | 0.272 | 0.186 | 0.012 |
| kNNw | 8 | 1 | 0.404 | 27.84 | 98.118 | 0.400 | 0.297 | 0.176 | 0.274 | 0.203 | 0.014 |
| loess | 8 | 1 | 0.295 | 41.59 | 140.246 | 0.303 | 0.230 | 0.115 | 0.204 | 0.264 | 0.014 |
| rf | 8 | 1 | 0.476 | 12.29 | 52.282 | 0.520 | 0.422 | 0.318 | 0.394 | 0.165 | 0.020 |
| rf_uncertainty | 8 | 1 | 0.460 | 24.95 | 77.322 | 0.497 | 0.398 | 0.308 | 0.385 | 0.153 | 0.024 |
| dal_rf | 8 | 1 | 0.461 | 12.67 | 55.652 | 0.513 | 0.414 | 0.312 | 0.387 | 0.123 | 0.018 |

## Fallback-Level Summary With System-Normalized MRE/RMSE

Excluded fallbacks for summary tables: `worst`.
MRE and RMSE are min-max normalized within each system/scale before averaging across systems: 0 is the best fallback in that system/scale and 1 is the worst. Other metrics keep their original scale.

| Fallback | Systems | Scales | RQ2 Spearman | MRE norm | RMSE norm | Config Spearman | Config Kendall | Top-5% overlap | Top-10% overlap | Auto-corr error | Local-optima ratio error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1NN | 8 | 1 | 0.039 | 0.47 | 0.537 | 0.275 | 0.201 | 0.130 | 0.212 | 0.130 | 0.030 |
| kNN | 8 | 1 | 0.473 | 0.38 | 0.226 | 0.397 | 0.294 | 0.175 | 0.272 | 0.186 | 0.012 |
| kNNw | 8 | 1 | 0.404 | 0.37 | 0.222 | 0.400 | 0.297 | 0.176 | 0.274 | 0.203 | 0.014 |
| loess | 8 | 1 | 0.295 | 0.94 | 0.737 | 0.303 | 0.230 | 0.115 | 0.204 | 0.264 | 0.014 |
| rf | 8 | 1 | 0.476 | 0.00 | 0.000 | 0.520 | 0.422 | 0.318 | 0.394 | 0.165 | 0.020 |
| rf_uncertainty | 8 | 1 | 0.460 | 0.32 | 0.325 | 0.497 | 0.398 | 0.308 | 0.385 | 0.153 | 0.024 |
| dal_rf | 8 | 1 | 0.461 | 0.02 | 0.041 | 0.513 | 0.414 | 0.312 | 0.387 | 0.123 | 0.018 |

## Fallback-Level Ranks

Rank 1 is best for each metric after applying the metric direction above.

| fallback | RQ2 Spearman | MRE | RMSE | Config Spearman | Config Kendall | Top-5% overlap | Top-10% overlap | Auto-corr error | Local-optima ratio error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1NN | 7 | 6 | 7 | 7 | 7 | 6 | 6 | 2 | 7 |
| kNN | 2 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 1 |
| kNNw | 5 | 4 | 4 | 4 | 4 | 4 | 4 | 6 | 2 |
| loess | 6 | 7 | 6 | 6 | 6 | 7 | 7 | 7 | 3 |
| rf | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 4 | 5 |
| rf_uncertainty | 4 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 6 |
| dal_rf | 3 | 2 | 2 | 2 | 2 | 2 | 2 | 1 | 4 |

## Fallback-Level Summary Excluding Compilers

Excluded fallbacks for summary tables: `worst`.
Excluded systems for this fallback-level view: `gcc`, `clang`.
Each cell is the mean across the remaining systems after each system/fallback row has already been averaged over train fractions and repeats.

| Fallback | Systems | Scales | RQ2 Spearman | MRE | RMSE | Config Spearman | Config Kendall | Top-5% overlap | Top-10% overlap | Auto-corr error | Local-optima ratio error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1NN | 6 | 1 | 0.062 | 36.90 | 188.061 | 0.367 | 0.268 | 0.157 | 0.249 | 0.138 | 0.034 |
| kNN | 6 | 1 | 0.621 | 35.15 | 132.070 | 0.530 | 0.392 | 0.215 | 0.329 | 0.160 | 0.011 |
| kNNw | 6 | 1 | 0.542 | 34.77 | 130.821 | 0.533 | 0.395 | 0.217 | 0.332 | 0.169 | 0.011 |
| loess | 6 | 1 | 0.478 | 40.23 | 186.980 | 0.405 | 0.307 | 0.138 | 0.240 | 0.206 | 0.008 |
| rf | 6 | 1 | 0.736 | 14.01 | 69.706 | 0.694 | 0.563 | 0.407 | 0.492 | 0.077 | 0.016 |
| rf_uncertainty | 6 | 1 | 0.737 | 29.53 | 103.092 | 0.663 | 0.531 | 0.393 | 0.480 | 0.066 | 0.022 |
| dal_rf | 6 | 1 | 0.696 | 14.29 | 74.200 | 0.683 | 0.551 | 0.400 | 0.483 | 0.065 | 0.014 |

## Fallback-Level Ranks Excluding Compilers

Rank 1 is best for each metric after applying the metric direction above.

| fallback | RQ2 Spearman | MRE | RMSE | Config Spearman | Config Kendall | Top-5% overlap | Top-10% overlap | Auto-corr error | Local-optima ratio error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1NN | 7 | 6 | 7 | 7 | 7 | 6 | 6 | 4 | 7 |
| kNN | 4 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 2 |
| kNNw | 5 | 4 | 4 | 4 | 4 | 4 | 4 | 6 | 3 |
| loess | 6 | 7 | 6 | 6 | 6 | 7 | 7 | 7 | 1 |
| rf | 2 | 1 | 1 | 1 | 1 | 1 | 1 | 3 | 5 |
| rf_uncertainty | 1 | 3 | 3 | 3 | 3 | 3 | 3 | 2 | 6 |
| dal_rf | 3 | 2 | 2 | 2 | 2 | 2 | 2 | 1 | 4 |

## System-Controlled Metric-RQ2 Relation

For each system/scale, each metric is first oriented so that larger means better and then correlated with RQ2 Spearman across fallback strategies. The table reports the average relation across systems, keeping all systems in the analysis while controlling for system-level difficulty and metric scale. Rows are sorted by mean Spearman relation.

| Metric | Better | Systems | Mean Spearman | Median Spearman | Positive Systems | Mean Pearson | Median Pearson |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Top-5% overlap | higher | 8 | 0.597 | 0.714 | 6 | 0.573 | 0.696 |
| MRE | lower | 8 | 0.565 | 0.625 | 8 | 0.376 | 0.306 |
| RMSE | lower | 8 | 0.525 | 0.547 | 8 | 0.376 | 0.273 |
| Config Kendall | higher | 8 | 0.503 | 0.762 | 6 | 0.524 | 0.555 |
| Config Spearman | higher | 8 | 0.499 | 0.762 | 6 | 0.517 | 0.550 |
| Top-10% overlap | higher | 8 | 0.449 | 0.798 | 5 | 0.500 | 0.615 |
| Auto-corr error | lower | 8 | 0.426 | 0.518 | 7 | 0.293 | 0.530 |
| Local-optima ratio error | lower | 8 | 0.243 | 0.268 | 6 | 0.442 | 0.407 |
