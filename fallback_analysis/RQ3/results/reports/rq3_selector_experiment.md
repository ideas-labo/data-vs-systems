# RQ3 Fallback Selector Experiment

This report evaluates fallback recommendation strategies using leave-one-system-out validation.
RQ2 Spearman is used only as validation signal, not as an input feature.

## Data

- Input rows after exclusion: `56`
- Supervised rows: `56`
- Systems: `8`
- Candidate fallbacks per system: `7`
- Excluded fallbacks before normalization: `worst`

Quality features are normalized within each system and larger is always better:

- `q_mre`: inverse normalized MRE
- `q_rmse`: inverse normalized RMSE
- `q_spearman`: normalized configuration-level Spearman
- `q_kendall`: normalized configuration-level Kendall tau
- `q_top5`: normalized Top-5% overlap
- `q_top10`: normalized Top-10% overlap
- `q_auto_corr`: inverse normalized auto-correlation absolute error
- `q_local_optima`: inverse normalized local-optima ratio absolute error

## Strategies

### MRE-only

Selects the fallback with the lowest MRE, i.e., the largest `q_mre`.

Held-out system results:

| System | Selected | Selected RQ2 | RQ2 rank | Oracle | Oracle RQ2 | Regret |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| mysql | rf | 0.588 | 2 | dal_rf | 0.615 | 0.038 |
| postgresql | rf | 0.560 | 6 | kNN | 0.774 | 0.185 |
| httpd | rf | 0.584 | 2 | rf_uncertainty | 0.674 | 0.169 |
| tomcat | rf | 0.923 | 1 | rf | 0.923 | 0.000 |
| gcc | kNN | 0.190 | 1 | kNN | 0.190 | 0.000 |
| clang | kNN | -0.135 | 4 | dal_rf | 0.048 | 0.446 |
| x264 | rf | 0.814 | 1 | rf | 0.814 | 0.000 |
| x265 | dal_rf | 0.854 | 4 | rf | 0.946 | 0.150 |

### RMSE-only

Selects the fallback with the lowest RMSE, i.e., the largest `q_rmse`.

Held-out system results:

| System | Selected | Selected RQ2 | RQ2 rank | Oracle | Oracle RQ2 | Regret |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| mysql | rf | 0.588 | 2 | dal_rf | 0.615 | 0.038 |
| postgresql | rf | 0.560 | 6 | kNN | 0.774 | 0.185 |
| httpd | rf | 0.584 | 2 | rf_uncertainty | 0.674 | 0.169 |
| tomcat | rf | 0.923 | 1 | rf | 0.923 | 0.000 |
| gcc | kNN | 0.190 | 1 | kNN | 0.190 | 0.000 |
| clang | kNN | -0.135 | 4 | dal_rf | 0.048 | 0.446 |
| x264 | rf | 0.814 | 1 | rf | 0.814 | 0.000 |
| x265 | rf | 0.946 | 1 | rf | 0.946 | 0.000 |

### Spearman-only

Selects the fallback with the highest configuration-level Spearman, i.e., the largest `q_spearman`.

Held-out system results:

| System | Selected | Selected RQ2 | RQ2 rank | Oracle | Oracle RQ2 | Regret |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| mysql | rf | 0.588 | 2 | dal_rf | 0.615 | 0.038 |
| postgresql | rf | 0.560 | 6 | kNN | 0.774 | 0.185 |
| httpd | rf | 0.584 | 2 | rf_uncertainty | 0.674 | 0.169 |
| tomcat | rf | 0.923 | 1 | rf | 0.923 | 0.000 |
| gcc | dal_rf | -0.531 | 7 | kNN | 0.190 | 1.000 |
| clang | dal_rf | 0.048 | 1 | dal_rf | 0.048 | 0.000 |
| x264 | rf | 0.814 | 1 | rf | 0.814 | 0.000 |
| x265 | rf | 0.946 | 1 | rf | 0.946 | 0.000 |

### Kendall-only

Selects the fallback with the highest configuration-level Kendall tau, i.e., the largest `q_kendall`.

Held-out system results:

| System | Selected | Selected RQ2 | RQ2 rank | Oracle | Oracle RQ2 | Regret |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| mysql | rf | 0.588 | 2 | dal_rf | 0.615 | 0.038 |
| postgresql | rf | 0.560 | 6 | kNN | 0.774 | 0.185 |
| httpd | rf | 0.584 | 2 | rf_uncertainty | 0.674 | 0.169 |
| tomcat | rf | 0.923 | 1 | rf | 0.923 | 0.000 |
| gcc | dal_rf | -0.531 | 7 | kNN | 0.190 | 1.000 |
| clang | dal_rf | 0.048 | 1 | dal_rf | 0.048 | 0.000 |
| x264 | rf | 0.814 | 1 | rf | 0.814 | 0.000 |
| x265 | rf | 0.946 | 1 | rf | 0.946 | 0.000 |

### Top5-only

Selects the fallback with the highest Top-5% overlap, i.e., the largest `q_top5`.

Held-out system results:

| System | Selected | Selected RQ2 | RQ2 rank | Oracle | Oracle RQ2 | Regret |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| mysql | rf | 0.588 | 2 | dal_rf | 0.615 | 0.038 |
| postgresql | rf | 0.560 | 6 | kNN | 0.774 | 0.185 |
| httpd | rf_uncertainty | 0.674 | 1 | rf_uncertainty | 0.674 | 0.000 |
| tomcat | dal_rf | 0.739 | 4 | rf | 0.923 | 0.286 |
| gcc | kNNw | 0.089 | 2 | kNN | 0.190 | 0.140 |
| clang | kNN | -0.135 | 4 | dal_rf | 0.048 | 0.446 |
| x264 | rf | 0.814 | 1 | rf | 0.814 | 0.000 |
| x265 | rf | 0.946 | 1 | rf | 0.946 | 0.000 |

### Top10-only

Selects the fallback with the highest Top-10% overlap, i.e., the largest `q_top10`.

Held-out system results:

| System | Selected | Selected RQ2 | RQ2 rank | Oracle | Oracle RQ2 | Regret |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| mysql | rf | 0.588 | 2 | dal_rf | 0.615 | 0.038 |
| postgresql | rf | 0.560 | 6 | kNN | 0.774 | 0.185 |
| httpd | rf_uncertainty | 0.674 | 1 | rf_uncertainty | 0.674 | 0.000 |
| tomcat | rf | 0.923 | 1 | rf | 0.923 | 0.000 |
| gcc | rf_uncertainty | -0.375 | 6 | kNN | 0.190 | 0.783 |
| clang | kNN | -0.135 | 4 | dal_rf | 0.048 | 0.446 |
| x264 | rf | 0.814 | 1 | rf | 0.814 | 0.000 |
| x265 | rf | 0.946 | 1 | rf | 0.946 | 0.000 |

### AutoCorr-only

Selects the fallback with the lowest auto-correlation error, i.e., the largest `q_auto_corr`.

Held-out system results:

| System | Selected | Selected RQ2 | RQ2 rank | Oracle | Oracle RQ2 | Regret |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| mysql | 1NN | -0.079 | 7 | dal_rf | 0.615 | 1.000 |
| postgresql | 1NN | -0.384 | 7 | kNN | 0.774 | 1.000 |
| httpd | dal_rf | 0.498 | 3 | rf_uncertainty | 0.674 | 0.330 |
| tomcat | rf_uncertainty | 0.901 | 2 | rf | 0.923 | 0.033 |
| gcc | 1NN | -0.108 | 3 | kNN | 0.190 | 0.413 |
| clang | 1NN | 0.045 | 2 | dal_rf | 0.048 | 0.007 |
| x264 | dal_rf | 0.706 | 3 | rf | 0.814 | 0.141 |
| x265 | dal_rf | 0.854 | 4 | rf | 0.946 | 0.150 |

### LocalOptima-only

Selects the fallback with the lowest local-optima ratio error, i.e., the largest `q_local_optima`.

Held-out system results:

| System | Selected | Selected RQ2 | RQ2 rank | Oracle | Oracle RQ2 | Regret |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| mysql | kNN | 0.406 | 5 | dal_rf | 0.615 | 0.301 |
| postgresql | loess | 0.627 | 4 | kNN | 0.774 | 0.127 |
| httpd | loess | 0.475 | 4 | rf_uncertainty | 0.674 | 0.374 |
| tomcat | loess | 0.691 | 6 | rf | 0.923 | 0.360 |
| gcc | kNN | 0.190 | 1 | kNN | 0.190 | 0.000 |
| clang | 1NN | 0.045 | 2 | dal_rf | 0.048 | 0.007 |
| x264 | loess | 0.222 | 6 | rf | 0.814 | 0.777 |
| x265 | dal_rf | 0.854 | 4 | rf | 0.946 | 0.150 |

### Equal-weight

Averages all eight quality features with equal weights.

Held-out system results:

| System | Selected | Selected RQ2 | RQ2 rank | Oracle | Oracle RQ2 | Regret |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| mysql | rf | 0.588 | 2 | dal_rf | 0.615 | 0.038 |
| postgresql | rf | 0.560 | 6 | kNN | 0.774 | 0.185 |
| httpd | rf | 0.584 | 2 | rf_uncertainty | 0.674 | 0.169 |
| tomcat | rf | 0.923 | 1 | rf | 0.923 | 0.000 |
| gcc | kNN | 0.190 | 1 | kNN | 0.190 | 0.000 |
| clang | kNN | -0.135 | 4 | dal_rf | 0.048 | 0.446 |
| x264 | rf | 0.814 | 1 | rf | 0.814 | 0.000 |
| x265 | rf | 0.946 | 1 | rf | 0.946 | 0.000 |

### Pairwise-random-forest

Trains a random-forest pairwise ranker. For fallback A and B, the model receives `quality(A) - quality(B)` and learns whether A has higher RQ2 Spearman than B.

Average RF feature importance across LOSO folds:

| Feature | Importance |
| --- | ---: |
| MRE quality | 0.139 |
| RMSE quality | 0.083 |
| Config-Spearman quality | 0.123 |
| Config-Kendall quality | 0.126 |
| Top-5 quality | 0.231 |
| Top-10 quality | 0.131 |
| Auto-corr quality | 0.054 |
| Local-optima quality | 0.113 |

Held-out system results:

| System | Selected | Selected RQ2 | RQ2 rank | Oracle | Oracle RQ2 | Regret |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| mysql | rf | 0.588 | 2 | dal_rf | 0.615 | 0.038 |
| postgresql | rf | 0.560 | 6 | kNN | 0.774 | 0.185 |
| httpd | rf_uncertainty | 0.674 | 1 | rf_uncertainty | 0.674 | 0.000 |
| tomcat | rf | 0.923 | 1 | rf | 0.923 | 0.000 |
| gcc | dal_rf | -0.531 | 7 | kNN | 0.190 | 1.000 |
| clang | kNN | -0.135 | 4 | dal_rf | 0.048 | 0.446 |
| x264 | rf | 0.814 | 1 | rf | 0.814 | 0.000 |
| x265 | rf | 0.946 | 1 | rf | 0.946 | 0.000 |

### LambdaRank

Learns a linear scoring function with LambdaRank-style pairwise gradients. Pairs with larger potential NDCG impact receive larger training weight.

Average learned feature weight across LOSO folds:

| Feature | Weight |
| --- | ---: |
| MRE quality | 0.240 |
| RMSE quality | 0.190 |
| Config-Spearman quality | 0.224 |
| Config-Kendall quality | 0.244 |
| Top-5 quality | 0.292 |
| Top-10 quality | 0.229 |
| Auto-corr quality | 0.160 |
| Local-optima quality | 0.189 |

Held-out system results:

| System | Selected | Selected RQ2 | RQ2 rank | Oracle | Oracle RQ2 | Regret |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| mysql | rf | 0.588 | 2 | dal_rf | 0.615 | 0.038 |
| postgresql | rf | 0.560 | 6 | kNN | 0.774 | 0.185 |
| httpd | rf | 0.584 | 2 | rf_uncertainty | 0.674 | 0.169 |
| tomcat | rf | 0.923 | 1 | rf | 0.923 | 0.000 |
| gcc | kNN | 0.190 | 1 | kNN | 0.190 | 0.000 |
| clang | kNN | -0.135 | 4 | dal_rf | 0.048 | 0.446 |
| x264 | rf | 0.814 | 1 | rf | 0.814 | 0.000 |
| x265 | rf | 0.946 | 1 | rf | 0.946 | 0.000 |

## Overall Comparison

Top-1 and Top-2 measure whether the selected fallback is in the oracle best one or two fallbacks by RQ2 Spearman.
Mean regret is normalized within each system, where lower is better.

| Selector | N | Top-1 | Top-2 | Mean regret | Mean selected RQ2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| MRE-only | 8 | 0.375 | 0.625 | 0.123 | 0.547 |
| RMSE-only | 8 | 0.500 | 0.750 | 0.105 | 0.559 |
| Spearman-only | 8 | 0.500 | 0.750 | 0.174 | 0.491 |
| Kendall-only | 8 | 0.500 | 0.750 | 0.174 | 0.491 |
| Top5-only | 8 | 0.375 | 0.625 | 0.137 | 0.534 |
| Top10-only | 8 | 0.500 | 0.625 | 0.181 | 0.499 |
| AutoCorr-only | 8 | 0.000 | 0.250 | 0.384 | 0.304 |
| LocalOptima-only | 8 | 0.125 | 0.250 | 0.262 | 0.439 |
| Equal-weight | 8 | 0.500 | 0.750 | 0.105 | 0.559 |
| Pairwise-random-forest | 8 | 0.500 | 0.625 | 0.209 | 0.480 |
| LambdaRank | 8 | 0.500 | 0.750 | 0.105 | 0.559 |
