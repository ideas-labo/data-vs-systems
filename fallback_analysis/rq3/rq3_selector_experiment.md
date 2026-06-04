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
| mysql | rf | 0.563 | 1 | dal_rf | 0.563 | 0.000 |
| postgresql | rf | 0.345 | 6 | kNN | 0.631 | 0.452 |
| httpd | rf | 0.873 | 2 | rf_uncertainty | 0.908 | 0.068 |
| tomcat | rf | 0.916 | 2 | dal_rf | 0.954 | 0.070 |
| gcc | kNN | -0.411 | 3 | 1NN | -0.100 | 0.402 |
| clang | kNN | 0.500 | 1 | kNN | 0.500 | 0.000 |
| x264 | rf | 0.845 | 2 | rf_uncertainty | 0.918 | 0.070 |
| x265 | dal_rf | 0.821 | 4 | rf | 0.955 | 0.186 |

### RMSE-only

Selects the fallback with the lowest RMSE, i.e., the largest `q_rmse`.

Held-out system results:

| System | Selected | Selected RQ2 | RQ2 rank | Oracle | Oracle RQ2 | Regret |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| mysql | rf | 0.563 | 1 | dal_rf | 0.563 | 0.000 |
| postgresql | rf | 0.345 | 6 | kNN | 0.631 | 0.452 |
| httpd | rf | 0.873 | 2 | rf_uncertainty | 0.908 | 0.068 |
| tomcat | rf | 0.916 | 2 | dal_rf | 0.954 | 0.070 |
| gcc | kNN | -0.411 | 3 | 1NN | -0.100 | 0.402 |
| clang | kNN | 0.500 | 1 | kNN | 0.500 | 0.000 |
| x264 | rf | 0.845 | 2 | rf_uncertainty | 0.918 | 0.070 |
| x265 | rf | 0.955 | 1 | rf | 0.955 | 0.000 |

### Spearman-only

Selects the fallback with the highest configuration-level Spearman, i.e., the largest `q_spearman`.

Held-out system results:

| System | Selected | Selected RQ2 | RQ2 rank | Oracle | Oracle RQ2 | Regret |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| mysql | rf | 0.563 | 1 | dal_rf | 0.563 | 0.000 |
| postgresql | rf | 0.345 | 6 | kNN | 0.631 | 0.452 |
| httpd | rf | 0.873 | 2 | rf_uncertainty | 0.908 | 0.068 |
| tomcat | rf | 0.916 | 2 | dal_rf | 0.954 | 0.070 |
| gcc | dal_rf | -0.874 | 6 | 1NN | -0.100 | 1.000 |
| clang | dal_rf | -0.095 | 4 | kNN | 0.500 | 0.675 |
| x264 | rf | 0.845 | 2 | rf_uncertainty | 0.918 | 0.070 |
| x265 | rf | 0.955 | 1 | rf | 0.955 | 0.000 |

### Kendall-only

Selects the fallback with the highest configuration-level Kendall tau, i.e., the largest `q_kendall`.

Held-out system results:

| System | Selected | Selected RQ2 | RQ2 rank | Oracle | Oracle RQ2 | Regret |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| mysql | rf | 0.563 | 1 | dal_rf | 0.563 | 0.000 |
| postgresql | rf | 0.345 | 6 | kNN | 0.631 | 0.452 |
| httpd | rf | 0.873 | 2 | rf_uncertainty | 0.908 | 0.068 |
| tomcat | rf | 0.916 | 2 | dal_rf | 0.954 | 0.070 |
| gcc | dal_rf | -0.874 | 6 | 1NN | -0.100 | 1.000 |
| clang | dal_rf | -0.095 | 4 | kNN | 0.500 | 0.675 |
| x264 | rf | 0.845 | 2 | rf_uncertainty | 0.918 | 0.070 |
| x265 | rf | 0.955 | 1 | rf | 0.955 | 0.000 |

### Top5-only

Selects the fallback with the highest Top-5% overlap, i.e., the largest `q_top5`.

Held-out system results:

| System | Selected | Selected RQ2 | RQ2 rank | Oracle | Oracle RQ2 | Regret |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| mysql | rf | 0.563 | 1 | dal_rf | 0.563 | 0.000 |
| postgresql | rf | 0.345 | 6 | kNN | 0.631 | 0.452 |
| httpd | rf_uncertainty | 0.908 | 1 | rf_uncertainty | 0.908 | 0.000 |
| tomcat | dal_rf | 0.954 | 1 | dal_rf | 0.954 | 0.000 |
| gcc | kNNw | -0.250 | 2 | 1NN | -0.100 | 0.194 |
| clang | kNN | 0.500 | 1 | kNN | 0.500 | 0.000 |
| x264 | rf | 0.845 | 2 | rf_uncertainty | 0.918 | 0.070 |
| x265 | rf | 0.955 | 1 | rf | 0.955 | 0.000 |

### Top10-only

Selects the fallback with the highest Top-10% overlap, i.e., the largest `q_top10`.

Held-out system results:

| System | Selected | Selected RQ2 | RQ2 rank | Oracle | Oracle RQ2 | Regret |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| mysql | rf | 0.563 | 1 | dal_rf | 0.563 | 0.000 |
| postgresql | rf | 0.345 | 6 | kNN | 0.631 | 0.452 |
| httpd | rf_uncertainty | 0.908 | 1 | rf_uncertainty | 0.908 | 0.000 |
| tomcat | rf | 0.916 | 2 | dal_rf | 0.954 | 0.070 |
| gcc | rf_uncertainty | -0.461 | 4 | 1NN | -0.100 | 0.466 |
| clang | kNN | 0.500 | 1 | kNN | 0.500 | 0.000 |
| x264 | rf | 0.845 | 2 | rf_uncertainty | 0.918 | 0.070 |
| x265 | rf | 0.955 | 1 | rf | 0.955 | 0.000 |

### AutoCorr-only

Selects the fallback with the lowest auto-correlation error, i.e., the largest `q_auto_corr`.

Held-out system results:

| System | Selected | Selected RQ2 | RQ2 rank | Oracle | Oracle RQ2 | Regret |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| mysql | 1NN | -0.101 | 7 | dal_rf | 0.563 | 1.000 |
| postgresql | 1NN | 0.000 | 7 | kNN | 0.631 | 1.000 |
| httpd | dal_rf | 0.617 | 4 | rf_uncertainty | 0.908 | 0.561 |
| tomcat | rf_uncertainty | 0.906 | 3 | dal_rf | 0.954 | 0.088 |
| gcc | 1NN | -0.100 | 1 | 1NN | -0.100 | 0.000 |
| clang | 1NN | 0.306 | 2 | kNN | 0.500 | 0.220 |
| x264 | dal_rf | 0.736 | 3 | rf_uncertainty | 0.918 | 0.175 |
| x265 | dal_rf | 0.821 | 4 | rf | 0.955 | 0.186 |

### LocalOptima-only

Selects the fallback with the lowest local-optima ratio error, i.e., the largest `q_local_optima`.

Held-out system results:

| System | Selected | Selected RQ2 | RQ2 rank | Oracle | Oracle RQ2 | Regret |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| mysql | kNN | 0.090 | 6 | dal_rf | 0.563 | 0.712 |
| postgresql | loess | 0.536 | 3 | kNN | 0.631 | 0.150 |
| httpd | loess | 0.468 | 5 | rf_uncertainty | 0.908 | 0.848 |
| tomcat | loess | 0.406 | 7 | dal_rf | 0.954 | 1.000 |
| gcc | kNN | -0.411 | 3 | 1NN | -0.100 | 0.402 |
| clang | 1NN | 0.306 | 2 | kNN | 0.500 | 0.220 |
| x264 | loess | 0.296 | 6 | rf_uncertainty | 0.918 | 0.597 |
| x265 | dal_rf | 0.821 | 4 | rf | 0.955 | 0.186 |

### Equal-weight

Averages all eight quality features with equal weights.

Held-out system results:

| System | Selected | Selected RQ2 | RQ2 rank | Oracle | Oracle RQ2 | Regret |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| mysql | rf | 0.563 | 1 | dal_rf | 0.563 | 0.000 |
| postgresql | rf | 0.345 | 6 | kNN | 0.631 | 0.452 |
| httpd | rf | 0.873 | 2 | rf_uncertainty | 0.908 | 0.068 |
| tomcat | rf | 0.916 | 2 | dal_rf | 0.954 | 0.070 |
| gcc | kNN | -0.411 | 3 | 1NN | -0.100 | 0.402 |
| clang | kNN | 0.500 | 1 | kNN | 0.500 | 0.000 |
| x264 | rf | 0.845 | 2 | rf_uncertainty | 0.918 | 0.070 |
| x265 | rf | 0.955 | 1 | rf | 0.955 | 0.000 |

### Pairwise-random-forest

Trains a random-forest pairwise ranker. For fallback A and B, the model receives `quality(A) - quality(B)` and learns whether A has higher RQ2 Spearman than B.

Average RF feature importance across LOSO folds:

| Feature | Importance |
| --- | ---: |
| MRE quality | 0.166 |
| RMSE quality | 0.050 |
| Config-Spearman quality | 0.095 |
| Config-Kendall quality | 0.084 |
| Top-5 quality | 0.264 |
| Top-10 quality | 0.136 |
| Auto-corr quality | 0.062 |
| Local-optima quality | 0.143 |

Held-out system results:

| System | Selected | Selected RQ2 | RQ2 rank | Oracle | Oracle RQ2 | Regret |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| mysql | rf | 0.563 | 1 | dal_rf | 0.563 | 0.000 |
| postgresql | rf | 0.345 | 6 | kNN | 0.631 | 0.452 |
| httpd | rf_uncertainty | 0.908 | 1 | rf_uncertainty | 0.908 | 0.000 |
| tomcat | dal_rf | 0.954 | 1 | dal_rf | 0.954 | 0.000 |
| gcc | kNN | -0.411 | 3 | 1NN | -0.100 | 0.402 |
| clang | kNN | 0.500 | 1 | kNN | 0.500 | 0.000 |
| x264 | rf | 0.845 | 2 | rf_uncertainty | 0.918 | 0.070 |
| x265 | rf | 0.955 | 1 | rf | 0.955 | 0.000 |

### LambdaRank

Learns a linear scoring function with LambdaRank-style pairwise gradients. Pairs with larger potential NDCG impact receive larger training weight.

Average learned feature weight across LOSO folds:

| Feature | Weight |
| --- | ---: |
| MRE quality | 0.253 |
| RMSE quality | 0.197 |
| Config-Spearman quality | 0.234 |
| Config-Kendall quality | 0.264 |
| Top-5 quality | 0.357 |
| Top-10 quality | 0.283 |
| Auto-corr quality | 0.200 |
| Local-optima quality | 0.223 |

Held-out system results:

| System | Selected | Selected RQ2 | RQ2 rank | Oracle | Oracle RQ2 | Regret |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| mysql | rf | 0.563 | 1 | dal_rf | 0.563 | 0.000 |
| postgresql | rf | 0.345 | 6 | kNN | 0.631 | 0.452 |
| httpd | rf | 0.873 | 2 | rf_uncertainty | 0.908 | 0.068 |
| tomcat | dal_rf | 0.954 | 1 | dal_rf | 0.954 | 0.000 |
| gcc | kNN | -0.411 | 3 | 1NN | -0.100 | 0.402 |
| clang | kNN | 0.500 | 1 | kNN | 0.500 | 0.000 |
| x264 | rf | 0.845 | 2 | rf_uncertainty | 0.918 | 0.070 |
| x265 | rf | 0.955 | 1 | rf | 0.955 | 0.000 |

## Overall Comparison

Top-1 and Top-2 measure whether the selected fallback is in the oracle best one or two fallbacks by RQ2 Spearman.
Mean regret is normalized within each system, where lower is better.

| Selector | N | Top-1 | Top-2 | Mean regret | Mean selected RQ2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| MRE-only | 8 | 0.250 | 0.625 | 0.156 | 0.557 |
| RMSE-only | 8 | 0.375 | 0.750 | 0.133 | 0.573 |
| Spearman-only | 8 | 0.250 | 0.625 | 0.292 | 0.441 |
| Kendall-only | 8 | 0.250 | 0.625 | 0.292 | 0.441 |
| Top5-only | 8 | 0.625 | 0.875 | 0.090 | 0.603 |
| Top10-only | 8 | 0.500 | 0.750 | 0.132 | 0.572 |
| AutoCorr-only | 8 | 0.125 | 0.250 | 0.404 | 0.398 |
| LocalOptima-only | 8 | 0.000 | 0.125 | 0.514 | 0.314 |
| Equal-weight | 8 | 0.375 | 0.750 | 0.133 | 0.573 |
| Pairwise-random-forest | 8 | 0.625 | 0.750 | 0.115 | 0.583 |
| LambdaRank | 8 | 0.500 | 0.750 | 0.124 | 0.578 |
