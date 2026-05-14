# Task: Extend Fallback Analysis with Landscape Preservation Metrics

只在 `fallback_analysis` 下扩展分析，不要修改 `datasets/`、`results/`、`results_system/` 等原始数据或历史实验输出。

## Background

项目研究的问题是：当 tuner 在离线表格数据集上优化配置时，如果生成的 configuration 不存在于表中，就需要 fallback strategy 估计该 configuration 的 performance。不同 fallback strategy 会改变 tuner 的评价结果，进一步改变 tuner 的排名。

已有 RQ：

1. RQ1: 在相同数据集、相同 tuner 下，替换不同 fallback strategy，比较 Scott-Knott ranking 是否一致。当前发现 ranking 差异明显。
2. RQ2: 在真实系统上运行 tuner，不需要 fallback，得到 ground-truth tuner ranking。然后比较不同 fallback 下得到的 tuner ranking 与 ground truth 的相关性。
3. RQ3: 解释为什么某些 fallback 更接近 ground truth，以及如何设计更可靠的 fallback guideline。

当前 `fallback_analysis/analysis.py` 已经做 sparse-to-full reconstruction：

- 从完整 table 中抽取一部分作为 train/observed table。
- 使用 train table + fallback 预测 holdout configurations。
- 比较 predicted performance 和 true performance。
- 当前已有指标包括 MRE、MAE、RMSE、rank correlation、top-k overlap 和 NDCG。

现在需要把它扩展为 RQ3 的 landscape-preservation analysis。

## Overall Goal

新增 landscape metrics，用来判断 fallback 是否保留真实 performance landscape 的关键结构，而不是只看 pointwise prediction accuracy。

核心思想：

- 对同一批 holdout configurations，分别有 `true_perfs` 和 `pred_perfs`。
- Landscape 的邻接结构必须只由 configurations 决定，不能由 performance 决定。
- 因此 true 和 pred 应该共享同一个 adaptive radius 和 neighbor graph。
- 然后在同一个 neighbor graph 上分别计算 true landscape metrics 和 predicted landscape metrics，再比较二者差异。

## Files To Modify Or Add

允许新增或修改：

- `fallback_analysis/analysis.py`
- `fallback_analysis/metric_visualization.py`
- `fallback_analysis/landscape_metrics.py`
- `fallback_analysis/accuracy_metrics.py` 我感觉可以把analysis里面计算mre 等指标的都放到这里面来，便于代码管理
- 如确有必要，可新增一个简单 visualization 脚本。

不要修改：

- `datasets/`
- `results/`
- `results_system/`
- tuner 实现
- utils 实现，除非绝对必要

## 1. Add `fallback_analysis/landscape_metrics.py`

新建 `fallback_analysis/landscape_metrics.py`。

该文件只负责 landscape 指标计算：

- 不读写 CSV。
- 不调用 fallback。
- 不画图。
- 不引入复杂 class。

### Main Inputs

主要函数应接收：

- `configs`: `pd.DataFrame` 或 `np.ndarray`，表示 configurations，不包含 performance 列。
- `perfs`: `np.ndarray` 或 list，表示每个 configuration 的 performance。
- `maximize`: bool，表示 raw performance 是否越大越好。

该模块既要支持计算真实 performance landscape：

```python
compute_landscape_metrics(configs, true_perfs, maximize=True)
不同的landscape metric 不同的一个方法. 先实现一个最简单的统计local optima 的数量
```

也要支持计算预测 performance landscape：

```python
compute_landscape_metrics(configs, pred_perfs, maximize=True)
```

更推荐实现 pair function：

```python
compute_landscape_pair_metrics(configs, true_perfs, pred_perfs, maximize=True) -> dict
```

该函数内部：

1. 编码 configs。
2. 根据 configs 自适应选择 radius。
3. 构建一次 neighbor graph。
4. 在同一个 graph 上分别计算 true 和 pred 的 landscape metrics。
5. 返回 true/pred/diff 指标。

## 2. Distance Encoding

不要使用 Hamming distance。

原因：配置空间包含 numeric/non-categorical 参数，Hamming distance 不合适。

请使用 normalized Manhattan distance。

要求：

- 对 numeric 列：min-max normalize 到 `[0, 1]`。
- 对 categorical/binary 列：实现要简单。可以用 `OrdinalEncoder` 后再归一化。
- 最终每个维度应该尽量处于相近尺度，避免大数值列支配距离。

建议实现：

```python
encode_configs_for_distance(configs: pd.DataFrame | np.ndarray) -> np.ndarray
```

返回二维 numeric matrix。

注意：

- 对 NaN 做简单处理，例如转成字符串类别或用列中位数/0 填充。
- 不要让 encoding 因单个异常值中断整个分析。

## 3. Adaptive Neighbor Radius

不要写死 radius。

需要根据数据自动选择 radius，使得每个 configuration 的平均邻居数接近配置维度数。

例如：

- 如果 config dimension = 20，则希望 average neighbor count 接近 20。
- `target_neighbors = config_dim`
- 允许一定误差，例如 `tolerance = 0.1` 或至少 `±1`。

请实现：

```python
find_adaptive_radius(encoded_configs, target_neighbors=None, tolerance=0.1) -> float
```

要求：

- 使用 normalized Manhattan distance。
- 不包含 self-neighbor。
- `target_neighbors` 默认等于配置维度数。
- 如果数据太小，`target_neighbors` 不应超过 `n - 1`。
- 可以用 binary search，也可以基于 pairwise distance candidates 搜索。
- 返回使 average neighbor count 最接近 target_neighbors 的 radius。
- 如果无法精确达到，返回误差最小的 radius。
- 注意性能，不要写三层循环。
- 可使用 `sklearn.neighbors.NearestNeighbors(metric="manhattan")` 或 `sklearn.metrics.pairwise_distances`。

## 4. Neighbor Graph

基于 adaptive radius 构建邻接关系。

建议实现：

```python
build_neighbor_graph(encoded_configs, radius) -> list[np.ndarray]
```

要求：

- 返回每个 configuration 的邻居索引列表。
- 不包含自己。
- 允许某些点没有邻居。
- 后续指标应跳过无邻居点或返回 NaN，不要报错。

## 5. Landscape Metrics

在 `landscape_metrics.py` 中实现：

```python
compute_landscape_metrics(configs, perfs, maximize=True) -> dict
```

至少返回以下指标：

- `radius`: 自适应得到的邻居半径。
- `avg_neighbor_count`: 平均邻居数。
- `isolated_ratio`: 没有邻居的 configuration 占比。
- `neighbor_diff_mean`: 所有邻接边上的平均绝对 score 差异，用于衡量 ruggedness。
- `neighbor_diff_std`: 邻接边 score 差异的标准差。
- `local_optima_ratio`: 局部最优点比例。一个点如果 score 大于等于所有邻居 score，则视为 local optimum。没有邻居的点不要计入分母。
- `local_optima_count`: 局部最优点数量。
- `smoothness_correlation`: 对所有邻接边，计算两端 score 的 Spearman correlation。越高表示邻域越平滑。若无法计算则返回 NaN。

所有 landscape 指标都应基于统一方向的 score：

- 如果 `maximize=True`，`score = perfs`。
- 如果 `maximize=False`，`score = -perfs`。

对 NaN/inf perfs 要过滤或跳过，不能让整个脚本崩溃。

## 6. Pair Metrics Output

在 `compute_landscape_pair_metrics(configs, true_perfs, pred_perfs, maximize=True)` 中返回字段建议如下：

```text
true_radius
pred_radius
true_avg_neighbor_count
pred_avg_neighbor_count
true_isolated_ratio
pred_isolated_ratio
true_neighbor_diff_mean
pred_neighbor_diff_mean
neighbor_diff_abs_error
neighbor_diff_rel_error
true_neighbor_diff_std
pred_neighbor_diff_std
true_local_optima_ratio
pred_local_optima_ratio
local_optima_ratio_abs_error
true_local_optima_count
pred_local_optima_count
true_smoothness_correlation
pred_smoothness_correlation
smoothness_correlation_abs_error
```

重要：

- true 和 pred 必须使用同一批 configs。
- true 和 pred 必须共享同一个 radius 和 neighbor graph。
- 因为 graph 由 configs 决定，所以 `true_radius` 和 `pred_radius` 实际应相同。
- 保留 `true_radius` 和 `pred_radius` 字段是为了 CSV 结构清晰，也可以只计算一次后写入两个字段。

## 7. Integrate Into `analysis.py`

现有 `analysis.py` 在每个 split/fallback 下已经得到：

- `holdout_df`
- `y_true`
- `y_pred`

请新增逻辑：

```python
configs = holdout_df.iloc[:, :-1]
maximize = infer_optimization_goal(system) == "maximum"
landscape_dict = compute_landscape_pair_metrics(
    configs=configs,
    true_perfs=y_true,
    pred_perfs=y_pred,
    maximize=maximize,
)
```

然后把 `landscape_dict` 合并进当前 row。

新增指标需要自动进入：

- 每个 system 的 raw CSV。
- `fallback_reconstruction_raw_all.csv`
- `fallback_reconstruction_summary.csv`

每一行仍必须包含：

```text
system
scale
train_frac
repeat
fallback
```

## 8. Add Accuracy Metrics

在现有 reconstruction metrics 中新增：

- `median_mre`
- `p90_mre`
- `bias`
- `abs_bias`
- `error_std`

保留已有指标：

- `mre`
- `mae`
- `rmse`
- `nmae_iqr`
- `nrmse_iqr`
- `spearman_perf`
- `kendall_perf`
- `pearson_perf`
- `top5_overlap`
- `top10_overlap`
- `ndcg_top5`
- `ndcg_top10`

定义建议：

```text
median_mre = median(abs(y_pred - y_true) / max(abs(y_true), eps)) * 100
p90_mre = 90th percentile of MRE values
bias = mean(y_pred - y_true)
abs_bias = abs(bias)
error_std = std(y_pred - y_true)
```

## 9. Training Fraction

将 `TRAIN_FRACS` 改为更稀疏场景：

```python
TRAIN_FRACS = [0.05]
```

代码仍应支持多个比例，例如未来改成：

```python
TRAIN_FRACS = [0.05, 0.1, 0.2]
```

不需要改其他逻辑。

## 10. Output Paths

继续使用现有输出目录：

```text
fallback_analysis/analysis_results/fallback_reconstruction/
```

继续保存：

```text
*_reconstruction_raw.csv
fallback_reconstruction_raw_all.csv
fallback_reconstruction_summary.csv
```

新增指标应进入这些 CSV。

不要覆盖原始 datasets，不要删除旧结果。

## 11. Visualization

更新或新增 visualization 脚本，至少输出以下指标图。

Accuracy:

- `mre`
- `median_mre`
- `p90_mre`
- `top5_overlap`

Landscape:

- `neighbor_diff_abs_error`
- `neighbor_diff_rel_error`
- `local_optima_ratio_abs_error`
- `smoothness_correlation_abs_error`
- `true_avg_neighbor_count`
- `true_isolated_ratio`

输出目录建议：

```text
fallback_analysis/visualization/accuracy/
fallback_analysis/visualization/landscape/
```

图的要求：

- x-axis: fallback
- y-axis: metric
- 每个 system 单独输出
- 同时保存画图用的 summary CSV
- 使用 matplotlib 即可，不需要复杂交互图

## 12. Code Style Requirements

- 不要引入复杂 class。
- 函数保持短小清晰。
- 不重构整个项目。
- 优先复用现有 `analysis.py` 的 helper，例如 `infer_optimization_goal()` 和 `objective_to_score()`。
- 对 NaN、constant vector、样本太少、无邻居等情况返回 NaN 或合理默认值，不要中断整个实验。
- 添加必要但简短的注释。
- 不要修改原始数据。
- 不要引入不必要的新依赖。

## Expected Interpretation

最终分析应能支持以下论文论点：

- Pointwise accuracy 低并不一定足以解释 fallback 是否可靠。
- 更关键的是 fallback 是否保留 optimization-relevant landscape structure。
- 如果 fallback 过度平滑 landscape，可能减少 local optima，但也可能破坏真实 top region。
- 如果 fallback 制造过多局部最优或邻域噪声，可能让 local search / evolutionary tuner 被误导。
- 如果 top region overlap 和 landscape preservation 更能解释 RQ2 ground-truth correlation，则说明这些指标比 MRE 更适合作为 fallback guideline。
