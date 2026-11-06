# Discussion

## 第一部分：AC 与最大单参数效应

分析完整真实 dataset 的两个特征，与论文双子图一一对应：

- (a) AC：沿用 RQ3 的邻域及 lag-1 随机游走自相关估计，描述局部性能关联。
- (b) 最大单参数效应：先将性能转换为百分位，分别计算 20 个 option 的单因素 ANOVA η²，只保留最大值及对应 option。图中乘 100，显示为百分数。

### 运行顺序

在项目根目录运行：

```sh
.venv/bin/python statistical_tools/discussion/scripts/discussion_3_analyze_ac.py
.venv/bin/python statistical_tools/discussion/scripts/discussion_4_analyze_parameter_effects.py
.venv/bin/python statistical_tools/discussion/scripts/discussion_5_export_ac_parameter_tikz.py
```

AC 输入为完整 dataset 和 RQ3 缓存的 radius，默认 seed=0；不读取 MDS 点，不重新选择邻域。全部记录保留。AC 计算口径不变，已有八系统 AC 结果继续使用。

### 结果

- [AC 数据](results/ac/large/ac_summary.csv)：八系统各一行。
- [最大效应数据](results/parameter_effects/large/system_summary.csv)：八系统各一行，效应保存为 0–1 比例。
- [单参数效应方法与结果说明](results/parameter_effects/README.md)。
- [论文双子图及编译说明](results/ac_parameter_figure/README.md)。

这部分不计算 FDC、raw 效应、平均/中位效应、交互或分箱敏感性分析，不生成 LaTeX 表格或单独的参数图。唯一的展示图是自包含 TeX 的双子图。

### 解释

AC 较低的系统仍可能存在清楚的参数方向规律。PostgreSQL/Tomcat 的最大边际效应高于 GCC/Clang，为 fallback 可能利用的性能排序规律提供补充线索。这是描述性解释，不证明 fallback 实际利用了这些规律，也不以 η² 判定 tuner 评价可靠性。

验证：

```sh
.venv/bin/python -m unittest discover -s statistical_tools/discussion/tests -v
```

## 其他 discussion 材料

`discussion_1_export_system_level_summary.py` 及其 `results/latex_tables/` 汇总，以及 `discussion_2_visualize_true_vs_fallback_landscape.py` 的 fallback 对比案例，属于其他 discussion 材料，独立保留。其指标不进入上述第一部分。
