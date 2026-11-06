# Discussion 最终双子图

使用一个自包含 TeX：`large/ac_parameter_effects.tex`，包含全部数据及文档结构，无需外部 CSV 或其他 TeX 文件。图底部分别仅标注 (a)、(b)。

- (a)：八系统全数据 AC，非 holdout 平均 AC。
- (b)：每系统 20 个 option 中最大的百分位 η²，乘 100，以百分数显示。

数据来源：`../ac/large/ac_summary.csv` 与 `../parameter_effects/large/system_summary.csv`。完整精度及最强 option 名称见 `large/ac_parameter_data.csv`。最大值为描述性选择，不是全部参数联合解释率或预测精度。

从项目根目录生成和编译：

```sh
.venv/bin/python statistical_tools/discussion/scripts/discussion_5_export_ac_parameter_tikz.py
pdflatex -output-directory=statistical_tools/discussion/results/ac_parameter_figure/large statistical_tools/discussion/results/ac_parameter_figure/large/ac_parameter_effects.tex
```

依赖 standalone、pgfplots 及其 groupplots 库。生成的 PDF 为矢量图，PNG 为预览。
