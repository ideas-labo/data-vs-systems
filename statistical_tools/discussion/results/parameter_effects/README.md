# 最大单参数效应

## 目的

检查局部不平滑的系统是否仍有可辨识的单参数性能排序规律。每个系统分别计算 20 个 option 的边际效应，只报告最大的一个，与论文双子图 (b) 对应。

## 计算

输入 `datasets/{system}/{system}_large.csv`，前 20 列为 option，最后一列为性能，保留全部记录。

1. 将性能统一为越大越好的 score：MySQL/PostgreSQL/Httpd/Tomcat 用原性能，其余用负性能。
2. 计算平均秩并除以样本数，得到性能百分位 q；并列值用平均秩。
3. 对每个 option 按取值分组。数值取值超过六个时，固定使用最多六个配置值等频组（重复边界合并）；其余保留原取值。分组不使用性能，不做箱数敏感性分析。
4. 计算单因素 ANOVA 的 η²：

$$\eta_j^2=\frac{\sum_g n_g(\bar q_g-\bar q)^2}{\sum_i(q_i-\bar q)^2}.$$

5. 取二十个值中的最大值，记录其 option 名称。图中显示 `100 × η²`，单位为 %。

例如四条记录，A 为 `0,0,1,1`，性能为 `10,10,20,30`。百分位为 `0.375,0.375,0.75,1`，总体均值为 `0.625`。两组均值为 `0.375,0.875`，组间平方和为 `0.25`，总平方和为 `0.28125`，因此 A 的 η² 为 `0.8889`，图上显示 `88.889%`。它不是性能提升百分比。

## 八系统结果

| 系统 | 最大 η²（%） | 对应 option |
| --- | ---: | --- |
| MySQL | 33.916 | innodb_flush_log_at_trx_commit |
| PostgreSQL | 2.277 | seq_page_cost |
| Httpd | 42.105 | KeepAlive |
| Tomcat | 13.325 | minSpareThreads |
| GCC | 0.045 | fno-inline-small-functions |
| Clang | 0.017 | funroll-loops |
| x264 | 25.773 | me |
| x265 | 36.992 | rd |

[精确结果](large/system_summary.csv)。PostgreSQL/Tomcat 即使 AC 较低，仍有比 GCC/Clang 更明显的边际关联。这些规律可能为 fallback 提供线索；未验证 fallback 实际利用它们。

η² 表示已采样数据中组均值的区分程度，不是控制其他 option 后的因果作用、联合解释率或预测准确率。该分析不作 F 检验，也不排除其他形式的结构。性能恒定时效应未定义，输出 NaN。

## 运行与输出

```sh
.venv/bin/python statistical_tools/discussion/scripts/discussion_4_analyze_parameter_effects.py
```

唯一输出：`large/system_summary.csv`，每系统一行，包含 `system`、`strongest_parameter`、`max_percentile_eta_squared`。支持指定 systems、scale、dataset-root、output-dir；分箱固定为六组。

参考：[NIST 单因素 ANOVA 平方和分解](https://itl.nist.gov/div898/handbook/prc/section4/prc431.htm)；[effectsize 的 η² 定义](https://cran.r-project.org/web/packages/effectsize/vignettes/anovaES.html)。
