# RQ3 Analysis Plan: Fallback Reliability for Tuner Evaluation

## Goal

RQ3 should answer:

> Why do fallback strategies lead to unreliable tuner evaluation, and how can we mitigate this risk?

The core idea is not only to ask whether a fallback has low prediction error, but whether it preserves the optimizer-relevant properties needed for reliable tuner comparison.

Current hypothesis:

> If a fallback has better accuracy, better top-region fidelity, and better landscape preservation, then the tuner ranking obtained under that fallback should be more consistent with the ground-truth ranking from real-system evaluation.

## Why Accuracy Alone Is Not Enough

A simple analysis like “lower MRE means better fallback” is useful but too limited.

Fallback-based evaluation can fail even when pointwise prediction accuracy looks acceptable, because tuners do not care equally about every configuration. Tuners mainly exploit promising regions, local structures. Therefore, a fallback can have reasonable global error but still distort the top region or create false optima that change tuner rankings.

---

For example, in Clang and Gcc, we found the MRE is acceptable but the overlap ratio in top region is clearly lower than other systems. Correspondingly, the ranking is also distort. Deriving from this observation, top region might also be an important factor for fallback strategy.

---

RQ3 should therefore analyze fallback reliability as a multi-dimensional problem.

## Dimension 1: Pointwise Accuracy

Purpose:

Measure whether fallback predictions are numerically close to true performance.

Potential metrics:

- `mre`
- `mae`
- `rmse`

[//]: # (- `median_mre`)
[//]: # (- `p90_mre`)
[//]: # (- `nmae_iqr`)
[//]: # (- `nrmse_iqr`)
[//]: # (- `bias`)
[//]: # (- `abs_bias`)
[//]: # (- `error_std`)
[//]: # (- `spearman_perf`)
[//]: # (- `kendall_perf`)
[//]: # (- `pearson_perf`)

Interpretation:

- Low pointwise error is a important condition for a trustworthy fallback.

[//]: # (- High tail error, such as high `p90_mre`, can indicate that a fallback occasionally makes severe mistakes.)
- Bias is important because tuners can exploit systematic overestimation or underestimation.

Limitation:

Accuracy alone does not explain whether the fallback preserves the search-relevant structure of the configuration space.

## Dimension 2: Top-Region Fidelity

Purpose:

Measure whether the fallback correctly identifies promising configurations.

Potential metrics:

- `top5_overlap`
- `top10_overlap`
- `ndcg_top5`
- `ndcg_top10`





Interpretation:

- Tuner evaluation is more sensitive to the top region than to average global prediction error.
- A fallback can have acceptable MRE but still fail if it misplaces the elite region.
- Top-region fidelity may explain ground-truth ranking correlation better than global accuracy.

Guideline:

When selecting a fallback, do not only check global error. Check whether it preserves the best-performing region.

## Dimension 3: Landscape Structure Preservation

Purpose:

Measure whether the fallback preserves the structure of the performance landscape.

Current implemented metrics: (we should carefully choose landscape metrics. Remove local optima and keep auto-correlation.)

- `local_optima_ratio`
- `local_optima_count`
- `auto_correlation`


Interpretation:

- `local_optima_ratio` and `local_optima_count` measure how many local traps exist in the landscape.
- `auto_correlation` measures ruggedness along a random walk:
  - high autocorrelation means smoother landscape;
  - low autocorrelation means more rugged landscape.

Important comparison:

For each fallback, compare:

```text
true_landscape_metric vs predicted_landscape_metric
```

Then compute:

```text
landscape_metric_abs_error
```

Potential explanation:

- If a fallback over-smooths the landscape, model-based or local-search tuners may appear artificially strong.
- If a fallback creates too many local optima, evolutionary or local-search methods may be misled.
- If the fallback changes landscape ruggedness, the relative ranking of tuners can change even when pointwise accuracy is not terrible.

Guideline:

A fallback should preserve optimizer-relevant landscape structure, not only pointwise performance values.

## Discussion: Tuner-Fallback Interaction

Purpose:

Analyze why different tuners are affected differently by the same fallback.

Tuner behavior can differ substantially:

- Random Search samples broadly.
- GA may exploit local or evolutionary structure.
- FLASH/SMAC/HEBO rely on model-guided search.
- BestConfig may depend more strongly on observed configurations.
- LINE/EZR/PromiseTune may have local or structured exploration bias.

Suggested metrics per tuner:

- fallback usage rate
- distance from tuner-generated configs to observed table
- predicted-vs-true rank of tuner-selected configs
- trajectory regret
- best-so-far curve difference between fallback and true evaluation
- fraction of selected configs located in fallback-created false optima

Interpretation:

- Fallback error is not uniform across tuners.
- A fallback can favor tuners whose search behavior aligns with the artifacts introduced by that fallback.
- This interaction can explain ranking flips.

Case study idea:

Choose a system where two fallback strategies produce clearly different tuner rankings. Compare:

- true landscape
- fallback A landscape
- fallback B landscape
- tuner trajectories
- top-region overlap
- local optima ratio
- autocorrelation

Then explain why a specific tuner benefits under one fallback but not under another.

---

## Resolution

We can resolve RQ3 with three connected parts:

1. RQ3 main analysis: explain which fallback quality metrics are associated with ground-truth-consistent tuner rankings.
2. Discussion: explain specific cases where fallback strategies change tuner behavior.
3. Guideline: propose an automatic fallback selection strategy based on the measured metrics.

### Layer 1: RQ3 Main Analysis

Question:

Which fallback quality metrics best explain consistency with ground-truth tuner ranking?

Compare RQ2 ground-truth ranking correlation against:

- pointwise accuracy metrics
- top-region fidelity metrics
- landscape preservation metrics

Expected outcome:

Top-region fidelity and landscape preservation may explain ranking reliability better than global MRE alone.

### Layer 2: Discussion and Interpretability

Question:

Why do specific fallback strategies change tuner rankings?

Use selected case studies:

- one system where fallback rankings differ strongly;
- one reliable fallback;
- one unreliable fallback;
- selected tuners with large rank changes.

Analyze:

- how fallback changes top region;
- how fallback changes local optima;
- how fallback changes autocorrelation;
- whether specific tuners benefit from the landscape shaped by a fallback.

Expected outcome:

Provide concrete causal explanations for ranking changes.

This layer should be placed in the discussion section. It is useful for explaining why fallback A and fallback B produce different tuner rankings on specific systems, but it is not the main quantitative answer to RQ3.

### Layer 3: Guideline and Automatic Fallback Selection

Question:

How should researchers and practitioners design or validate fallback strategies before using them for tuner evaluation?

The practical guideline is to build an automatic fallback selection procedure. Different fallback strategies can perform best on different datasets because dataset characteristics differ. Therefore, instead of recommending one universal fallback, we can select a fallback adaptively using the proposed metrics.

Proposed adaptive selection strategy:

1. Split table data into observed and holdout configurations.
2. Evaluate all candidate fallback strategies on holdout configurations.
3. Rank fallbacks by pointwise accuracy metrics and keep the top 3.
4. Among the top 3, rank fallbacks by top-region fidelity and keep the top 2.
5. Among the top 2, select the fallback with the best landscape preservation.
6. Use the selected fallback for tuner evaluation.

Example:

```text
candidate_fallbacks
  -> top 3 by accuracy
  -> top 2 by top-region fidelity
  -> top 1 by landscape preservation
```

This procedure reflects the idea that fallback reliability is hierarchical:

- first, the fallback should be reasonably accurate globally;
- second, it should preserve the promising region;
- third, it should avoid distorting the landscape structure.

Possible scoring choices:

- Accuracy stage:
  - lower `mre`, `mae`, or `rmse`
- Top-region stage:
  - higher `top5_overlap`, `top10_overlap`, `ndcg_top5`, or `ndcg_top10`
- Landscape stage:
  - lower `local_optima_ratio_abs_error`
  - lower `local_optima_count_abs_error`
  - lower `auto_correlation_abs_error`

Expected outcome:

The guideline does not claim that a single fallback is always best. Instead, it provides a data-adaptive way to choose a fallback before using fallback-based tuner evaluation.

### We can say our method can be an automatic tools. The users only need to feed their dataset, we will recommend a fallback to each dataset/system. Alternatively, users can design their fallback strategy and embed into this framework.


## Practical Design Guidelines

A fallback is more reliable when:

- it has low global prediction error;
- it preserves top-performing regions;
- it preserves local optima structure and ruggedness;
- it does not strongly distort the landscape in ways that change tuner ranking.

## Implementation Roadmap

Current implemented pieces:

- prediction storage:
  - `statistical_tools/RQ3/results/predictions/{fallback}/{system}_{scale}_predictions.csv`
- accuracy metrics:
  - `statistical_tools/RQ3/scripts/accuracy_metrics.py`
- landscape metrics:
  - `statistical_tools/RQ3/scripts/landscape_metrics.py`
- metric computation:
  - `statistical_tools/RQ3/scripts/rq3_2_compute_accuracy_landscape_metrics.py`

Recommended next additions:

1. Add top-region regret metrics to `accuracy_metrics.py`.
2. Store or reconstruct tuner-generated configuration traces for discussion case studies.
3. Add case-study scripts for selected systems and fallback strategies.
