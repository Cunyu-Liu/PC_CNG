# PC-CNG v3 Optimization C reaction-class 弱类诊断

日期：2026-07-11

## 目标

落实 SOTA 差距分析文档中的 Optimization C：

```text
reaction-class targeted generator expansion
```

当前先完成可复现弱类诊断门禁：同一脚本对 Morgan、graph-stats、external validity-aware 结果按 reaction class 输出 Top-1/MRR/NDCG、样本量、低支持标记和候选配额建议。

## 代码落地

新增：

```text
pc_cng/analyze_reaction_class_benchmark.py
tests/test_reaction_class_benchmark.py
```

功能：

```text
--score-csv NAME=PATH[:SCORE_COLUMN]
--min-groups 20
--weak-top1 0.80
--weak-mrr 0.85
```

输出：

```text
reaction_class_benchmark.json
reaction_class_summary.csv/md
reaction_class_by_split.csv/md
```

诊断规则：

```text
missing: no evaluable groups
low_support: groups < min_groups, recommendation = add_candidate_quota_to_reach_N_groups
weak_performance: top1 < weak_top1 or mrr < weak_mrr, recommendation = class_targeted_generator_or_error_analysis
ok: keep_monitoring
```

## 服务器运行

结果路径：

```text
/home/cunyuliu/pc_cng_research/results/reaction_class_benchmark_20260711
```

比较模型：

```text
morgan_seed20260710_same
graph_stats_ensemble_same
morgan_validity_external
graph_stats_validity_external
```

验证：

```text
local compileall: pass
local unittest test_reaction_class_benchmark.py: OK
remote compileall: pass
remote unittest test_reaction_class_benchmark.py: OK
```

## Same-context 弱类结论

Same-context candidate reranking 中，当前真正问题不是大类性能，而是多个 HiTEA reaction classes 的候选组数不足。

| Class | Morgan seed groups | Morgan Top-1 | Graph-stats groups | Graph-stats Top-1 | Status | Recommendation |
|---|---:|---:|---:|---:|---|---|
| Hydrogenation | 6 | 33.33 | 6 | 33.33 | low_support | add at least 14 groups to reach 20 |
| Ni coupling | 0 | 0.00 | 0 | 0.00 | missing | add at least 20 evaluable groups |
| Amide coupling | 3 | 100.00 | 3 | 100.00 | low_support | add at least 17 groups |
| Cu coupling | 2 | 100.00 | 2 | 100.00 | low_support | add at least 18 groups |
| Rh coupling | 2 | 100.00 | 2 | 100.00 | low_support | add at least 18 groups |
| Alkylation | 5 | 100.00 | 5 | 100.00 | low_support | add at least 15 groups |
| Cabonylation | 5 | 100.00 | 5 | 100.00 | low_support | add at least 15 groups |

Interpretation:

```text
Hydrogenation 的 33.33% Top-1 不应直接解读为模型系统性失败，因为只有 6 个 evaluable groups。
Optimization C 的优先事项是 class-aware minimum candidate quota，而不是先调模型。
```

## External validity-aware 弱类结论

在完整 Chemformer beam validity-aware benchmark 中，类别覆盖充足：

| Model | Weak / low-support finding |
|---|---|
| Morgan validity-aware | all classes ok |
| Graph-stats validity-aware | RegioSQM20 weak_performance, Top-1 79.71, MRR 88.60 |

关键对比：

| Class | Morgan validity-aware Top-1 | Graph-stats validity-aware Top-1 | Recommendation |
|---|---:|---:|---|
| RegioSQM20 | 93.66 | 79.71 | keep Morgan as external main branch; graph-stats needs regio error analysis |
| Alkylation | 82.21 | 99.79 | graph-stats improves this class |
| Untagged | 98.32 | 94.80 | Morgan stronger |
| Pd coupling | 100.00 | 100.00 | stable |
| Hydrogenation | 100.00 | 100.00 | stable in external validity-aware setting |

Interpretation:

```text
Graph-stats fixes several same-context generalization issues but weakens RegioSQM20 in the external benchmark.
Therefore graph-stats remains an architecture supplement, not the main external branch.
```

## Optimization C 下一步

Class-aware generator expansion should target same-context low-support classes:

```text
Hydrogenation, Ni coupling, Amide coupling, Cu coupling, Rh coupling, Alkylation, Cabonylation.
```

Minimum quota:

```text
At least 20 evaluable groups per class in same-context candidate reranking.
```

Concrete implementation recommendation:

1. During hard-negative generation, maintain `class -> generated_groups` counters.
2. For classes below quota, relax diverse-anchor thresholds in a controlled order:
   `max_candidates_per_reaction`, then `max_anchor_distance`, then `min_product_similarity`.
3. Keep known-positive filtering mandatory.
4. Re-run `analyze_reaction_class_benchmark.py` and require all manuscript-visible classes to be either `ok` or explicitly marked `low_support`.

## Class-quota supplement 实验

代码变更：

```text
pc_cng/run_hard_negative_actions.py
  - --include-reaction-class
  - --exclude-candidate-csv
  - canonical duplicate exclusion against existing candidate CSVs
  - output reaction_class in generated candidates

pc_cng/hard_negative_actions.py
  - HardNegativeCandidate.reaction_class

scripts_run_type1_class_quota_supplement.sh
  - reproducible augmented train/eval runner
```

服务器输出：

```text
/home/cunyuliu/pc_cng_research/results/type1_class_quota_supplement_20260711
```

目标类别：

```text
Hydrogenation, Ni coupling, Amide coupling, Cu coupling, Rh coupling, Alkylation, Cabonylation
```

生成设置：

```text
max_candidates_per_reaction: 32
max_candidates_per_pair: 64
max_anchor_distance: 12
min_product_similarity: 0.25
max_product_similarity: 0.999
min_atom_balance: 0.15
known-positive filtering: enabled
exclude existing diverse-anchor candidates: enabled
```

生成/审核结果：

| Class | Generated rows | Strong keep rows |
|---|---:|---:|
| Alkylation | 1 | 1 |
| Amide coupling | 3 | subset of total keep |
| Cabonylation | 2 | subset of total keep |
| Rh coupling | 2 | subset of total keep |
| Hydrogenation | 0 | 0 |
| Ni coupling | 0 | 0 |
| Cu coupling | 0 | 0 |
| Total | 8 | 5 |

Interpretation:

```text
Even with relaxed thresholds, class-quota generation produced only 8 non-duplicate candidates and
5 strong reviewed negatives. The current terminal-substituent/diverse-anchor action space is not
sufficient to solve low-support HiTEA classes. Further progress requires new class-specific action
families or learned candidate generation, not only threshold relaxation.
```

Augmented training/evaluation:

```text
path:
/home/cunyuliu/pc_cng_research/results/type1_class_quota_supplement_20260711/augmented_pairwise_seed20260710
```

| Model | Overall Top-1 | Held-out test Top-1 | Synthetic Top-1 | Binary test ROC-AUC |
|---|---:|---:|---:|---:|
| Base Morgan seed20260710 | 97.40 | 83.58 | 96.50 | 85.29 |
| Class-quota augmented Morgan seed20260710 | 97.59 | 88.06 | 96.67 | 85.67 |

Class support after augmentation:

```text
Hydrogenation remains 6 groups / 13 rows / 33.33 Top-1.
Ni coupling remains 0 evaluable groups.
Cu coupling remains 2 groups.
Rh coupling candidate rows increase from 6 to 8 but groups remain 2.
```

Conclusion:

```text
The small supplement improves held-out test Top-1 by +4.48 pp in seed20260710,
but it does not resolve the class-support gate. It should be treated as evidence that
targeted candidates can help, while the generator itself must be expanded with new
class-specific edits for Hydrogenation/Ni/Cu/Rh/Amide classes.
```

## Class-fallback action family

新增 action：

```text
class_fallback
```

核心思想：

```text
For low-support classes where anchor migration rarely applies, generate no-conversion /
partial-conversion candidates by using reactant-side molecules as same-context candidate
products. This is chemically meaningful for hydrogenation and coupling classes because
unreacted starting materials are plausible failed outcomes but not the observed product.
```

代码：

```text
pc_cng/hard_negative_actions.py
  - class_fallback_actions(...)

pc_cng/run_hard_negative_actions.py
  - --action class_fallback

tests/test_class_fallback_actions.py

scripts_run_type1_class_fallback_supplement.sh
```

服务器输出：

```text
/home/cunyuliu/pc_cng_research/results/type1_class_fallback_supplement_20260711
```

生成/审核结果：

| Class | Generated rows | Reviewed strong negatives |
|---|---:|---:|
| Alkylation | 115 | included in total keep |
| Amide coupling | 29 | included in total keep |
| Cabonylation | 114 | included in total keep |
| Cu coupling | 28 | included in total keep |
| Hydrogenation | 30 | included in total keep |
| Ni coupling | 12 | included in total keep |
| Rh coupling | 39 | included in total keep |
| Total | 367 | 326 |

Support after adding fallback candidates:

| Class | Before groups | After groups | Status after fallback |
|---|---:|---:|---|
| Alkylation | 5 | 53 | ok support, but harder task |
| Amide coupling | 3 | 13 | still low_support |
| Cabonylation | 5 | 44 | ok support, but harder task |
| Cu coupling | 2 | 14 | still low_support |
| Hydrogenation | 6 | 15 | still low_support, Top-1 improved after training |
| Ni coupling | 0 | 4 | still low_support |
| Rh coupling | 2 | 19 | near quota |

Fallback augmented training/evaluation:

```text
path:
/home/cunyuliu/pc_cng_research/results/type1_class_fallback_supplement_20260711/fallback_augmented_pairwise_seed20260710
```

| Model | Groups | Candidate rows | Overall Top-1 | Held-out test Top-1 | HITEA Top-1 | Synthetic Top-1 |
|---|---:|---:|---:|---:|---:|---:|
| Base Morgan seed20260710 | 1077 | 6166 | 97.40 | 83.58 | 93.24 | 96.50 |
| Quota-only augmented seed20260710 | 1077 | 6171 | 97.59 | 88.06 | 93.24 | 96.67 |
| Fallback augmented seed20260710 | 1216 | 6636 | 97.45 | 86.42 | 96.71 | 96.89 |

Class-level trained result after fallback:

| Class | Groups | Top-1 | MRR | Status |
|---|---:|---:|---:|---|
| Alkylation | 53 | 100.00 | 100.00 | ok |
| Cabonylation | 44 | 100.00 | 100.00 | ok |
| Hydrogenation | 15 | 66.67 | 83.33 | still low_support |
| Rh coupling | 19 | 100.00 | 100.00 | near quota |
| Amide coupling | 13 | 100.00 | 100.00 | still low_support |
| Cu coupling | 14 | 100.00 | 100.00 | still low_support |
| Ni coupling | 4 | 100.00 | 100.00 | still low_support |

Interpretation:

```text
class_fallback successfully changes Optimization C from "no candidate support" to a harder
class-level stress test. It substantially improves HITEA coverage and raises Hydrogenation
from 33.33 to 66.67 Top-1 after training, but some classes remain below the 20-group quota.
The next generator step should target Ni/Cu/Amide/Rh specifically, not broad threshold relaxation.
```

## 论文表述

可写：

```text
Reaction-class diagnostics show that the previously weak Hydrogenation result is primarily a support issue
(6 evaluable groups), not a statistically stable model failure. We therefore report class-level support
and introduce a class-aware quota gate for future generator expansion.
```

需要谨慎：

```text
Do not claim all reaction classes are solved. Same-context HiTEA classes still need targeted candidate
generation to reach top-journal robustness standards.
```
