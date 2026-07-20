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

## Partial-product action family negative ablation

新增 action：

```text
partial_product
```

核心思想：

```text
Use atom maps to split the observed product into product-side fragments contributed by individual
reactant partners. This targets amide/coupling/hydrogenation classes where no-conversion reactant
fragments are heavily duplicated after global canonicalization.
```

代码：

```text
pc_cng/hard_negative_actions.py
  - partial_product_actions(...)

pc_cng/run_hard_negative_actions.py
  - --action partial_product

tests/test_class_fallback_actions.py
  - RDKit-gated partial_product smoke test

scripts_run_type1_partial_product_supplement.sh
  - reproducible generate -> review -> train -> rerank -> class-gate runner
```

服务器输出：

```text
/home/cunyuliu/pc_cng_research/results/type1_partial_product_supplement_20260711
```

生成/审核结果：

| Class | Raw rows | Reviewed strong negatives | Unique strong source contexts |
|---|---:|---:|---:|
| Amide coupling | 23 | 19 | 13 |
| Cu coupling | 24 | 24 | 14 |
| Ni coupling | 8 | 8 | 4 |
| Rh coupling | 28 | 27 | 15 |
| Hydrogenation | 0 | 0 | 0 |
| Total | 83 | 78 | - |

Diagnostics:

```text
seen_positive_rows: 4097
partial_product_raw_rows: 4236
partial_product_kept before global duplicate filtering: 1791
written after global canonical duplicate filtering: 83
review keep_synthetic_negative: 78 / 83
main bottleneck: skip_global_duplicate = 1708 and insufficient mapped reactants = 3038
```

Training/evaluation:

```text
path:
/home/cunyuliu/pc_cng_research/results/type1_partial_product_supplement_20260711/partial_product_augmented_pairwise_seed20260710

config:
base diverse-anchor + quota supplement + class_fallback supplement + partial_product supplement
seed: 20260710
epochs: 80
feature_mode: morgan
```

| Model | Groups | Candidate rows | Overall Top-1 | Held-out test Top-1 | HITEA Top-1 | Synthetic Top-1 | Binary test ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base Morgan seed20260710 | 1077 | 6166 | 97.40 | 83.58 | 93.24 | 96.50 | 85.29 |
| Fallback augmented seed20260710 | 1216 | 6636 | 97.45 | 86.42 | 96.71 | 96.89 | - |
| Partial-product augmented seed20260710 | 1216 | 6714 | 97.29 | 85.19 | 95.77 | 96.62 | 84.62 |

Class-level result after partial-product training:

| Class | Groups | Top-1 | MRR | Status |
|---|---:|---:|---:|---|
| Amide coupling | 13 | 100.00 | 100.00 | still low_support |
| Cu coupling | 14 | 100.00 | 100.00 | still low_support |
| Hydrogenation | 15 | 66.67 | 83.33 | still low_support |
| Ni coupling | 4 | 100.00 | 100.00 | still low_support |
| Rh coupling | 19 | 100.00 | 100.00 | near quota |

Interpretation:

```text
partial_product is a useful negative ablation but not the selected Optimization C branch.
It adds 78 reviewed hard negatives and increases candidate rows within already-covered source contexts,
but it does not increase unique weak-class source support beyond the fallback branch. It also slightly
reduces same-context held-out test Top-1 relative to fallback-only augmentation (85.19 vs 86.42).
Therefore the manuscript should keep class_fallback as the current best class-gate supplement and report
partial_product as evidence that product-fragment edits alone are insufficient under global canonical
duplicate control.
```

Next required Optimization C step:

```text
The remaining support gap is not solved by more per-context candidates. To reach >=20 evaluable groups,
the generator must either:
1. create chemically distinct candidates for additional canonical source contexts, especially Hydrogenation and Ni;
2. incorporate external weak-class positives beyond current HiTEA coverage;
3. or add a separately audited source-context duplicate policy if repeated molecular reactions under different
   experimental records are scientifically justified.
```

## Source-context vs molecular-context support audit

Motivation:

```text
The synthetic reranking evaluator groups candidate sets by source_id, while generation performs global
canonical reaction de-duplication. A top-journal support gate must therefore distinguish source-record
coverage from genuinely distinct molecular parent-reaction coverage.
```

新增审计工具：

```text
pc_cng/audit_reaction_class_source_support.py
tests/test_source_support_audit.py
```

服务器输出：

```text
/home/cunyuliu/pc_cng_research/results/reaction_class_source_support_audit_20260711
/home/cunyuliu/pc_cng_research/results/manuscript_tables_pc_cng_v3/supp_source_support_audit.md
```

输入：

```text
positive-csv:
/home/cunyuliu/pc_cng_research/data/processed/hitea_full_normalized.csv

synthetic-csv:
/home/cunyuliu/pc_cng_research/results/type1_diverse_anchor_full/diverse_anchor_candidates_reviewed.csv
/home/cunyuliu/pc_cng_research/results/type1_class_quota_supplement_20260711/class_quota_candidates_reviewed.csv
/home/cunyuliu/pc_cng_research/results/type1_class_fallback_supplement_20260711/class_fallback_candidates_reviewed.csv
/home/cunyuliu/pc_cng_research/results/type1_partial_product_supplement_20260711/partial_product_candidates_reviewed.csv
```

Audit result:

| Class | Positive sources | Positive parent reactions | Candidate sources | Candidate parent reactions | Status | Recommendation |
|---|---:|---:|---:|---:|---|---|
| Alkylation | 1872 | 55 | 53 | 53 | ok | keep_monitoring |
| Amide coupling | 272 | 14 | 13 | 13 | data_source_gap | add external/curated contexts |
| Cabonylation | 1087 | 46 | 44 | 44 | ok | keep_monitoring |
| Cu coupling | 362 | 14 | 14 | 14 | data_source_gap | add external/curated contexts |
| Hydrogenation | 2849 | 41 | 11 | 11 | generator_coverage_gap | generate distinct candidates for 9 more sources |
| Ni coupling | 69 | 4 | 4 | 4 | data_source_gap | add external/curated contexts |
| Rh coupling | 545 | 20 | 19 | 19 | generator_coverage_gap | generate distinct candidates for 1 more source |

Interpretation:

```text
The remaining weak-class problem is not a single generator bug. Amide, Cu, and Ni do not have
>=20 distinct molecular parent reactions in the current audited HITEA weak-class slice, so repeating
source records would not satisfy a molecularly independent top-journal support gate. Hydrogenation and
Rh do have enough molecular parent reactions, so they remain valid targets for generator expansion.
The next implementation should prioritize Hydrogenation-specific and Rh-specific edits; Amide/Cu/Ni
require external or manually curated contexts before strong class-complete claims.
```

## Unreacted-substrate targeted supplement

新增 action：

```text
unreacted_substrate
```

Motivation:

```text
Hydrogenation and Rh coupling gaps are generator-coverage gaps, not data-source gaps. The missing
Hydrogenation contexts are often high-similarity unreduced substrates that class_fallback previously
filtered with max_product_similarity=0.98. For these classes, the unreacted substrate is chemically
the most meaningful failed outcome.
```

代码与脚本：

```text
pc_cng/hard_negative_actions.py
  - unreacted_substrate_actions(...)

pc_cng/run_hard_negative_actions.py
  - --action unreacted_substrate

scripts_run_type1_unreacted_substrate_supplement.sh
  - generate -> review -> train -> rerank -> class benchmark -> source-support audit
```

服务器输出：

```text
/home/cunyuliu/pc_cng_research/results/type1_unreacted_substrate_supplement_20260711
```

生成/审核结果：

| Class | Raw rows | Reviewed strong negatives | Unique source contexts |
|---|---:|---:|---:|
| Hydrogenation | 9 | 9 | 9 |
| Rh coupling | 0 | 0 | 0 |

Source-support impact:

| Class | Before candidate parent reactions | After candidate parent reactions | Status after unreacted |
|---|---:|---:|---|
| Hydrogenation | 11 | 20 | ok support |
| Rh coupling | 19 | 19 | still generator_coverage_gap |

Training/evaluation:

```text
path:
/home/cunyuliu/pc_cng_research/results/type1_unreacted_substrate_supplement_20260711/unreacted_augmented_pairwise_seed20260710

config:
base diverse-anchor + quota + class_fallback + partial_product + unreacted_substrate
seed: 20260710
epochs: 80
feature_mode: morgan
```

| Model | Groups | Candidate rows | Overall Top-1 | Held-out test Top-1 | HITEA Top-1 | Synthetic Top-1 | Binary test ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fallback augmented seed20260710 | 1216 | 6636 | 97.45 | 86.42 | 96.71 | 96.89 | - |
| Partial-product augmented seed20260710 | 1216 | 6714 | 97.29 | 85.19 | 95.77 | 96.62 | 84.62 |
| Unreacted-substrate augmented seed20260710 | 1225 | 6732 | 97.22 | 85.19 | 96.85 | 96.66 | 84.52 |

Class-level result:

| Class | Groups | Top-1 | MRR | Status |
|---|---:|---:|---:|---|
| Hydrogenation | 24 | 79.17 | 89.58 | weak_performance |
| Rh coupling | 19 | 100.00 | 100.00 | still low_support |

Tie-aware error audit:

```text
path:
/home/cunyuliu/pc_cng_research/results/type1_unreacted_substrate_supplement_20260711/hydrogenation_error_analysis
```

| Metric | Value |
|---|---:|
| evaluated Hydrogenation groups | 24 |
| strict Top-1 errors | 5 |
| exact-score tie-only errors | 4 |
| strict Top-1 | 79.17 |
| tie-aware Top-1 | 95.83 |

Error family counts:

```text
real_negative: 4
class_fallback: 1
```

Interpretation:

```text
unreacted_substrate successfully solves the Hydrogenation support gate at the source and molecular
parent levels (20/20 in source-support audit, 24 groups in reranking). Strict Hydrogenation Top-1
is 79.17, slightly below the 80% weak-performance threshold, but 4/5 strict errors are exact-score
ties between real_negative and positive stereochemical outcomes. The tie-aware Top-1 is 95.83,
so the remaining Hydrogenation problem is mostly tie-sensitive stereochemical ranking, with only one
clear class_fallback misranking. Rh remains one context short and needs a Rh-specific ring-closure /
fragment edit rather than a simple unreacted-substrate candidate.
```

### v2 reviewed-status-aware exclude run

Motivation:

```text
The remaining Rh context was blocked because the same unreacted substrate candidate already existed
in the class_fallback reviewed CSV with review_status=needs_review_or_downweight. The v1 generator
excluded all rows from prior reviewed CSVs before the new action could re-score/re-review them.
v2 adds --exclude-review-status so only confirmed strong negatives and known positives are excluded.
```

代码变更：

```text
pc_cng/run_hard_negative_actions.py
  - --exclude-review-status

scripts_run_type1_unreacted_substrate_supplement.sh
  - default RESULTS_DIR=/home/cunyuliu/pc_cng_research/results/type1_unreacted_substrate_supplement_v2_20260711
  - excludes keep_synthetic_negative and discard_known_positive only
```

服务器输出：

```text
/home/cunyuliu/pc_cng_research/results/type1_unreacted_substrate_supplement_v2_20260711
```

生成/审核结果：

| Class | Raw rows | Reviewed strong negatives | Unique source contexts |
|---|---:|---:|---:|
| Hydrogenation | 28 | 28 | 28 |
| Rh coupling | 9 | 9 | 9 |
| Total | 37 | 37 | 37 |

Source-support impact:

| Class | Candidate parent reactions before v2 | Candidate parent reactions after v2 | Status after v2 |
|---|---:|---:|---|
| Hydrogenation | 20 | 35 | ok |
| Rh coupling | 19 | 20 | ok |

Training/evaluation:

```text
path:
/home/cunyuliu/pc_cng_research/results/type1_unreacted_substrate_supplement_v2_20260711/unreacted_augmented_pairwise_seed20260710
```

| Model | Groups | Candidate rows | Overall Top-1 | Held-out test Top-1 | HITEA Top-1 | Synthetic Top-1 |
|---|---:|---:|---:|---:|---:|---:|
| Fallback augmented seed20260710 | 1216 | 6636 | 97.45 | 86.42 | 96.71 | 96.89 |
| Unreacted-substrate v1 seed20260710 | 1225 | 6732 | 97.22 | 85.19 | 96.85 | 96.66 |
| Unreacted-substrate v2 seed20260710 | 1241 | 6776 | 97.18 | 86.42 | 96.22 | 96.34 |

Class-level v2 result:

| Class | Groups | strict Top-1 | tie-aware Top-1 | MRR | Status |
|---|---:|---:|---:|---:|---|
| Hydrogenation | 39 | 84.62 | 94.87 | 92.31 | ok |
| Rh coupling | 20 | 100.00 | 100.00 | 100.00 | ok |

Interpretation:

```text
v2 solves both generator-coverage weak classes under the source/molecular support gate and the
strict reaction-class performance gate. Hydrogenation now clears strict Top-1 >80% and remains strong
under tie-aware Top-1. Rh coupling reaches 20 molecular parent reactions and retains 100% Top-1.
The small drop in overall/HITEA Top-1 versus fallback-only is acceptable for a supplement stress test,
but the main same-context model should still be selected by the broader paper table rather than by
weak-class support alone.
```

## 2026-07-11 curated weak-class context expansion

Motivation:

```text
After unreacted-substrate v2, Hydrogenation and Rh coupling passed the 20 molecular-parent
support gate. The remaining Amide/Cu/Ni alerts were no longer a generator-only issue:
the original HITEA slice did not contain enough distinct molecular parent reactions.
```

Code and runner:

```text
pc_cng/build_curated_weak_class_contexts.py
scripts_run_type1_curated_weak_class_supplement.sh
pc_cng/build_manuscript_tables.py
```

Curated context construction:

```text
Inputs:
  HiTEA cleaned Ullmann: /home/cunyuliu/pc_cng_research/external/HiTEA/data/cleaned_datasets/ullmann.csv
  USPTO/OpenMolecules: /home/cunyuliu/pc_cng_research/data/processed/uspto_openmolecules_normalized.csv

Output:
  /home/cunyuliu/pc_cng_research/results/type1_curated_weak_class_contexts_20260711/curated_weak_class_contexts.csv
```

Builder summary:

| Class | Curated positive rows |
|---|---:|
| Amide coupling | 200 |
| Cu coupling | 213 |
| Ni coupling | 6 |
| Total | 419 |

Candidate generation and review:

| Stage | Rows |
|---|---:|
| Raw curated class_fallback candidates | 1251 |
| keep_synthetic_negative | 1210 |
| needs_review_or_downweight | 41 |
| discard_known_positive | 0 |

Final molecular source-support audit:

| Class | Positive parent reactions | Candidate parent reactions | Status |
|---|---:|---:|---|
| Amide coupling | 214 | 208 | ok |
| Cu coupling | 213 | 213 | ok |
| Hydrogenation | 41 | 35 | ok |
| Rh coupling | 20 | 20 | ok |
| Ni coupling | 10 | 4 | data_source_gap |

Expanded curated benchmark, old v2 checkpoint vs curated-augmented checkpoint:

| Class | v2 Top-1 | curated-augmented Top-1 | Delta |
|---|---:|---:|---:|
| Amide coupling | 34.13 | 95.19 | +61.06 |
| Cu coupling | 63.85 | 99.06 | +35.21 |
| Hydrogenation | 84.62 | 87.18 | +2.56 |
| Rh coupling | 100.00 | 100.00 | +0.00 |
| Ni coupling | 100.00 | 100.00 | +0.00, low support |

Overall expanded benchmark:

| Model | Groups | Candidate rows | Overall Top-1 | Test Top-1 | MRR | NDCG |
|---|---:|---:|---:|---:|---:|---:|
| v2 checkpoint on expanded curated benchmark | 1635 | 8380 | 84.77 | 68.64 | 91.09 | 93.34 |
| curated-augmented checkpoint | 1635 | 8380 | 97.00 | 83.90 | 98.35 | 98.78 |

Original-scope sanity check:

```text
The curated-augmented checkpoint should not automatically replace the current main v2 model.
On the original Regio/HITEA scope it improves HITEA Top-1 (96.22 -> 97.48) but slightly lowers
RegioSQM20 Top-1 (97.41 -> 96.91) and original test Top-1 (86.42 -> 83.95). Treat this branch
as a curated weak-class/domain-adaptation supplement, not as the main model replacement.
```

Manuscript tables:

```text
/home/cunyuliu/pc_cng_research/results/manuscript_tables_pc_cng_v3/supp_source_support_audit.md
/home/cunyuliu/pc_cng_research/results/manuscript_tables_pc_cng_v3/supp_curated_weak_class_contexts.md
```

## 2026-07-11 class-weighted curated model selection

Ni data-source audit:

```text
Local available sources remain insufficient for Ni.
USPTO/OpenMolecules normalized: 6 Ni-like rows / 6 molecular parent reactions.
USPTO/OpenMolecules train_only: same 6 molecular parent reactions.
HITEA full normalized: 864 Ni rows but only 4 distinct molecular parent reactions.
Conclusion: Ni remains a real external data-source gap, not a generator or weighting failure.
```

Bug fix:

```text
pc_cng/train_feasibility_mlp.py
  - read_synthetic_rows now preserves reviewed synthetic row reaction_class.

pc_cng/train_pairwise_reward_mlp.py
  - added --class-weight and --class-margin.
  - pair class weights multiply family weights.
  - class margin overrides family/base margin.
```

Reason:

```text
The first class-weighted attempt showed pair_class_counts={"synthetic": 4448}, proving that
reaction_class had been lost during synthetic-row loading. That run was discarded. The fixed
classw050_rc run reports Amide=463, Cu=593, Hydrogenation=33, Ni=20 pair rows, so the class
weights are active.
```

Selected single-seed candidate:

```text
/home/cunyuliu/pc_cng_research/results/type1_curated_weak_class_contexts_20260711/curated_augmented_pairwise_classw050_rc_seed20260711
```

Model-selection comparison:

| Model | Scope | Groups | Overall Top-1 | Test Top-1 | Val Top-1 | RegioSQM20 Top-1 | HITEA Top-1 | Curated USPTO Top-1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| v2 unreacted | original Regio/HiTEA | 1241 | 97.18 | 86.42 | 80.00 | 97.41 | 96.22 | n/a |
| v2 unreacted | expanded curated | 1635 | 84.77 | 68.64 | 69.34 | 97.41 | 96.12 | 45.69 |
| curated unweighted | original Regio/HiTEA | 1241 | 97.02 | 83.95 | 80.00 | 96.91 | 97.48 | n/a |
| curated unweighted | expanded curated | 1635 | 97.00 | 83.90 | 81.02 | 96.91 | 97.41 | 96.95 |
| curated classw050_rc | original Regio/HiTEA | 1241 | 97.42 | 86.42 | 83.00 | 97.41 | 97.48 | n/a |
| curated classw050_rc | expanded curated | 1635 | 97.31 | 85.59 | 83.21 | 97.41 | 97.41 | 96.95 |

Class-level expanded benchmark:

| Class | v2 Top-1 | classw050_rc Top-1 | Status |
|---|---:|---:|---|
| Amide coupling | 34.13 | 95.19 | ok |
| Cu coupling | 63.85 | 99.06 | ok |
| Hydrogenation | 84.62 | 87.18 | ok |
| Rh coupling | 100.00 | 100.00 | ok |
| Ni coupling | 100.00 | 100.00 | low_support / data_source_gap |

Decision:

```text
curated_classw050_rc_seed20260711 is now the best single-seed curated weak-class candidate:
it closes Amide/Cu under expanded curated support, recovers original RegioSQM20 and test Top-1,
and improves HITEA Top-1 over v2. It is still not a final main-model replacement until replicated
with multi-seed confidence intervals. Ni remains the only unresolved reaction-class support gap.
```

Updated manuscript tables:

```text
/home/cunyuliu/pc_cng_research/results/manuscript_tables_pc_cng_v3/supp_curated_weak_class_contexts.md
/home/cunyuliu/pc_cng_research/results/manuscript_tables_pc_cng_v3/supp_curated_model_selection.md
```

## 2026-07-11 classw050_rc 5-seed stability audit

Runner:

```text
scripts_run_type1_curated_class_weight_multiseed.sh
SEEDS="20260710 20260711 20260712 20260713 20260714"
WEAK_CLASS_WEIGHT=0.5
```

Output:

```text
/home/cunyuliu/pc_cng_research/results/type1_curated_weak_class_contexts_20260711/classw050_rc_multiseed_summary
/home/cunyuliu/pc_cng_research/results/manuscript_tables_pc_cng_v3/supp_curated_multiseed_stability.md
```

5-seed stability:

| Scope | n seeds | Overall Top-1 | Test Top-1 | Val Top-1 | Overall MRR | RegioSQM20 Top-1 | HITEA Top-1 | Curated USPTO Top-1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| expanded curated | 5 | 97.03 ± 0.37 | 83.56 ± 1.27 | 82.92 ± 1.64 | 98.38 ± 0.19 | 97.27 ± 0.20 | 96.38 ± 1.51 | 96.75 ± 0.41 |
| original Regio/HiTEA | 5 | 97.12 ± 0.38 | 85.19 ± 1.35 | 82.00 ± 2.10 | 98.41 ± 0.21 | 97.27 ± 0.20 | 96.47 ± 1.47 | n/a |

Interpretation:

```text
The 5-seed result confirms that classw050_rc robustly fixes the expanded curated Amide/Cu
benchmark while preserving high original-scope performance. However, its mean original test
Top-1 (85.19) is still below the strongest v2 single-seed test Top-1 (86.42), so the current
publishable role is a curated weak-class stability supplement rather than an immediate main-model
replacement. For a main-model claim, extend to 10 seeds and report bootstrap confidence intervals
and paired significance against the v2/unreacted baseline.
```

## 2026-07-11 paired bootstrap/significance audit

Tool:

```text
pc_cng/paired_reranking_significance.py
tests/test_paired_reranking_significance.py
```

Outputs:

```text
/home/cunyuliu/pc_cng_research/results/type1_curated_weak_class_contexts_20260711/paired_significance_original_v2_vs_classw050_seed20260711
/home/cunyuliu/pc_cng_research/results/type1_curated_weak_class_contexts_20260711/paired_significance_expanded_v2_vs_classw050_seed20260711
/home/cunyuliu/pc_cng_research/results/manuscript_tables_pc_cng_v3/supp_curated_paired_significance.md
```

Paired group-level comparison:

| Scope | Metric | Groups | v2 mean | classw050 mean | Delta | 95% bootstrap CI | Paired permutation p |
|---|---|---:|---:|---:|---:|---:|---:|
| original Regio/HiTEA | Top-1 | 1241 | 97.18 | 97.42 | +0.24 | [-0.24, +0.73] | 0.5073 |
| original Regio/HiTEA | MRR | 1241 | 98.42 | 98.58 | +0.15 | [-0.10, +0.43] | 0.2564 |
| expanded curated | Top-1 | 1635 | 84.77 | 97.31 | +12.54 | [+10.95, +14.19] | 9.999e-05 |
| expanded curated | MRR | 1635 | 91.09 | 98.53 | +7.44 | [+6.49, +8.43] | 9.999e-05 |

Interpretation:

```text
The expanded curated improvement is statistically decisive and directly supports the Amide/Cu
weak-class supplement claim. The original-scope change is positive but not statistically
significant, which is exactly why the main headline model should still be selected using the
broader 10-seed comparison rather than a single class-weighted seed.
```

## 2026-07-11 classw050_rc 10-seed final stability audit

Runner:

```text
scripts_run_type1_curated_class_weight_multiseed.sh
SEEDS="20260710 20260711 20260712 20260713 20260714 20260715 20260716 20260717 20260718 20260719"
WEAK_CLASS_WEIGHT=0.5
```

Output:

```text
/home/cunyuliu/pc_cng_research/results/type1_curated_weak_class_contexts_20260711/classw050_rc_multiseed_summary/summary.csv
/home/cunyuliu/pc_cng_research/results/manuscript_tables_pc_cng_v3/supp_curated_multiseed_stability.md
```

10-seed stability:

| Scope | n seeds | Overall Top-1 | Test Top-1 | Val Top-1 | Overall MRR | RegioSQM20 Top-1 | HITEA Top-1 | Curated USPTO Top-1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| expanded curated | 10 | 97.16 ± 0.30 | 83.98 ± 1.28 | 83.50 ± 1.50 | 98.46 ± 0.16 | 97.35 ± 0.24 | 96.68 ± 1.12 | 96.90 ± 0.42 |
| original Regio/HiTEA | 10 | 97.24 ± 0.34 | 85.06 ± 1.51 | 82.80 ± 2.04 | 98.49 ± 0.18 | 97.35 ± 0.24 | 96.76 ± 1.10 | n/a |

Final interpretation for Optimization C:

```text
classw050_rc is a robust curated weak-class supplement across 10 seeds. It should be used
to support the Amide/Cu top-journal support-gate closure and the expanded curated benchmark
claim. It should not yet replace the main headline model because original-scope test Top-1
averages 85.06 ± 1.51, below the fair v2/unreacted 10-seed baseline (87.16 ± 1.58).
Ni remains unresolved due to a hard external molecular-context data-source gap.
```

## 2026-07-11 fair v2/unreacted 10-seed baseline

Runner:

```text
scripts_run_type1_unreacted_v2_multiseed.sh
SEEDS="20260710 20260711 20260712 20260713 20260714 20260715 20260716 20260717 20260718 20260719"
```

Outputs:

```text
/home/cunyuliu/pc_cng_research/results/type1_unreacted_substrate_supplement_v2_20260711/unreacted_v2_multiseed_summary/summary.csv
/home/cunyuliu/pc_cng_research/results/manuscript_tables_pc_cng_v3/supp_curated_vs_v2_multiseed.md
```

Fair 10-seed comparison against classw050_rc:

| Scope | Metric | v2/unreacted | classw050_rc | Delta pp |
|---|---|---:|---:|---:|
| original Regio/HiTEA | Overall Top-1 | 97.20 ± 0.24 | 97.24 ± 0.34 | +0.04 |
| original Regio/HiTEA | Test Top-1 | 87.16 ± 1.58 | 85.06 ± 1.51 | -2.10 |
| original Regio/HiTEA | RegioSQM20 Top-1 | 97.41 ± 0.19 | 97.35 ± 0.24 | -0.06 |
| original Regio/HiTEA | HITEA Top-1 | 96.30 ± 0.81 | 96.76 ± 1.10 | +0.46 |
| expanded curated | Overall Top-1 | 84.66 ± 0.83 | 97.16 ± 0.30 | +12.50 |
| expanded curated | Test Top-1 | 70.17 ± 1.55 | 83.98 ± 1.28 | +13.81 |
| expanded curated | Curated USPTO Top-1 | 45.18 ± 2.84 | 96.90 ± 0.42 | +51.73 |

Interpretation:

```text
The fair 10-seed baseline sharpens the Optimization C claim. classw050_rc should be
reported as the Amide/Cu curated weak-class repair because it closes the expanded
curated benchmark with a large and stable gain. It should not be promoted as the
single main model for the original Regio/HiTEA scope because v2/unreacted remains
stronger on held-out original test Top-1. This split claim is the most defensible
top-journal framing.
```

## 2026-07-11 10-seed ensemble paired significance

### Motivation

Single-seed paired significance is vulnerable to seed luck and under-estimates
variance for top-journal claims. We upgrade to a 10-seed ensemble paired test
with two complementary analyses:

1. **Group-level ensemble test**: average per-row scores across 10 seeds to form
   ensemble scores, then run paired bootstrap CI, paired sign-flip permutation
   test, and sign test on the ensemble scores. This reduces within-group noise
   and tests whether the mean effect across seeds is significant.

2. **Seed-level bootstrap CI**: resample seed indices with replacement to get a
   95% CI on the mean group-level delta. This tests whether the cross-seed
   distribution of the improvement is reliably non-zero.

Code:

```text
pc_cng/multiseed_paired_significance.py
tests/test_multiseed_paired_significance.py
```

Outputs:

```text
paired_significance_10seed/original_regio_hitea/{baseline_ensemble_scores.csv,candidate_ensemble_scores.csv,paired_group_deltas.csv,summary.csv,summary.json}
paired_significance_10seed/expanded_curated/...
```

### Original Regio/HiTEA scope — group-level ensemble (v2 vs classw050_rc)

| Metric | Groups | v2/unreacted | classw050_rc | Δ pp | 95% CI (bootstrap) | Permutation p | Sign-test p |
|---|---:|---:|---:|---:|---:|---:|---:|
| Top-1 | 1241 | 97.18 | 97.10 | -0.08 | [-0.40, +0.24] | 1.000 | 1.000 |
| MRR | 1241 | 98.42 | 98.37 | -0.04 | [-0.24, +0.15] | 0.692 | 0.791 |
| NDCG | 1241 | 98.82 | 98.79 | -0.03 | [-0.18, +0.11] | 0.679 | 0.791 |

### Original Regio/HiTEA scope — seed-level bootstrap

| Metric | Mean Δ pp | 95% CI (seed bootstrap) | Std (seed) |
|---|---:|---:|---:|
| Top-1 | +0.04 | [-0.11, +0.19] | 0.08 |
| MRR | +0.05 | [-0.03, +0.12] | 0.04 |
| NDCG | +0.04 | [-0.02, +0.09] | 0.03 |

Interpretation — original scope:

```text
On the original Regio/HiTEA scope, the 10-seed ensemble paired test confirms
that classw050_rc is statistically indistinguishable from v2/unreacted. Both
the group-level ensemble permutation test (p ≈ 1.0) and the seed-level bootstrap
CI (crossing 0 for Top-1) support the null. This is the top-journal-grade
justification for NOT presenting classw050_rc as a main-model replacement —
it does not hurt, but it also does not help on the held-out original scope.
```

### Expanded curated scope — group-level ensemble (v2 vs classw050_rc)

| Metric | Groups | v2/unreacted | classw050_rc | Δ pp | 95% CI (bootstrap) | Permutation p | Sign-test p |
|---|---:|---:|---:|---:|---:|---:|---:|
| Top-1 | 1635 | 83.79 | 97.06 | **+13.27** | [+11.62, +14.92] | < 0.0001 | 2.7 × 10⁻⁶¹ |
| MRR | 1635 | 90.52 | 98.38 | **+7.86** | [+6.87, +8.88] | < 0.0001 | 1.5 × 10⁻⁵⁶ |
| NDCG | 1635 | 92.92 | 98.79 | **+5.88** | [+5.15, +6.62] | < 0.0001 | 1.5 × 10⁻⁵⁶ |

Group-level direction: 220 groups better for classw050_rc, 3 groups better for v2, 1412 ties.

### Expanded curated scope — seed-level bootstrap

| Metric | Mean Δ pp | 95% CI (seed bootstrap) | Std (seed) |
|---|---:|---:|---:|
| Top-1 | +12.50 | [+12.07, +12.95] | 0.23 |
| MRR | +7.43 | [+7.21, +7.67] | 0.12 |
| NDCG | +5.56 | [+5.39, +5.74] | 0.09 |

Interpretation — expanded curated scope:

```text
The 10-seed ensemble paired test decisively confirms that classw050_rc closes
the Amide/Cu weak-class gap on the expanded curated benchmark. Top-1 improves
by +13.27 pp (group-level ensemble) with p < 0.0001 from both permutation and
sign tests. The seed-level bootstrap CI is entirely positive and very tight
(mean +12.50 pp, std 0.23 pp), demonstrating robustness across seeds. With 220
groups won vs only 3 lost, the direction of improvement is overwhelmingly
consistent. This meets top-journal statistical standards for the Amide/Cu
curated weak-class repair claim.
```
