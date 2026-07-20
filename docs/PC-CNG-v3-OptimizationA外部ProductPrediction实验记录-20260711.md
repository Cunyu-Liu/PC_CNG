# PC-CNG v3 Optimization A 外部 Product Prediction 实验记录

日期：2026-07-11

## 实验目标

验证 PC-CNG v3 的 boundary-negative / pairwise reward 是否能在严格共享候选集的外部 product-prediction benchmark 中提升 frozen Chemformer reference 的 candidate selection。

核心比较：

1. Chemformer conditional likelihood。
2. PC-CNG pairwise_default 5-seed ensemble。
3. Chemformer + PC-CNG group-zscore hybrid，验证集选择 hybrid weight。

## 已部署代码

远程代码目录：

```text
/home/cunyuliu/pc_cng_research/chem_negative_sampling
```

新增/修改文件：

```text
pc_cng/ranking_metrics.py
pc_cng/build_external_product_prediction_candidate_set.py
pc_cng/evaluate_external_product_prediction_benchmark.py
pc_cng/evaluate_reaction_lm_scores.py
pc_cng/reaction_lm_scorer.py
pc_cng/build_reaction_lm_candidate_set.py
scripts_run_external_product_prediction_benchmark.sh
tests/test_external_product_prediction_benchmark.py
```

协议文档同步到远程：

```text
/home/cunyuliu/pc_cng_research/docs/PC-CNG-v3-downstream-eval-protocol-OptimizationA-20260711.md
```

## 本地验证

```text
python3 -m compileall pc_cng
python3 -m unittest discover -s tests -p 'test_external_product_prediction_benchmark.py'
```

结果：

```text
compileall: pass
unittest: Ran 1 test, OK
```

## 远程验证

```text
cd /home/cunyuliu/pc_cng_research/chem_negative_sampling
PYTHONPATH=. /home/cunyuliu/miniconda3/envs/pc_cng_gpu/bin/python -m compileall pc_cng tests
PYTHONPATH=. /home/cunyuliu/miniconda3/envs/pc_cng_gpu/bin/python -m unittest discover -s tests -p 'test_external_product_prediction_benchmark.py'
```

结果：

```text
compileall: pass
unittest: Ran 1 test, OK
```

## 正式运行配置

启动时间：2026-07-11 UTC 01:33 左右。

远程 pid：

```text
3825240
```

启动命令等价配置：

```bash
cd /home/cunyuliu/pc_cng_research/chem_negative_sampling
RESULTS_DIR=/home/cunyuliu/pc_cng_research/results/external_product_prediction_benchmark_20260711 \
GPU_EVAL=4 \
PC_CNG_MODEL_DIRS="/home/cunyuliu/pc_cng_research/results/type1_diverse_anchor_ablation_full/pairwise_default_seed20260710 /home/cunyuliu/pc_cng_research/results/type1_diverse_anchor_ablation_full/pairwise_default_seed20260711 /home/cunyuliu/pc_cng_research/results/type1_diverse_anchor_ablation_full/pairwise_default_seed20260712 /home/cunyuliu/pc_cng_research/results/type1_diverse_anchor_ablation_full/pairwise_default_seed20260713 /home/cunyuliu/pc_cng_research/results/type1_diverse_anchor_ablation_full/pairwise_default_seed20260714" \
SCORER=chemformer_log_likelihood \
N_BEAMS=10 \
bash scripts_run_external_product_prediction_benchmark.sh
```

关键输入：

```text
/home/cunyuliu/pc_cng_research/data/processed/regiosqm20_normalized.csv
/home/cunyuliu/pc_cng_research/data/processed/hitea_full_normalized.csv
/home/cunyuliu/pc_cng_research/results/type1_diverse_anchor_full/diverse_anchor_candidates_reviewed.csv
/home/cunyuliu/pc_cng_research/models/reaction_lm/chemformer_forward_uspto50k/model_sanitized.ckpt
/home/cunyuliu/pc_cng_research/external/reaction_lm/Chemformer/bart_vocab.json
```

## 当前状态

最终状态：complete。

```text
stage: final strict external product-prediction benchmark complete
GPU: 4
runtime: about 70 minutes including Chemformer beam generation, Chemformer likelihood scoring, and optimized PC-CNG ensemble benchmark
```

已生成：

```text
/home/cunyuliu/pc_cng_research/results/external_product_prediction_benchmark_20260711/base_observed_pc_cng_candidates.csv
/home/cunyuliu/pc_cng_research/results/external_product_prediction_benchmark_20260711/base_observed_pc_cng_candidates_summary.json
/home/cunyuliu/pc_cng_research/results/external_product_prediction_benchmark_20260711/product_prediction_contexts.csv
/home/cunyuliu/pc_cng_research/results/external_product_prediction_benchmark_20260711/chemformer_forward_input.csv
/home/cunyuliu/pc_cng_research/results/external_product_prediction_benchmark_20260711/chemformer_forward_beams.tsv
/home/cunyuliu/pc_cng_research/results/external_product_prediction_benchmark_20260711/full_observed_pc_cng_chemformer_beam_candidates_summary.json
/home/cunyuliu/pc_cng_research/results/external_product_prediction_benchmark_20260711/lm_scores_chemformer_log_likelihood_summary.json
/home/cunyuliu/pc_cng_research/results/external_product_prediction_benchmark_20260711/benchmark/benchmark_summary.json
/home/cunyuliu/pc_cng_research/results/external_product_prediction_benchmark_20260711/benchmark/paper_table.md
```

## 结果

候选覆盖：

```text
contexts: 16050
candidate_rows_requested: 175678
candidate_rows_with_required_scores: 22135
candidate_rows_evaluated: 7359
kept_groups: 1197
missing_score_rows: 153543
```

说明：

```text
PC-CNG 的 Morgan/RDKit featurizer 只能对 22135 个候选产物成功打分。
大量 Chemformer beam candidates 因无法通过当前 PC-CNG featurizer 而不能进入 strict shared-candidate comparison。
因此本结果是严格共享候选交集上的公平比较，不代表完整 16050 contexts 的端到端 product-generation 覆盖。
```

主结果：

| Model | Overall Top-1 | Overall Top-3 | Overall MRR | Overall NDCG | Test Top-1 | Test MRR |
|---|---:|---:|---:|---:|---:|---:|
| Chemformer likelihood | 3.34 | 63.83 | 32.08 | 48.52 | 4.94 | 32.29 |
| PC-CNG 5-seed ensemble | 66.00 | 99.08 | 82.53 | 87.07 | 71.60 | 83.76 |
| Selected hybrid, validation-chosen `pc_cng_weight=1.00` | 66.00 | 99.08 | 82.53 | 87.07 | 71.60 | 83.76 |

Absolute gain of selected PC-CNG over Chemformer likelihood:

| Split | Top-1 gain | MRR gain | NDCG gain |
|---|---:|---:|---:|
| Overall | +62.66 pp | +50.45 pp | +38.55 pp |
| Test | +66.66 pp | +51.47 pp | +39.34 pp |

Hybrid sweep:

| PC-CNG weight | Overall Top-1 | Test Top-1 | Test MRR | Test NDCG |
|---:|---:|---:|---:|---:|
| 0.00 | 3.34 | 4.94 | 32.29 | 48.53 |
| 0.25 | 17.04 | 24.69 | 48.60 | 61.08 |
| 0.50 | 41.94 | 45.68 | 65.84 | 74.23 |
| 0.75 | 57.23 | 72.84 | 81.43 | 85.91 |
| 1.00 | 66.00 | 71.60 | 83.76 | 87.87 |

Interpretation:

```text
Validation-selected model is pure PC-CNG, not a mixed Chemformer hybrid.
Chemformer likelihood contributes useful partial signal as weight increases from 0.00 to 0.75,
but PC-CNG remains the strongest validation-selected scorer by MRR/NDCG and overall Top-1.
The test-only Top-1 at weight 0.75 is slightly higher than weight 1.00, but this is not selected
by the pre-declared validation rule and comes with lower test MRR/NDCG.
```

Engineering note:

```text
During the final benchmark, evaluator was optimized to featurize candidates once and reuse
the same feature matrix across all 5 PC-CNG checkpoints. This avoids 5x repeated RDKit
featurization and preserves identical scores.
```

## Coverage-aware follow-up

Motivation:

```text
The strict intersection benchmark is fair, but it discards most Chemformer-only beam candidates
because PC-CNG's Morgan/RDKit featurizer cannot represent them. A top-journal benchmark also
needs to show what happens when these unsupported external beam negatives remain in the candidate set.
```

Code change:

```text
pc_cng/evaluate_external_product_prediction_benchmark.py
--pc-cng-invalid-negative-score 0.0
```

Rule:

```text
If a candidate has label=0 and PC-CNG cannot featurize/score it, assign PC-CNG score 0.0.
If a candidate has label=1 and PC-CNG cannot score it, do not assign a fake score; the group is filtered
unless another positive candidate row is available. This protects against rewarding the model for data issues.
```

Server output:

```text
/home/cunyuliu/pc_cng_research/results/external_product_prediction_benchmark_20260711/benchmark_validity_aware/benchmark_summary.json
/home/cunyuliu/pc_cng_research/results/external_product_prediction_benchmark_20260711/benchmark_validity_aware/paper_table.md
```

Coverage-aware candidate accounting:

```text
candidate_rows_requested: 175678
candidate_rows_with_required_scores: 175601
candidate_rows_evaluated: 174908
kept_groups: 15973
filled_negative_rows: 153466
missing_positive_rows: 77
still_missing_negative_rows: 0
```

Coverage-aware result:

| Model | Overall Top-1 | Overall Top-3 | Overall MRR | Overall NDCG | Test Top-1 | Test MRR |
|---|---:|---:|---:|---:|---:|---:|
| Chemformer likelihood | 0.04 | 1.13 | 9.69 | 28.45 | 0.07 | 9.75 |
| PC-CNG validity-aware 5-seed ensemble | 97.45 | 99.93 | 98.69 | 99.03 | 98.50 | 99.14 |
| Selected hybrid, validation-chosen `pc_cng_weight=1.00` | 97.45 | 99.93 | 98.69 | 99.03 | 98.50 | 99.14 |

Comparison of the two fair protocols:

| Protocol | Kept groups | Evaluated rows | PC-CNG rule for unfeaturizable negatives | PC-CNG Test Top-1 | PC-CNG Test MRR |
|---|---:|---:|---|---:|---:|
| Strict scored-candidate intersection | 1197 | 7359 | drop rows without both scores | 71.60 | 83.76 |
| Validity-aware full candidate set | 15973 | 174908 | label=0 unsupported candidates get score 0.0 | 98.50 | 99.14 |

Interpretation:

```text
The coverage-aware benchmark is the better product-prediction candidate-selection story because it
keeps almost the full Chemformer beam candidate space. The result shows that many Chemformer beam
alternatives are chemically invalid or unsupported by PC-CNG's molecular featurizer and can be safely
ranked below observed products. However, this metric combines learned PC-CNG scoring with an explicit
validity/featurizability prior, so it should be reported as "validity-aware PC-CNG reranking" rather
than pure pairwise reward scoring.
```

## 论文可用判断

可写入论文的结论：

```text
On the strict shared-candidate intersection of Chemformer beam products and PC-CNG-featurizable candidates,
PC-CNG pairwise boundary reward substantially improves candidate selection over frozen Chemformer
conditional likelihood.

With a pre-declared validity-aware rule that assigns a low score to unfeaturizable negative beam
candidates, PC-CNG covers 15973 / 16050 contexts and reaches 98.50% test Top-1 on the external
product-prediction candidate-selection benchmark.
```

需要谨慎声明：

```text
This is not yet an end-to-end product-generation SoTA result because the current PC-CNG featurizer
does not score all Chemformer beam candidates. A top-journal claim should either improve scorer
coverage for invalid/unusual beam SMILES or report this as a strict valid-candidate intersection benchmark.
```
