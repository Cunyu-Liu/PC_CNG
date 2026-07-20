# PC-CNG v3 Expanded Actions 全量训练结果

## 实验目的

验证 expanded action generator 是否能缓解 v3 的候选生成空间瓶颈。

本轮接入四类 action：

```text
heteroatom
regio
tautomer
low_yield_seed
```

训练矩阵：

1. expanded actions + pairwise reward
2. expanded actions + direct BCE
3. stacked ensemble 自动纳入上述两个新 run

## 运行路径

A100 workspace：

```text
/home/cunyuliu/pc_cng_research
```

核心输出：

```text
results/expanded_hard_negative_actions_full/
results/expanded_actions_pairwise_reward_h2048_n4096_e80/
results/expanded_actions_direct_bce_h2048_n4096_e80/
results/stacked_ensemble_summary.json
```

本地脚本：

```text
code/chem_negative_sampling/scripts_run_expanded_hard_negative_actions_pipeline.sh
code/chem_negative_sampling/pc_cng/analyze_action_family_contribution.py
code/chem_negative_sampling/scripts_run_type1_diverse_anchor_pipeline.sh
```

## 候选生成结果

输入 positive rows：

```text
HiTEA positives: 15,498
RegioSQM20 positives: 552
total seen_positive_rows: 16,050
```

生成候选：

```text
total written: 3,953
regio: 27
tautomer: 1,340
low_yield_seed: 2,586
heteroatom: 0 kept as negative
```

False-negative review：

```text
keep_synthetic_negative: 2,690
needs_review_or_downweight: 845
discard_known_positive: 418
```

说明：

heteroatom action 在 smoke test 和 raw candidate 层面可生成，但全量 HiTEA 中 raw heteroatom alternatives 大量命中 known-positive set。抽查前 2,000 个 HiTEA positive：

```text
hetero_raw: 397
hetero_known: 397
hetero_unknown: 0
```

因此本轮没有把 heteroatom alternatives 强行作为负样本，这是正确的 false-negative 防护行为。

## Action-family contribution 审计

审计输出：

```text
A100:
/home/cunyuliu/pc_cng_research/results/expanded_hard_negative_actions_full/action_family_contribution

files:
action_family_contribution.json
generation_family_table.md
score_family_table.md
```

生成、审查与 split 映射：

| Family | Total | Keep | Keep train | Keep val | Keep test | Needs review | Discard known-positive | Keep rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| heteroatom | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0000 |
| regio | 27 | 27 | 27 | 0 | 0 | 0 | 0 | 1.0000 |
| tautomer | 1,340 | 495 | 390 | 59 | 46 | 845 | 0 | 0.3694 |
| low_yield_seed | 2,586 | 2,168 | 1,723 | 220 | 225 | 0 | 418 | 0.8384 |

Same-split candidate reranking family challenge：

| Model | Family | Family rows | Groups | Family-only Top-1 | Family-only MRR | Mean margin | Hard negatives beating positive |
|---|---|---:|---:|---:|---:|---:|---:|
| expanded pairwise | regio | 27 | 12 | 1.0000 | 1.0000 | 0.9358 | 0 |
| expanded pairwise | tautomer | 495 | 266 | 1.0000 | 1.0000 | 0.8762 | 1 |
| expanded BCE synth=0.5 | regio | 27 | 12 | 1.0000 | 1.0000 | 0.9755 | 0 |
| expanded BCE synth=0.5 | tautomer | 495 | 266 | 0.9962 | 0.9981 | 0.8780 | 2 |

解释：

```text
1. tautomer 是当前真正提供规模化 type-1 reranking challenge 的主 family。
2. regio 质量较好，但仅 27 条、12 个 candidate groups，数量不足以支撑强泛化结论。
3. low_yield_seed 有 1,723 条进入 train，是主要 type-2 feasibility supervision；它没有 parent-positive candidate reranking groups，因此不能用同一个 Top-1 口径评价。
4. heteroatom 在 known-positive-aware review 后没有保留负样本，说明当前 heteroatom generator 更容易生成已知正样本替代产物，后续应优先改成 unknown-positive-masked candidate discovery。
```

## Relaxed anchor-only 诊断

为确认 `regio/heteroatom` 覆盖率低是否只是阈值过严，已给 anchor generator 增加可配置阈值和诊断计数：

```text
pc_cng/hard_negative_actions.py
pc_cng/run_hard_negative_actions.py

new args:
--min-product-similarity
--max-product-similarity
--min-atom-balance
```

full relaxed anchor-only run：

```text
A100:
/home/cunyuliu/pc_cng_research/results/type1_anchor_relaxed_full

actions: heteroatom, regio
seen_positive_rows: 16,050
min_product_similarity: 0.45
max_product_similarity: 0.995
min_atom_balance: 0.35
max_anchor_distance: 8
max_candidates_per_pair: 24
```

结果：

```text
raw_family:heteroatom: 397
skip_known_positive:heteroatom: 397

candidate_seen:regio: 2,909
skip_global_duplicate:regio: 2,882
written unique regio: 27

review:
keep_synthetic_negative: 27 / 27
```

结论：

```text
阈值放宽没有增加最终可用 type-1 negatives。
heteroatom 的瓶颈是 known-positive overlap；
regio 的瓶颈是 global duplicate collapse。
因此下一步应设计新的 diversity-aware regio/heteroatom generator，
而不是继续放宽 similarity / distance / atom-balance 阈值。
```

## Diversity-aware regio/heteroatom generator

已实现：

```text
pc_cng/hard_negative_actions.py
pc_cng/run_hard_negative_actions.py
scripts_run_type1_diverse_anchor_pipeline.sh

flag:
--diverse-anchor
```

该策略直接在 product graph 中寻找 terminal substituent attached to aromatic/ring scaffold，并将该 terminal substituent 迁移到同环或近邻可成键 anchor：

```text
same-ring / same-atom anchor -> regio
N/O/P/S anchor -> heteroatom
known-positive filtering remains mandatory
```

Full run：

```text
A100:
/home/cunyuliu/pc_cng_research/results/type1_diverse_anchor_full

seen_positive_rows: 16,050
generated written: 4,545
  regio: 3,988
  heteroatom: 557

review:
  keep_synthetic_negative: 3,829
  needs_review_or_downweight: 716

kept train:
  regio: 2,748
  heteroatom: 351
```

Full-parameter pairwise reward：

```text
/home/cunyuliu/pc_cng_research/results/type1_diverse_anchor_pairwise_reward_h2048_n4096_e80

pair_family_counts:
  regio: 2,748
  heteroatom: 351

test ROC-AUC: 0.8529
test AUPRC: 0.8017
test F1: 0.7242
```

同一 diverse-anchor candidate set 的 apples-to-apples reranking：

| Model | Overall Top-1 | Synthetic Top-1 | Synthetic MRR | Heteroatom Top-1 | Regio Top-1 |
|---|---:|---:|---:|---:|---:|
| real-only MLP | 0.8839 | 0.8017 | 0.8846 | 0.9349 | 0.8078 |
| old expanded pairwise | 0.9285 | 0.8817 | 0.9339 | 0.9726 | 0.8836 |
| new diverse-anchor pairwise | **0.9740** | **0.9650** | **0.9803** | **0.9863** | **0.9646** |

5-seed / 80-epoch 消融：

```text
/home/cunyuliu/pc_cng_research/results/type1_diverse_anchor_ablation_full

pairwise_default: BCE anchor + pairwise margin
family_margin: BCE anchor + pairwise margin + family-specific margin/weight
completed runs: 10 / 10
```

| Setting | Overall Top-1 | Test Top-1 | Synthetic Top-1 | Regio Top-1 | Heteroatom Top-1 |
|---|---:|---:|---:|---:|---:|
| pairwise_default | **0.9749 +/- 0.0006** | 0.8507 +/- 0.0094 | **0.9663 +/- 0.0007** | **0.9659 +/- 0.0007** | **0.9863 +/- 0.0000** |
| family_margin | 0.9747 +/- 0.0016 | **0.8537 +/- 0.0174** | 0.9660 +/- 0.0017 | 0.9656 +/- 0.0017 | **0.9863 +/- 0.0000** |

Chemformer-reference DPO matrix：

```text
/home/cunyuliu/pc_cng_research/results/type1_diverse_anchor_dpo_reference

Chemformer scored candidate rows: 35,917
frozen Chemformer overall Top-1: 0.4608
frozen Chemformer synthetic Top-1: 0.2833
completed reward runs: 15 / 15
```

| Setting | Reward Overall Top-1 | Synthetic Top-1 | Test Top-1 | Delta vs frozen Chemformer overall |
|---|---:|---:|---:|---:|
| frozen Chemformer LL | 0.4608 | 0.2833 | 0.4433 | - |
| dpo_only_synth | 0.9074 +/- 0.0093 | 0.9513 +/- 0.0149 | 0.7670 +/- 0.0124 | +0.4465 +/- 0.0093 |
| dpo_pairwise_synth | 0.9069 +/- 0.0052 | 0.9513 +/- 0.0069 | 0.7629 +/- 0.0065 | +0.4461 +/- 0.0052 |
| pairwise_only_synth | **0.9107 +/- 0.0038** | **0.9567 +/- 0.0055** | **0.7711 +/- 0.0152** | **+0.4498 +/- 0.0038** |

## Type-2 low-yield branch

已将 `low_yield_seed` 单独拆为 type-2 feasibility branch：

```text
code:
scripts_run_type2_low_yield_branch.sh
pc_cng/train_feasibility_mlp.py

A100:
/home/cunyuliu/pc_cng_research/results/type2_low_yield_branch_full

filter:
--synthetic-family low_yield_seed

completed runs: 15 / 15
```

结果：

| Setting | Test ROC-AUC | Test AUPRC | Test F1 | HiTEA ROC-AUC | RegioSQM20 ROC-AUC |
|---|---:|---:|---:|---:|---:|
| low_yield_synth02 | 0.8539 +/- 0.0027 | 0.7956 +/- 0.0031 | 0.7126 +/- 0.0046 | 0.8520 +/- 0.0033 | 0.8167 +/- 0.0148 |
| low_yield_synth05 | **0.8556 +/- 0.0021** | **0.7993 +/- 0.0008** | 0.7214 +/- 0.0043 | **0.8547 +/- 0.0026** | 0.8337 +/- 0.0051 |
| low_yield_synth10 | 0.8553 +/- 0.0022 | 0.7979 +/- 0.0023 | **0.7230 +/- 0.0008** | 0.8545 +/- 0.0024 | **0.8361 +/- 0.0029** |

解释：

```text
1. low_yield_seed 的价值主要是 type-2 failed/low-yield feasibility regularization。
2. 权重 0.5/1.0 略优于 0.2，但提升幅度温和。
3. 该分支应作为 supplementary feasibility evidence，不参与 type-1 reranking 主表。
```

## Manuscript tables

已生成：

```text
/home/cunyuliu/pc_cng_research/results/manuscript_tables_pc_cng_v3

main_type1_reranking.md/csv
supp_type2_low_yield.md/csv
supp_action_family_generation.md/csv
supp_type1_dataset.md/csv
supp_type1_reaction_class.md/csv
supp_type2_reaction_class.md/csv
manifest.json
```

结论：

```text
1. diversity-aware generator 将 regio/heteroatom 可用负样本从 27/0 提升到 3,393/436。
2. 在同一更难 candidate set 上，新 pairwise scorer 明显优于 real-only 和旧 expanded pairwise。
3. 主要收益仍体现在 candidate reranking / preference boundary，而不是 binary ROC-AUC。
4. 这一步解决了 expanded-actions 早期版本暴露的 type-1 regio/heteroatom 覆盖率瓶颈。
5. Chemformer-reference DPO matrix 显著超过 frozen Chemformer，但没有超过 pairwise-only；因此 DPO-reference 是补充佐证，不是当前主模型。
```

## Downstream Pairwise Reward

目录：

```text
results/expanded_actions_pairwise_reward_h2048_n4096_e80/
```

训练计数：

```text
pair_rows_requested: 418
pair_rows_featurized: 417
real_train_rows_featurized: 34,927
```

Validation：

```text
ROC-AUC: 0.8569
AUPRC: 0.7983
F1: 0.7131
```

Test：

```text
ROC-AUC: 0.8590
AUPRC: 0.8015
F1: 0.7219
```

解读：

pairwise reward 单模型的 validation 不如 v2/rule-hard，但 test ROC-AUC 略高于 v2 pairwise 和 rule-hard pairwise，说明 expanded actions 中存在一定互补泛化信号。

## Downstream Direct BCE

目录：

```text
results/expanded_actions_direct_bce_h2048_n4096_e80/
```

训练计数：

```text
synthetic_rows: 2,140
train_rows_featurized: 37,064
train_positive: 13,152
train_negative: 23,912
```

Validation：

```text
ROC-AUC: 0.8672
AUPRC: 0.8109
F1: 0.7368
```

Test：

```text
ROC-AUC: 0.8525
AUPRC: 0.7967
F1: 0.7151
```

解读：

direct BCE 在 validation AUC 略高，但 test 明显下降，说明直接混入 expanded negatives 仍存在分布偏移或权重过强问题。该路线不适合作为单模型主结论。

## Stacked Ensemble

新增两个 run 后，stacked ensemble 自动纳入：

```text
expanded_actions_direct_bce_h2048_n4096_e80
expanded_actions_pairwise_reward_h2048_n4096_e80
```

当前 best by validation ROC-AUC：

```text
method: logreg_C1.0
validation ROC-AUC: 0.8774
validation AUPRC: 0.8244
test ROC-AUC: 0.8826
test AUPRC: 0.8244
test F1: 0.7581
```

对比上一轮 known-positive masked hard-negative ensemble：

```text
old test ROC-AUC: 0.8818
old test AUPRC: 0.8234
new test ROC-AUC: 0.8826
new test AUPRC: 0.8244
```

结论：

expanded actions 没有显著提升单模型，但作为 ensemble feature 提供了稳定增益，刷新当前项目最好结果。

## 下一步判断

1. low_yield_seed 已作为 type-2 feasibility branch 单独建模；建议作为补充表。
2. heteroatom action 必须继续 known-positive-aware，不应把已观测 hetero alternative 当负样本。
3. regio/heteroatom 的 5-seed pairwise / family-margin 消融已完成；family-margin 没有明显优于 default pairwise。
4. DPO-reference reward matrix 已完成，作为 external-reference preference tuning 补充表，不建议替代 pairwise_default 主表。
5. low_yield_seed 与 type-1 regio/heteroatom 已拆成不同训练分支；后续若扩展 low-yield，应继续保持独立评价口径。
