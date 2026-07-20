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

1. low_yield_seed 不应直接强 BCE 加权过高，建议做 `origin_weight=synthetic=0.2/0.5` 消融。
2. heteroatom action 应保留为 known-positive-aware candidate，不应把已观测 hetero alternative 当负样本。
3. pairwise reward 可以进一步扩展成 action-family aware loss，对 `regio/tautomer/low_yield_seed` 分别设 margin。
4. 下一轮优先做 expanded actions 的 weighted BCE 与 family ablation，而不是继续盲目扩大候选数量。
