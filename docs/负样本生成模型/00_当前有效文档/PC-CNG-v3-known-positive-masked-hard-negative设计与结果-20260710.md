# PC-CNG v3：Known-Positive Masked Hard-Negative Decoder

## 背景

上一版可训练 reaction-center edit decoder 学到的是：

```text
observed positive anchor > all alternative anchors
```

这导致模型最高分候选往往是另一个已知 positive regioisomer。严格 known-positive 过滤后没有可用负样本；放宽后生成的 C-anchor relaxed negatives 质量偏低，导致 downstream validation ROC-AUC 下降到约 0.84。

因此，本轮目标改为：

```text
observed / known-positive candidates 与 hard-negative candidates 分头建模
```

即模型不能把 known-positive alternative 当负样本，也不能只拟合 observed positive。

## 候选角色定义

在 `build_edit_decoder_dataset.py` 中，每个 candidate anchor 被标注为：

- `observed_positive`：原始反应实际观察到的 anchor。
- `known_positive_alt`：不是当前 source 的 observed anchor，但在全局 known-positive set 中存在。
- `hard_negative`：未观测、非 known-positive、与产物相似、原子平衡较高、距离反应中心较近。
- `artifact`：其余低质量/远距离候选。

当前全量候选统计：

```text
candidate_rows: 3754
candidate_groups: 448
observed_positive: 448
known_positive_alt: 397
hard_negative: 2733
artifact: 176
```

## 新训练目标

模型由单头变为双头：

```text
positive_head(x)      -> 可行/已知正样本得分
hard_negative_head(x) -> 可作为 hard negative 的得分
```

### Loss 1：known-positive masked positive BCE

```text
y_pos = 1 for observed_positive and known_positive_alt
y_pos = 0 for hard_negative and artifact
```

目的：

- known-positive alternative 不再被当负样本；
- positive head 学到 observed/known positive 的可行性。

### Loss 2：hard-negative BCE

```text
y_hard = 1 for hard_negative
y_hard = 0 for observed_positive, known_positive_alt, artifact
```

目的：

- 单独训练 hard-negative head；
- 避免模型只学 known positive。

### Loss 3：positive ranking

```text
positive_head(observed or known_positive) > positive_head(hard_negative/artifact)
```

### Loss 4：hard-negative ranking

```text
hard_negative_head(hard_negative) > hard_negative_head(observed/known_positive/artifact)
```

总损失：

```text
L = w1 * BCE_pos
  + w2 * BCE_hard
  + w3 * Rank_pos
  + w4 * Rank_hard
```

当前默认：

```text
w1=1.0, w2=1.0, w3=0.5, w4=1.0
```

## 已实现代码

- `pc_cng/build_edit_decoder_dataset.py`
- `pc_cng/train_reaction_center_edit_decoder.py`
- `pc_cng/export_rule_hard_negatives.py`
- `scripts_run_masked_hard_decoder_pipeline.sh`

## A100 执行结果

### Masked hard-negative decoder

路径：

```text
/home/cunyuliu/pc_cng_research/results/masked_hard_decoder_full/train_masked_hard_decoder
```

训练集指标：

- positive head top1 accuracy: 1.0
- hard-negative head top1 accuracy: 0.8476
- hard-negative row ROC-AUC: 0.9976
- hard-negative row AUPRC: 0.9992

注意：candidate groups 仍集中在 train split，decoder 本身还缺少独立 val/test group。

### 规则 hard-negative 对照集

路径：

```text
/home/cunyuliu/pc_cng_research/results/masked_hard_decoder_full/rule_hard_negatives_reviewed.csv
```

统计：

- candidate groups: 420
- exported negatives: 840
- false-negative review: keep 840 / 840

### Downstream 对照结果

#### Rule hard negatives + BCE

路径：

```text
/home/cunyuliu/pc_cng_research/results/rule_hard_negatives_direct_bce_h2048_n4096_e80
```

最终 validation：

- ROC-AUC: 0.8524
- AUPRC: 0.7866

#### Rule hard negatives + pairwise reward

路径：

```text
/home/cunyuliu/pc_cng_research/results/rule_hard_negatives_pairwise_reward_h2048_n4096_e80
```

最终 validation：

- ROC-AUC: 0.8436
- AUPRC: 0.7719

结果仍低于当前 real-only / stacked ensemble。

### Stacked ensemble 更新

加入 rule-hard 相关模型后：

- validation ROC-AUC: 0.8772
- validation AUPRC: 0.8242
- test ROC-AUC: 0.8818
- test AUPRC: 0.8234

这是当前最高 ensemble 指标。

## 结论

1. 新 loss 解决了已知 positive 被误当负样本的问题。
2. Hard-negative head 可以在候选集内部有效学习 hard-negative 角色。
3. 但规则 hard negatives 作为 downstream 训练数据仍没有单模型提升，说明候选空间的化学质量仍是瓶颈。
4. 当前最强结果仍来自 ensemble，说明这些模型提供了互补信号，但负样本生成器还没达到单模型 SOTA 水平。

## 下一步

下一步不应继续只调 loss，而应扩展 candidate construction：

- 从 C-anchor relaxed candidates 转向更化学合理的 heteroatom/regio/tautomer candidates；
- 引入真实低产率 HiTEA case 作为 hard-negative seed；
- 为 decoder 构建 val/test candidate groups，避免只在 train split 拟合；
- 将 hard-negative head 用作生成器排序器，而不是把所有规则 candidates 硬塞给 downstream BCE。
