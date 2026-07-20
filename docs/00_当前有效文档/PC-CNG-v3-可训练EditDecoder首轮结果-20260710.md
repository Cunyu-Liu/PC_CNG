# PC-CNG v3 可训练 Edit Decoder 首轮结果

## 本轮目标

按照最新架构方向，把 v2 的规则 reaction-center migration 升级为可训练的 reaction-center edit decoder：

```text
normalized reactions
-> atom-mapped reaction center extraction
-> candidate anchor dataset
-> trainable edit decoder
-> learned boundary negative generation
-> downstream pairwise/BCE training
```

## 已实现代码

- `pc_cng/reaction_center_edit_decoder.py`
- `pc_cng/build_edit_decoder_dataset.py`
- `pc_cng/train_reaction_center_edit_decoder.py`
- `pc_cng/run_learned_boundary_generation.py`
- `scripts_run_edit_decoder_v3_pipeline.sh`

## 数据预处理兼容性检查

当前 `data_ingestion.py` 输出字段满足新 decoder 的最低要求：

- `source_id`
- `reaction_smiles`
- `label_type`
- `split`
- `source`

新增兼容：

- 可识别 `mapped_rxn`
- 可识别 `mapped_reaction_smiles`
- 可识别 `label_type`

小样本 smoke：

- 输入 500 行 RegioSQM20 + HiTEA normalized CSV。
- 成功构建 397 个 candidate groups。
- 生成 3573 行 anchor candidates。
- 跳过原因可解释：real_negative 被跳过、部分 positive 无可迁移 formed bond。

## 全量 candidate dataset

A100 路径：

```text
/home/cunyuliu/pc_cng_research/results/edit_decoder_v3_full/edit_decoder_candidates.csv
```

统计：

- seen rows: 41,970
- candidate rows: 3,754
- candidate groups: 448
- groups_by_source: HiTEA 448

跳过原因：

- not_positive: 25,920
- no_candidate_anchor: 11,297
- no_formed_bond: 3,676
- mapping_failed: 629

解释：当前 v3 decoder 只覆盖“formed-bond substituent migration”这类 edit，因此覆盖范围很窄，主要集中在 HiTEA 的 N-alkylation 场景。

## Edit decoder 训练

A100 路径：

```text
/home/cunyuliu/pc_cng_research/results/edit_decoder_v3_full/reaction_center_edit_decoder
```

训练结果：

- train groups: 448
- val groups: 0
- test groups: 0
- train top1 accuracy: 1.0
- train MRR: 1.0

注意：这只是链路跑通和训练集拟合，不是独立泛化证明。当前可构建 candidate groups 的 split 全在 train，说明还需要扩展 candidate construction 才能获得 val/test group。

## Learned generation 结果

严格生成模式：

- 要求 candidate anchor 与真实 anchor 同原子类型；
- 跳过 known positives。

结果：

- HiTEA generated: 0
- Regio generated: 0

原因：decoder 最高分的 N->N regioisomer 候选全部是数据集中已知 positive，被 known-positive review 正确丢弃。

放宽生成模式：

- 允许不同原子类型 anchor；
- 保留 known-positive 过滤；
- 生成 HiTEA learned negatives: 896
- review 后 keep: 896

但这些 relaxed negatives 多包含 C-anchor methylation 等低置信候选，化学质量弱于 N->N regioisomer。

## 下游训练结果

### v3 relaxed pairwise reward

路径：

```text
/home/cunyuliu/pc_cng_research/results/v3_learned_relaxed_pairwise_reward_h2048_n4096_e80
```

结果：

- validation ROC-AUC: 0.8439
- validation AUPRC: 0.7723

### v3 relaxed direct BCE

路径：

```text
/home/cunyuliu/pc_cng_research/results/v3_learned_relaxed_direct_bce_h2048_n4096_e80
```

结果：

- validation ROC-AUC: 0.8427
- validation AUPRC: 0.7699

两者都明显低于当前 real-only/stacked ensemble。

## 结论

本轮不是最终成功模型，但有两个重要结论：

1. 工程链路跑通了。
   数据预处理、candidate dataset、edit decoder 训练、learned generation、review、downstream training 全部可运行。

2. 当前训练目标不对。
   decoder 学到的是“观察到的真实 anchor 概率”，最高分候选往往就是已知 positive。跳过 known-positive 后，剩下候选质量明显降低。

因此，下一版 decoder 不能只训练 `observed anchor > alternative anchor`。它需要直接学习：

```text
high-quality unobserved type-1 boundary negative
```

也就是目标应从 observed-anchor ranking 改为 hard-negative generation/ranking。

## 下一步修复方向

1. 扩展 candidate construction
   - regio shift；
   - competing heteroatom anchors；
   - leaving-group alternatives；
   - acyl/alkyl migration；
   - product tautomer/chemoselective alternatives；
   - real low-yield anchors from HiTEA。

2. 加入 positive exclusion-aware loss
   - observed positive anchors 不应直接成为生成目标；
   - 已知 positive candidates 应作为 masked candidates；
   - 训练目标应鼓励模型选择“接近 positive 但不在 known positive set 中”的候选。

3. 引入三层排序目标

```text
observed positive > plausible unobserved boundary negative > artifact/random
```

4. 保留当前 best 模型
   当前最强仍是包含 v2 信号的 stacked ensemble：

```text
validation ROC-AUC: 0.8740
test ROC-AUC: 0.8816
test AUPRC: 0.8184
```

v3 learned relaxed negatives 暂不纳入主结果。
