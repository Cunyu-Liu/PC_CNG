# 负样本生成模型文档索引

## 当前有效文档

建议优先阅读 `00_当前有效文档/`：

- `PC-CNG-v2-反应中心边界生成器方案.md`：当前最新架构方案。
- `SCIADV_NEGATIVE_DATA_REASSESSMENT_20260710.md`：重新核查 Science Advances 论文后的负样本定义与问题复盘。
- `EXPERIMENT_ANALYSIS_20260710.md`：已有实验结果、失败路线与当前最佳模型总结。
- `顶刊论文核心思想与从0到1落地方案.md`：论文级落地路线。
- `模型架构设计-物理约束反事实负反应生成器.md`：PC-CNG v1/v2 的早期架构基础。

## 历史参考文档

`99_历史参考文档/` 存放早期想法验证、旧策略和调研草稿。它们不再作为当前实现依据，但可用于追溯想法来源。

## 当前执行方向

当前方向已经从“规则字符串扰动生成大量负样本”切换为：

```text
atom-mapped reaction center extraction
-> reaction-center local alternative product generation
-> type-1 boundary negative filtering/review
-> pairwise reward / DPO-style training
-> real validation/test evaluation
```

核心代码位于：

```text
../code/chem_negative_sampling/pc_cng/reaction_boundary_generator.py
../code/chem_negative_sampling/pc_cng/run_boundary_generation.py
../code/chem_negative_sampling/pc_cng/train_pairwise_reward_mlp.py
```
