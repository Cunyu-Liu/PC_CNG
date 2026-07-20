# PC-CNG v2：反应中心边界负样本生成器方案

## 1. 为什么要从 MVP 升级

重新核查 Science Advances 2025 论文后，需要修正原先的假设：真正有价值的负样本不是随机错配，也不是任意字符串扰动，而是**接近成功反应流形边界的失败实验**。

论文最强调的 type-1 negative 是：

- 同一或相近反应上下文；
- 生成了意外但化学合理的 alternative product；
- 能揭示 regioselectivity、chemoselectivity、reaction-center choice 的边界。

我们 MVP 的负样本主要来自：

- `product:=reactants`
- `retro_no_disconnection`
- `retro_missing_reactant`
- `append:O`
- Br/Cl、N/O 字符串替换

这些样本大多不是高质量 type-1 negatives，因此直接混入训练会伤害 ROC-AUC。实验证据也支持这一点：direct PC-CNG、weighted PC-CNG、paper-aligned filtering、pairwise reward training 都没有超过 real-only baseline。

## 2. v2 目标

PC-CNG v2 的目标不是“多生成负样本”，而是生成**反应中心附近的化学合理竞争产物**。

目标负样本形式：

```text
same reactants / same context -> alternative product
```

其中 alternative product 应满足：

- 产物合法；
- 反应中心局部变化；
- 与真实 positive product 有中等相似度；
- 不是已知 positive；
- 不是 obvious artifact；
- 可作为 reward / preference 信号，而不一定作为 hard negative label。

## 3. 新架构

### 3.1 Reaction-center encoder

输入 atom-mapped reaction：

```text
mapped reactants >> mapped product
```

抽取：

- formed bonds；
- broken bonds；
- changed bonds；
- reacting atom map numbers；
- reaction-center signature。

如果 reaction 未映射，则优先调用 RXNMapper。

### 3.2 Boundary generator

在 product 的 reaction-center atoms 附近做局部图编辑：

- center atom transmutation：如 Br->Cl、O->N、O->S；
- center bond order perturbation：如 SINGLE->DOUBLE、DOUBLE->SINGLE；
- 后续可加入 regio shift、leaving-group competition、nucleophile competition。

输出 type-1 counterfactual：

```text
reactants >> locally edited alternative product
```

### 3.3 Quality filter

保留标准：

- RDKit valid；
- product similarity 在中间区间；
- false-negative risk 不过高；
- atom-balance 不过低；
- edit locality 靠近 reaction center。

### 3.4 Reward training

不再把 generated negative 简单当 BCE 负类。更合理的训练信号是：

```text
score(observed positive) > score(type-1 counterfactual) > score(random/artifact)
```

当前已实现 `train_pairwise_reward_mlp.py` 作为过渡版本。后续应升级为 graph encoder + DPO/RL-style objective。

## 4. 已落地代码

代码路径：

- `pc_cng/reaction_boundary_generator.py`
- `pc_cng/run_boundary_generation.py`
- `pc_cng/filter_paper_aligned_negatives.py`
- `pc_cng/train_pairwise_reward_mlp.py`

运行示例：

```bash
PYTHONPATH=. python -m pc_cng.run_boundary_generation \
  --input /path/to/positives.csv \
  --output /path/to/v2_boundary_negatives.csv \
  --summary /path/to/v2_boundary_summary.json \
  --limit 1000
```

## 5. 下一轮验证

最小闭环：

1. 对 RegioSQM20 / HiTEA positive train reactions 生成 v2 boundary negatives。
2. false-negative review。
3. pairwise reward training。
4. 与以下模型比较：
   - real-only MLP；
   - direct PC-CNG MVP；
   - paper-aligned type-1 filtered MVP；
   - stacked ensemble；
   - v2 boundary pairwise reward。

成功标准：

- validation ROC-AUC/AUPRC 超过 current best real-only；
- test 不靠反选提升；
- generated negatives 的人工抽样化学合理性明显高于 MVP；
- 在 RegioSQM20 的 regio/chemoselectivity case 上有可解释提升。

## 6. 仍需升级的部分

当前 v2 是第一版工程实现，还不是最终 SOTA：

- 生成器仍是规则/局部图编辑，不是训练得到的生成模型；
- regio shift 还未系统实现；
- 没有显式条件/催化剂上下文建模；
- reward model 仍是 fingerprint MLP，不是 graph transformer；
- 还没有对 retrosynthesis reranking 做闭环验证。

真正 SOTA 版本应进一步实现：

```text
USPTO pretraining -> atom-mapped graph encoder -> reaction-center edit decoder
-> type-1 boundary generator -> reward/DPO training -> retrosynthesis reranking evaluation
```
