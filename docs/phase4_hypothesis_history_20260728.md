# Phase 4 假设历史与状态冻结记录

**冻结日期**：2026-07-28  
**用途**：保存 H1/H2/H3 的原始定义、后续 amendment 和证据边界；本文件不得覆盖历史假设。

## 1. 原始假设（Phase 4 v4.1）

| 假设 | 原始定义 | 原始主要比较 |
|---|---|---|
| H1 | learned structured negatives 在固定 semi-hard test pool 上优于主要单一 negative source | learned vs rule / random / shuffled-parent |
| H2 | shuffled-parent 作为 hard/null control，不应保留可迁移的化学特异性信号 | shuffled-parent 的 source-macro AUPRC 接近 slice base rate |
| H3 | semi-hard negatives 存在 inverted-U utility peak | diff_semihard > diff_easy 且 diff_semihard > diff_hard |

原始 difficulty 定义在 test 结果前冻结为 Morgan radius-2/2048 Tanimoto：easy `<0.40`、semi-hard `[0.40,0.75]`、hard `>0.75`。后续不得因结果改变阈值或重新分箱。

## 2. 2026-07-28 amendment history

### A. fixed-pool contamination

固定池被用于 Union、Union_v2、shuffled-parent 修复和第二 scorer 设计。因此所有固定池结果统一标记：

```text
DEVELOPMENT_SET_USED_FOR_METHOD_DESIGN
```

这些结果可以用于审计、失败分析和方法消融，不能用于新增方法的 sealed confirmatory claim。

### B. H1 status

canonical `phase4_v41_aggregation.json` 的最终主分析为 learned SOTA `1/8`；Union 与 Union_v2 不是 learned 的确认性替代。H1 当前为：

```text
NO_GO_DEVELOPMENT
```

### C. H2 status

修复版 shuffled-parent 同时打乱 reactants 和 products，并恢复正负训练配对后，AUPRC 仍明显高于 literal null。formula-preserved 与 formula-changed diagnostics 分别为 `0.9095` 和 `0.7795`。因此 shuffled-parent 学到了广义 compatibility-transfer prior，H2 原 hard/null 解释不成立。

随机标签 arm 的 median AUPRC 约为 `0.58`，只能作为 development leakage check；它是 literal null，不等于 external utility evidence。

H2 当前为：

```text
NOT_A_NULL_CONTROL
```

### D. H3 status

fixed development pool 中仅 `2/8` 场景同时满足 semi-hard > easy 和 semi-hard > hard，不能支持普遍 inverted-U 机制。RegioSQM20 外部 replication 在单一 scenario 上满足两项，但该结果不能泛化为机制确认。

H3 当前为：

```text
UNSUPPORTED_DEVELOPMENT
```

### E. post-hoc arms

Union、Union_v2、source-aware gate 和组合 negative-source arms 均在查看 fixed-pool 结果或其诊断后形成，统一标记：

```text
EXPLORATORY_POST_HOC
```

它们不得改写原 H1/H2/H3，也不得回写为预注册 confirmatory hypothesis。

## 3. 当前 claim 映射

- C11：H1，`NO_GO_DEVELOPMENT`
- C12：H3，`UNSUPPORTED_DEVELOPMENT`
- C13：H2 修复后的 compatibility-transfer prior，`PASS_DEVELOPMENT_ONLY`
- C14：Union/source complementarity，`EXPLORATORY_POST_HOC`
- C15：randomized-label literal null，`PASS_DEVELOPMENT_ONLY`

## 4. 重新进入 confirmatory 阶段的条件

只有在新数据或 sealed split 上，先冻结 source policy、endpoint、difficulty 分析和统计方案，再运行一次盲测，才能重新提出 confirmatory H1/H3 或新的 source-gate hypothesis。当前 fixed pool 不得重新充当 confirmatory test。
