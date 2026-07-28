# NMI_FINAL_GOAL v2.0 Amendment

**日期**：2026-07-28
**基线**：`docs/00_当前有效文档/NMI_FINAL_GOAL.md` v1.0
**来源**：`提示词/pccng3.md` Phase A 与 Phase 4 v4.1 交接修订

## 状态重冻结

Phase 4 fixed pools 已经被用于设计 Union/Union_v2 和 shuffled-parent 修复，因此从本修订起统一标记为：

```text
DEVELOPMENT_SET_USED_FOR_METHOD_DESIGN
```

它们仍可用于调试、失败分析和方法消融，但不能再作为 Union/gate 的 confirmatory test。

旧 G8-C tiny self-built utility gate 统一标记为：

```text
DEPRECATED_INVALID_EVALUATION
```

## Phase 4 当前证据边界

- H1 learned-structured SOTA：固定池开发结果不支持；不得写成主方法优势。
- H3 semi-hard inverted-U：开发结果不支持；不得通过事后改分箱追求转正。
- Union：6/8 数值领先、仅少数比较经 Holm 后显著的旧结果属于 exploratory post-hoc evidence。
- randomized-label null：开发集 control 结果可保留为泄漏检查，不等于外部效用证据。
- shuffled-parent：旧 one-class/feature-contract 不一致的 rescore 作废；修复版 8/8 已完成，最终主分析 alignment 128/128；它仍不是随机 null，而是 development-only compatibility-transfer 证据。formula-preserved AUPRC=0.9095，formula-changed AUPRC=0.7795。
- Union_v2：主恢复目录 8/8 完成；frozen Morgan radius-2/2048 Tanimoto 区间为 `[0.40, 0.75]`，matched_fraction=0.866–0.976，fallback_count=12–67；5/8 点估计胜出，但 H1u_v2=0/8 Holm-confirmed SOTA，仍为 post-hoc development-only。
- 第二 scorer：EnhancedMLP 已在 GPU 6 完成当前 fixed-pool 8/8 exploratory replication；H1=1/8、H2 shuffled median=0.8149、H3=2/8；没有 randomized-label arm，不作 null-control 结论。

## North-Star 改写

### English

> Develop a reaction-conditioned, source-aware counterfactual learning framework that estimates candidate-level false-negative risk and adaptively selects among structured, rule-based, retrieval-based and observed-product-derived negatives under a matched training budget; validate the policy on a sealed real-HTE/OOD test and retain negative-transfer outcomes.

### 中文

> 构建以完整反应上下文为条件、能够感知负样本来源差异的反事实学习框架，在严格匹配训练预算下估计逐候选假阴性风险，并自适应选择结构化、规则、检索和 observed-product-derived 负样本；最终在全新封存的真实 HTE/OOD 测试上验证，并保留负迁移结果。

learned structured generator 从唯一成败点降级为 source expert；只有在新盲测中证明 gate 优于最佳单一来源和 uniform union，才恢复其 NMI 主方法地位。

## pccng3/G8-C 实施状态

已完成一项不改变科学结论、但必须先修复的运行语义问题：

- 正式模式增加 `--formal-run` fail-closed；缺少真实 edit target、rule proposal、competing outcome 或 preference pair 时终止运行，不生成伪监督 checkpoint。
- Stage 4 的 frozen reference policy 改为在当前轮 Stage 3 完成后复制。
- 远端语法检查、既有四阶段回归测试和缺 cache 最小验收通过。

这只证明正式训练路径不会静默退化为伪监督；不证明 learned generator 已经优于任何 negative source。

同时已加入 `pc_cng/source_aware_policy.py` 的 development-only softmax gate 原型：

- 输入 reaction representation、逐 source candidate statistics 和 availability mask；
- 输出在 matched one-negative budget 下的 source distribution；
- 对 unavailable source 做显式 mask，并拒绝“所有 source 均不可用”的输入；
- 不读取 test label、不内置 source ratio，也未接入 confirmatory benchmark。

该模块及 3 项单元测试通过；在 sealed validation/test 和预注册 endpoint 冻结前，不将其称为 adaptive-policy 结果。

## pccng3/G6 证据边界

G6 的现有实现已通过回归/数据 schema sanity tests，但这只证明旧评估管线可运行，不能把它升级为 NMI 级任务模型。代码审计仍确认：

- T1–T5 的主分数仍主要来自 product Morgan fingerprint；
- T2 的“ordinal”仍是单一分数上的 OvR macro-AUPRC，而不是 cumulative-link/CORAL/CORN 输出；
- T3 仍把分类分数乘以 100 作为 yield regression proxy；
- T4 仍由 pointwise 分数计算 NDCG，不是训练得到的 pairwise/listwise ranker；
- collision sensitivity 仍是占位常数，未由独立 collision stress set 计算。

因此 G6 当前状态继续标记为 `INVALID_PENDING_REANALYSIS`：不得把回归测试通过、旧 summary 或 proxy 指标写成外部效用结论。下一次正式 G6 运行必须先完成真正的 task-specific heads、完整 reaction/condition representation、独立 stress set 和冻结的 paired cluster analysis。

## 新增 claim registry 条目

| claim_id | claim | status | boundary |
|---|---|---|---|
| C11 | Phase 4 fixed-pool learned SOTA | `NO_GO_DEVELOPMENT` | 仅开发集，0/8 主比较支持；不得外推 |
| C12 | semi-hard inverted-U utility | `UNSUPPORTED_DEVELOPMENT` | 当前数据不支持；需新盲测 controlled replication |
| C13 | shuffled-real compatibility transfer | `PASS_DEVELOPMENT_ONLY` | repaired 8/8 rescore + final alignment 128/128；formula-preserved AUPRC=0.9095, changed=0.7795；仅支持 development-only compatibility prior 结论 |
| C14 | union source complementarity | `EXPLORATORY_POST_HOC` | 旧 fixed pool 用于方法设计，不能作确认性结果 |
| C15 | randomized-label null control | `PASS_DEVELOPMENT_ONLY` | 可检查泄漏；不支持 external utility claim |

## 本次交接已完成

1. 修复版 shuffled-parent rescore、配对身份检查和 H2/source-transfer 诊断已完成；
2. 真正按 frozen semi-hard 区间筛选的 Union_v2 已在 `/mnt/cunyuliu_pc_cng_phase4_v2_20260728` 完成 8/8，并记录 matched fraction/fallback；
3. 最终 analyzer 已完成；权威文件为 `phase4_v41_aggregation.json`，恢复目录中旧复制分析文件不参与最终判定。

## 下一阶段执行顺序

1. 完成 Phase A 状态同步：README、NMI goal amendment、claim registry 和结果 provenance；
2. 仅在开发集完成 source gate 原型，不在 fixed pool 上宣称确认性 SOTA；
3. 新建 sealed test split、冻结 gate/endpoint/统计方案后再进入 confirmatory evaluation；
4. G7 专家 pilot 与 prospective experiment 仍是独立阻塞项。

## 投稿门槛

在下列证据完成前，不启动 NMI manuscript：

- gate 优于最佳 single source 和 uniform union；
- 至少两个独立数据集、两个 downstream backbone；
- 至少一个真实 HTE primary endpoint 和一个严格 OOD endpoint；
- paired cluster CI、多重比较和 null control 均通过；
- 专家或前瞻实验验证；
- 所有负结果、旧作废结果和方法设计污染边界均公开。
