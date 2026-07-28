# Phase D source-aware policy：开发协议与证据边界

> 状态：`DEVELOPMENT_PROTOCOL_FROZEN_BEFORE_PHASE_D_RUN`
> 日期：2026-07-29
> 前置结果：Phase 4 fixed pools 已用于方法设计；Phase C v2 仅通过 source-expert 内部验证。

## 1. 目标

Phase D 检验的不是“哪个 generator 普遍最好”，而是：

> 在每条 positive 只允许一个 synthetic negative、训练更新数和下游架构固定的条件下，reaction-conditioned source policy 能否根据完整反应上下文、候选难度和逐候选 FNR，从多种 negative source 中选择更有效的训练信号。

source gate 是训练数据选择器，不是测试时 classifier ensemble。gate 冻结后，每条 parent reaction 只选择一个 source，再从零训练下游模型。

## 2. 当前证据状态

当前 `phase4_fixed_testset_v41` 已经被用于：

- 分析 learned/rule/random/shuffled 的差异；
- 设计 Union_v1、Union_v2；
- 形成 source complementarity 假设；
- 设计当前 reaction-conditioned gate。

因此该 pool 永久标记为：

```text
DEVELOPMENT_SET_USED_FOR_METHOD_DESIGN
```

本阶段在该 pool 上的任何结果只能用于开发、消融和失败诊断。即使开发集 paired CI 全正，也不能称为 confirmatory、blind、SOTA 或 NMI Gate 通过。

## 3. 固定 source 集合

Phase D 使用六个 source expert：

1. `random_mismatch`；
2. `shuffled_real`；
3. `similarity_retrieval`；
4. `template_perturbation`；
5. `rule_pc_cng`；
6. `learned_structured`。

所有 arm 使用同一组 complete-case parent reactions。只有当一条 parent 的六个 source 都能产生不同于 gold、RDKit-valid、可 featurize 的 candidate 时，该 parent 才进入比较。

每个非 positive-only arm 均满足：

```text
n_positive = n_negative = n_complete_parent
one negative per parent
same optimizer epochs
same downstream backbone
same fixed evaluation records
```

`positive_only` 是无负样本对照，不伪装为预算匹配的 negative-source arm。

## 4. Gate 监督信号

### 4.1 交叉拟合

训练 parent 按 `sha256(experimental_group) mod K` 分 fold。

对每个 held-out fold：

1. 在其余 fold 上按均衡的 uniform source allocation 训练下游模型；
2. 对 held-out parent 的六个 source candidate 评分；
3. 将 candidate 被判为 feasible 的概率作为 out-of-fold hardness；
4. 该 parent 从未进入生成其 reward 的下游模型训练。

这避免用同一 parent 的训练拟合分数作为 gate target。

### 4.2 Risk-adjusted reward

FNR 模型只使用当前 scenario training rows 中的 observed positive/zero-yield outcomes 校准。synthetic candidate 的 source label 不作为 FNR 监督。

开发 reward 固定为：

```text
OOF hardness
× (1 − candidate FNR)
× (0.5 + 0.5 × frozen boundary closeness)
```

其中 boundary closeness 使用 Phase 4 已冻结的 similarity 中心，不根据 Phase D evaluation 结果重调。该 reward 是训练期 proxy，不是真实实验效用。

### 4.3 Gate

输入包括：

- reaction-enhanced fingerprint；
- reaction-family hash；
- source identity；
- positive similarity；
- boundary closeness；
- candidate FNR 与 FNR uncertainty；
- validity；
- atom-balance quality；
- family support；
- nearest observed positive/negative similarity。

输出为 `p(source | reaction, task, backbone)`。训练使用 soft target、source dropout 和 entropy regularization；评估时 deterministic argmax，严格选择一个 source。

## 5. 比较 arm

- `positive_only`；
- 六个 single-source arm；
- `uniform_union`；
- `validation_selected_global_mixture`；
- `learned_source_gate`；
- `oracle_source_policy`；
- `randomized_label_null`。

`validation_selected_global_mixture` 和 validation-selected best single 只根据 out-of-fold training reward 确定，不读取 fixed evaluation pool 的 arm 排名。

`oracle_source_policy` 使用每条 training parent 的 OOF reward argmax，仅作为开发上界，不是可部署方法。

## 6. 消融

在预先指定的 ablation backbone 上运行：

- gate 无 reaction context；
- gate 无 FNR；
- gate 无 difficulty；
- gate 无 family；
- gate 无 learned source；
- gate 无 shuffled-real source；
- gate 无 source dropout；
- gate 无 entropy regularization。

`uniform_union` 同时是 equal-source-budget control。

## 7. 统计与主比较

开发 primary endpoint 沿用 fixed pool 的：

```text
source-macro AUPRC
```

预先指定比较：

1. gate − validation-selected best single；
2. gate − uniform union；
3. gate − validation-selected global mixture；
4. gate − gate without learned source。

统计使用同一 experimental cluster 的 paired cluster bootstrap。best single 必须由 OOF training reward 选择，禁止按 evaluation pool 点估计选择。

randomized-label null 与每个 source slice 的理论 positive prevalence 比较，而不是机械地与 0.5 比较。

## 8. Development Exit 与正式 Exit

### Development completion

- 两个数据集完成；
- MLP 与 reaction-aware GNN 两个 backbone 完成；
- 六源 complete-case candidate contract 可审计；
- gate、global mixture、oracle、single、uniform、null 全部完成；
- 指定消融完成；
- 每个非 positive-only arm 预算严格相同；
- 独立 verifier 重建指标、预算、记录对齐与 checkpoint hash。

### Confirmatory Exit

只有在全新 sealed test 上同时满足以下条件，才可把 Phase D 判为正式 GO：

- 至少两个独立真实数据集；
- 至少两个 backbone；
- gate vs validation-frozen best single paired CI 全正；
- gate vs uniform union paired CI 全正；
- leave-one-learned-source-out paired CI 全正；
- randomized-label null 与 source-macro chance 一致；
- candidate budget 完全匹配；
- 模型、threshold、source schema 和统计计划在 labels 解封前冻结。

当前 development pool 无论结果如何，`confirmatory_exit_met` 必须为 `false`。

## 9. Fail-closed 规则

- 没有 CUDA：立即失败；
- Phase C v2 checkpoint 缺失或不兼容：立即失败；
- 任一 parent 缺 source：从所有 arm 的共同 parent 集排除并记录；
- complete-case parent 少于 40：立即失败；
- FNR calibration 缺任一 observed class：立即失败；
- 任一 OOF parent 被用于生成其 reward 的模型训练：禁止；
- 任一非 positive-only arm 不是 1:1：立即失败；
- formal run 缺 sealed manifest：立即失败；
- 当前 Phase 4 fixed pool 被指定为 formal：立即失败；
- 运行中出现 NaN/Inf、CUDA 错误或 artifact alignment 失败：保留日志和 partial artifact，停止该 run，不生成成功 verdict。

## 10. 权威入口与 artifacts

代码入口：

```bash
python -m pc_cng.run_phase_d_source_policy --development-run --gpu 0
python -m pc_cng.verify_phase_d_source_policy <result_dir>
```

每个结果目录必须包含：

- `run_manifest.json`；
- `candidate_cache/*.json`；
- `gate_checkpoints/<scenario>/<backbone>/*.pt`；
- `policy_maps/*.csv`；
- `scored_records/*.csv`；
- `partial_results.json`；
- `phase_d_results.json`；
- `verdict.json`；
- `independent_verification.json`；
- GPU 运行日志。
