# Phase 4 v4.1 最终交接

**完成日期**：2026-07-29
**仓库**：`Cunyu-Liu/PC_CNG`，`main`
**服务器**：`cunyuliu@36.137.135.49:/home/cunyuliu/pc_cng_research`
**开发集标记**：`DEVELOPMENT_SET_USED_FOR_METHOD_DESIGN`

> 本文是 Phase 4 当前权威交接。历史 H1/H2/H3、旧结果和失败运行均保留，但不再把已看过的 fixed pool 用作确认性 SOTA 证据。

---

## 1. 最终结论

Phase 4 的工程任务与开发集分析已经闭合，但机制 Exit Criterion **未满足**。当前可发表级结论边界是：

- fixed difficulty 在看结果前已冻结；
- easy / semi-hard / hard 的受控训练臂已经运行，样本预算和候选有效性匹配；
- GNN 与 EnhancedMLP 两个 scorer 上，semi-hard inverted-U 均只在 `2/8` 场景成立，不是普遍机制；
- continuous spline 分析提供了 reaction-group bootstrap 95% 置信带，但没有 inverted-U 特征跨数据集、跨 scorer 复现；
- family/source 分布没有严格匹配，真实 edit count、scorer margin、reaction-centre locality 和 candidate-level FNR 也未完成可用控制；
- G8-B 正式 full run 为 `NO_GO`：`0/7` 迁移方向存在 CI 全正的方法；
- 因此 Phase 4 状态为 `EXPLORATORY_ONLY_EXIT_NOT_MET`，不得宣称“semi-hard boundary negative 的因果最优区间”。

这不是回退。它把未被数据支持的机制主张正式关闭，并将下一阶段转向可证伪的 source complementarity / adaptive policy。

---

## 2. 本轮完成事项

### 2.1 shuffled_parent 重评分闭环

`rescore_shuffled_parent.py` 已在 8 个场景完成 GPU 重评分。修复后训练与评分都使用同时打乱 reactants + products 的相同特征语义；8 个 authoritative CSV 已同步回 fixed-pool 主目录并逐文件核对。

| 场景 | shuffled_parent source-macro AUPRC |
|---|---:|
| author_lab | 0.997808 |
| condition_space | 0.889271 |
| ni_coupling | 0.940787 |
| random | 0.869487 |
| reaction_family | 0.902358 |
| scaffold | 0.887675 |
| time | 0.857936 |
| uspto_patent | 0.793657 |

高 AUPRC 在完全打乱 parent 后仍存在，因此 shuffled_parent 不能被解释为 chance null。它更接近 broad compatibility / dataset-distribution prior。真正的 chance 检查是 randomized-label arm；其结果接近各 slice 的正例 base rate。

### 2.2 Union_v1 与 Union_v2

Union 系列是查看 fixed-pool 结果后提出的 post-hoc development arm，只能用于方法发现。

- `Union_v1`：learned + rule + shuffled 的均匀可用源采样，固定 1:1 正负预算；
- `Union_v2`：使用冻结的 `[0.40, 0.75]` Tanimoto 区间优先选择 semi-hard 候选，并显式记录 fallback；
- 旧 Union_v1 汇总已保留为 `union_arm_results_v1_before_v2_20260729.json`；
- Union_v2 使用 Phase C v2 checkpoint。一次旧 checkpoint 不兼容运行被立即停止，日志和无效输出另存，不进入正式结果；
- Union runner 已改为 learned source 加载失败时 fail closed，禁止静默退化为 rule+shuffled。

| 场景 | Union_v2 AUPRC | matched fraction | 相对全部单源基线 |
|---|---:|---:|---|
| author_lab | 0.9993 | 0.868 | 未超过 learned |
| condition_space | 0.9531 | 0.860 | 数值领先，CI 未形成全胜 |
| ni_coupling | 0.9906 | 0.958 | 数值领先，vs learned CI 跨 0 |
| random | 0.9776 | 0.876 | 数值领先，vs diff_semihard CI 跨 0 |
| reaction_family | 0.9136 | 0.864 | 数值领先，多个 CI 跨 0 |
| scaffold | 0.9356 | 0.836 | 数值领先，多个 CI 跨 0 |
| time | 0.7993 | 0.850 | 低于 learned / shuffled / diff_semihard |
| uspto_patent | 0.7868 | 0.860 | 低于 learned / shuffled / diff_semihard |

Union_v2 在 5/8 场景数值超过全部单源基线，但没有任何场景同时显著超过全部预注册比较臂，因此 H1u_v2 为 `0/8 SOTA`。独立 verifier 的全部检查通过：8 场景齐全、checkpoint 加载成功、每场景实际选择 learned source、500 negatives、逐记录完全对齐、score 有限、difficulty metadata 完整。

### 2.3 连续机制重分析

新增 `pc_cng.analyze_phase4_mechanism_continuous`：

- 不再使用十等分任意分箱；
- 用 cubic regression spline 拟合连续关系；
- 用 reaction-group cluster bootstrap 生成 95% 置信带；
- 观测终点定义为“负候选在同一 reaction group 内的 score percentile”，只表示 negative hardness，不冒充 downstream utility；
- 15/18 个 dataset × scorer × feature 曲线单元可估计；
- 3 个 FNR 单元不可估计，因为历史 candidate-level FNR 全为常数 0.5；
- 仅 2 个单独曲线单元满足严格 interior-peak 判据，跨数据集、跨 scorer 复现的 feature 为 0。

权威 artifact：

```text
chem_negative_sampling/results/phase4_mechanism_continuous_v2_20260729/
├── continuous_curves.json
├── curve_summary.csv
├── driver_effects.json
├── mechanism_status.json
├── input_hashes.json
└── run_manifest.json
```

### 2.4 competing explanations 与匹配审计

已覆盖：

- candidate source 异质性；
- reaction family 异质性；
- randomized-label null；
- shuffled transfer 的 formula-preserved / formula-changed 分层；
- G8-B 跨家族与跨数据集负迁移。

未满足：

- family distribution exact matching；
- source distribution exact matching；
- true edit count matching；
- scorer margin pre-pool matching；
- reaction-centre locality；
- usable candidate-level FNR；
- known-positive collision 的双层估计（正例过少）；
- family support density。

因此当前 controlled intervention 仍有 competing explanations，不能升级为机制 GO。

### 2.5 G8-B 负迁移保留

正式 full run：

```text
results/p4_cross_family_transfer_v2/
```

结果：

- 7 个迁移方向；
- 6 种方法；
- 10 seeds；
- `directions_with_any_positive_ci = 0`；
- verdict：`NO_GO`。

该结果不删除、不改口径。它支持的是“failure knowledge 具有 family/domain 依赖，直接迁移、LoRA、EWC、risk-aware 或 multitask 都未稳定解决负迁移”，而不是普遍可迁移主张。

---

## 3. Phase 4 假设最终状态

| 假设 | 最终结果 | 判定 |
|---|---|---|
| H1 learned structured SOTA | 1/8 场景 | `NO_GO_DEVELOPMENT` |
| H1u Union_v1 SOTA | 0/8 场景 | `EXPLORATORY_POST_HOC` |
| H1u_v2 difficulty-matched Union SOTA | 0/8 场景；5/8 仅数值领先 | `EXPLORATORY_POST_HOC` |
| H2 shuffled_parent 是 chance null | 否；AUPRC 0.794–0.998 | 改释为 compatibility/distribution prior |
| literal randomized-label null | 接近 slice base rate | leakage audit 通过 |
| H3 semi-hard inverted-U | GNN 2/8；EnhancedMLP 2/8；连续特征复现 0 | `UNSUPPORTED_DEVELOPMENT` |
| G8-B transferable failure knowledge | 0/7 positive-CI directions | `NO_GO_NEGATIVE_TRANSFER` |

旧 H1/H2/H3 历史保存在 `docs/phase4_hypothesis_history_20260728.md`，没有被 Union 或连续分析覆盖。

---

## 4. 原 Todo 完成审计

| Todo | 状态 | 证据边界 |
|---|---|---|
| 在看 test 前冻结 difficulty | 完成 | `difficulty_v1_frozen_20260726` |
| 避免任意分箱，使用 spline + CI | 完成 | continuous mechanism v2 |
| 构造 easy / semi-hard / hard pools | 完成 | fixed-pool v4.1 |
| 匹配 validity | 完成 | 全池 1.0 |
| 匹配 candidate budget | 完成 | 训练臂相同 `n_diff_per_arm` |
| 匹配 family/source/edit count/scorer margin | 未满足 | fail-closed matching audit |
| 随机抽样不同 difficulty negatives 训练 | 完成 | diff_easy / diff_semihard / diff_hard |
| 检验 inverted-U | 完成，结果不支持普遍机制 | 两个 scorer 均 2/8 |
| 第二 dataset / scorer replication | 部分 | 第二 scorer 为同一 development pool，不是 sealed confirmation |
| 分析 boundary utility drivers | 部分 | source/family/similarity/uncertainty 可分析；FNR/locality/edit type 不可用 |
| 保留 G8-B 负迁移 | 完成 | full run `NO_GO`, 0/7 |

---

## 5. 下一阶段交接

Phase 4 不再进行 test-guided 阈值、分箱或 fixed-pool 调参。下一阶段进入 `pccng3.md` Phase D：

1. 在 development / validation 上实现训练期 source-aware policy；
2. 明确区分“选择训练 negatives”与“推理期多个 classifier 集成”；
3. 比较 single source、uniform union、global mixture、learned gate 和 oracle；
4. 所有 arm 保持相同 candidate budget 和 optimizer budget；
5. 当前 fixed pools 只用于开发，最终主张必须等待全新的 sealed test；
6. 若 gate 不优于 uniform union，立即按预注册失败分支降级主张。

---

## 6. 常用入口

```bash
# 重建 Phase 4 聚合
python -m pc_cng.analyze_phase4_v41 \
  --base-results results/phase4_fixed_testset_v41

# 重建连续机制分析
python -m pc_cng.analyze_phase4_mechanism_continuous \
  --per-candidate-metrics ../results/p4_mechanism_curve/per_candidate_metrics.csv \
  --manifest ../data/p4/manifests/hte_feasibility_v2.json \
  --phase4-aggregation results/phase4_fixed_testset_v41/phase4_v41_aggregation.json \
  --phase4-per-scenario results/phase4_fixed_testset_v41/per_scenario_results.json \
  --second-scorer-aggregation /mnt/cunyuliu_pc_cng_phase4_mlp_20260728/phase4_fixed_testset_v41_mlp_20260728/phase4_v41_aggregation.json \
  --g8b-transfer ../results/p4_cross_family_transfer_v2/transfer_analysis.json \
  --output-dir results/phase4_mechanism_continuous_v2_20260729

# 独立验证 Union_v2
python -m pc_cng.verify_phase4_union_v2 \
  --base-results results/phase4_fixed_testset_v41 \
  --checkpoint results/p4_g8c_formal_v2_20260729/model_checkpoint.pt \
  --log results/phase4_fixed_testset_v41/logs/union_v2_phasec_v2_gpu0_20260729.log \
  --repo-root .. \
  --output results/phase4_fixed_testset_v41/union_v2_verification.json
```
