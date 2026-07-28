# PC-CNG → Nature Machine Intelligence：唯一有效目标文档

> **文档版本**：v1.0（Phase 0 证据冻结）
> **创建日期**：2026-07-24
> **审计依据**：`pccng 的分阶段提示词 2.md` §一/§二/§三
> **核心原则**：不为了让 Gate 变绿而调指标；只接受能够经受独立审稿、复现和外部验证的证据。
> **文档地位**：本文件是 PC-CNG 项目当前唯一有效的目标与状态文档。自此日期起，`docs/00_当前有效文档/` 下其他历史规划文档（`顶刊论文核心思想与从0到1落地方案.md`、`P4_GOAL_20260721.md` 等）仅作历史参考，不再驱动当前决策。新增决策以 amendment 形式追加至本文件 §八。

---

## 一、North-Star Goal

> **Develop a reaction-conditioned learned structured counterfactual generator that learns chemically meaningful failure directions from abundant positive reactions and scarce observed negative outcomes, generates valid and non-trivial boundary candidates with calibrated false-negative risk, and improves independent real-HTE and out-of-distribution reaction-learning tasks over strong matched negative-sampling baselines.**

中文表述：构建一个以完整反应上下文为条件的学习型结构化反事实生成器。模型从大量成功反应中学习反应图语法，从少量真实失败或竞争产物中学习失败方向，在控制假阴性风险的前提下生成有效、非平凡、位于可行性边界附近的反事实候选，并在独立真实 HTE 与 OOD 任务中稳定优于严格匹配的强负采样基线。

---

## 二、论文只保留三个核心主张

### Claim 1：方法创新

PC-CNG 是一个真正学习的、reaction-conditioned、reaction-center-aware structured generator，而不是规则库的包装。

### Claim 2：外部效用

PC-CNG 生成的 negatives 在至少两个独立数据集、两个下游 backbone 和预注册的主要任务上，显著优于：random mismatch；fingerprint retrieval；template perturbation；unconstrained structural edit；rule PC-CNG；现有 reaction-center-aware negative sampling；matched hard-negative mining。

"负数据有用"已由已有研究证明，PC-CNG 的新增价值必须是：**在真实负数据稀缺时，如何生成比既有负样本更可信、更有用的合成边界样本。**

### Claim 3：可信性与机制

PC-CNG 能够显式控制 false-negative risk，并通过真实 HTE、OOD、专家或前瞻实验表明：
- 不只是生成"容易被模型识别的假负样本"；
- 不只是提升自建 benchmark；
- utility 来源于接近反应边界的结构化编辑；
- 模型在训练分布之外仍具有可迁移性。

---

## 三、修正后的 Gate 状态（Phase 0 审计冻结）

| Gate | 原始判断（tracked artifact） | 审计后判断 | 原因 |
| ---- | -------------------- | ----------------------- | ------------------------------------------------ |
| G3   | NO_GO（v1 manifest，A6=A2 重复） | **REDO / PROMISING** | v2 manifest 修复后有正向信号，但训练输入、预算匹配和统计单位不足以支持 Strong GO |
| G4   | GO                   | **SUPPORTIVE GO**       | 能说明 generator×scorer 存在交互，但不能证明 PC-CNG 普遍优于简单负样本 |
| G5   | PARTIAL_GO           | **SAFETY PARTIAL_GO**   | 证明 calibration/abstention 改善，不等于生成器有效性提升 |
| G6   | WEAK_GO              | **INVALID_PENDING_REANALYSIS** | 任务定义错误、训练组混杂、delta CI 计算不正确；整体重做 |
| G7   | DEFERRED             | **DEFERRED**            | 必须获得真实专家或实验数据 |
| G8-A | GO                   | **EXPLORATORY_MECHANISM** | 当前曲线不足以构成机制证据，仅作探索性分析 |
| G8-B | 运行中                | **保持独立判定**         | 不应预设必须 GO；负迁移本身可能是重要科学结论 |
| G8-C | NO_GO                | **PROTOTYPE_NO_GO**     | 当前训练阶段并未真正实现所声称的四阶段监督，仅为结构化生成器 prototype |

### 状态漂移修复记录

1. **README 状态漂移**：README.md 原标记"P1 阶段（2026-07-19 启动）"，与仓库实际推进至 P4/G8 不一致。Phase 0 已更新 README 仅描述当前阶段与可验证结果。
2. **G3 artifact 漂移**：`results/p4_augmentation/go_no_go.json` 原为 v1 manifest 的 NO_GO（A6 与 A2 bit-identical），最新提交已描述 v2 为 WEAK_GO。Phase 0 已在 `results/p4_augmentation/nmi_audit_status.json` 记录审计状态，原始 v1 verdict 保留为历史记录。
3. **G6 verdict 失效**：`results/p4_hte_external_validation/go_no_go.json` 原 WEAK_GO 因任务定义、实验混杂、delta CI 计算问题而失效。Phase 0 已在 `results/p4_hte_external_validation/nmi_audit_status.json` 标记为 INVALID_PENDING_REANALYSIS。
4. **G8-A verdict 降级**：`results/p4_mechanism_curve/go_no_go.json` 原 GO 降级为 EXPLORATORY_MECHANISM。Phase 0 已在 `results/p4_mechanism_curve/nmi_audit_status.json` 记录。
5. **G8-C verdict 重分类**：`results/p4_learned_proposal_full/go_no_go.json` 原 NO_GO 重分类为 PROTOTYPE_NO_GO（接口完成但训练语义需重建）。Phase 0 已在 `results/p4_learned_proposal_full/nmi_audit_status.json` 记录。

---

## 四、Phase 0 证据冻结 Exit Criterion

- [x] 仓库只有一个当前状态（本文件为唯一有效目标文档）
- [x] 所有 headline claim 都能一键定位到代码、输入和结果（见 `docs/claim_registry.csv`）
- [x] 不存在最新文档与 tracked artifact 相互冲突（Phase 0 已通过 `nmi_audit_status.json` 覆盖文件消除冲突）
- [x] 历史结果仍可审计，但不再参与当前 Gate（原始 go_no_go.json 保留，审计状态以 override 文件为准）

---

## 五、分阶段执行路径

### Phase 1：G3/G6 评测与统计系统重建
- 模型输入改为完整反应上下文（reactants + conditions + candidate product）
- 所有 arm 严格匹配训练预算（样本数/batch 数/optimizer updates/evaluations）
- G6 五个任务建立真正独立的任务头（T1 二分类、T2 ordinal、T3 回归、T4 ranking、T5 condition-aware）
- 统计改为 cluster-aware paired bootstrap + permutation test + 多重比较校正
- Exit：paired cluster CI 可由独立脚本重建；重跑后重新判定

### Phase 2：学习型结构化生成器重建
- 四阶段训练语义重建（真实 edit reconstruction → rule imitation → observed competing outcome → risk-adjusted preference）
- 至少 11 个 baseline 对比
- Exit：full learned PC-CNG 在 ≥2 数据集、≥2 backbone 上 paired CI 全正

### Phase 3：真实外部验证与 G7
- ≥2 真实外部 HTE 数据集 + OOD splits（time/patent/author/family/scaffold/condition）
- 三专家 pilot（80-100 条，双盲）→ main review（200-300 条）
- 小规模前瞻实验（预注册、盲法、冻结分析）
- Exit：预注册 primary endpoint 显著改善 + 专家或前瞻验证

### Phase 4：机制与迁移
- 冻结 difficulty 定义后做 controlled intervention
- difficulty-controlled candidate pools（easy/semi-hard/hard）匹配后检验 inverted-U utility
- held-out replication
- Exit：observational + intervention + replication + uncertainty + ablation 同时具备

### Phase 5：NMI 投稿 Gate
- 方法/外部效用/可信性/机制/可复现性五类必达条件全部满足后才启动 manuscript

---

## 六、立即停止的错误方向

1. 不再针对 G3 的 0.06pp 阈值做 test-guided 调参
2. 不再通过改变 clustering 方式让 CI 下界变正
3. 不再把更多 bootstrap iterations 描述为"收窄真实 CI"
4. 不再把 G5 calibration improvement 当作 external utility
5. 不再把 G8-A 当前结果称为 mechanism GO
6. 不再把 G8-C 当前四阶段训练描述为真实 reconstruction/imitation/competing-outcome/DPO
7. 不再使用 manuscript 字节数或 GO task 数决定期刊

---

## 七、失败分支（不逃避，但也不伪造成功）

- **分支 A**：learned generator 优于规则版本 → 继续 NMI 冲刺
- **分支 B**：learned 不优于规则，但规则 PC-CNG 外部效用稳定 → 转 Chemical Science / JACS Au
- **分支 C**：只改善 calibration/selective risk，不改善外部 utility → 转安全学习/不确定性论文
- **分支 D**：外部 HTE 结果不稳定 → 研究适用边界、负迁移和 failure taxonomy
- **分支 E**：专家与模型判断不一致 → 分析分歧来源，可能成为最有价值的科学发现

---

## 八、Amendment 记录

| 日期 | Amendment | 说明 |
| ---- | --------- | ---- |
| 2026-07-24 | v1.0 创建 | Phase 0 证据冻结，建立唯一有效目标文档 |

## 九、2026-07-28 Amendment：从单一生成器转向 source-aware policy

### 状态边界

Phase 4 fixed pools 已用于设计 Union/Union_v2 和 shuffled-parent 修复，因此从本 amendment 起统一标记为 `DEVELOPMENT_SET_USED_FOR_METHOD_DESIGN`。它们可用于调试、失败分析和探索性消融，但不能作为新 gate 的 confirmatory test。

旧 G8-C tiny self-built utility gate 标记为 `DEPRECATED_INVALID_EVALUATION`；正式结果必须迁移到固定 external pools、真实 HTE、RegioSQM、第二独立 HTE 或 sealed OOD test。

### 当前 Phase 4 证据边界

- H1 learned-structured SOTA：开发结果不支持，不得写成主方法优势。
- H3 semi-hard inverted-U：开发结果不支持，不得通过事后改分箱追求转正。
- Union：旧 fixed-pool 数值领先属于 exploratory post-hoc evidence。
- randomized-label null：可用于泄漏检查，不等于 external utility 证据。
- shuffled-parent：旧 rescore 曾存在训练类不平衡和 train/score feature contract 问题；修复版已独立完成 8/8，最终主分析 alignment 128/128，formula-preserved AUPRC=0.9095、formula-changed AUPRC=0.7795；它不是随机 null，只支持 development-only compatibility-transfer 结论。
- Union_v2：主恢复目录已完成 8/8；frozen semi-hard 区间为 Morgan radius-2/2048 Tanimoto `[0.40, 0.75]`，matched_fraction=0.866–0.976，fallback_count=12–67；5/8 点估计胜出但 0/8 Holm-confirmed SOTA，不能作为确认性证据。

### v2.0 North-Star Goal

> Develop a reaction-conditioned, source-aware counterfactual learning framework that estimates candidate-level false-negative risk and adaptively selects among structured, rule-based, retrieval-based and observed-product-derived negatives under a matched training budget; validate the policy on a sealed real-HTE/OOD test and retain negative-transfer outcomes.

learned structured generator 从唯一成败点降级为 source expert；只有在新盲测中证明 gate 优于最佳单一来源和 uniform union，才恢复其 NMI 主方法地位。

### 新增 claim registry 条目

| Claim | Status | Boundary |
|---|---|---|
| C11 Phase 4 fixed-pool learned SOTA | `NO_GO_DEVELOPMENT` | 仅开发集；不得外推 |
| C12 semi-hard inverted-U utility | `UNSUPPORTED_DEVELOPMENT` | 需新盲测 controlled replication |
| C13 shuffled-real compatibility transfer | `PASS_DEVELOPMENT_ONLY` | 修复版 8/8、final alignment 128/128；formula-preserved=0.9095、changed=0.7795；仅支持 compatibility-transfer prior |
| C14 union source complementarity | `EXPLORATORY_POST_HOC` | fixed pool 已参与方法设计 |
| C15 randomized-label null control | `PASS_DEVELOPMENT_ONLY` | 仅作泄漏检查 |

### 已完成的本次交接

1. 修复版 shuffled-parent rescore、paired identity check 和 transfer diagnosis 已完成；
2. 真正按 frozen semi-hard 区间筛选的 Union_v2 已完成 8/8 并记录 matched fraction/fallback；
3. 最终 analyzer 权威输出为 `/mnt/cunyuliu_pc_cng_phase4_v2_20260728/phase4_fixed_testset_v41_rescorefix_20260728/phase4_v41_aggregation.json`。

### 后续执行顺序

1. 完成第二独立数据集/第二 scorer 的冻结 replication；当前 RegioSQM20 入口已建立，结果未出前不查看 test outcome；
2. 新建 sealed test split，冻结 gate、endpoint 和统计方案；
3. 完成 G7 专家 pilot 或 prospective experiment；
4. 仅当外部盲测与可信性闭环通过后启动 manuscript。
