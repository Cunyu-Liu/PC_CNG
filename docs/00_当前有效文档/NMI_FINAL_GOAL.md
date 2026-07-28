# PC-CNG → Nature Machine Intelligence：唯一有效目标文档

> **文档版本**：v2.0（Phase A 状态重冻结）
> **创建日期**：2026-07-24
> **审计依据**：`pccng 的分阶段提示词 2.md` §一/§二/§三
> **核心原则**：不为了让 Gate 变绿而调指标；只接受能够经受独立审稿、复现和外部验证的证据。
> **文档地位**：本文件是 PC-CNG 项目当前唯一有效的目标与状态文档。v1.0 的原始目标和 H1/H2/H3 定义保留为历史基线；当前状态以本文 v2.0 状态冻结段、`docs/phase4_hypothesis_history_20260728.md` 和 `docs/claim_registry.csv` 为准。其他历史规划文档仅作历史参考，不再驱动当前决策。

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
| G6   | WEAK_GO              | **FORMAL_NO_GO_SINGLE_SOURCE** | v3 已完成任务/统计重建与独立重建；PC-CNG 对 random/template 的 superiority CI 均跨 0，且当前仅一个 HTE publication source |
| G7   | DEFERRED             | **DEFERRED**            | 必须获得真实专家或实验数据 |
| G8-A | GO                   | **EXPLORATORY_MECHANISM** | 当前曲线不足以构成机制证据，仅作探索性分析 |
| G8-B | 运行中                | **保持独立判定**         | 不应预设必须 GO；负迁移本身可能是重要科学结论 |
| G8-C | NO_GO                | **FORMAL_SOURCE_EXPERT_PARTIAL_EXPERT_LABELS_PENDING** | Phase C v2 已完成真实监督与内部 holdout 核心验证；专家标签仍为 0，且尚无 source-superiority 证据 |

### 状态漂移修复记录

1. **README 状态漂移**：README.md 原标记"P1 阶段（2026-07-19 启动）"，与仓库实际推进至 P4/G8 不一致。Phase 0 已更新 README 仅描述当前阶段与可验证结果。
2. **G3 artifact 漂移**：`results/p4_augmentation/go_no_go.json` 原为 v1 manifest 的 NO_GO（A6 与 A2 bit-identical），最新提交已描述 v2 为 WEAK_GO。Phase 0 已在 `results/p4_augmentation/nmi_audit_status.json` 记录审计状态，原始 v1 verdict 保留为历史记录。
3. **G6 verdict 失效**：`results/p4_hte_external_validation/go_no_go.json` 原 WEAK_GO 因任务定义、实验混杂、delta CI 计算问题而失效。Phase 0 已在 `results/p4_hte_external_validation/nmi_audit_status.json` 标记为 INVALID_PENDING_REANALYSIS。
4. **G8-A verdict 降级**：`results/p4_mechanism_curve/go_no_go.json` 原 GO 降级为 EXPLORATORY_MECHANISM。Phase 0 已在 `results/p4_mechanism_curve/nmi_audit_status.json` 记录。
5. **G8-C 历史 verdict 重分类**：`results/p4_learned_proposal_full/go_no_go.json` 原 NO_GO 在 Phase A 重分类为 PROTOTYPE_NO_GO；该历史状态现已由本文件末尾的 Phase C completion amendment 取代。Phase 0 审计仍保留在 `results/p4_learned_proposal_full/nmi_audit_status.json`。

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
- shuffled-parent：旧 rescore 曾存在训练类不平衡和 train/score feature contract 问题；修复版已独立完成 8/8 并同步回 authoritative fixed-pool records，formula-preserved AUPRC=0.9095、formula-changed AUPRC=0.7795；它不是随机 null，只支持 development-only compatibility-transfer 结论。
- Union_v2：使用 Phase C v2 checkpoint 完成 8/8；frozen semi-hard 区间为 Morgan radius-2/2048 Tanimoto `[0.40, 0.75]`，matched_fraction=0.836–0.958，fallback_count=21–82；5/8 点估计胜出但 0/8 Holm-confirmed SOTA，不能作为确认性证据。

### v2.0 North-Star Goal

> Develop a reaction-conditioned, source-aware counterfactual learning framework that estimates candidate-level false-negative risk and adaptively selects among structured, rule-based, retrieval-based and observed-product-derived negatives under a matched training budget; validate the policy on a sealed real-HTE/OOD test and retain negative-transfer outcomes.

learned structured generator 从唯一成败点降级为 source expert；只有在新盲测中证明 gate 优于最佳单一来源和 uniform union，才恢复其 NMI 主方法地位。

### 新增 claim registry 条目

| Claim | Status | Boundary |
|---|---|---|
| C11 Phase 4 fixed-pool learned SOTA | `NO_GO_DEVELOPMENT` | 仅开发集；不得外推 |
| C12 semi-hard inverted-U utility | `UNSUPPORTED_DEVELOPMENT` | 需新盲测 controlled replication |
| C13 shuffled-real compatibility transfer | `PASS_DEVELOPMENT_ONLY` | 修复版 8/8；formula-preserved=0.9095、changed=0.7795；仅支持 compatibility-transfer prior |
| C14 union source complementarity | `EXPLORATORY_POST_HOC` | fixed pool 已参与方法设计 |
| C15 randomized-label null control | `PASS_DEVELOPMENT_ONLY` | 仅作泄漏检查 |

### 已完成的本次交接

1. 修复版 shuffled-parent rescore、paired identity check 和 transfer diagnosis 已完成；
2. 真正按 frozen semi-hard 区间筛选的 Union_v2 已完成 8/8 并记录 matched fraction/fallback；
3. 最终 analyzer 权威输出为 `chem_negative_sampling/results/phase4_fixed_testset_v41/phase4_v41_aggregation.json`。

### 后续执行顺序

1. 完成第二独立数据集/第二 scorer 的冻结 replication；当前 RegioSQM20 入口已建立，结果未出前不查看 test outcome；
2. 新建 sealed test split，冻结 gate、endpoint 和统计方案；
3. 完成 G7 专家 pilot 或 prospective experiment；
4. 仅当外部盲测与可信性闭环通过后启动 manuscript。

---

## 十、Phase A：状态重冻结 v2.0（2026-07-28）

### 当前唯一有效状态

- 当前阶段：`Phase 4 evidence-driven method redesign`。
- Phase 4 fixed pools、Union、Union_v2、shuffled-parent 修复和第二 scorer 均已看过或参与方法设计，统一标记为 `DEVELOPMENT_SET_USED_FOR_METHOD_DESIGN`。
- 旧 G8-C tiny self-built utility gate 统一标记为 `DEPRECATED_INVALID_EVALUATION`，不再作为 learned generator 的科学效用证据。
- 任何通过查看 fixed pool 后新增的 Union、Union_v2、source-gate 或组合 arm 均为 `EXPLORATORY_POST_HOC`，不能称为 sealed confirmatory SOTA。
- `claim_registry.csv` 是 headline claim 的唯一索引；每个 claim 必须同时给出 endpoint、split、代码入口、artifact、统计方法和 limitation。

### H1/H2/H3 不覆盖原则

原始 H1/H2/H3 定义、后续 amendment、判定结果和证据污染边界保存在 `docs/phase4_hypothesis_history_20260728.md`。本 v2.0 只记录当前解释，不删除或改写原假设：

| 假设 | 当前状态 | 当前可支持的最强表述 |
|---|---|---|
| H1 learned structured SOTA | `NO_GO_DEVELOPMENT` | learned 在 fixed development pool 上不支持普遍优于主要单一来源 |
| H2 shuffled-parent hard/null control | `NOT_A_NULL_CONTROL` | repaired shuffled-parent 保留 compatibility-transfer prior；randomized-label 才是 literal null |
| H3 semi-hard inverted-U | `UNSUPPORTED_DEVELOPMENT` | fixed pool 仅 2/8 场景满足；RegioSQM20 单场景 replication 不足以泛化机制 |

### 外部复制边界

RegioSQM20 的 GNN 和 EnhancedMLP 各完成一个既有 split 的外部 scenario。结果只能作为独立 replication/exploratory robustness evidence：没有证明 learned SOTA，没有恢复一般 inverted-U 主张，也没有运行外部 randomized-label null arm。

### Phase A Exit Criterion

- [x] 唯一有效 NMI goal 已升级为 v2.0，并保留 v1 历史内容。
- [x] README 当前阶段为 `Phase 4 evidence-driven method redesign`。
- [x] C11–C15 已登记，并分别标注 development-only、unsupported 或 exploratory 边界。
- [x] fixed Phase 4 pool、旧 G8-C utility gate 和 post-hoc arms 已显式标注。
- [x] 原始 H1/H2/H3 与后续判定历史已保存，不覆盖原定义。
- [x] 当前文档不在已看过的 pool 上宣称新方法 confirmatory SOTA。

Phase A 完成不等于 Phase 4 scientific exit；sealed source-gate evaluation、candidate-level driver analysis、专家/前瞻验证仍是后续独立门槛。

---

## 十一、2026-07-29 Amendment：Phase B G6 formal benchmark 完成

### 完成范围

- T1 主阈值冻结为 `<10% yield`。
- T2 使用 cumulative-link ordinal head；T3 使用完整 reaction-conditioned regression；T4 使用 pairwise ranking loss；T5 显式编码 catalyst、solvent、reagent、temperature 和 time。
- T1–T5 共享冻结的 GPU Chemformer reaction encoder；product-only Morgan 不进入正式主模型。
- `pc_cng`、`random`、`template_rule`、`union` 均使用 76 个相同 parent、76 positive + 76 negative 的匹配预算。
- primary endpoint 冻结为 real-HTE condition-feasibility source-macro AUPRC；当前只有一个 evaluable publication source，因此不得声称 cross-publication replication。
- 完成 paired cluster bootstrap、seed×cluster hierarchical aggregation、paired permutation、Holm、多重比较 effect size、预注册 non-inferiority margin，以及 type-I/power operating-characteristic simulation。
- 独立脚本从 prediction artifact 完整重建 primary inference object；历史正式 JSON 中 NumPy boolean 被 `default=str` 写为字符串的问题仅在验证器中作窄范围兼容，新写入器已改为原生 JSON scalar。

### 正式主结果

| 预注册比较 | AUPRC delta | 95% paired CI | Holm p | Superiority | Non-inferiority (margin 0.02) |
|---|---:|---:|---:|---|---|
| PC-CNG − random | +0.02835 | [-0.01106, 0.08281] | 0.7695 | 否 | 是 |
| PC-CNG − template/rule | +0.02112 | [-0.01408, 0.06832] | 0.4417 | 否 | 是 |
| union − PC-CNG | -0.02399 | [-0.06768, 0.00584] | 0.8597 | 否 | 否 |

### 判定

`Phase B benchmark engineering = PASS`；`PC-CNG superiority claim = NO-GO`。

非劣效不等于优越性。当前结果不能支持 PC-CNG negatives 优于匹配随机或 template negatives，也不能支持 union 优于 PC-CNG。次要 endpoint 仅作诊断；其中 T3 Spearman 为负，进一步阻止任何广义效用表述。

### Artifact contract

- Formal result：`/mnt/cunyuliu_pc_cng_phaseb_v3_20260728/g6_v3_formal_a602a41_20260728/formal_result_v3.json`
- Prediction artifact：`/mnt/cunyuliu_pc_cng_phaseb_v3_20260728/g6_v3_formal_a602a41_20260728/predictions_t5_v3.json`
- Independent inference：`/mnt/cunyuliu_pc_cng_phaseb_v3_20260728/g6_v3_formal_a602a41_20260728/independent_primary_inference_v3_c7b0263.json`
- Reconstruction verification：`/mnt/cunyuliu_pc_cng_phaseb_v3_20260728/g6_v3_formal_a602a41_20260728/reconstruction_verification_v3.json`
- Operating characteristics：`/mnt/cunyuliu_pc_cng_phaseb_v3_20260728/g6_v3_inference_operating_characteristics_a602a41_20260728.json`
- Detailed audit：`docs/phase_b_g6_completion_audit_20260729.md`

### Phase B Exit Criterion

- [x] 五个 task definition 与正式模型输出一致。
- [x] primary comparison arms 只改变 negative source，parent/candidate/update budget 匹配。
- [x] primary inference 可由独立脚本完整重建。
- [x] baseline 与比较顺序由冻结分析计划确定，无 test-driven selection。
- [x] 科学负结果和单一 publication-source 限制已进入 README、claim registry 与唯一有效目标文档。

## Phase C completion amendment（2026-07-29）

### 已完成的正式语义重建

- `--formal-run` 强制 GPU、固定 Stage 1→2→3→4 顺序并要求 edit、rule、competing、preference 与 risk cache 全部存在。
- 正式路径不再调用 batch-halves、随机跨反应 pair 或零 reference log-prob fallback。
- Stage 1 使用真实 bond-form、bond-break、bond-order-change 的 locus/type/argument 联合监督。
- Stage 2 使用实际 rule action；`NOT_APPLICABLE` 与 `NO_EDIT` 分离。
- Stage 3 只使用同 reactants/catalysts/solvent/temperature/time 的 observed competing outcomes。
- Stage 4 在 Stage 3 后复制并哈希冻结 reference，使用完整真实 action log-prob；reference 前后哈希一致。
- risk head 使用 known-positive collision、observed competing product 和 held-out HTE outcome；专家表单目前没有任何真实完成行，因此 expert count 保持 0。
- 正式评估禁用旧 tiny self-built MLP。

### v1 负结果与 v2 修复

commit `91dfdf7` 的首个 full formal run 保留为 `FORMAL_SOURCE_EXPERT_NO_GO`：locus=0.069、type=0。根因是 real reconstruction 与 counterfactual proposal 共享互相冲突的动作头。

commit `1256081` 将 reconstruction/proposal heads 分离，在 Stage 2–4 加入固定权重 real-edit rehearsal，并让 risk 路径覆盖所有可解析产品而非只接受 formed-bond reactions。v1 source validation 被整体排除；v2 从此前未单独评估的 source-training groups 预注册新 holdout，source test 继续 sealed。

### v2 one-shot 正式结果

| Endpoint | v2 result | Frozen threshold | Pass |
|---|---:|---:|---|
| edit-locus accuracy | 0.583 (14/24) | >=0.20 | 是 |
| edit-type accuracy | 1.000 (24/24) | >=0.50 | 是 |
| valid edit rate | 1.000 (976/976) | >=0.95 | 是 |
| candidate coverage | 0.984 (126/128) | >=0.80 | 是 |
| calibrated FNR ECE | 0.0668 (n=84 eval) | <=0.15 | 是 |
| max absolute preference log-ratio | 0.140 | <=5.0 | 是 |
| action-type entropy | 1.283 | >=0.50 | 是 |
| frozen reference hash | unchanged | required | 是 |

独立 verifier 已重建全部 Gate、检查输入哈希、checkpoint action schema，并确认正式结果目录不存在旧 comparison/tiny-MLP artifacts。

### 判定与边界

`Phase C core source-expert validation = PASS`；`expert-label requirement = PENDING`；综合状态为 `FORMAL_SOURCE_EXPERT_PARTIAL_EXPERT_LABELS_PENDING`。

该结果不能支持 learned source 优于 rule、random、shuffled 或 union。只有 24 个 holdout reactions 进入 edit 指标，Stage 4 reward-hacking validation 只有 3 个可用 pair，且 expert labels=0。这些限制必须保留。

下一阶段为 Phase D：在 development-only benchmark 上接入 source-aware gate，并在新冻结的 external blind test 前完成方法选择。Phase C holdout 不得再用于 Phase D 调参或 source-superiority 主张。

## Phase 4 completion amendment（2026-07-29）

### 完成范围

- shuffled_parent 8 场景 GPU rescore 已完成并同步回 authoritative fixed-pool CSV；
- Union_v1 历史汇总已保护，Union_v2 使用 Phase C v2 source-expert checkpoint 在相同 1:1 预算下完成；
- Union runner 增加 learned checkpoint fail-closed，旧动作空间不兼容时禁止静默退化为两源 arm；
- 新增 continuous cubic-spline mechanism analysis，使用 reaction-group cluster bootstrap 95% bands；
- 完成 matching audit、source/family heterogeneity、randomized-label null、formula-preserving transfer 与 G8-B negative-transfer 审计；
- Phase 4 fixed pools 始终保持 `DEVELOPMENT_SET_USED_FOR_METHOD_DESIGN`。

### 机制结果

| 证据项 | 结果 | 当前状态 |
|---|---|---|
| GNN H3 semi-hard > easy & hard | 2/8 场景 | 不支持普遍 inverted-U |
| EnhancedMLP H3 | 2/8 场景 | 同一 development pool 的 exploratory replication |
| continuous spline | 15/18 单元可估计 | 只有 2 个单独 interior peaks |
| replicated inverted-U feature | 0 | 未跨 dataset/scorer 复现 |
| candidate-level FNR | 历史表为常数 0.5 | 不可用于 driver claim |
| G8-B full transfer | 0/7 directions with any positive CI | `NO_GO_NEGATIVE_TRANSFER` |

### Matching audit

已匹配：

- RDKit validity；
- 每正例、每来源的候选预算；
- 三个 difficulty 训练臂的总训练样本预算；
- classifier 架构与 optimizer budget。

未匹配或未测量：

- family distribution；
- source distribution；
- true edit count；
- pre-pool scorer margin；
- reaction-centre locality；
- family support density；
- usable candidate-level FNR；
- sufficient known-positive collision strata。

因此 controlled intervention 仍存在 competing explanations，不能解释为因果边界机制。

### Union_v2 边界

Union_v2 使用 Phase C v2 checkpoint，8 场景均完成 500-negative、1:1 预算训练，semi-hard matched fraction 为 0.836–0.958。它在 5/8 场景数值超过全部单源基线，但 `time` 与 `uspto_patent` 明确落后，且没有任何场景同时显著超过全部预注册比较臂；H1u_v2 为 `0/8 SOTA`。独立 verifier 确认 8 场景记录完全对齐、每场景实际选择 learned source、checkpoint SHA256 固定且没有 silent fallback。

Union_v1/Union_v2 都是在查看 fixed development pool 后设计的。无论开发集数值如何，它们都不能支持 blind confirmatory SOTA。其价值仅在于为 Phase D 的 source-aware policy 提供开发假设与工程数据契约。

### Phase 4 最终判定

`Phase 4 engineering and development analysis = COMPLETE`；`causal mechanism Exit Criterion = NOT MET`；综合状态为：

```text
EXPLORATORY_ONLY_EXIT_NOT_MET
```

当前允许的表述是：difficulty、source、family 和 scorer 之间存在显著异质性，synthetic failure knowledge 的迁移也可能为负。当前不允许的表述是：semi-hard negatives 存在已被确认的普遍因果最优区间。

下一阶段为 Phase D development：比较最佳 single source、uniform union、validation-selected global mixture、learned source gate 和 oracle。任何 gate 结构与超参数必须在新的 sealed test 之前冻结；当前 Phase 4 fixed pools 不得再用于 confirmatory headline claim。
