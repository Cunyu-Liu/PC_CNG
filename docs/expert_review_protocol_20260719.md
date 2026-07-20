# PC-CNG 专家审查协议：双盲合成负样本真实性评估

- 协议版本：v1.0
- 日期：2026-07-19
- 对应任务：P1-08（Section 9 E3 真实性 + False-Negative 三层控制）
- 状态：**协议已制定，执行推迟至论文修订阶段（protocol proposed, execution deferred to revision）**
- 适用范围：PC-CNG v3 pipeline 产出的 reviewed synthetic negatives（hitea_full_generation + regiosqm20_full，合计 64,648 行）

---

## 1. 目标

对 PC-CNG pipeline 产出的合成负样本（synthetic reaction-boundary negatives）进行双盲专家审查，量化以下三类风险：

1. **Chemical validity risk**：合成的 candidate reaction SMILES 在化学上不合法或物理不可行。
2. **Mechanistic plausibility risk**：candidate reaction 的反应机理不合理（例如禁用官能团、违反价键规则）。
3. **False-negative risk**：candidate reaction 实际上是可发生的真实反应（即被错误标为负样本），这会污染下游 reranker 训练标签。

协议同时作为 P1-08 三层 False-Negative 控制的 Layer 3 输入（`false_negative_three_layer_control.py`）。在专家审查实际未执行的当前阶段，Layer 3 自动降级为 rule-based plausibility check（见第 7 节）。

## 2. 抽样规则

### 2.1 抽样框

输入文件（合并后的 reviewed synthetic negatives）：
- `results/hitea_full_generation/pc_cng_synthetic_negatives_reviewed.csv`（61,620 行）
- `results/regiosqm20_full/pc_cng_synthetic_negatives_reviewed.csv`（3,028 行）
- 合计：64,648 行（与 Section 22.1 描述的 "约 64,646 行" 一致，差异 2 行来自 header 对齐）

### 2.2 分层随机抽样

- **分层变量**：`failure_type`（合成负样本的失败类型，如 `retro_missing_reactant`、`retro_wrong_functional_group`、`regio_wrong_position` 等）；当 `failure_type` 缺失时退化为 `task` 列（如 `retro_precursor`、`regioselectivity`）。
- **目标样本量**：400 条（位于协议规定的 200–500 区间中点，兼顾统计功效与审查成本）。
- **配额分配**：按各层在总体中的占比分配配额（proportional allocation）；对占比 < 1% 的稀有层，强制至少抽 1 条以保证覆盖。
- **随机种子**：`seed=20260719`（与 PC-CNG v3 multiseed 协议一致），保证可复现。
- **对照样本注入（双盲关键）**：在 400 条 PC-CNG 合成负样本之外，额外注入 100 条**真实负样本对照**（从 USPTO normalized 中 yield=0 / explicit-failure 的真实反应中抽取），与合成样本混合后打乱顺序，供审查者判断 "是否为 PC-CNG 合成"。对照样本用于校准审查者的判断基线与 false-positive 估计。

### 2.3 输出

抽样结果写入 `results/expert_review_20260719/sampled_for_review.csv`，列包括：
- `sample_id`：匿名化的样本编号（如 `S0001`…`S0500`），不暴露来源。
- `reaction_smiles`：candidate reaction SMILES（待审查）。
- `parent_reaction_smiles`：parent positive reaction（仅用于上下文，审查时不提供）。
- `failure_type`、`task`：分层信息（审查时不提供）。
- `source_origin`：`pc_cng_synthetic` 或 `real_negative_control`（**审查时隐藏**，仅事后分析用）。
- `true_label`：1 = real negative control，0 = PC-CNG synthetic（**审查时隐藏**）。
- 5 项审查评分字段（见第 3 节，审查者填写）。
- `reviewer_id`、`review_timestamp`：审查元数据。

## 3. 审查表单（5 项评分 + 自由评论）

每位审查者对每条样本独立填写以下 5 项评分（1–5 分 Likert scale）+ 1 项自由评论。**所有评分均在审查者不知样本来源的情况下进行。**

| # | 评估项 | 1 分（最差） | 5 分（最好） | 说明 |
|---|--------|--------------|--------------|------|
| 1 | **Chemical validity**（化学合法性） | SMILES 不合法 / 违反价键规则 | SMILES 合法且符合化合价 | RDKit 能否解析 + 原子价是否合法 |
| 2 | **Mechanistic plausibility**（机理合理性） | 反应机理完全不可能 | 反应机理完全合理 | 考虑官能团兼容性、离去基团、电子效应 |
| 3 | **Side product likelihood**（副产物可能性） | 必然产生大量副产物 | 几乎无副产物 | 评估主产物选择性 |
| 4 | **Feasibility score**（合成可行性） | 实验条件下完全不可行 | 标准条件下即可实现 | 考虑温度、压力、催化剂可得性 |
| 5 | **Overall verdict**（综合判定） | 1 = 明确是真实可发生反应（false negative） | 5 = 明确是合成负样本（真负样本） | 综合判断该反应是否会被误标 |

**自由评论**（free-form comment）：审查者需简要说明判定理由，特别是对 overall verdict ≤ 2 的样本，必须列出 "为什么认为这是真实可发生反应" 的具体机理依据。

### 3.1 双盲流程

1. **审查者招募**：≥2 名具备有机化学博士学位（或同等经验）的审查者，签署利益冲突声明。
2. **样本去标识化**：sampled_for_review.csv 移除 `source_origin`、`true_label`、`failure_type` 列后分发；reaction_smiles 顺序随机打乱（seed=20260719）。
3. **独立审查**：每位审查者独立填写 5 项评分 + 评论，禁止讨论。
4. **第三审查者仲裁（可选）**：当 2 名审查者 overall verdict 差距 ≥ 2 分时，引入第三审查者仲裁；此时启用 Fleiss' κ（3 名审查者）。
5. **解盲**：所有审查提交后，按 sample_id 合并 `source_origin` / `true_label`，进行第 4 节的一致性分析。

## 4. Inter-annotator agreement 计算方法

### 4.1 Cohen's κ（2 名审查者）

用于量化 2 名审查者在 **overall_verdict** 上的判定一致性。步骤：

1. 将 overall_verdict（1–5 分）二值化为 `verdict_binary`：1–2 = "false negative risk"（认为是真实反应），3 = "uncertain"，4–5 = "true negative"（认为是合成负样本）。
2. 构造 2×2 混淆矩阵（审查者 A vs 审查者 B，类别 = {false_neg_risk, true_neg}；uncertain 单独报告比例，不进入 κ 主矩阵，但作为敏感性分析）。
3. 计算：
   - 观察一致率 `Po = (n_agree) / N`
   - 期望一致率 `Pe = Σ(行边缘 × 列边缘) / N²`
   - `Cohen's κ = (Po - Pe) / (1 - Pe)`
4. 解释标准（Landis-Koch 1977）：κ < 0.20 = slight，0.21–0.40 = fair，0.41–0.60 = moderate，0.61–0.80 = substantial，0.81–1.00 = almost perfect。
5. **验收阈值**：κ ≥ 0.40（moderate 以上）方可作为 Layer 3 输入；κ < 0.40 时协议判定为 "inter-annotator agreement 不足，需补充审查者培训或修订评分标准"。

实现函数：`cohens_kappa(rater_a: list[int], rater_b: list[int]) -> tuple[float, dict]`，返回 `(kappa, contingency_table_dict)`。

### 4.2 Fleiss' κ（≥3 名审查者）

当引入第三审查者仲裁时使用。步骤：

1. 对每个样本，统计 K 名审查者在每个类别上的投票数（类别同 4.1 的二值化 + uncertain）。
2. 构造 N × C 矩阵 `n_ij`（N 样本，C 类别），`Σ_j n_ij = K`。
3. 计算每类别的边缘比例 `p_j = (Σ_i n_ij) / (N·K)`。
4. 每样本的一致性 `P_i = (Σ_j n_ij² - K) / (K·(K-1))`。
5. 观察一致率 `P_bar = (Σ_i P_i) / N`；期望一致率 `P_e = Σ_j p_j²`。
6. `Fleiss' κ = (P_bar - P_e) / (1 - P_e)`。

实现函数：`fleiss_kappa(ratings: list[list[int]], n_categories: int) -> tuple[float, list[float]]`，其中 ratings 为 N×K 矩阵，返回 `(kappa, per_category_proportions)`。

### 4.3 报告内容

专家审查执行后，需在论文修订稿中报告：
- Cohen's κ（主审查者对）与 Fleiss' κ（含仲裁者）。
- 二值化混淆矩阵 + 各类别的 precision / recall。
- 对 `source_origin = real_negative_control` 的 100 条对照样本，报告审查者正确识别率（应 ≥ 70% 才说明审查者具备区分能力）。
- 各评分项（5 项）的 Cronbach's α 内部一致性。

## 5. 当前阶段降级策略

**专家审查实际未执行**（论文当前阶段无可用审查者资源）。降级策略如下：

1. Layer 3（expert review）在 `false_negative_three_layer_control.py` 中降级为 **rule-based plausibility check**，规则见 `false_negative_three_layer_control.py::rule_based_plausibility_check()`：
   - 排除候选：`valid == 0` 或 `atom_balance < 0.5` 或（`review_status == "needs_review_or_downweight"` 且 `false_negative_risk > 0.7`）。
   - 保留候选：`review_status == "keep_synthetic_negative"` 或（`false_negative_risk < 0.5` 且 `atom_balance ≥ 0.7` 且 `valid == 1`）。
   - 中间候选：进入 "uncertain" 池，不作为 high-confidence negative，但保留用于敏感性分析。
2. sampled_for_review.csv 仍产出（400 条合成 + 100 条对照 = 500 条），供论文修订阶段直接执行。
3. 论文 Limitations 段明确声明：
   > "The expert review protocol (Section 9 E3) is fully specified and the stratified sample (n=500, including 100 real-negative controls) is pre-drawn for reproducibility, but execution is deferred to the revision stage due to reviewer availability. Layer 3 of the false-negative control therefore uses a conservative rule-based plausibility check as a interim surrogate; the resulting high-confidence negative rate is a lower bound on what expert adjudication would yield."

## 6. Go/No-Go 判定

- **No-Go 触发条件**：三层控制后 high-confidence negatives 占比 < 30%（相对输入 64,648 行）。
- **No-Go 处置**：论文明确写入 "high false-negative risk is a known limitation"，并在 Discussion 段讨论对 reranker 训练标签噪声的影响估计。
- **Go 条件**：high-confidence 占比 ≥ 30% 且 Layer 1（ensemble agreement）排除率 ≤ 50%（避免 ensemble 过度激进排除）。

## 7. 协议产物清单

执行本协议将产出以下文件（当前阶段仅前两项已生成）：

| 文件 | 当前状态 | 路径 |
|------|----------|------|
| 本协议文档 | ✅ 已生成 | `docs/expert_review_protocol_20260719.md` |
| 待审查抽样样本 | ✅ 已生成（500 条，双盲去标识化） | `results/expert_review_20260719/sampled_for_review.csv` |
| 审查者原始评分表 | ⏳ 待执行（修订阶段） | `results/expert_review_20260719/reviewer_ratings_raw.csv` |
| 一致性分析报告 | ⏳ 待执行 | `results/expert_review_20260719/inter_annotator_agreement.json` |
| 解盲后对照样本识别率 | ⏳ 待执行 | `results/expert_review_20260719/control_identification_rate.json` |

## 8. 引用

- Landis, J. R., & Koch, G. G. (1977). The measurement of observer agreement for categorical data. *Biometrics*, 33(1), 159–174.
- Fleiss, J. L. (1971). Measuring nominal scale agreement among many raters. *Psychological Bulletin*, 76(5), 378–382.
- Cohen, J. (1960). A coefficient of agreement for nominal scales. *Educational and Psychological Measurement*, 20(1), 37–46.
