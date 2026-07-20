# PC-CNG P1 Baseline Metrics（2026-07-19）

**目的**：记录 P1 阶段开始时的当前最佳主张与负结果，作为 P1 结束时对比基线。本文件由 P1-00b 任务产出，对应 `docs/P1_initial_state_manifest_20260719.json`。

**生成时间**：2026-07-19
**项目根**：`/home/cunyuliu/pc_cng_research`
**对应文档**：`docs/00_当前有效文档/顶刊论文核心思想与从0到1落地方案.md` Section 20.1

---

## 一、当前最佳主张（论文级口径）

### 1.1 Type-1 Same-Context Candidate Reranking（主任务）

| 指标 | 值 | seeds | paired CI | 备注 |
|---|---|---|---|---|
| Overall Top-1 | 97.49 ± 0.06% | 10 (20260710–20260719) | — | pairwise margin loss 默认 |
| Held-out Test Top-1 | 85.07 ± 0.94% | 10 | — | 与 RegioSQM20 SOTA 92.7% 差 7.63 pp |
| v2/unreacted Test Top-1 | 87.16 ± 1.58% | 10 | — | unreacted substrate supplement |
| Combined Morgan+GraphStats expanded curated Top-1 | 89.82 ± 0.42% | 10 | paired CI [+4.32, +6.00] pp；p < 0.0001；10/10 seeds 正向 | +5.16 pp over Morgan-only 84.66% |
| classw050_rc expanded curated Top-1 | 97.16 ± 0.30% | 10 | — | Amide/Cu 弱类修复 |

### 1.2 External Product-Selection Bridge

| 协议 | PC-CNG Top-1 | Chemformer Top-1 | groups | rows | paired CI |
|---|---|---|---|---|---|
| Validity-aware（不 apples-to-apples） | 98.50% | 0.07% | 15,973 | 174,908 | — |
| Strict shared intersection（apples-to-apples） | 71.60% | 4.94% | 1,197 | — | — |
| Repaired 25k strict（PC-CNG 作为主打分器） | 13.59% | 57.00% | — | — | **负结果**：PC-CNG < Chemformer |

### 1.3 Science Advances K_low/K_high 复现

| Split | K_low ΔTop-1 | K_high ΔTop-1 | paired CI |
|---|---|---|---|
| 10-seed benchmark | +2.83 ± 1.78 pp | +1.91 ± 1.20 pp | K_low CI [1.72, 3.94]；K_high CI [1.20, 2.67]（全正） |

### 1.4 Type-2 Low-Yield Binary Classification

| 指标 | 值 | seeds |
|---|---|---|
| Test ROC-AUC | 85.56 ± 0.21% | 10 |
| Test AUPRC | 79.93 ± 0.08% | 10 |
| Test F1 | 72.30 ± 0.08% | 10 |

### 1.5 Held-out 5k External Calibration（仅 base-only diagnostic，非 full-beam SOTA）

| 指标 | 值 | 备注 |
|---|---|---|
| base-only MLP Top-1 | 94.51% | positive diagnostic，**非 full-beam SOTA evidence** |
| full-beam frozen MLP Top-1 | **pending** | 由 P1-01 闭环 |

---

## 二、当前负结果与已知 limitation

### 2.1 架构 ablation 全部不显著（p > 0.05）

| Ablation | Test Top-1 | paired p | 判定 |
|---|---|---|---|
| hidden_dim 4096 | — | 0.629 | 拒绝 |
| dropout 0.4 | — | — | 拒绝 |
| nbits 8192 | — | 0.489 | 拒绝 |
| Cosine LR + warmup | — | 1.000 | 拒绝 |
| Pairwise weight 0.20 margin 0.005 | — | — | 拒绝 |
| DPO beta050 | 91.16% overall | — | 不超 pairwise default 97.49% |

### 2.2 External repaired 25k strict 主打分器负结果

PC-CNG 13.59% < Chemformer 57.00%。论文需明确：PC-CNG 作为外部独立打分器在 strict 全候选场景下劣于 Chemformer。当前 external bridge 主张只能基于 validity-aware 或 strict shared intersection。

### 2.3 RegioSQM20 SOTA gap

Held-out Test Top-1 85.07% < RegioSQM20 SOTA 92.7%，差 7.63 pp。论文需明确：PC-CNG 不主张端到端 site-selectivity SOTA。

### 2.4 Ni coupling 数据缺口

HITEA 0 个、USPTO/OpenMolecules 6 个 Ni coupling 反应。论文需明确：Ni coupling 是已知数据源 limitation。

### 2.5 缺失的能力（论文不可主张）

- 真实 HTE 实验验证（无）
- 专家双盲审查 synthetic negatives（无）
- 三层 false-negative 控制（仅 ensemble 一层）
- 可学习 graph edit decoder（仍是规则）
- 失败原型 prototype learning（未实现）
- Semi-hard 课程控制器（未实现）
- ORD / Reaxys / Pistachio / ChEMBL 数据接入（无）
- 跨数据集迁移评估（无）
- OOD scaffold/template 完整执行（无）
- Calibration error 报告（无）
- Retrosynthesis route ranking 任务（无）

---

## 三、Reproducibility 验证

### 3.1 Manuscript Tables 可复现性

| 项目 | 值 |
|---|---|
| 原 v3 manuscript tables 数 | 10 张主/补表（含 csv + md） |
| P1 baseline 重生成 tables 数 | 同 10 张 |
| 行数 diff | 0（全部匹配） |
| 可复现 | **True** |

### 3.2 单元测试

| 项目 | 值 |
|---|---|
| 测试文件数 | 17 |
| 测试用例数 | 25 |
| 通过率 | 100% (25/25) |
| 失败数 | 0 |

### 3.3 Active Processes（P1 启动时）

| PID | 名称 | 状态 | 备注 |
|---|---|---|---|
| 2544995 | calibrate-inference | alive | GPU 4，禁止占用 |
| 1437378 | RF-CF5 scheduler | alive | reactflow_c0_stage_20260718，禁止占用 |
| 2042374 | RF-CF5 child | alive | 同上 |

---

## 四、P1 结束时对比验收项

P1 阶段结束时，需对照以下指标判定是否达成场景 A/B/C：

### 场景 A（顶刊冲刺成功）判定
- [ ] P1-05 learned graph edit decoder 通过 Go/No-Go
- [ ] P1-01 held-out 5k frozen MLP ≥ Chemformer + 1.0 pp 且 CI 全正
- [ ] P1-08 三层 false-negative 控制完成
- [ ] P1-09 ORD 接入完成
- [ ] P1-12 manuscript v1 完成

### 场景 B（强计算化学期刊）判定
- [ ] P1-05 learned decoder 至少作为 supplementary
- [ ] P1-01 held-out 5k 结果明确（无论正负）
- [ ] P1-08 三层 pipeline 实现但专家审查仅占位
- [ ] P1-09 ORD 接入完成
- [ ] P1-12 manuscript v1 完成

### 场景 C（降级）判定
- [ ] P1-05 / P1-06 / P1-07 全部失败
- [ ] P1-01 held-out 5k 仍 pending
- [ ] P1-10 完全无计算/实验验证

---

## 五、P1 启动时 result 目录 inventory 摘要

- 总 result 子目录数：158
- 关键 artifact SHA-256 数：28
- 详细列表见 `docs/P1_initial_state_manifest_20260719.json`

---

## 六、版本

| 版本 | 日期 | 变更 |
|---|---|---|
| v1.0 | 2026-07-19 | P1-00b 初版，由 P1 启动基线锁定任务产出 |
