# PC-CNG: PhysChem-Constrained Counterfactual Negative Generator

**项目状态**：Phase D source-policy development 已完成并独立重建，但 adaptive-gate 主假设为开发集 NO-GO；Phase E sealed-test 工程与 Phase F 三专家 pilot 材料已就绪，当前等待独立数据托管和真实专家回收（2026-07-29）
**唯一有效目标文档**：[docs/00_当前有效文档/NMI_FINAL_GOAL.md](docs/00_当前有效文档/NMI_FINAL_GOAL.md)
**Claim 注册表**：[docs/claim_registry.csv](docs/claim_registry.csv)
**Phase A 状态冻结**：v2.0；H1/H2/H3 历史见 [docs/phase4_hypothesis_history_20260728.md](docs/phase4_hypothesis_history_20260728.md)

## 当前阶段

PC-CNG 已完成研究基础设施、Phase 4 机制审计、Phase C learned source-expert 重建和 Phase D matched-budget source-policy 开发实验。当前 Phase 4 fixed pools 已用于方法设计，统一视为 `DEVELOPMENT_SET_USED_FOR_METHOD_DESIGN`；Union、Union_v2、source-aware gate 及其组合臂均不能作为 confirmatory evidence，旧 G8-C tiny utility gate 为 `DEPRECATED_INVALID_EVALUATION`。

Phase D 在 GPU 上完成两个数据集、两个 backbone、20 个预先指定 arm/消融，并由独立 verifier 完整重建。gate 对 validation-frozen best single 的 paired CI 在 3/4 单元为正，但对 uniform union 仅 1/4 为正；去除 learned source 的增量仅在 Ni-coupling 两个 backbone 上成立。因此 adaptive gate 不再作为当前 headline contribution，source diversity/heterogeneity 只保留为待新盲测验证的开发假设。

Phase E 已建立 fail-closed sealed-test 与独立 label-custody contract，并只登记外部数据元数据，未下载或查看 test labels。Phase F 已生成 80 条、8 个 strata、3 名专家的双盲 pilot 材料；真实专家评分尚未返回。距离 Nature Machine Intelligence 仍缺少一次独立托管的外部盲测和一条真实专家或前瞻实验验证链。

当前所有 Gate 已按 NMI 审计标准重新判定（见 NMI_FINAL_GOAL.md §三）：

| Gate | 审计后状态 | 说明 |
| ---- | --------- | ---- |
| G3   | REDO_PROMISING | v2 manifest 修复后有正向信号，需重建无混杂实验 |
| G4   | SUPPORTIVE_GO | generator×scorer 交互存在 |
| G5   | SAFETY_PARTIAL_GO | 校准改善，仅作风险控制证据 |
| G6   | FORMAL_NO_GO_SINGLE_SOURCE | v3 formal benchmark 已完成并独立重建；PC-CNG 对 random/template 的 superiority CI 均跨 0，且当前 HTE 仅一个 publication source |
| G7   | DEFERRED | 待真实专家或实验数据 |
| G8-A | EXPLORATORY_ONLY_EXIT_NOT_MET | continuous spline 有 reaction-group bootstrap CI，但 inverted-U 未跨 dataset/scorer 复现，且匹配审计不完整 |
| G8-B | NO_GO_NEGATIVE_TRANSFER | full run 7 directions × 6 methods × 10 seeds，0/7 方向存在 CI 全正的方法 |
| G8-C | FORMAL_SOURCE_EXPERT_PARTIAL_EXPERT_LABELS_PENDING | v2 unseen holdout 的 edit/validity/coverage/FNR/reward-safety 核心阈值全部通过；专家标签为 0，且未比较其他 source，不能称完整 GO 或 SOTA |

Phase 4 v4.1 的 fixed-pool、Union/Union_v2 和 H3 结论均为开发/探索性证据。权威交接见 `docs/phase4_v41_handover.md`：GNN 与 EnhancedMLP 的 H3 都仅 2/8；continuous spline 的 15/18 单元可估计，但没有 feature 跨 dataset/scorer 复现严格 inverted-U；family/source/edit-count/scorer-margin 未完成严格匹配。阶段结论是 **development analysis complete / causal mechanism Exit Criterion not met**。

Phase B 的权威边界见 `docs/phase_b_g6_completion_audit_20260729.md`：五任务共享完整反应上下文编码器、预算匹配、预注册比较和独立重建均已完成；正式结论是 **benchmark engineering PASS / PC-CNG superiority NO-GO**，不得将非劣效或点估计优势表述为外部效用成功。

Phase C 的权威边界见 `docs/phase_c_g8c_completion_audit_20260729.md`：正式模式已 fail-closed，四阶段使用真实 edit/rule/same-context/preference 监督，Stage 4 reference 在 Stage 3 后冻结；v1 的动作头覆盖失败保留为 NO_GO，v2 在未单独评估的 group-hash holdout 上达到 locus 0.583、type 1.000、validity 1.000、coverage 0.984、FNR ECE 0.0668。该证据只支持“可信 source expert 的内部 validation”，不支持 learned source 优于 rule/random/union；真实专家标签仍为 0。

Phase D 的权威边界见 `docs/phase_d_source_policy_completion_audit_20260729.md`：工程验收通过，adaptive-policy development exit 未通过，不能继续声称 gate 优于 uniform mixture。Phase E/F/G 的交接与外部阻塞见 `docs/phase_efg_external_evidence_handover_20260729.md`。

## 项目简介

PC-CNG 是面向化学反应预测的物理化学约束反事实负反应生成器。核心思想是通过少量真实负样本校准失败方向，生成 boundary negatives 用于下游反应预测/重排序模型训练。

**North-Star Goal v2.1**：严格检验结构化、规则、检索和 observed-product-derived 负样本是否提供可迁移的互补监督；在固定预算、独立托管的真实 HTE/OOD 盲测中，比较最佳单源、uniform mixture 与冻结的 adaptive policy。除非新盲测同时支持效用、风险控制和 learned-source 独特增量，否则不恢复 adaptive gate 或 learned generator 的 headline claim。

## 快速开始

```bash
# 激活环境
source /home/cunyuliu/miniconda3/etc/profile.d/conda.sh
conda activate pc_cng_gpu

# 运行测试
cd chem_negative_sampling
python -m pytest tests/ -q --tb=no

# 干净环境安装测试
bash scripts/smoke_test_install.sh
```

## 目录结构

```text
pc_cng_research/
├── README.md                              # 本文件（仅描述当前阶段与可验证结果）
├── LICENSE                                # MIT
├── pyproject.toml                         # 项目 packaging
├── .github/workflows/ci.yml               # CI 自动测试
├── docs/
│   ├── 00_当前有效文档/
│   │   └── NMI_FINAL_GOAL.md              # 唯一有效目标文档
│   ├── claim_registry.csv                 # 所有 headline claim 注册表
│   └── manuscript_*/                      # 历史 manuscript 草稿（带 checksum）
├── chem_negative_sampling/                # 主代码
│   ├── pc_cng/                            # 主模块
│   ├── tests/                             # 单元测试
│   └── requirements.txt
├── results/                               # 结果目录（每个含 run_provenance.json）
├── data/                                  # 数据
├── scripts/                               # 执行脚本
└── tools/                                 # 工具
```

## 可复现性

每个 `results/` 子目录包含 `run_provenance.json`，记录：
- immutable run ID（内容哈希）
- git commit SHA
- 环境锁哈希
- 输入哈希
- 精确命令
- 随机种子
- 输出 schema
- 冻结分析 spec 引用

Gate 审计状态以各目录下 `nmi_audit_status.json` 为准（覆盖原始 `go_no_go.json`）。

## 历史文档说明

`docs/00_当前有效文档/` 下其他历史规划文档（`顶刊论文核心思想与从0到1落地方案.md`、`P4_GOAL_20260721.md` 等）仅作历史参考，不再驱动当前决策。当前决策以 `NMI_FINAL_GOAL.md` 为唯一权威。
