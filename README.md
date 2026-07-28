# PC-CNG: PhysChem-Constrained Counterfactual Negative Generator

**项目状态**：Phase 4 evidence-driven method redesign（2026-07-28）；修复版 shuffled-parent 与 Union v2 已完成 8/8，最终分析已审计；第二 scorer 仅作 development-pool exploratory replication
**唯一有效目标文档**：[docs/00_当前有效文档/NMI_FINAL_GOAL.md](docs/00_当前有效文档/NMI_FINAL_GOAL.md)
**Claim 注册表**：[docs/claim_registry.csv](docs/claim_registry.csv)
**Phase A 状态冻结**：v2.0；H1/H2/H3 历史见 [docs/phase4_hypothesis_history_20260728.md](docs/phase4_hypothesis_history_20260728.md)

## 当前阶段

PC-CNG 已完成研究基础设施和若干探索性实验，但当前 Phase 4 fixed pools 已用于方法设计，统一视为 `DEVELOPMENT_SET_USED_FOR_METHOD_DESIGN`。Union、Union_v2、source-aware gate 及其组合臂均为 exploratory/post-hoc；旧 G8-C tiny utility gate 为 `DEPRECATED_INVALID_EVALUATION`。距离 Nature Machine Intelligence 仍需一次核心方法重建 + 一次冻结后的外部盲测 + 一条可信外部验证证据链。

当前所有 Gate 已按 NMI 审计标准重新判定（见 NMI_FINAL_GOAL.md §三）：

| Gate | 审计后状态 | 说明 |
| ---- | --------- | ---- |
| G3   | REDO_PROMISING | v2 manifest 修复后有正向信号，需重建无混杂实验 |
| G4   | SUPPORTIVE_GO | generator×scorer 交互存在 |
| G5   | SAFETY_PARTIAL_GO | 校准改善，仅作风险控制证据 |
| G6   | INVALID_PENDING_REANALYSIS | 任务定义/混杂/统计问题，整体重做 |
| G7   | DEFERRED | 待真实专家或实验数据 |
| G8-A | EXPLORATORY_MECHANISM | 仅探索性分析，非机制证据 |
| G8-B | RUNNING | 跨家族迁移实验独立判定中；负迁移也必须保留 |
| G8-C | PROTOTYPE_NO_GO | 已增加 formal fail-closed 和正确 reference snapshot；科学效用仍未证明 |

Phase 4 v4.1 的旧 fixed-pool、Union 和 H3 结论均为开发/探索性证据；详见 `docs/phase4_v41_amendment_20260728.md` 和 `docs/NMI_FINAL_GOAL_v2_amendment_20260728.md`。

`pc_cng/source_aware_policy.py` 是 pccng3 Phase D 的 development-only softmax source-gate 原型；它尚未接入 sealed benchmark，也不构成 adaptive-policy 性能结论。

## 项目简介

PC-CNG 是面向化学反应预测的物理化学约束反事实负反应生成器。核心思想是通过少量真实负样本校准失败方向，生成 boundary negatives 用于下游反应预测/重排序模型训练。

**North-Star Goal v2.0**：构建以完整反应上下文为条件、感知负样本来源差异的反事实学习框架，在严格匹配训练预算下估计逐候选假阴性风险，并自适应选择结构化、规则、检索和 observed-product-derived 负样本；最终在封存的真实 HTE/OOD 测试上验证。

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
