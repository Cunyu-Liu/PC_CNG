# PC-CNG: PhysChem-Constrained Counterfactual Negative Generator

**项目状态**：P1 阶段（2026-07-19 启动）
**当前最佳主张**：详见 [docs/P1_baseline_metrics_20260719.md](docs/P1_baseline_metrics_20260719.md)
**P1 起点状态 manifest**：[docs/P1_initial_state_manifest_20260719.json](docs/P1_initial_state_manifest_20260719.json)
**核心方案文档**：[docs/00_当前有效文档/顶刊论文核心思想与从0到1落地方案.md](docs/00_当前有效文档/顶刊论文核心思想与从0到1落地方案.md)

## 项目简介

PC-CNG 是一个面向化学反应预测的**物理化学约束反事实负反应生成器**。核心思想是通过少量真实负样本校准失败方向，生成 boundary negatives（边界负样本）用于下游反应预测/重排序模型的训练。

**核心问题**：化学反应预测模型在少量正样本的低数据场景下泛化差，且缺乏失败方向的学习信号。
**核心方法**：基于反应中心边界的反事实编辑 + 学习型 reranker + 物理化学验证器。
**核心主张**：在 same-context candidate reranking 与 external validity-aware bridge 上达到 SOTA-positive，复现并扩展 Science Advances 2025 论文的 K_low/K_high 结论。

## 目录结构

```text
pc_cng_research/
├── README.md                              # 本文件
├── chem_negative_sampling/                # 主代码
│   ├── pc_cng/                           # 60+ Python 模块（主代码）
│   ├── tests/                            # 17 个测试文件，25 个测试用例
│   ├── docs/                             # chem_negative_sampling 文档
│   ├── evaluation/                       # 评估工具
│   ├── utils/                            # 工具
│   ├── phase1_bootstrap/                 # [DEPRECATED] 早期原型
│   ├── phase2_pretrain/                  # [DEPRECATED] 早期原型
│   ├── phase3_refinement/                # [DEPRECATED] 早期原型
│   ├── phase4_expansion/                 # [DEPRECATED] 早期原型
│   ├── pc_cng_tmp_sync/                  # [DEPRECATED] 2026-07-10 staging snapshot
│   ├── scripts_run_*.sh                  # 执行脚本
│   └── requirements.txt
├── data/
│   ├── raw/                              # 原始数据
│   ├── processed/                        # 处理后数据（3 个数据集）
│   │   ├── hitea_full_normalized.csv
│   │   ├── regiosqm20_normalized.csv
│   │   └── uspto_openmolecules_normalized.csv
│   ├── source_data/
│   └── summaries/
├── docs/
│   ├── 00_当前有效文档/                   # 当前唯一有效文档目录
│   ├── 99_历史参考文档/                   # 历史文档
│   ├── archive_20260719/                 # 归档目录（待 P1-13 决定）
│   ├── backup_*/                         # 8 个历史快照
│   ├── P1_initial_state_manifest_20260719.json
│   ├── P1_baseline_metrics_20260719.md
│   ├── file_reorganization_manifest_20260719.md
│   └── *.md / *.json                     # 38 个根目录文档（保留原位）
├── external/                              # 外部仓库（HiTEA, negative_learning, reaction_lm）
├── models/reaction_lm/                    # 反应语言模型
├── tools/                                 # git-lfs, openchemlib
├── envs/reaction_lm/
├── outputs/
├── results/                               # 158 个 result 子目录（禁止删除/重命名）
├── logs/
├── scripts/                               # 21 个 run/watch 脚本
├── evaluate_ensemble.py                   # 根级评估脚本（保留原位）
├── evaluate_stacked_ensemble.py
└── summarize_full_feasibility.py
```

## 快速开始

```bash
# 激活环境
source /home/cunyuliu/miniconda3/etc/profile.d/conda.sh
conda activate pc_cng_gpu

# 运行测试
cd chem_negative_sampling
python -m pytest tests/ -q --tb=no

# 主任务：Type-1 same-context candidate reranking（10-seed paired significance）
bash scripts/run_v2_pairwise_margin_10seed_selected.sh

# 生成 manuscript 表格
python -m pc_cng.build_manuscript_tables --root /home/cunyuliu/pc_cng_research \
  --output-dir results/manuscript_tables_latest
```

## 核心文档索引

| 文档 | 用途 |
|---|---|
| [docs/00_当前有效文档/顶刊论文核心思想与从0到1落地方案.md](docs/00_当前有效文档/顶刊论文核心思想与从0到1落地方案.md) | 核心方案 + P1 阶段目标 |
| [docs/P1_baseline_metrics_20260719.md](docs/P1_baseline_metrics_20260719.md) | P1 启动基线指标 |
| [docs/P1_initial_state_manifest_20260719.json](docs/P1_initial_state_manifest_20260719.json) | P1 起点状态 manifest（SHA-256 索引） |
| [docs/file_reorganization_manifest_20260719.md](docs/file_reorganization_manifest_20260719.md) | 文件整理 manifest |
| [docs/00_当前有效文档/PC-CNG-v3-项目进展汇报-20260712.md](docs/00_当前有效文档/PC-CNG-v3-项目进展汇报-20260712.md) | v3 项目进展汇报 |
| [docs/00_当前有效文档/PC-CNG-v3-SOTA-gap-analysis-updated-20260711.md](docs/00_当前有效文档/PC-CNG-v3-SOTA-gap-analysis-updated-20260711.md) | SOTA gap 分析 |
| [docs/PC-CNG-v3-reproducibility-manifest-20260712.{md,json}](docs/PC-CNG-v3-reproducibility-manifest-20260712.md) | 可复现性 manifest |

## 当前 P1 阶段任务

详见 [核心方案文档 Section 22](docs/00_当前有效文档/顶刊论文核心思想与从0到1落地方案.md)。P1 共 14 个任务（P1-00 至 P1-13），按 5 个 Session 推进：

- **Session 1**（进行中）：P1-00 基线锁定 + P1-13 文件整理
- **Session 2**：P1-05/06/07 算法三件套（learned graph edit decoder + prototype calibrator + semi-hard curriculum）
- **Session 3**：P1-01/02/03/04 评测四件套（held-out 5k + cross-dataset + calibration+OOD + retrosynthesis）
- **Session 4**：P1-08/09/10/11 真实性四件套（专家审查 + ORD + xTB/DFT + Ni coupling）
- **Session 5**：P1-12 manuscript v1 草稿

## 约束

- 禁止删除或重命名已有 `results/` 子目录
- 禁止修改 `docs/00_当前有效文档/` 下既有 .md 文件（仅追加新文档）
- 禁止占用 GPU 4（calibrate PID 2544995）
- 所有新代码必须配套单元测试
- 所有性能主张必须基于 10-seed paired significance test

## 版本

| 版本 | 日期 | 变更 |
|---|---|---|
| v1.0 | 2026-07-19 | P1-13a 创建项目根 README |
