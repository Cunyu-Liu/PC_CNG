# 项目文件结构整理 Manifest

**整理日期**：2026-07-19
**执行人**：trae（自动执行）
**整理范围**：pc_cng_research 项目根目录文件组织
**SHA-256 参考基线**：未修改任何 `results/` 子目录、`docs/00_当前有效文档/`、`docs/99_历史参考文档/`、`docs/backup_*/` 内容

## 一、整理目标

根据 `docs/00_当前有效文档/顶刊论文核心思想与从0到1落地方案.md` Section 22.3 的方案，对项目根目录文件进行整理，降低文件组织混乱度，提高可维护性。

## 二、已执行的移动操作

### 2.1 根目录 `run_*.sh` 与 `watch_*.sh` → `scripts/`

**原位置**：`/home/cunyuliu/pc_cng_research/run_*.sh`、`watch_*.sh`
**新位置**：`/home/cunyuliu/pc_cng_research/scripts/`

移动的 21 个文件：

| 原路径 | 新路径 |
|---|---|
| `run_descriptor_feature_matrix.sh` | `scripts/run_descriptor_feature_matrix.sh` |
| `run_expanded_m3_uspto_multiseed_eval.sh` | `scripts/run_expanded_m3_uspto_multiseed_eval.sh` |
| `run_full_feasibility_matrix.sh` | `scripts/run_full_feasibility_matrix.sh` |
| `run_masked_hard_decoder_and_rule_test.sh` | `scripts/run_masked_hard_decoder_and_rule_test.sh` |
| `run_paper_aligned_type1_train.sh` | `scripts/run_paper_aligned_type1_train.sh` |
| `run_real_only_extra.sh` | `scripts/run_real_only_extra.sh` |
| `run_uspto_full_train.sh` | `scripts/run_uspto_full_train.sh` |
| `run_uspto_pc_cng_generation.sh` | `scripts/run_uspto_pc_cng_generation.sh` |
| `run_v2_boundary_generation.sh` | `scripts/run_v2_boundary_generation.sh` |
| `run_v2_coslr_warm5_multiseed.sh` | `scripts/run_v2_coslr_warm5_multiseed.sh` |
| `run_v2_filtered_baseline_multiseed.sh` | `scripts/run_v2_filtered_baseline_multiseed.sh` |
| `run_v2_nbits8192_10seed.sh` | `scripts/run_v2_nbits8192_10seed.sh` |
| `run_v2_pairwise_margin_10seed_selected.sh` | `scripts/run_v2_pairwise_margin_10seed_selected.sh` |
| `run_v2_pairwise_margin_smoke.sh` | `scripts/run_v2_pairwise_margin_smoke.sh` |
| `run_v2_representation_scale_smoke.sh` | `scripts/run_v2_representation_scale_smoke.sh` |
| `run_v2_training.sh` | `scripts/run_v2_training.sh` |
| `run_v2_unreacted_expanded_m3_uspto_eval.sh` | `scripts/run_v2_unreacted_expanded_m3_uspto_eval.sh` |
| `run_v2_valtop1_ckpt_smoke.sh` | `scripts/run_v2_valtop1_ckpt_smoke.sh` |
| `run_v3_relaxed_downstream.sh` | `scripts/run_v3_relaxed_downstream.sh` |
| `run_weighted_improvement_matrix.sh` | `scripts/run_weighted_improvement_matrix.sh` |
| `watch_v2_pairwise_margin_10seed_selected.sh` | `scripts/watch_v2_pairwise_margin_10seed_selected.sh` |

**引用更新**：`scripts/watch_v2_pairwise_margin_10seed_selected.sh` 第 26 行的 `bash "$ROOT/run_v2_pairwise_margin_10seed_selected.sh"` 已更新为 `bash "$ROOT/scripts/run_v2_pairwise_margin_10seed_selected.sh"`。

**安全性验证**：所有 `run_*.sh` 均使用 `ROOT=${ROOT:-/home/cunyuliu/pc_cng_research}` 绝对路径引用其他文件，与脚本自身位置无关，移动后仍可正常执行。

### 2.2 `chem_negative_sampling/*.md` → `chem_negative_sampling/docs/`

**原位置**：`/home/cunyuliu/pc_cng_research/chem_negative_sampling/*.md`
**新位置**：`/home/cunyuliu/pc_cng_research/chem_negative_sampling/docs/`

移动的 5 个文件：

| 原路径 | 新路径 |
|---|---|
| `chem_negative_sampling/README.md` | `chem_negative_sampling/docs/README.md` |
| `chem_negative_sampling/EXPERIMENT_ANALYSIS_20260710.md` | `chem_negative_sampling/docs/EXPERIMENT_ANALYSIS_20260710.md` |
| `chem_negative_sampling/PC-CNG-v2-反应中心边界生成器方案.md` | `chem_negative_sampling/docs/PC-CNG-v2-反应中心边界生成器方案.md` |
| `chem_negative_sampling/SCIADV_NEGATIVE_DATA_REASSESSMENT_20260710.md` | `chem_negative_sampling/docs/SCIADV_NEGATIVE_DATA_REASSESSMENT_20260710.md` |
| `chem_negative_sampling/REMOTE_A100_RUNBOOK.md` | `chem_negative_sampling/docs/REMOTE_A100_RUNBOOK.md` |

**安全性验证**：grep 全项目未发现任何代码或脚本对这 5 个 .md 文件的相对路径引用（仅文档之间相互引用），移动后无破坏。

### 2.3 创建空目录 `docs/archive_20260719/`

**目的**：作为未来归档 `docs/` 根目录 38 个 .md 文件的目标目录。

**未实际移动 `docs/*.md` 的原因**：
1. `docs/PC-CNG-v3-benchmark-manifest-20260712.{md,json}` 与 `docs/PC-CNG-v3-reproducibility-manifest-20260712.{md,json}` 之间通过绝对路径相互引用，且记录了彼此的 SHA-256 哈希。
2. `results/benchmark_manifest_data_quality_20260712/benchmark_data_quality_audit.json` 通过绝对路径引用 `docs/PC-CNG-v3-benchmark-manifest-20260712.json`。
3. 8 个 `docs/backup_*/` 目录中也包含对这些文档的引用备份。
4. 移动会破坏 reproducibility manifest 中记录的绝对路径 → SHA-256 映射，影响论文复现性证据链。

**建议**：P1-13 后续执行时，若需移动 `docs/*.md` 到 `archive_20260719/`，必须同步更新以下文件中的绝对路径引用：
- `docs/PC-CNG-v3-reproducibility-manifest-20260712.{md,json}` 中的所有路径
- `results/benchmark_manifest_data_quality_20260712/benchmark_data_quality_audit.json`
- 所有 `docs/backup_*/` 内的同步副本（或选择不更新 backups，作为历史快照保留）

## 三、未执行的移动操作（保留原位）

### 3.1 根目录 3 个 `.py` 文件

**文件**：
- `evaluate_ensemble.py`
- `evaluate_stacked_ensemble.py`
- `summarize_full_feasibility.py`

**未移动原因**：以下 4 个脚本通过 `$ROOT/evaluate_stacked_ensemble.py` 引用：
- `chem_negative_sampling/scripts_run_expanded_actions_weighted_ablation.sh`
- `chem_negative_sampling/scripts_run_expanded_hard_negative_actions_pipeline.sh`
- `chem_negative_sampling/scripts_run_masked_hard_decoder_pipeline.sh`
- `chem_negative_sampling/scripts_run_edit_decoder_v3_pipeline.sh`

**建议**：P1-13 后续执行时，若要移动这 3 个 .py 到 `chem_negative_sampling/pc_cng/`，必须同步更新上述 4 个脚本中的 `$ROOT/evaluate_stacked_ensemble.py` → `$ROOT/chem_negative_sampling/pc_cng/evaluate_stacked_ensemble.py`，并确认无其他相对导入依赖。

### 3.2 `docs/` 根目录 38 个 `.md` / `.json` 文件

见 Section 2.3 说明，因 SHA-256 manifest 依赖未移动。

### 3.3 `chem_negative_sampling/phase1_bootstrap/` 至 `phase4_expansion/`

**未移动原因**：这些目录是早期原型，README 已在 `chem_negative_sampling/README.md`（已移至 `chem_negative_sampling/docs/README.md`）中说明。P1-13 后续可在每个 phase 目录的 README 顶部追加 `DEPRECATED` 标注。

### 3.4 `docs/backup_*/` 8 个备份目录

**未移动原因**：这些是 2026-07-12 的历史快照，作为可复现性证据保留。P1-13 后续可考虑统一移至 `docs/backups/archive_20260712/`，但不影响当前使用。

## 四、整理后的目录结构

```text
pc_cng_research/
├── README.md                              # 项目总览（待 P1-13 创建）
├── chem_negative_sampling/
│   ├── pc_cng/                           # 主代码（60+ Python 模块）
│   ├── tests/                            # 测试（17 个测试文件，25 个测试用例全通过）
│   ├── evaluation/
│   ├── utils/
│   ├── docs/                             # 【新】5 个 .md 文档
│   │   ├── README.md
│   │   ├── EXPERIMENT_ANALYSIS_20260710.md
│   │   ├── PC-CNG-v2-反应中心边界生成器方案.md
│   │   ├── SCIADV_NEGATIVE_DATA_REASSESSMENT_20260710.md
│   │   └── REMOTE_A100_RUNBOOK.md
│   ├── phase1_bootstrap/                 # 早期原型（建议 P1-13 标注 deprecated）
│   ├── phase2_pretrain/
│   ├── phase3_refinement/
│   ├── phase4_expansion/
│   ├── pc_cng_tmp_sync/                  # 临时同步目录（建议 P1-13 清理）
│   ├── data/                             # phase1 用的 demo 数据
│   ├── examples/
│   ├── outputs/
│   ├── results/
│   ├── requirements.txt
│   ├── filter_paper_aligned_negatives.py
│   ├── hard_negative_actions.py
│   ├── run_hard_negative_actions.py
│   ├── train_pairwise_reward_mlp.py
│   ├── run_eval_fallback.sh
│   ├── run_train_fallback.sh
│   ├── scripts_check_reaction_lm_env.sh
│   ├── scripts_download_reaction_lm_checkpoints.sh
│   ├── scripts_fetch_negative_learning_repo.sh
│   ├── scripts_prepare_public_data.sh
│   ├── scripts_run_*.sh                  # 29 个执行脚本
│   ├── scripts_setup_*.sh
│   └── scripts_test_*.sh
├── data/
│   ├── raw/
│   ├── processed/                        # 3 个数据集：hitea, regiosqm20, uspto_openmolecules
│   ├── source_data/
│   └── summaries/
├── docs/
│   ├── 00_当前有效文档/                   # 当前唯一有效文档目录（含已更新的顶刊论文核心思想文档）
│   ├── 99_历史参考文档/
│   ├── archive_20260719/                 # 【新】空目录，待 P1-13 决定是否归档根目录 .md
│   ├── backup_*/                         # 8 个历史快照
│   └── *.md / *.json                     # 38 个根目录文档（保留原位，见 Section 2.3）
├── external/                              # HiTEA, negative_learning, reaction_lm
├── models/reaction_lm/
├── tools/                                 # git-lfs, openchemlib
├── envs/reaction_lm/
├── outputs/2026-07-16/
├── results/                               # 161 个 result 子目录（禁止删除）
├── logs/
├── scripts/                               # 【新】21 个 run/watch 脚本
│   ├── run_*.sh
│   └── watch_v2_pairwise_margin_10seed_selected.sh
├── evaluate_ensemble.py                   # 保留原位（见 Section 3.1）
├── evaluate_stacked_ensemble.py           # 保留原位
└── summarize_full_feasibility.py          # 保留原位
```

## 五、验证结果

### 5.1 单元测试

```bash
cd /home/cunyuliu/pc_cng_research/chem_negative_sampling
/home/cunyuliu/miniconda3/envs/pc_cng_gpu/bin/python -m pytest tests/ -q --tb=no
```

结果：**25 passed in 6.76s**（与整理前一致，0 失败）

### 5.2 引用完整性

- `scripts/watch_v2_pairwise_margin_10seed_selected.sh` 中对 `run_v2_pairwise_margin_10seed_selected.sh` 的引用已更新为 `$ROOT/scripts/run_v2_pairwise_margin_10seed_selected.sh`，验证通过。
- `chem_negative_sampling/docs/` 下 5 个 .md 文件无任何代码或脚本的相对路径引用，移动后无破坏。
- 全项目 grep 未发现其他对已移动文件的相对路径引用。

### 5.3 历史进程未受影响

- SSH 服务器现有长进程（calibrate PID 2544995、RF-CF5 PID 1437378 / 子进程 2042374、sample2019 下载等）均未受文件整理影响。
- `results/` 子目录全部保留原位，无任何修改。

## 六、后续建议（P1-13 任务）

1. **创建项目根 README.md**：在 `/home/cunyuliu/pc_cng_research/README.md` 写入项目总览，含目录结构说明、快速开始、核心文档索引。
2. **标注 deprecated 目录**：在 `chem_negative_sampling/phase1_bootstrap/` 至 `phase4_expansion/` 每个 README 顶部加 `> **DEPRECATED**: Superseded by pc_cng/ modules.`。
3. **清理 `chem_negative_sampling/pc_cng_tmp_sync/`**：检查是否仍需保留，若否删除。
4. **移动根目录 .py 文件**：若 P1-13 决定移动 `evaluate_ensemble.py` 等 3 个 .py 到 `chem_negative_sampling/pc_cng/`，需同步更新 4 个 `scripts_run_*.sh` 中的 `$ROOT/evaluate_stacked_ensemble.py` 引用。
5. **归档 `docs/` 根目录 .md**：若决定移动到 `docs/archive_20260719/`，需同步更新 `docs/PC-CNG-v3-reproducibility-manifest-20260712.{md,json}` 与 `results/benchmark_manifest_data_quality_20260712/benchmark_data_quality_audit.json` 中的绝对路径。
6. **整理 `chem_negative_sampling/` 根目录的 .py 与 .sh**：当前仍有 `filter_paper_aligned_negatives.py`、`hard_negative_actions.py`、`run_hard_negative_actions.py`、`train_pairwise_reward_mlp.py` 等 4 个 .py 与 2 个 fallback .sh 在 `chem_negative_sampling/` 根目录，可考虑移入 `pc_cng/` 或 `pc_cng/legacy/`。

## 七、SHA-256 审计

| 文件 | SHA-256 |
|---|---|
| `docs/00_当前有效文档/顶刊论文核心思想与从0到1落地方案.md`（已更新） | `e927e90fa18c43de482b64d6bb86bf4152c9b1e3eac4d6fde22b2687a47c403f` |
| `docs/file_reorganization_manifest_20260719.md`（本文件） | 待计算 |

注：本次整理未修改任何 `results/` 子目录、`docs/00_当前有效文档/` 下既有文件（除 `顶刊论文核心思想与从0到1落地方案.md` 追加 Section 20–23）、`docs/99_历史参考文档/`、`docs/backup_*/` 内容。所有 reproducibility manifest 中记录的 SHA-256 仍然有效。
