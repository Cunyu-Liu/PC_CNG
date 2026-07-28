# Phase 4 v4.1 交接文档

**日期**: 2026-07-28
**仓库**: `git@github.com:Cunyu-Liu/PC_CNG.git` (branch: main, handover baseline: `a56feeb`)
**本次本地聚焦提交**: `21d13de2a41637d6cf2e1984f16b01dfbe15aaa7`（服务器本地，未 push）
**服务器**: `cunyuliu@36.137.135.49:/home/cunyuliu/pc_cng_research`

> **交接修订**：请同时阅读 `docs/phase4_v41_amendment_20260728.md`。本地审计发现旧 `rescore_shuffled_parent.py` 的训练集与评分特征不一致，且旧 `union_v2 --difficulty-match` 没有真正按 difficulty 筛选；修复版已同步到远端。当前修复版 rescore 已在独立目录完成 8/8，原结果目录和旧日志均保留。

> **最终运行记录**：首次 home 目录 Union v2 在 3/8 写盘时触发 `Errno 122: Disk quota exceeded`；部分输出保留为失败证据。完整运行迁移到 `/mnt/cunyuliu_pc_cng_phase4_v2_20260728`，主恢复目录 8/8 完成；结果仍属于 post-hoc development-only。

> **Provenance**：`/mnt/cunyuliu_pc_cng_phase4_v2_20260728/phase4_fixed_testset_v41_rescorefix_20260728/phase4_v41_provenance_20260728.json` 记录了 commit、seed、GPU 映射、冻结 difficulty 定义、精确命令、最终 artifact hash 和被 supersede 的旧分析文件。

---

## 1. 本阶段执行总结

### 1.1 已完成的工作

#### A. shuffled_parent 控制臂修复

**问题**: 原始 shuffled_parent 只打乱产物(products)，保留了反应物-产物兼容性，导致 AUPRC 虚高(0.80-0.99)，无法作为真正的 null 控制。

**修复**: 修改 `run_phase4_fixed_testset.py` 和 `rescore_shuffled_parent.py`，同时打乱反应物(reactants)和产物(products)，彻底破坏化学特异性信号。

**交接时状态**: 已通过 `ssh -p 22` 核验并完成修复版重算。修复版训练集补齐真实正样本与 shuffled negatives，且训练/评分均使用同一 reaction fingerprint 特征契约；8 个测试池均有正负类，`label/is_positive` 一致。最终恢复运行的主分析 alignment audit 为 128/128；结果目录为 `/mnt/cunyuliu_pc_cng_phase4_v2_20260728/phase4_fixed_testset_v41_rescorefix_20260728`。旧 `home` 目录结果和失败日志均保留，不覆盖。

**关键文件**:
- `pc_cng/rescore_shuffled_parent.py` — 独立重评分脚本
- `pc_cng/run_phase4_fixed_testset.py` — 主评估脚本(已修改 shuffled_parent 逻辑)

#### B. Union 负样本源混合臂

**设计**: 对每个训练正样本，从 learned_structured、rule_pc_cng、shuffled_parent 三种负源各生成候选，均匀采样一个作为负样本，保持与基线相同的 1:1 正负比。

**已完成场景及 source-macro AUPRC**:

| 场景 | Union | 最强基线 | 基线臂 | 显著? |
|---|---|---|---|---|
| author_lab | 0.9972 | 0.9993 | random_mismatch | Holm NS |
| condition_space | 0.9851 | 0.9360 | rule_pc_cng | Holm Sig |
| random | 0.9864 | 0.8971 | learned_structured | Holm Sig |
| reaction_family | 0.9386 | 0.9096 | learned_structured | Holm NS |
| scaffold | 0.9365 | 0.9142 | learned_structured | Holm NS |
| time | 0.8925 | 0.8539 | learned_structured | Holm NS |
| ni_coupling | 0.9923 | 0.9893 | learned_structured | Holm NS |
| uspto_patent | 0.8409 | 0.8579 | diff_semihard | Holm NS (输给diff_semihard) |

**关键发现**: Union 在 6/8 场景数值上超越所有基线，但 Holm 校正后仅 2/8 显著。uspto_patent 场景 union 输给 diff_semihard。

#### C. Union_v2 难度匹配臂

**设计**: 在 union 基础上，每源过生成 2 个候选，通过分数接近 0.5 筛选 semi-hard 负样本，随机选择加入训练。

**状态**: 修复版已在 GPU 7 主恢复目录完成 8/8。冻结定义为 Morgan radius-2/2048 Tanimoto semi-hard 区间 `[0.40, 0.75]`；matched_fraction 为 0.866–0.976，fallback_count 为 12–67。首次在 home 目录运行到 3/8 时触发 `Errno 122: Disk quota exceeded`，该失败证据被保留；完整运行迁移到 `/mnt`。结果仍不能作为 sealed confirmatory test。

**关键文件**: `pc_cng/run_phase4_union_arm.py` (`--difficulty-match` 标志)

#### D. 评估分析框架

**已完成**: `analyze_phase4_v41.py` 实现了:
- H1: learned_structured SOTA 检验(Holm 校正)
- H1u: union SOTA 检验
- H1u_v2: union_v2 SOTA 检验
- H2: shuffled_parent 硬控制检验(AUPRC ∈ [0.40, 0.70])
- H3: inverted-U 效用检验(diff_semihard > diff_easy & diff_hard)
- Paired cluster bootstrap CI + Holm 多重比较校正

最终权威分析文件为 `phase4_v41_aggregation.json`。恢复目录中随结果复制而来的旧
`verdict.json`、`comparison_table_semihard.json` 和 `holm_correction.json` 不参与最终判定。

#### E. Null 控制(随机标签)

已在 5 个场景完成(author_lab, condition_space, random, scaffold, time):
- median srcMacroAUPRC = 0.5808
- 符合预期(~slice base rate)，无评估泄漏迹象

#### F. pccng3/G8-C 正式运行安全修复

- `p4_g8c_learned_structured_proposal.py` 增加 `--formal-run` fail-closed 模式；缺少真实 edit/competing/preference supervision 时立即终止，不回退到 batch-half 伪监督。
- Stage 4 reference model 改为在当前轮 Stage 3 完成后复制，避免使用 Stage 1 前的错误 reference policy。
- 远端语法检查、既有 `test_g8c_four_stage_fix.py` 和缺 cache 最小运行时验收均通过。
- 该修复属于训练语义与运行安全，不等于 learned generator 已获得科学效用优势。

#### G. 独立数据复制入口

- `run_phase4_fixed_testset.py` 新增 `--external-csv --external-only`，只读取已有 normalized CSV 的 train/test split，不自行创建或调参 split。
- RegioSQM20 已完成只读输入审计：1936/246/242 train/val/test，2424 条记录，reaction SMILES 唯一；冻结 protocol 为 `/mnt/cunyuliu_pc_cng_phase4_regiosqm20_20260728/protocol_v1_20260728.json`。
- 外部 loader 两项单测通过；RegioSQM20 的 GNN 与 EnhancedMLP GPU 复制均已完成。
- ORD 当前只有 train split，暂不作为确认性复制数据。

#### H. 第二 scorer exploratory replication

- EnhancedMLP 已在 GPU 6 完成当前 fixed-pool 的 8/8 场景运行；统一 analyzer 权威文件为 `/mnt/cunyuliu_pc_cng_phase4_mlp_20260728/phase4_fixed_testset_v41_mlp_20260728/phase4_v41_aggregation.json`。
- 结果：H1 learned SOTA=1/8；H2 shuffled_parent median=0.8149，仍不是 null；H3=2/8，仍不支持一般 inverted-U。
- 本次没有 randomized-label arm，不能给出 null-control 结论；该结果只作为第二 scorer 的 development-only robustness check。

#### I. RegioSQM20 外部复制

- 冻结输入：`regiosqm20_normalized.csv`，train/val/test=`1936/246/242`，SHA-256=`32bc43589c8b77a332dd8516c68bf4e3256417c2a369bbf2770b2cc5f011208d`。
- 冻结 protocol：`/mnt/cunyuliu_pc_cng_phase4_regiosqm20_20260728/protocol_v1_20260728.json`；只读取既有 split，不重新切分，不用 test 结果选参数。
- GNN 外部结果（GPU 6，1 个 scenario）：learned `0.9076`，rule `0.9182`，random `0.8965`，shuffled `0.8919`，diff-semi `0.9510`；H1 `0/1`、H2 非 null、H3 `1/1` exploratory。canonical aggregation SHA-256=`a64bdcf41873ddcb205f1fd788e1e85fdb736b90ad3bd524d10272e69e3990c6`。
- EnhancedMLP 外部结果（GPU 6，1 个 scenario）：learned `0.9092`，rule `0.9149`，random `0.9156`，shuffled `0.9295`，diff-semi `0.9510`；H1 `0/1`、H2 非 null、H3 `1/1` exploratory。canonical aggregation SHA-256=`667d8dd250a11fc26247ae055148ccefb0f0970c4904d11cbdf1a5be2acdd9a7`。
- 外部复制只证明当前 frozen difficulty pool 在一个独立数据集上可复现部分 semi-hard 对比；它不证明 learned generator SOTA，也不足以恢复一般 inverted-U 机制主张。两个 scorer 均无 randomized-label arm，因此不报告外部 null-control 结论。

---

### 1.2 当前结果总览

#### 8 场景 source-macro AUPRC (semi_hard pool)

| 场景 | learned | rule | random | shuffled | diff_easy | diff_semi | diff_hard | union | union_v2 |
|---|---|---|---|---|---|---|---|---|---|
| author_lab | 1.0000 | 0.6979 | 0.9993 | 0.9978 | 0.8122 | 0.9064 | 0.7653 | 0.9972 | 1.0000 |
| condition_space | 0.8965 | 0.9360 | 0.8808 | 0.8893 | 0.9025 | 0.9190 | 0.8260 | 0.9851 | 0.9514 |
| random | 0.8971 | 0.8808 | 0.8621 | 0.8695 | 0.9194 | 0.9674 | 0.8786 | 0.9864 | 0.9739 |
| reaction_family | 0.9096 | 0.8144 | 0.8945 | 0.9024 | 0.8512 | 0.8861 | 0.8652 | 0.9386 | 0.9393 |
| scaffold | 0.9142 | 0.7821 | 0.8787 | 0.8877 | 0.8354 | 0.8595 | 0.8782 | 0.9365 | 0.9468 |
| time | 0.8539 | 0.7926 | 0.7945 | 0.8579 | 0.8288 | 0.8338 | 0.8043 | 0.8925 | 0.8029 |
| ni_coupling | 0.9893 | 0.9475 | 0.9622 | 0.9408 | 0.9558 | 0.9520 | 0.9014 | 0.9923 | 0.9936 |
| uspto_patent | 0.8223 | 0.7740 | 0.7615 | 0.7937 | 0.8131 | 0.8579 | 0.7971 | 0.8409 | 0.8440 |

列顺序为：learned、rule、random、shuffled、diff_easy、diff_semihard、diff_hard、union、union_v2。

#### 假设检验结果

| 假设 | 结果 | 说明 |
|---|---|---|
| H1 (learned SOTA) | **FAIL** | 最终 paired/Holm 仅 1/8 SOTA，仍不支持普遍优势 |
| H1u_v2 (difficulty-matched union) | **FAIL** | 0/8 SOTA；5/8 点估计胜出，但 Holm 后无场景达标 |
| H2 (hard control) | **NOT A NULL CONTROL** | median AUPRC=0.8885；formula-preserved=0.9095、formula-changed=0.7795，支持 compatibility transfer |
| H3 (inverted-U) | **NOT SUPPORTED** | repaired analysis 2/8 场景同时胜过 easy 和 hard，仍不足以支持机制主张 |
| H1u (union SOTA) | **FAIL** | repaired comparison 0/8 SOTA；旧 6/8 数值领先仅作 post-hoc exploratory |
| Null control | **PASS** | median=0.58, 无泄漏 |

---

## 2. 待完成事项

### 2.1 紧急(阻塞下一步)

1. **shuffled_parent rescore 完成**: 修复版已在隔离目录完成 8/8；class balance、feature contract 和新版输出均验收通过。
   - 结果: `results/phase4_fixed_testset_v41_rescorefix_20260728`
   - 验收: 8 个场景均有正负类、分数有限、`label/is_positive` 一致，最终 paired alignment 128/128
   - 已重新运行 `analyze_phase4_v41.py`；最终结论以 `phase4_v41_aggregation.json` 为准

2. **H2 修复后重评估**: 已完成；shuffled_parent 不是随机 null，而是 compatibility-transfer prior

### 2.2 重要(影响 SOTA 目标)

3. **Union_v2 完整运行**: 已完成；主恢复目录为 `/mnt/cunyuliu_pc_cng_phase4_v2_20260728/phase4_fixed_testset_v41_rescorefix_20260728`，输出包含 `matched_fraction` 和 `fallback_count`。

4. **SOTA 差距分析**:
   - learned 的最终 H1 判定为 1/8 场景 SOTA，不能支持普遍优势
   - Union_v2 的 H1u_v2 判定为 0/8 SOTA；5/8 为点估计胜出，但经 Holm 后没有场景确认
   - 不通过增加 seeds、调分箱或测试集导向的架构搜索制造确认性正结果；下一步转向第二独立数据集/第二 scorer 的冻结 replication

### 2.3 后续

5. **Phase 4 提示词剩余项**: 第二数据集/第二 scorer 的 replication 已完成，但每个 scorer 只有 1 个 RegioSQM20 scenario，仍不是 confirmatory evidence；H3 只在该外部场景成立，不能泛化
6. **uspto_patent 薄弱**: Union v2=0.8440，仍低于 diff_semihard=0.8579，保留为负结果
7. **代码清理**: 服务器上有多份 .bak 文件，待单独确认范围后处理；本次未删除任何用户文件

---

## 3. 关键代码文件说明

| 文件 | 用途 | 状态 |
|---|---|---|
| `pc_cng/run_phase4_fixed_testset.py` | Phase 4 主评估: 8 臂 × 8 场景 × 3 难度池 | 已修改(shuffled_parent fix) |
| `pc_cng/run_phase4_union_arm.py` | Union/Union_v2 训练+评估 | 已实现 `--difficulty-match` |
| `pc_cng/analyze_phase4_v41.py` | 汇总分析: H1/H1u/H1u_v2/H2/H3 + Holm | 已实现 |
| `pc_cng/rescore_shuffled_parent.py` | 独立重评分 shuffled_parent | 已实现，8/8 完成 |
| `pc_cng/run_phase4_null_control.py` | 随机标签 null 控制 | 已完成 5 场景 |
| `pc_cng/paired_cluster_inference.py` | Paired bootstrap CI + Holm 校正 | 稳定 |
| `pc_cng/run_phase3_external_validation.py` | 数据加载 + GNN 模型 | 稳定 |

---

## 4. 服务器运行环境

- **Conda env**: `pc_cng` (`/home/cunyuliu/miniconda3/envs/pc_cng`)
- **GPU**: NVIDIA A100-PCIE-40GB (MIG 1g.5gb 分区)
- **数据路径**:
  - Parquet: `/home/cunyuliu/pc_cng_research/data/processed/p4_hte_normalized.parquet`
  - NI CSV: `/home/cunyuliu/pc_cng_research/data/processed/ni_coupling_supplement.csv`
  - OOD splits: `/home/cunyuliu/pc_cng_research/data/ood_splits/`
  - USPTO: `/home/cunyuliu/pc_cng_research/data/processed/uspto_openmolecules_normalized.csv`
  - Checkpoint: `/home/cunyuliu/pc_cng_research/results/p4_g8c_phase2_full/model_checkpoint.pt`
- **结果目录**: `/home/cunyuliu/pc_cng_research/chem_negative_sampling/results/phase4_fixed_testset_v41/`

---

## 5. 常用命令

```bash
# 查看 rescore 进度
ssh cunyuliu@36.137.135.49 "tail -20 /home/cunyuliu/pc_cng_research/chem_negative_sampling/results/rescore_shuffled3.log"

# 重新运行分析(rescore 完成后)
ssh cunyuliu@36.137.135.49 "cd /home/cunyuliu/pc_cng_research/chem_negative_sampling && /home/cunyuliu/miniconda3/envs/pc_cng/bin/python -m pc_cng.analyze_phase4_v41 --base-results results/phase4_fixed_testset_v41"

# 运行 union 臂(新场景)
ssh cunyuliu@36.137.135.49 "cd /home/cunyuliu/pc_cng_research/chem_negative_sampling && CUDA_VISIBLE_DEVICES=7 /home/cunyuliu/miniconda3/envs/pc_cng/bin/python -m pc_cng.run_phase4_union_arm --base-results results/phase4_fixed_testset_v41 --use-gnn --scaffold --time"

# 运行 union_v2
ssh cunyuliu@36.137.135.49 "cd /home/cunyuliu/pc_cng_research/chem_negative_sampling && CUDA_VISIBLE_DEVICES=7 /home/cunyuliu/miniconda3/envs/pc_cng/bin/python -m pc_cng.run_phase4_union_arm --base-results results/phase4_fixed_testset_v41 --use-gnn --difficulty-match"
```
