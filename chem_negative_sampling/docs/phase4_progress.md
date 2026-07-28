# PC-CNG Phase 4 进度文档

**最后更新**: 2026-07-28（Union v2 最终验收；第二 scorer 与 RegioSQM20 外部复制完成）
**上游交接基线**: `a56feeb` (main)
**本次聚焦修复提交**: `21d13de2a41637d6cf2e1984f16b01dfbe15aaa7`（服务器本地，未 push）
**交接文档**: `docs/phase4_v41_handover.md`
**交接修订**: `docs/phase4_v41_amendment_20260728.md`

> 注意：修复版 rescore 在隔离目录运行，旧结果不覆盖；主恢复目录的 Union v2 已完成 8/8。最终 analyzer 的 alignment audit 为 128/128；旧 `verdict.json`、`comparison_table_semihard.json` 和 `holm_correction.json` 属于复制前旧分析，最终结论以 `phase4_v41_aggregation.json` 为准。

---

## 总体状态

| 阶段 | 状态 | 完成度 |
|---|---|---|
| Phase 4 v4.1 基线评估 (8臂×8场景) | ✅ 完成 | 100% |
| shuffled_parent 修复 | ✅ 远端修复版 8/8 完成 | 输出契约通过；rescore 子集 86/86，最终主分析 paired alignment 128/128 |
| Union 臂 (8场景) | ✅ 完成 | 100% |
| Union_v2 臂 | ✅ GPU 7 主恢复运行 8/8 完成 | post-hoc development-only; matched_fraction 0.866–0.976 |
| Null 控制 (5场景) | ✅ 完成 | 100% |
| 分析框架 (analyze_phase4_v41) | ✅ 完成 | 100% |
| SOTA 目标 | ❌ 未达成 | learned 1/8；Union v2 0/8 Holm-confirmed |
| 第二 scorer（EnhancedMLP） | ✅ GPU 6 exploratory 8/8 完成 | H1=1/8；H3=2/8；不作 confirmatory claim |
| RegioSQM20 外部复制 | ✅ GNN + EnhancedMLP 完成 | 各 1 个外部场景；仅作 replication/exploratory，不支持普遍机制结论 |

---

## 关键指标速查

### Source-macro AUPRC (semi_hard pool, 8 场景)

| 场景 | learned | rule | random | shuffled* | diff_semi | union | union_v2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| author_lab | 1.0000 | 0.6979 | 0.9993 | 0.9978 | 0.9064 | 0.9972 | 1.0000 |
| condition_space | 0.8965 | 0.9360 | 0.8808 | 0.8893 | 0.9190 | 0.9851 | 0.9514 |
| ni_coupling | 0.9893 | 0.9475 | 0.9622 | 0.9408 | 0.9520 | 0.9923 | 0.9936 |
| random | 0.8971 | 0.8808 | 0.8621 | 0.8695 | 0.9674 | 0.9864 | 0.9739 |
| reaction_family | 0.9096 | 0.8144 | 0.8945 | 0.9024 | 0.8861 | 0.9386 | 0.9393 |
| scaffold | 0.9142 | 0.7821 | 0.8787 | 0.8877 | 0.8595 | 0.9365 | 0.9468 |
| time | 0.8539 | 0.7926 | 0.7945 | 0.8579 | 0.8338 | 0.8925 | 0.8029 |
| uspto_patent | 0.8223 | 0.7740 | 0.7615 | 0.7937 | 0.8579 | 0.8409 | 0.8440 |

*shuffled_parent 为修复版结果；Union v2 使用同一 frozen difficulty definition。

### 假设检验汇总

| 假设 | 判定 | 说明 |
|---|---|---|
| H1: learned SOTA | ❌ FAIL | 最终 paired/Holm 仅 1/8 SOTA，仍不支持普遍优势 |
| H1u: union SOTA | ❌ FAIL | repaired comparison 0/8 SOTA；旧 6/8 数值领先仅作 post-hoc exploratory |
| H1u_v2: difficulty-matched union | ❌ FAIL | 0/8 SOTA；5/8 点估计胜出但经 Holm 后无场景达标 |
| H2: hard control | ❌ NOT A NULL CONTROL | median AUPRC=0.8885；formula-preserved=0.9095, changed=0.7795 |
| H3: inverted-U | ❌ NOT SUPPORTED | 2/8 场景同时胜过 easy/hard，仍不足以支持机制主张 |
| Null: 随机标签 | ✅ PASS | median=0.58, 无泄漏 |

---

## 阶段完成事项

### ✅ Phase 4 v4.1 基线 (commit `fb89cde`, 2026-07-26)
- 8 负样本臂 × 8 评估场景 × 3 难度池 (easy/semi_hard/hard)
- 冻结 difficulty 定义: Tanimoto similarity [0,0.4) / [0.4,0.75] / (0.75,1.0]
- source-macro AUPRC 消除池组成偏差
- Paired cluster bootstrap CI

### ✅ Union 臂 (commit `a56feeb`, 2026-07-28)
- 多源混合训练: learned + rule + shuffled_parent
- 8 场景全部完成, 6/8 数值超越所有基线
- uspto_patent 输给 diff_semihard (0.841 < 0.858)

### ✅ shuffled_parent 修复 (2026-07-28)
- 问题: 只打乱 products → AUPRC 0.80-0.99
- 修复: 同时打乱 reactants + products；rescore 训练集补齐 positive/negative 配对，并统一 train/score feature semantics
- 场景 seed 改为稳定 SHA-256 派生
- 修复版 8/8 已完成；8 个测试池均有正负类，分数有限，`label/is_positive` 一致；rescore 子集 paired alignment 86/86，最终主分析 paired alignment 128/128
- H2 原“hard control”假设不成立：formula-preserved AUPRC=0.9095，formula-changed AUPRC=0.7795，说明 shuffled-parent 学到广义 compatibility prior

### ⚠️ Union_v2 implementation amendment (2026-07-28)
- 旧 `--difficulty-match` 只重复调用缓存生成器并随机选样，没有实际按相似度筛选
- 本地代码现在复用 frozen v4.1 Tanimoto 区间，记录 `sim`、`difficulty_pool`、`difficulty_match` 和 fallback
- 主恢复目录已完成 8/8；matched_fraction=0.866–0.976，fallback_count=12–67
- 结果属于 post-hoc development-only，不能作为 sealed confirmatory test

### ✅ Null 控制 (commit `a56feeb`)
- 5 场景随机标签训练: median AUPRC=0.58
- 确认无评估泄漏

### ✅ 分析框架 (commit `a56feeb`)
- `analyze_phase4_v41.py`: H1/H1u/H1u_v2/H2/H3 + Holm 校正
- `rescore_shuffled_parent.py`: 独立重评分工具

### ✅ pccng3/G8-C 正式运行安全项
- `--formal-run` 缺少真实监督时 fail-closed，禁止回退到伪标签。
- Stage 4 reference snapshot 移到当前轮 Stage 3 之后。
- 远端 G8-C 语法检查、四阶段回归测试和缺 cache 最小验收通过。

### ✅ pccng3 Phase D development-only 原型
- 新增 `pc_cng/source_aware_policy.py`：reaction-conditioned softmax source gate，支持 source availability mask 和 one-negative selection。
- 不接入当前 fixed pool 的确认性结果；远端专用测试 3 项通过。

### ✅ Phase 4 外部数据入口
- `run_phase4_fixed_testset.py` 新增 `--external-csv --external-only`，只接受已有 frozen train/test split，不自行创建或调参切分。
- RegioSQM20 输入审计：1936/246/242 train/val/test，2424 条记录、reaction SMILES 唯一；ORD 当前仅 train，暂不作为确认性数据。
- 外部入口 loader 单测 2 项通过。

### ✅ 第二 scorer exploratory replication
- EnhancedMLP 在 GPU 6 完成 8/8 当前 fixed-pool 场景；统一 analyzer 输出为 `/mnt/cunyuliu_pc_cng_phase4_mlp_20260728/phase4_fixed_testset_v41_mlp_20260728/phase4_v41_aggregation.json`。
- H1 learned SOTA=1/8；H2 shuffled_parent median=0.8149（仍非 null）；H3=2/8（仍不支持一般 inverted-U）。
- 本次没有 randomized-label arm，因此不作 null-control 结论；完整 provenance 位于结果目录。

### ✅ RegioSQM20 外部复制（冻结 protocol）
- 数据：`regiosqm20_normalized.csv`，既有 split 1936/246/242 train/val/test；输入 SHA-256 为 `32bc43589c8b77a332dd8516c68bf4e3256417c2a369bbf2770b2cc5f011208d`。
- protocol：`/mnt/cunyuliu_pc_cng_phase4_regiosqm20_20260728/protocol_v1_20260728.json`；difficulty 与 primary metric 未因结果修改；每条 source 预算一致，`max_train=500`、`max_test=200`、`n_bootstrap=1000`、seed `20260726`。
- GNN（GPU 6，1/1 场景）：learned=0.9076、rule=0.9182、random=0.8965、shuffled=0.8919、semi-hard=0.9510；H1=0/1，H2 非 null，H3=1/1（仅 exploratory）。canonical aggregation SHA-256=`a64bdcf41873ddcb205f1fd788e1e85fdb736b90ad3bd524d10272e69e3990c6`。
- EnhancedMLP（GPU 6，1/1 场景）：learned=0.9092、rule=0.9149、random=0.9156、shuffled=0.9295、semi-hard=0.9510；H1=0/1，H2 非 null，H3=1/1（仅 exploratory）。canonical aggregation SHA-256=`667d8dd250a11fc26247ae055148ccefb0f0970c4904d11cbdf1a5be2acdd9a7`。
- 结论边界：RegioSQM20 的两个 scorer 都没有显示 learned 优于主要单一基线；H3 在单一外部场景成立，不能升级为一般 inverted-U 机制主张；本次没有 randomized-label arm。

---

## 待办事项

### 紧急
1. [x] 通过 `ssh -p 22` 确认修复版进程并保留旧 rescore 结果（不覆盖旧日志）
2. [x] 用修复后的 rescore 脚本完成并验证 AUPRC、class balance、feature contract（8/8）
3. [x] 重新运行 analyze_phase4_v41 更新所有结果，alignment audit 128/128 通过
4. [x] 运行修复后的 union_v2 全部场景 (`--difficulty-match`)

### 重要
4. [x] 分析 SOTA 差距: 不做 test-guided 调参；已完成第二 scorer 与 RegioSQM20 外部 replication，结果保留为受限证据
5. [x] uspto_patent 薄弱分析: Union v2=0.8440，仍低于 diff_semihard=0.8579
6. [x] H3 inverted-U 重新检验: 2/8 场景达标，仍不足以支持一般机制主张

### 后续
8. [x] Phase 4 提示词剩余项中的第二 scorer/第二数据集 replication；已完成，但仍需 sealed source-gate test 才能形成确认性结论
9. [ ] Phase 5 NMI 投稿 Gate 准备

---

## 代码仓库

- **GitHub**: `git@github.com:Cunyu-Liu/PC_CNG.git`
- **分支**: `main`
- **最新 commit**: `a56feeb` — feat(phase4-v41): fixed shuffled_parent control + union_v2 arm + analysis framework
- **服务器路径**: `cunyuliu@36.137.135.49:/home/cunyuliu/pc_cng_research`
