# PC-CNG Phase 4 最终进度

**最后更新**：2026-07-29
**权威交接**：`docs/phase4_v41_handover.md`
**数据用途**：development-only，已用于方法设计

## 状态看板

| 工作项 | 状态 | 结论 |
|---|---|---|
| fixed difficulty 冻结 | 完成 | `difficulty_v1_frozen_20260726`，不再调阈值 |
| 8 场景 fixed-pool GNN | 完成 | H1 learned SOTA 仅 1/8 |
| 第二 scorer EnhancedMLP | 完成（exploratory） | H1 1/8，H3 2/8 |
| shuffled_parent GPU rescore | 完成 | 8/8，同步 authoritative CSV |
| randomized-label null | 完成 | 接近 slice base rate，无明显评估泄漏 |
| Union_v1 | 完成（post-hoc） | 只能用于开发 |
| Union_v2 | 完成（post-hoc） | 5/8 数值领先，0/8 场景 SOTA；verifier 全通过 |
| continuous spline + CI | 完成 | 15/18 可估计；无 replicated inverted-U feature |
| matching audit | 完成 | family/source/edit-count/margin 未严格匹配 |
| G8-B full | 完成 | NO_GO，0/7 positive-CI directions |
| Phase 4 mechanism Exit Criterion | **未满足** | `EXPLORATORY_ONLY_EXIT_NOT_MET` |

## 关键结论

- H1 learned structured 并非普遍最优。
- H3 semi-hard inverted-U 在 GNN 与 EnhancedMLP 上都仅 `2/8` 场景成立。
- shuffled_parent 不是 chance null；它表现为 broad compatibility / distribution prior。
- randomized-label arm 才是 literal null，并通过 base-rate leakage audit。
- continuous spline 中只有 2 个单独单元出现严格 interior peak，跨数据集、跨 scorer 复现数为 0。
- G8-B 全量迁移没有任何方向 CI 全正，负迁移必须保留。
- Phase 4 已完成“回答问题”，答案是当前证据不支持因果机制 GO。

## 工程验收

- [x] 旧 Union_v1 汇总保留；
- [x] 无效旧 checkpoint Union_v2 运行留痕并排除；
- [x] Union runner learned-source fail closed；
- [x] Phase C v2 checkpoint 用于 Union_v2；
- [x] 连续机制脚本与聚焦测试；
- [x] Union_v2 independent verifier；
- [x] Phase 4 artifacts、claims 和文档同步；
- [ ] sealed external confirmatory test（属于 Phase E，不属于当前 development Phase 4）。

## 下一步

进入 `pccng3.md` Phase D：在严格 development/validation 边界内实现训练期 source-aware gate，比较最佳 single source、uniform union、validation-selected global mixture、learned gate 和 oracle。当前 fixed pools 不得再次用于确认性 SOTA。
