# Phase 4 v4.1 交接修订记录

**修订日期**：2026-07-28
**适用范围**：Phase 4 v4.1 修复版 rescore、Union v2、统一 analyzer、第二 scorer 和外部数据入口

## 最终状态

- 修复版 shuffled-parent rescore：8/8 完成；最终主分析 alignment 128/128。
- Union v2：GPU 7 主恢复目录 8/8 完成；frozen semi-hard `[0.40, 0.75]`，matched_fraction=0.866–0.976，fallback_count=12–67。
- 权威分析：`phase4_v41_aggregation.json`；恢复目录中复制而来的旧 `verdict.json`、`comparison_table_semihard.json` 和 `holm_correction.json` 不参与最终判定。
- H1 learned：1/8 SOTA；H1u_v2：0/8 Holm-confirmed SOTA；H2 不是 null；H3=2/8，机制不支持。
- EnhancedMLP 第二 scorer：GPU 6 完成当前 fixed-pool 8/8 exploratory replication；H1=1/8、H2 median=0.8149、H3=2/8；没有 randomized-label arm。

## 证据边界

当前 fixed pools 已用于 Union、Union v2、shuffled-parent 修复和第二 scorer 设计，统一为 `DEVELOPMENT_SET_USED_FOR_METHOD_DESIGN`。所有上述结果只能用于探索、失败分析和 robustness check，不得写成 sealed confirmatory SOTA 或机制证据。

## 新增外部入口

`pc_cng/run_phase4_fixed_testset.py` 新增 `--external-csv --external-only`。该入口只接受已有 normalized CSV 的 frozen train/test split，不自行创建切分、不读取 test outcome 选择参数。RegioSQM20 protocol 已冻结在：

```text
/mnt/cunyuliu_pc_cng_phase4_regiosqm20_20260728/protocol_v1_20260728.json
```

ORD 当前没有 test split，不进入确认性复制。

## 保留的失败证据

首次 home 目录 Union v2 运行在 3/8 写盘时触发 `Errno 122: Disk quota exceeded`；失败日志和部分输出保留。RegioSQM20 首次启动因相对输入路径错误在 5 秒内退出；空输出和日志保留，修正版使用绝对路径写入新的 `gnn_retry_20260728` 目录。
