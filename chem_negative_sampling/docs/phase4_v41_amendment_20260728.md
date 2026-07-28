# Phase 4 v4.1 交接修订记录

**修订日期**：2026-07-28
**适用范围**：Phase 4 v4.1 修复版 rescore、Union v2、统一 analyzer、第二 scorer 和外部数据入口

## 最终状态

- 修复版 shuffled-parent rescore：8/8 完成；最终主分析 alignment 128/128。
- Union v2：GPU 7 主恢复目录 8/8 完成；frozen semi-hard `[0.40, 0.75]`，matched_fraction=0.866–0.976，fallback_count=12–67。
- 权威分析：`phase4_v41_aggregation.json`；恢复目录中复制而来的旧 `verdict.json`、`comparison_table_semihard.json` 和 `holm_correction.json` 不参与最终判定。
- H1 learned：1/8 SOTA；H1u_v2：0/8 Holm-confirmed SOTA；H2 不是 null；H3=2/8，机制不支持。
- EnhancedMLP 第二 scorer：GPU 6 完成当前 fixed-pool 8/8 exploratory replication；H1=1/8、H2 median=0.8149、H3=2/8；没有 randomized-label arm。
- RegioSQM20 外部复制：GNN 与 EnhancedMLP 均在 GPU 6 完成 1/1 scenario；GNN learned/rule/random/shuffled/diff-semi=`0.9076/0.9182/0.8965/0.8919/0.9510`，EnhancedMLP=`0.9092/0.9149/0.9156/0.9295/0.9510`；两者 H1=0/1、H2 非 null、H3=1/1。结果只作 independent replication/exploratory，不能支持 learned SOTA 或一般 inverted-U。

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

外部复制 canonical aggregation：

- GNN：`/mnt/cunyuliu_pc_cng_phase4_regiosqm20_20260728/gnn_retry_20260728/phase4_v41_aggregation.json`，SHA-256=`a64bdcf41873ddcb205f1fd788e1e85fdb736b90ad3bd524d10272e69e3990c6`
- EnhancedMLP：`/mnt/cunyuliu_pc_cng_phase4_regiosqm20_20260728/mlp_retry_20260728/phase4_v41_aggregation.json`，SHA-256=`667d8dd250a11fc26247ae055148ccefb0f0970c4904d11cbdf1a5be2acdd9a7`
- 两个运行均使用既有 RegioSQM20 split、seed `20260726`、`n_bootstrap=1000`、`max_train=500`、`max_test=200`，没有通过 test 结果改 protocol。
