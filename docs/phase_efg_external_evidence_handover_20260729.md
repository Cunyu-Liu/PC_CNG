# Phase E/F/G external evidence handover

> 日期：2026-07-29
> 原则：准备工作不等于外部证据；独立托管标签和真实专家评分不得由项目代码代替。

## 1. Phase E：sealed external evaluation

已完成：

- fail-closed sealed-test contract；
- development artifact/hash exclusion；
- model、source schema、endpoint 与统计计划冻结接口；
- 独立 label receipt schema；
- label receipt 的路径、hash、schema、row identity 和 development-collision 验证；
- 只登记外部候选元数据，未下载、打开或使用 test labels。

候选：

- primary：JACS 2025 Pd C-N HTE Figshare 数据；
- secondary：`bh-hte-ood` 中独立 JnJ source-only rows，禁止使用混合后的统一数据；
- 其他候选必须通过 exact record/hash isolation，避免与当前 ORD development 数据重叠。

当前状态：

```text
ENGINEERING_READY_EXTERNAL_CUSTODIAN_PENDING
```

缺失的外部动作：

1. 独立数据托管人下载和隔离数据；
2. 托管人生成不含明文标签的 receipt；
3. 项目方在模型、阈值和分析冻结后仅消费验证通过的 receipt；
4. 盲测只运行一次并披露全部结果。

没有独立托管人时，不得宣称 sealed、blind 或 confirmatory evidence。

## 2. Phase F / G7：三专家 pilot

已完成：

- 20 条真实 HTE controls：10 条实测 yield ≥80% positive controls，10 条 reported zero-yield obvious-negative controls；
- 80 条 pilot，八个 strata 各 10 条；
- 3 份独立随机顺序的双盲 reviewer forms；
- source、FNR、模型分数和 observed label 全部隐藏；
- 80 个唯一 parent/control identity；
- strict agreement analyzer：weighted kappa、ordinal Krippendorff alpha、control discrimination、reviewer drift 和 source effect；
- 少于三位完整专家、缺评分、重复 item、未通过 controls 时 fail closed。

八个 strata：

1. positive control；
2. obvious-negative control；
3. random mismatch；
4. rule PC-CNG；
5. learned structured；
6. shuffled real；
7. uniform union；
8. learned source gate。

权威目录：

```text
/mnt/cunyuliu_pc_cng_g7_pilot_v2_20260729/pilot_v3
```

核心校验：

| Artifact | SHA256 |
|---|---|
| `pilot_manifest.json` | `8dae9e29e97d5e17b9a817eac8c682e55f172003bf9d3be027524d3c47b417d6` |
| `reviewer_instructions.md` | `8ffbb34baa1a44ebb1a9c56d83730ab60dcd81cf73aa4c43a6eb85114e263e31` |
| `verified_controls.csv` | `f330465cbe2e5f4617e8a55ec06a3de23773b49b13fa98655e2fbe6124d9bd87` |
| `verified_controls_manifest.json` | `f63a8097b07f092a2e5ffae44ca2af8dcc8fbf7a3453199600b779b96852b3cc` |

为保持 blinding，unblinded sampling key 和 reviewer forms 不进入公共 Git 仓库。

当前状态：

```text
PILOT_MATERIALS_READY_EXPERT_RESPONSES_PENDING
G7 = DEFERRED
```

只有至少三名独立化学专家返回真实、完整评分并通过预注册 agreement/control gate 后，才可启动 main review 或将 G7 升级。

## 3. Phase G：source complementarity mechanism

开发分析发现：

- random/MLP 的 source-selection normalized entropy 为 0.962，family NMI 为 0.322；
- random/GNN 为 0.633 和 0.212；
- Ni-coupling/MLP 为 0.560 和 0.024；
- Ni-coupling/GNN 为 0.546 和 0.034；
- random 与 Ni-coupling 的 policy Jensen-Shannon divergence：MLP 0.169，GNN 0.259；
- learned-source leave-one-out 的正 CI 仅在 Ni-coupling 两个 backbone 上成立。

这些结果支持“source selection 存在数据集/backbone 异质性”的探索性描述，但不证明 source complementarity 的因果机制。当前缺少：

- matched source intervention；
- held-out replication；
- 专家或真实实验 outcome；
- competing-explanation ablation 的跨数据集确认。

因此：

```text
status = EXPLORATORY_SOURCE_COMPLEMENTARITY_ONLY
mechanism_exit_met = false
```

`phase_g_source_complementarity.json` SHA256：

```text
16a930b84a416f99c4cd142915f3ac557dc67d6312e2968d4cc70bc5fb7c49b5
```

## 4. 下一次可授权执行

外部依赖就绪后的顺序固定为：

1. 接收并验证独立 custodian receipt；
2. 再次核验 frozen model/endpoint/statistics hash；
3. 运行一次 sealed Phase E evaluation；
4. 收回三份真实专家表单；
5. 运行 G7 agreement/control/source analysis；
6. 根据盲测与专家结果决定是否进入 prospective experiment。

当前阶段不能用内部自动化、LLM 或规则填充专家表单，也不能由项目方先查看外部 labels 再构造 receipt。
