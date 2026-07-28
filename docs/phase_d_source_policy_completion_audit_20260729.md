# Phase D source-aware policy completion audit

> 日期：2026-07-29
> 证据级别：`DEVELOPMENT_SET_USED_FOR_METHOD_DESIGN`
> 科学判定：`ADAPTIVE_GATE_NO_GO_DEVELOPMENT`

## 1. 完成范围

Phase D 在 commit `f779a875fb3fd82dcd66bac7a6deb4be5b6d390b` 上完成了真实 GPU 开发运行：

- GPU：NVIDIA A100-PCIE-40GB；
- 数据场景：`random`、`ni_coupling`；
- downstream backbone：MLP、reaction-aware GNN；
- complete-case parents：分别为 88 和 118；
- 每个非 positive-only arm：每个 parent 恰好一个 positive 和一个 negative；
- 20 个 arm/消融，包括六个 single source、uniform union、validation-selected global mixture、learned gate、oracle、randomized-label null 和八个预先指定消融；
- 运行时间：2358.84 秒。

所有比较使用同一 fixed evaluation records、相同 parent 集和相同 1:1 candidate budget。开发 pool 已参与方法设计，因此本次结果不能称为 blind、confirmatory 或 SOTA。

## 2. Primary result

| Dataset | Backbone | Gate | Uniform | Validation-frozen best single | Gate − best single, 95% CI | Gate − uniform, 95% CI | Gate − no learned, 95% CI |
|---|---|---:|---:|---:|---|---|---|
| random | MLP | 0.7579 | 0.6947 | 0.7604 (rule) | -0.0025 [-0.0528, 0.0472] | +0.0632 [0.0246, 0.0968] | +0.0109 [-0.0105, 0.0300] |
| random | GNN | 0.8950 | 0.9009 | 0.7008 (rule) | +0.1942 [0.1593, 0.2211] | -0.0060 [-0.0418, 0.0288] | +0.0134 [-0.0019, 0.0262] |
| ni_coupling | MLP | 0.9616 | 0.9651 | 0.9197 (rule) | +0.0419 [0.0241, 0.0596] | -0.0035 [-0.0203, 0.0141] | +0.0251 [0.0143, 0.0435] |
| ni_coupling | GNN | 0.9677 | 0.9752 | 0.9383 (learned) | +0.0294 [0.0082, 0.0538] | -0.0074 [-0.0271, 0.0101] | +0.0217 [0.0056, 0.0369] |

汇总：

- gate vs validation-frozen best single：3/4 paired CI 全正；
- gate vs uniform union：1/4 paired CI 全正；
- gate vs no-learned-source：2/4 paired CI 全正，且仅发生在 Ni-coupling；
- randomized-label null：4/4 与各 source slice chance 一致；
- candidate budget：4/4 精确匹配；
- `development_statistical_exit_met=false`；
- `confirmatory_exit_met=false`。

## 3. Stop-rule decision

预注册主假设要求 gate 同时优于 validation-frozen best single 和 uniform union。该条件未满足；gate 在 3/4 单元的点估计低于 uniform union，且只有 1/4 的 paired CI 全正。

因此：

1. 停止把 adaptive source policy 作为当前 headline contribution；
2. 不在同一 fixed pool 上继续调 gate 以追求转正；
3. uniform source diversity 保留为 exploratory hypothesis；
4. learned structured source 的增量只能表述为 Ni-coupling-specific development signal；
5. 只有全新 sealed external test 才能重新判定 source diversity、adaptive policy 或 learned-source increment。

## 4. Independent verification

`independent_verification.json` 给出：

- `verified=true`；
- `failures=[]`；
- candidate cache hash、六源 schema、checkpoint hash、scored-record alignment、source-macro AUPRC、paired inference、null 判定和预算均重建通过。

核心 artifacts：

| Artifact | SHA256 |
|---|---|
| `phase_d_results.json` | `0b83c1f26c77f2d8a0f1574831994b5fa72dc782d3210a1651719cc77bc533d7` |
| `independent_verification.json` | `b9b28d6cd9e23966fef35f9398ed03b407d898e5ef27e8fe6565f0d8ed6d835c` |
| `run_manifest.json` | `a7f11296c1c59d1f433ba6642ba2725dc217477dde273c309d9d5ba1f2ae1cdf` |

权威结果目录：

```text
/mnt/cunyuliu_pc_cng_phase_d_dev_f779a87_20260729
```

## 5. Phase D final status

```text
engineering_and_reconstruction = PASS
adaptive_gate_primary_hypothesis = NO_GO_DEVELOPMENT
learned_source_increment = PARTIAL_NI_ONLY_DEVELOPMENT
confirmatory_claim = NOT_TESTED
```
