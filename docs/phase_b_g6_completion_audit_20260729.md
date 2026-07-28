# Phase B G6 formal benchmark completion audit

> Date: 2026-07-29
> Scientific boundary: benchmark engineering verified; PC-CNG superiority not supported
> Frozen schema: `g6_v3_formal_20260728`

## 1. Outcome

Phase B completed the task-definition, reaction-context, matched-budget and paired-inference rebuild requested for G6. The primary inference was reconstructed from the frozen prediction artifact by an independent entrypoint and verified as a complete object match after narrowly normalizing the documented legacy `"True"`/`"False"` serialization from commit `a602a41`.

The 2026-07-29 completion re-audit found and repaired one reproducibility gap:
the formal CLI imported the shared encoder through the repository-parent
namespace, so the documented module command failed from
`chem_negative_sampling/`. The canonical import is now the packaged
`models.pretrained_backbone`, `models*` is included in the wheel, and a
subprocess regression test runs without adding the repository parent to
`PYTHONPATH`. This packaging repair does not change the frozen model,
predictions or statistical result.

This is not an efficacy GO. All three preregistered superiority intervals cross zero.

## 2. Requirement audit

| Requirement | Evidence | Status |
|---|---|---|
| T1 frozen low-yield endpoint | `<10% yield` in frozen plan | PASS |
| T2 true ordinal model | cumulative-link thresholds and ordinal loss | PASS |
| T3 reaction-conditioned regression | shared full reaction/condition representation with regression head | PASS |
| T4 true ranking model | within-plate pair construction and pairwise ranking loss | PASS |
| T5 explicit conditions | catalyst, solvent, reagent, temperature and time fields encoded | PASS WITH DATA LIMITATION |
| Shared pretrained encoder | frozen Chemformer checkpoint shared across T1–T5 on CUDA | PASS |
| Product-only Morgan only weak baseline | excluded from v3 formal main model | PASS |
| Matched source arms | 76 common parents; each source arm 76 positive + 76 negative | PASS |
| One preregistered primary endpoint | real-HTE condition-feasibility source-macro AUPRC | PASS WITH SINGLE-SOURCE LIMITATION |
| Paired cluster bootstrap | same clusters resampled for challenger and baseline | PASS |
| Hierarchical seed×cluster inference | seed and cluster uncertainty aggregated in frozen procedure | PASS |
| Paired permutation + Holm | frozen three-comparison family | PASS |
| Effect and non-inferiority | raw delta, paired CI and margin 0.02 recorded | PASS |
| Type-I and power simulations | familywise type-I = 0.0; power at delta 0.08 = 1.0 over 80 simulations | PASS AS DESIGN CHECK |
| Independent reconstruction | complete primary inference object verified | PASS |
| No test-driven baseline selection | comparison order frozen before formal evaluation | PASS |
| Documented CLI and wheel import | `pc_cng` and shared `models` import from source checkout and wheel | PASS AFTER RE-AUDIT FIX |

`reagent_fraction=0.0` in the available HTE records, despite the model having an explicit reagent channel. This is a data-availability limitation, not evidence that reagent effects were validated. Catalyst availability is 0.5982; solvent 0.9917; temperature and reaction time 1.0.

The legacy v2 sanity suite also contained an invalid AUPRC power generator: it shifted every score by the same constant while the baseline ranking was already perfectly separated. A constant shift cannot change AUPRC, so the old coverage/power assertions necessarily reported zero. The repaired tests use a location metric for CI coverage and an overlapping, label-directional score shift for AUPRC power; the effect grid is deliberately unsaturated.

## 3. Formal dataset and run contract

- Input records: 39,546
- Included after reaction-context validation: 39,450
- Excluded for missing reaction context: 96
- Validation records: 4,291
- Test records: 5,898
- Test clusters: 48
- Endpoint-evaluable clusters: 25
- Evaluable source-publication slices: 1
- Seeds: 5
- Formal device: GPU (`CUDA_VISIBLE_DEVICES=2`)
- Shared encoder: frozen Chemformer checkpoint, `product_only_baseline=false`

Because only one publication source is evaluable, the current endpoint is numerically source-macro but does not establish cross-publication replication. A second real HTE source remains mandatory.

## 4. Preregistered primary inference

| Comparison | Delta | 95% paired CI | Permutation p | Holm p | Superiority | Non-inferiority |
|---|---:|---:|---:|---:|---|---|
| PC-CNG − random | +0.02835 | [-0.01106, 0.08281] | 0.3848 | 0.7695 | NO | YES |
| PC-CNG − template/rule | +0.02112 | [-0.01408, 0.06832] | 0.4417 | 0.4417 | NO | YES |
| union − PC-CNG | -0.02399 | [-0.06768, 0.00584] | 0.2866 | 0.8597 | NO | NO |

Interpretation:

1. PC-CNG is not proven superior to random mismatch or template perturbation.
2. PC-CNG meets the preregistered 0.02 non-inferiority criterion against those two sources.
3. Uniform union is not superior or non-inferior to PC-CNG in this run.
4. These results do not justify source-aware or learned-generator SOTA claims.

## 5. Secondary diagnostics

Five-seed means:

| Arm | T1 low-yield AUPRC | T2 ordinal MAE | T3 yield MAE | T3 Spearman | T4 plate NDCG | T5 primary AUPRC | T5 ECE | T5 selective risk |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| positive-only | 0.8349 | 3.6706 | 74.9811 | -0.1417 | 0.4922 | 0.1272 | 0.5944 | 0.9427 |
| PC-CNG | 0.8636 | 0.3223 | 30.6008 | -0.1418 | 0.4868 | 0.1260 | 0.4135 | 0.5371 |
| random | 0.8394 | 0.3223 | 30.5918 | -0.1517 | 0.4891 | 0.0976 | 0.3531 | 0.5879 |
| template/rule | 0.8505 | 0.3223 | 30.5917 | -0.1450 | 0.4849 | 0.1049 | 0.3646 | 0.5741 |
| union | 0.8405 | 0.3223 | 30.5940 | -0.1499 | 0.4912 | 0.1020 | 0.3867 | 0.5500 |

Secondary metrics are diagnostic and were not used to select the winner. Negative T3 Spearman and weak T4 discrimination are explicit failure evidence for future model work.

## 6. Independent reconstruction and hashes

- `formal_result_v3.json`: `7e1dc807077aaab51ec078ca38360dcfe809f6724c6323c728fb340db893a1e6`
- `predictions_t5_v3.json`: `1ab507bbbbf74a2ffca5594ba661dddc2559e324a399394b37cc77ba11b7f028`
- `independent_primary_inference_v3_c7b0263.json`: `de581aedc6dbd6d16a4bf85bb1ac39639baf3dd49a07b4b1e028950bf41d9dc4`
- frozen analysis plan: `567a4fe8d5a99b350d29c8ceba1e2f80ab950c2bfc19085dc8dccdf1f9c97f82`

The original formal writer used `default=str`, which represented NumPy booleans as strings. No numeric statistic changed. The verifier accepts only the exact legacy `"True"` and `"False"` forms, compares the complete primary object, and rejects any mutated inference value. New formal writes use native JSON scalar conversion.

## 7. Artifact locations

- `/mnt/cunyuliu_pc_cng_phaseb_v3_20260728/g6_v3_formal_a602a41_20260728/formal_result_v3.json`
- `/mnt/cunyuliu_pc_cng_phaseb_v3_20260728/g6_v3_formal_a602a41_20260728/predictions_t5_v3.json`
- `/mnt/cunyuliu_pc_cng_phaseb_v3_20260728/g6_v3_formal_a602a41_20260728/independent_primary_inference_v3_c7b0263.json`
- `/mnt/cunyuliu_pc_cng_phaseb_v3_20260728/g6_v3_formal_a602a41_20260728/reconstruction_verification_v3.json`
- `/mnt/cunyuliu_pc_cng_phaseb_v3_20260728/g6_v3_inference_operating_characteristics_a602a41_20260728.json`

## 8. Exit decision

- Benchmark and statistical-system rebuild: **PASS**
- Independent reconstruction: **PASS**
- PC-CNG superiority over matched simple sources: **NO-GO**
- Cross-publication external replication: **NOT YET AVAILABLE**

Phase B is complete without changing the frozen endpoint or statistical interpretation. Phase C may proceed, but it must not reuse this result as evidence that the learned structured source is already superior.
