# Phase B G6 formal benchmark completion audit

> Date: 2026-07-29
> Scientific boundary: corrected benchmark engineering verified; PC-CNG superiority not supported
> Frozen schema: `g6_v3_corrected_reanalysis_20260729`
> Formal run commit: `080236bd11893e7fccf0a7f3a173250c11a404ff`
> Analysis status: `CORRECTED_REANALYSIS_TEST_OUTCOMES_PREVIOUSLY_OBSERVED`

## 1. Outcome

Phase B completed the task-definition, reaction-context, matched-budget and
paired-inference rebuild requested for G6. The authoritative result is the
corrected GPU reanalysis from clean commit `080236b`. The fixed test outcomes
had already been viewed during earlier development and audit runs, so this
artifact is explicitly a corrected reanalysis rather than a blind
confirmatory test.

The 2026-07-29 completion re-audit found and repaired one reproducibility gap:
the formal CLI imported the shared encoder through the repository-parent
namespace, so the documented module command failed from
`chem_negative_sampling/`. The canonical import is now the packaged
`models.pretrained_backbone`, `models*` is included in the wheel, and a
subprocess regression test runs without adding the repository parent to
`PYTHONPATH`. The `chem` installation extra now declares the formal runner's
direct `pyarrow` dependency. This packaging repair does not change the frozen
model, predictions or statistical result.

The first current-main GPU smoke then exposed a second non-scientific artifact
edge case: a one-seed standardized effect was written as the non-standard JSON
token `Infinity`. The statistic is undefined with no estimable between-seed
variance, so the implementation now writes `null` and all new formal/smoke and
independent-reconstruction writers reject any remaining NaN/Inf value. The
five-seed frozen formal result has finite between-seed variance and is
numerically unchanged.

The re-audit also found that the Holm step-down rejection decisions were
correct, but the reported adjusted p-values omitted the required cumulative
maximum of the ordered Bonferroni products. The implementation and regression
test now follow the standard definition. The corrected GPU run supplies the
authoritative adjusted p-values below. This correction does not change any
rejection decision or the scientific NO-GO.

The original v3 no-signal operating-characteristic generator also gave both
arms identical predictions at `delta=0`. Its reported family-wise type-I error
of zero was therefore tautological rather than a useful calibration check.
The completion re-audit replaces it with exchangeable, non-identical paired
predictions that share label and cluster signal but receive independent
symmetric arm noise. The corrected simulation is explicitly a post-
implementation design check; it does not retroactively strengthen or
preregister the efficacy result.

The original primary AUPRC helper sorted tied scores record by record and
therefore depended on input order. A historical prediction audit showed
single-seed differences of roughly 0.003--0.004 relative to
`sklearn.metrics.average_precision_score`, which is material at the observed
effect scale. The corrected amendment uses sklearn's threshold-level tie
handling for every primary, bootstrap, permutation and simulation metric call,
and a regression test requires permutation invariance when scores are tied.
All earlier AUPRC inference artifacts are deprecated.

The same audit replaced the secondary hand-written Spearman ranks and NDCG
ordering with `scipy.stats.spearmanr` and
`sklearn.metrics.ndcg_score(ignore_ties=False)`. These secondary corrections
cannot rescue or overturn a primary superiority claim, but they prevent
record-order-dependent diagnostic reporting.

The earlier 256-token context setting truncated 129 of 5,898 test records:
109 candidate products were only partially visible and 20 were not visible to
the encoder. The frozen checkpoint natively supports 512 tokens. At 512, the
maximum complete sequence lengths are 354 (train), 337 (validation) and 382
(test), so all observed records fit. The corrected runner audits every matched
train arm as well as validation and test and fails closed unless all segments
are fully visible.

The corrected formal contract also distinguishes recorded seeds from
bitwise-reproducible CUDA execution. Formal and GPU smoke runs now fail unless
`CUBLAS_WORKSPACE_CONFIG=:4096:8` is present, enable PyTorch deterministic
algorithms and deterministic cuDNN, disable cuDNN benchmarking, and serialize
all states in runtime provenance. An initial two-run smoke with only those four
controls preserved identical metadata but differed by at most
`4.20e-05` in predicted probabilities, so it was not reported as bitwise
reproducible. The contract was tightened to disable TF32, flash SDP and
memory-efficient SDP and to require math SDP before final repeat testing.

The tightened A/B repeat still produced different prediction hashes. Record
metadata were identical, and across 400 arm-record scores the maximum absolute
difference was `3.92e-05` and the mean absolute difference was `7.62e-06`.
The benchmark therefore records a strict deterministic-kernel policy and
tolerance-level same-environment reproducibility, not bitwise reproducibility.
Five-seed hierarchical inference remains necessary; deterministic flags are
not used to pretend that training variability is zero.

This is not an efficacy GO. All three preregistered superiority intervals cross zero.

Four completion-re-audit attempts were intentionally stopped before result
acceptance and remain preserved as failed-run evidence:

- `f752be6`: the manifest attempted to hash a non-existent legacy task-head
  path at finalization;
- `1615593`: a training-manifest audit found one template candidate identical
  to its observed parent product.
- `2521572`: the primary AUPRC implementation was found to be input-order
  dependent when predictions were tied;
- `aaa8746`: the 256-token context setting was found to partially or fully
  hide candidate products.

Neither attempt is a completed scientific result. The authoritative corrected
reanalysis originates from clean commit `080236b`, excludes parent-positive
collisions under the v2 amendment, and is subject to complete-object
independent reconstruction.

## 2. Requirement audit

| Requirement | Evidence | Status |
|---|---|---|
| T1 frozen low-yield endpoint | `<10% yield` in frozen plan | PASS |
| T2 true ordinal model | cumulative-link thresholds and ordinal loss | PASS |
| T3 reaction-conditioned regression | shared full reaction/condition representation with regression head | PASS |
| T4 true ranking model | within-plate pair construction and pairwise ranking loss | PASS |
| T5 explicit conditions | catalyst, solvent, reagent, temperature and time fields encoded | PASS WITH DATA LIMITATION |
| Shared pretrained encoder | frozen Chemformer checkpoint shared across T1–T5 on CUDA | PASS |
| Pretrained checkpoint provenance | immutable checkpoint hash and conversion metadata | PASS WITH CORPUS-PROVENANCE GAP |
| Product-only Morgan only weak baseline | excluded from v3 formal main model | PASS |
| Matched source arms | 75 common parents; each source arm 75 positive + 75 negative | PASS |
| Split independence | zero record, experimental-group or plate overlap across train/val/test | PASS |
| One preregistered primary endpoint | real-HTE condition-feasibility source-macro AUPRC | PASS WITH SINGLE-SOURCE LIMITATION |
| Paired cluster bootstrap | same clusters resampled for challenger and baseline | PASS |
| Hierarchical seed×cluster inference | seed and cluster uncertainty aggregated in frozen procedure | PASS |
| Paired permutation + Holm | frozen three-comparison family | PASS |
| Effect and non-inferiority | raw delta, paired CI and margin 0.02 recorded | PASS WITH RATIONALE NEEDED |
| Type-I and power simulations | exchangeable paired null FWER 0.0125 (Wilson 95% CI 0.0022–0.0675); power 0.45/0.925/1.0 at synthetic deltas 0.02/0.04/0.08 | PASS AS POST-IMPLEMENTATION DESIGN CHECK |
| Independent reconstruction | separate-process reconstruction from the frozen prediction artifact; complete `primary_inference` equality, `verified=true` | PASS |
| No test-driven baseline selection | comparison order frozen before formal evaluation | PASS |
| Documented CLI and wheel import | `pc_cng` and shared `models` import from source checkout and wheel | PASS AFTER RE-AUDIT FIX |

`reagent_fraction=0.0` in the available HTE records, despite the model having an explicit reagent channel. This is a data-availability limitation, not evidence that reagent effects were validated. Catalyst availability is 0.5982; solvent 0.9917; temperature and reaction time 1.0.

The 0.02 non-inferiority margin was frozen before test evaluation, but the
current analysis plan does not give a domain or decision-theoretic rationale
for why a two-point AUPRC loss is scientifically acceptable. Non-inferiority
is therefore secondary and should not be a headline claim until that rationale
is supplied independently of the observed test result.

The checkpoint conversion summaries record the source file, tensor count,
hyperparameter keys and sanitized output, but do not identify the original
pretraining corpus version, license, split manifest or a reaction-level overlap
audit against the HTE test set. The checkpoint hash makes this run repeatable,
but those missing provenance fields remain reviewer-facing inputs before an
external-generalization claim can be manuscript-eligible.

The legacy v2 sanity suite also contained an invalid AUPRC power generator: it shifted every score by the same constant while the baseline ranking was already perfectly separated. A constant shift cannot change AUPRC, so the old coverage/power assertions necessarily reported zero. The repaired tests use a location metric for CI coverage and an overlapping, label-directional score shift for AUPRC power; the effect grid is deliberately unsaturated.

## 3. Formal dataset and run contract

- Input records: 39,546
- Included after reaction-context validation: 39,450
- Excluded for missing reaction context: 96
- Validation records: 4,291
- Test records: 5,898
- T5 positives (`yield >= 50%`): 337 (5.71%)
- T1 positives (`yield < 10%`): 4,979 (84.42%)
- Test clusters: 48
- Test cluster size: min 1, median 102, max 670 records
- Endpoint-evaluable clusters: 25
- Evaluable source-publication slices: 1
- Seeds: 5
- Formal device: GPU (`CUDA_VISIBLE_DEVICES=6`; A100-PCIE-40GB MIG 1g.5gb)
- Shared encoder: frozen Chemformer checkpoint, `product_only_baseline=false`
- Matched training parents: 75
- Candidate collision exclusion: one template candidate equalled its parent positive and was removed; the unmatched parent was excluded from every source arm
- Context visibility: zero truncated segments in every train arm, validation and test at the checkpoint-native 512-token limit

Because only one publication source is evaluable, the current endpoint is numerically source-macro but does not establish cross-publication replication. A second real HTE source remains mandatory.

## 4. Frozen corrected primary inference

| Comparison | Delta | 95% paired CI | Permutation p | Holm p | Superiority | Non-inferiority |
|---|---:|---:|---:|---:|---|---|
| PC-CNG − random | -0.00208 | [-0.01858, 0.01439] | 0.7590 | 1.0000 | NO | YES |
| PC-CNG − template/rule | -0.00676 | [-0.02557, 0.00379] | 0.2145 | 0.6434 | NO | NO |
| union − PC-CNG | -0.00152 | [-0.01547, 0.00825] | 0.7726 | 1.0000 | NO | YES |

Interpretation:

1. PC-CNG is not proven superior to random mismatch or template perturbation;
   both corrected point estimates are negative.
2. PC-CNG meets the frozen 0.02 numerical non-inferiority criterion against
   random, but not against template/rule. The margin lacks an independent
   domain rationale and is not a headline claim.
3. Uniform union is neither superior to PC-CNG nor demonstrably worse by more
   than the frozen numerical margin.
4. These results do not justify source-aware or learned-generator SOTA claims.

## 5. Secondary diagnostics

Five-seed means:

| Arm | T1 low-yield AUPRC | T2 ordinal MAE | T3 yield MAE | T3 Spearman | T4 plate NDCG | T5 primary AUPRC | T5 ECE | T5 selective risk |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| positive-only | 0.8361 | 3.6706 | 71.6500 | -0.1019 | 0.4951 | 0.0834 | 0.5945 | 0.9474 |
| PC-CNG | 0.8237 | 0.7744 | 36.5611 | -0.0731 | 0.4945 | 0.0894 | 0.3701 | 0.5958 |
| random | 0.8176 | 0.7536 | 36.5512 | -0.0748 | 0.4953 | 0.0915 | 0.3625 | 0.5969 |
| template/rule | 0.8238 | 0.7576 | 36.5519 | -0.0708 | 0.4937 | 0.0962 | 0.3644 | 0.5949 |
| union | 0.8099 | 0.7725 | 36.5546 | -0.0765 | 0.4984 | 0.0879 | 0.3659 | 0.5974 |

Secondary metrics are diagnostic and were not used to select the winner. Negative T3 Spearman and weak T4 discrimination are explicit failure evidence for future model work.

## 6. Independent reconstruction and hashes

- `formal_result_v3.json`: `e7274a4653deefc8f413d73cbb67a233f790c01dbf8dc472469eaf9e5f9100fa`
- `predictions_t5_v3.json`: `ad9eb4a85813896944bc6e27616302131d3da72744bba85e38d144fc53448cff`
- `independent_primary_inference_v3_080236b.json`: `7a67aef34af3c39b6b9b8e1b643fcd04c6838b3baf82eb9ad93d0c0816776cbd`
- `reconstruction_verification_v3_080236b.json`: `bea634f3ea2452f7a871f53a49e0a6023f7daf992536ca6033e58b8983f37885`
- `g6_v3_inference_operating_characteristics_080236b_20260729.json`: `f121405c4a8b17ec9ca2071cb427b5016dfecb37f0b1ed616b1da90b037853a7`
- frozen corrected analysis plan: `e91f56e5a6431606e45d0b2154fe01e19ab3d78fbc1219bba7f8dd6cfd7950b9`
- `run_manifest_v3.json`: `426160f530930a84d0ddacd8b3ed18d4d1209d18175a000b27ee12acbefe25aa`

The original formal writer used `default=str`, which represented NumPy booleans as strings. No numeric statistic changed. The verifier accepts only the exact legacy `"True"` and `"False"` forms, compares the complete primary object, and rejects any mutated inference value. New formal writes use native JSON scalar conversion.

The reconstruction entrypoint is process-independent and starts from the
frozen prediction JSON, but it intentionally reuses the repository's inference
library. It validates artifact reproducibility and complete-object equality;
it is not a fully independent reimplementation of the statistics. Standard
library implementations and focused regression cases anchor metric and Holm
semantics, while a third-party end-to-end reproduction remains future work.

## 7. Artifact locations

- `/mnt/cunyuliu_pc_cng_phaseb_v4_20260729/g6_v3_corrected_080236b_20260729/formal_result_v3.json`
- `/mnt/cunyuliu_pc_cng_phaseb_v4_20260729/g6_v3_corrected_080236b_20260729/predictions_t5_v3.json`
- `/mnt/cunyuliu_pc_cng_phaseb_v4_20260729/g6_v3_corrected_080236b_20260729/independent_primary_inference_v3_080236b.json`
- `/mnt/cunyuliu_pc_cng_phaseb_v4_20260729/g6_v3_corrected_080236b_20260729/reconstruction_verification_v3_080236b.json`
- `/mnt/cunyuliu_pc_cng_phaseb_v4_20260729/g6_v3_inference_operating_characteristics_080236b_20260729.json`
- `/mnt/cunyuliu_pc_cng_phaseb_v4_20260729/g6_v3_corrected_080236b_20260729/{pip_freeze,conda_explicit,environment_provenance}_080236b.txt`

## 8. Exit decision

- Benchmark and statistical-system rebuild: **PASS**
- Independent reconstruction: **PASS**
- PC-CNG superiority over matched simple sources: **NO-GO**
- Cross-publication external replication: **NOT YET AVAILABLE**

Phase B is complete without changing the frozen endpoint or statistical interpretation. Phase C may proceed, but it must not reuse this result as evidence that the learned structured source is already superior.
