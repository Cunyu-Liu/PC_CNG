# Phase C G8-C formal source-expert completion audit

Date: 2026-07-29  
Implementation commits: `91dfdf7`, `1256081`  
Scientific status: `FORMAL_SOURCE_EXPERT_PARTIAL_EXPERT_LABELS_PENDING`

## Outcome

Phase C has completed the engineering and internal validation needed to move
G8-C from a pseudo-supervised prototype to a real-supervision source expert.
The result is not a source-superiority claim and is not a complete scientific
GO because no real expert labels are available.

## What changed

- Formal runs require CUDA and all real supervision caches.
- Unmapped edit targets are `NOT_APPLICABLE`; they are not fabricated
  `NO_EDIT` targets.
- Stage 1 reconstructs real bond-form, bond-break and bond-order-change
  locus/type/arguments.
- Stage 2 imitates actual rule actions, including explicit rule
  non-applicability.
- Stage 3 uses only observed products from the same complete reaction context.
- Stage 4 snapshots the reference after Stage 3 and scores complete real
  actions under policy and reference.
- Reconstruction and proposal heads are separate; fixed-weight real-edit
  rehearsal prevents later stages from overwriting reconstruction semantics.
- Candidate-level risk uses all parseable product outcomes, full-pool
  class-balanced training and independent group-disjoint temperature
  calibration.
- Formal mode does not run the legacy tiny Morgan MLP utility proxy.

## Frozen data contract

The source validation split was consumed by v1 and by development diagnosis,
then excluded from v2. V2 used fixed SHA-256 partitions created from source
training groups:

| Data role | Train | V2 holdout |
|---|---:|---:|
| Reactions | 582 | 137 |
| Same-context competing/preference pairs | 81 | 15 |
| Risk examples after featurization | 758 | 165 |

The source test split remained sealed.

## Preserved v1 negative result

The first full formal run at commit `91dfdf7` failed:

| Metric | v1 |
|---|---:|
| Edit-locus accuracy | 0.069 |
| Edit-type accuracy | 0.000 |
| Valid edit rate | 1.000 |
| Candidate coverage | 1.000 |
| Calibrated FNR ECE | 0.113 |

This result is retained. It established the real failure mode: rule imitation
overwrote the shared reconstruction action head.

## V2 one-shot holdout result

| Endpoint | Result | 95% interval / sample | Threshold | Pass |
|---|---:|---|---:|---|
| Edit-locus accuracy | 0.583 | Wilson [0.388, 0.755], 14/24 | >=0.20 | yes |
| Edit-type accuracy | 1.000 | Wilson [0.862, 1.000], 24/24 | >=0.50 | yes |
| Joint locus/type | 0.542 | Wilson [0.351, 0.721], 13/24 | secondary | — |
| Argument accuracy after joint match | 1.000 | Wilson [0.772, 1.000], 13/13 | secondary | — |
| Valid edit rate | 1.000 | Wilson [0.996, 1.000], 976/976 | >=0.95 | yes |
| Candidate coverage | 0.984 | Wilson [0.945, 0.996], 126/128 | >=0.80 | yes |
| Calibrated FNR ECE | 0.0668 | 84 evaluation examples | <=0.15 | yes |
| Max absolute preference log-ratio | 0.140 | 3 usable pairs | <=5.0 | yes |
| Mean action-type entropy | 1.283 | 6 actions from 3 pairs | >=0.50 | yes |
| Frozen reference | unchanged | SHA-256 equality | required | yes |

Risk calibration used 81 calibration and 84 evaluation examples from
group-disjoint partitions. Raw ECE was 0.0958 and calibrated ECE was 0.0668.

## Risk-source audit

Available labels across the prepared data:

- known-positive collision: 4;
- observed competing product: 54;
- held-out HTE outcome: 1,079;
- completed expert label: 0.

No expert form was automatically filled. Because expert labels remain absent,
the strongest permissible status is
`FORMAL_SOURCE_EXPERT_PARTIAL_EXPERT_LABELS_PENDING`.

## Verification

- 25 Phase C tests passed on GPU 6.
- The independent verifier rebuilt all threshold checks and the final status.
- All current input hashes matched the formal run.
- The checkpoint action schema and stored status matched the result.
- No formal `comparison_results.csv`, Pareto proxy or non-empty legacy
  `raw_predictions` artifact was present.

Authoritative run:

`/mnt/cunyuliu_pc_cng_phasec_20260729/g8c_formal_v2_unseen_holdout_1256081`

Independent verification:

`/mnt/cunyuliu_pc_cng_phasec_20260729/g8c_formal_v2_unseen_holdout_1256081/independent_verification.json`

## Exit decision

Phase C engineering and core source-expert validation are complete. The
expert-label sub-criterion is externally blocked and remains explicitly open.
Phase D may use the learned generator as one source expert, but it must not:

- claim that learned is better than another source;
- tune on the Phase C v2 holdout;
- treat internal source-expert validation as external HTE/OOD utility;
- convert the missing expert evidence into a synthetic label.

