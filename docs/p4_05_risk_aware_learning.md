# P4-G5: Risk-Aware Counterfactual Learning — Report

**Phase**: P4-G5 (Risk-aware counterfactual learning)
**Date**: 2026-07-23
**Verdict**: **NO_GO** (0/6 criteria satisfied; NO-GO forensic rule triggered: simultaneous external+calibration degradation)
**Entry conditions**: P4-G3 (v2 re-run) = **WEAK_GO** ✓; P4-G4 = completed (DEFERRED, failure-diagnostic) ✓

---

## 1. Objective

Convert synthetic PC-CNG candidates from absolute negatives into risk-weighted
`counterfactual_unknown` samples, controlling false-negative damage, per the
P4-G5 spec. Five label-treatment methods were compared against the frozen
P4-G2 C3 backbone (Chemformer-LoRA, 180,737 trainable params) on the A6 arm
(gold + rule_pc_cng) of the **v2 manifest** (`hte_feasibility_v2.json`,
17 reaction-SMARTS rules, 100% SMILES validity).

## 2. Label and risk-model contract (spec-mandated)

- Four label classes used throughout: `positive_observed`, `negative_observed`,
  `counterfactual_unknown`, `known_positive_collision`. Synthetic candidates
  were **never** assigned `yield=0` / `true_negative` semantics outside the
  `hard_binary` ablation arm (which is the spec-allowed hard-label ablation).
- The false-negative-risk (FNR) model is a 13-feature logistic regression
  calibrated **only** on observed HTEa train-split rows
  (14,071 positive_observed vs 19,112 negative_observed, downsampled 2:1,
  val/test split_keys excluded) — synthetic self-labels were never used for
  risk-model training (policy embedded in `risk_model_manifest.json`).
- Calibration: train AUROC **0.8959**, logloss 0.3974 (n = 33,183).
- Feasibility ensemble (model-based signals): 5-member bagged MLP on
  Morgan-1024 fingerprints, bootstrap-balanced, observed data only.
- Sample weight = chemical_validity × data_support × boundary_value ×
  (1 − false_negative_risk), floored at 1e-4. All 4 components were
  individually ablated (§6).

## 3. Experimental setup

- **Runs**: 5 methods × 10 pre-declared seeds (20260721–20260730) = 50 main
  runs + 4 weight-component ablations × 10 seeds = 40 ablation runs
  (risk_weighted_pairwise), all with raw predictions persisted.
- **Protocol**: identical to P4-G3 — 5 epochs, batch 16, AdamW lr 1e-4,
  early stopping on val MRR (patience 2), same seeds, group-aware batching
  for pairwise/InfoNCE losses.
- **Stress tests** (built once, deterministic seed 20260723):
  known_positive (disguised observed positives), near_positive (high-FNR
  counterfactuals), ood_family (scaffold-level fallback — all HTEa val/test
  reaction families are covered by the 394 train groups, so family-level OOD
  is empty on v2), plus database-collision sensitivity.
- **Statistics**: seed-paired bootstrap CI (10k resamples) vs `hard_binary`
  per metric; verdict per the pre-declared 6-criterion gate
  (`results/p4_risk_aware/go_no_go.json`).

## 4. Main results (test split, mean over 10 seeds)

| method | MRR | AUPRC | ECE | fixed-fwd MRR | KP recovery@1 | selective risk@0.8 |
|---|---|---|---|---|---|---|
| hard_binary (baseline) | 0.4060 | 0.1960 | 0.3734 | 0.2181 | 0.454 | 0.248 |
| label_smoothing | 0.4048 | 0.1958 | 0.3736 | 0.2181 | 0.453 | 0.248 |
| pu_nnpu | 0.3918 | 0.1918 | 0.3966 | 0.2172 | 0.446 | 0.264 |
| risk_weighted_pairwise | 0.4035 | 0.1953 | 0.4056 | 0.2158 | 0.463 | 0.272 |
| risk_weighted_infonce | 0.4035 | 0.1949 | 0.4068 | 0.2159 | 0.459 | 0.272 |

Paired-CI verdicts vs hard_binary:

- **label_smoothing**: every metric CI straddles 0 — null effect.
- **pu_nnpu**: AUPRC CI [−0.0084, −0.0008] (significantly worse) **and** ECE
  CI [+0.0021, +0.0448] (significantly worse) → triggers the pre-declared
  NO-GO rule "risk-aware 方法同时降低性能和校准".
- **risk_weighted_pairwise / infonce**: ECE CI all-positive (worse
  calibration: +0.059 / +0.061 relative +8.6%/+8.9%), selective-risk CI
  all-positive (worse); external metrics unchanged (CI straddles 0).
- 0/6 GO criteria satisfied; `next_phase_allowed = false`.

Forensic checks passed (no artifacts driving the verdict):
FNR self-score coefficient share = **0.622** (< 0.8 dominance threshold);
best known-positive recovery@1 = **0.4625** (> 0.2 severe-failure threshold);
collision hard-reject rate = 0.000 for all methods.

## 5. Root cause: FNR inversion on v2 counterfactuals

The decisive diagnostic is in `risk_artifacts.json`:

| cohort | mean FNR = P(actually positive) |
|---|---|
| gold observed positives (sanity) | 0.4557 |
| **train rule_pc_cng synthetic negatives** | **0.6011** (median 0.8415) |

The observed-data-calibrated risk model assigns **higher** probability of
being positive to synthetic rule-PC-CNG counterfactuals than to the actual
observed positives. This is not a bug but a structural property: the v2
reaction-SMARTS rules generate *near-gold, fully valid* edits (by design they
mimic feasible functional-group interconversions), so the ensemble — trained
only on observed HTEa chemistry — scores them exactly like feasible products.
The observed-data domain provides no signal to distinguish "feasible-looking
but unobserved" from "observed positive".

Consequences, all directly observed:

1. **Weight collapse**: train A6 negatives get mean sample_weight 0.0019
   (median 0.0006, floor 1e-4) — risk-weighted losses effectively train on
   positives only, reproducing but not improving the baseline.
2. **PU prior saturation**: mean FNR ≥ 0.5 → pi capped at 0.5, destabilizing
   nnPU (its degradation is the NO-GO trigger).
3. **Ablation confirms direction**: removing the `one_minus_fnr` component
   **improves** risk_weighted_pairwise by +0.0148 MRR / +0.0126 AUPRC — the
   FNR signal is actively anti-predictive on v2. `chemical_validity` ablation
   has exactly 0 effect (v2 validity is 100% — constant component).

## 6. Ablation summary (risk_weighted_pairwise, vs full 4-component model)

| ablated component | ΔMRR | ΔAUPRC | ΔECE |
|---|---|---|---|
| chemical_validity | 0.000 | 0.000 | 0.000 (constant on v2) |
| data_support | −0.0011 | −0.0008 | +0.0055 |
| boundary_value | −0.0009 | −0.0010 | −0.0089 |
| one_minus_fnr | **+0.0148** | **+0.0126** | +0.0001 |

## 7. Interpretation and relation to prior phases

- The phase hypothesis — observed-data-calibrated risk weighting controls
  false-negative damage from rule PC-CNG — is **rejected** for the v2
  manifest with the C3 backbone.
- This is consistent with P4-G4's confirmed source×scorer interaction and
  the v2 audit: valid, near-positive rule negatives are the hardest
  counterfactuals, precisely the regime where an observed-only risk model
  cannot generalize (domain shift between observed and counterfactual
  distributions).
- The risk-model machinery itself is validated (AUROC 0.8959 on observed
  data; known-positive stress recovery 0.46 ≫ 0.2 floor; self-score
  dominance below threshold). The failure is **transfer** of the risk
  estimate to counterfactuals, not calibration on observed data.
- Future risk-aware attempts would need a counterfactual-domain calibration
  source (e.g., real failed-reaction records beyond HTEa, or experimental
  validation of a synthetic subset) — i.e., new data, not new losses.

## 8. Artifacts

- Code: `chem_negative_sampling/pc_cng/models/risk_aware_scorer.py`,
  `chem_negative_sampling/pc_cng/training/train_risk_aware.py`,
  `chem_negative_sampling/pc_cng/evaluation/false_negative_stress_test.py`,
  `chem_negative_sampling/pc_cng/run_p4_risk_aware.py`,
  `chem_negative_sampling/pc_cng/aggregate_p4_g5.py`
- Tests: `chem_negative_sampling/tests/test_risk_aware_loss.py`,
  `tests/test_false_negative_stress_test.py`, `tests/test_aggregate_p4_g5.py`
  — **67 passed** (pc_cng_gpu env).
- Results: `results/p4_risk_aware/` — `risk_model_manifest.json`,
  `risk_artifacts.json` (per-candidate 13 signals + weights), `stress_sets.json`,
  `summary.csv` (50 runs), `ablation.csv` (4 components × 10 seeds),
  `raw_predictions/` (test+val per run), `go_no_go.json`,
  plus contract files `run_manifest.json`, `environment.json`,
  `input_hashes.json`, `commands.log`.

## 9. Verdict

**NO_GO** — risk-aware label treatments do not improve over hard binary on
v2; nnPU significantly degrades both external performance and calibration
(pre-declared NO-GO rule). Root cause identified and evidenced: FNR inversion
on near-positive rule counterfactuals (§5). `next_phase_allowed = false`.
