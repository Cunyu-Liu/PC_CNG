# P4-G3 (v2 re-run): PC-CNG Augmentation on Remediated v2 Manifest — Report

**Phase**: P4-G3 re-run (v2 candidate manifest)
**Date**: 2026-07-23
**Verdict**: **WEAK_GO** (`next_phase_allowed = true`)
**Prior state**: P4-G3 on v1 = NO_GO (commit `0470e94`) because A6 ≡ A2 in the
frozen v1 manifest; P4-G4 = DEFERRED (commit `3c0d22b`) pending this re-run.
**Entry conditions**: v2 manifest built and audited (14/14 checks PASS),
G2 backbone config unchanged (C3 LoRA, `results/p4_lora_ablation/selected_backbone.json`).

---

## 1. Why a v2 re-run exists

The v1 manifest (`hte_feasibility_v1.json`) is immutable. Its `rule_pc_cng`
candidates were byte-identical to `random_corruption` in 500/500 groups
(string-replace artifact on atom-mapped SMILES), and SMILES validity of the
corruption/edit arms was ~7.4% / 5.0%. The v1 A6 arm therefore tested
*random corruption*, not rule PC-CNG (v1 report §4).

Remediation (this phase, new namespace, v1 untouched):

- `chem_negative_sampling/pc_cng/build_p4_candidate_manifests_v2.py`
  — 17 reaction-SMARTS PC-CNG rules (acid→amide, ester→amide,
  alcohol→ketone, aryl-Cl→amine, …) applied via RDKit RWMol with
  sanitization; per-group **sibling exclusion** so no two candidates in a
  group share a canonical SMILES; fallback generators
  (valid_corruption / valid_unconstrained_edit) produce only
  RDKit-parseable outputs; known-positive collision screening retained.
- `data/p4/manifests/hte_feasibility_v2.json`
  (manifest_hash `20a06d19…2481b6`, 500 groups × 8 candidates, same splits,
  same gold candidates as v1).
- Audit: `results/p4_manifest_v2_audit/` — 14/14 checks PASS, including
  A6 ≡ A2 in **0/500** groups and 100% sampled SMILES validity for
  random_corruption / rule_pc_cng / unconstrained_edit.
- Tests: `chem_negative_sampling/tests/test_p4_manifest_v2.py` (25 tests).

## 2. Experimental setup

Identical protocol to the v1 run (fairness controls unchanged):

- Backbones: Chemformer-LoRA (C3, 180,737 trainable params) and in-repo
  GAT GNN (187,265 params).
- Arms: A0 positive-only, A1 random mismatch, A2 random corruption,
  A3 Tanimoto, A4 template perturbation, A5 unconstrained edit,
  A6 rule PC-CNG.
- 10 pre-declared seeds `20260721..20260730`; same budget (5 epochs,
  AdamW 1e-4, batch 16, early stop patience 2 on val MRR); same split;
  394 train / 51 val / 55 test groups; per-sample predictions saved for all
  140 runs.
- Hardware: both backbones on GPU 6 (GPU 7 occupied by external jobs).
- Note: v2 regenerated the corruption/edit candidates in **all** groups,
  so the test candidate pool differs from v1; A0 baselines therefore shift
  (Chemformer 0.2935→0.3769, GNN 0.4121→0.2841). v1 and v2 numbers are not
  pool-comparable; only within-manifest arm contrasts are valid.

## 3. Results (test MRR, mean over 10 seeds)

| Backbone | A0 | A6 (rule PC-CNG) | Δ A6−A0 | 95% CI | p |
|---|---|---|---|---|---|
| Chemformer-LoRA | 0.3769 | **0.4060** (best arm) | +2.90 pp | [+0.30, +5.49] | 0.013 |
| GNN (GAT) | 0.2841 | 0.3528 | +6.87 pp | [+4.51, +9.17] | <0.001 |

All arms, Δ vs A0 (pp):

| Arm | Chemformer | GNN |
|---|---|---|
| A1 random_mismatch | −1.83 | +7.45 |
| A2 random_corruption | +1.63 | +5.23 |
| A3 tanimoto_retrieval | −5.72 | +1.43 |
| A4 template_perturbation | −0.97 | +4.36 |
| A5 unconstrained_edit | +2.47 | +4.62 |
| **A6 rule_pc_cng** | **+2.90** | **+6.87** |

A6 vs best non-PC-CNG baseline (pre-declared Strong-GO gate, ≥0.5 pp on
every backbone):

| Backbone | Best baseline | A6 − best | 95% CI | p | ≥0.5 pp? |
|---|---|---|---|---|---|
| Chemformer | A5 (0.4016) | +0.44 pp | [−0.42, +1.30] | 0.162 | **No** |
| GNN | A1 (0.3586) | −0.58 pp | [−2.13, +0.57] | 0.794 | **No** |

## 4. Verdict: WEAK_GO

Per the pre-declared P4-G3 thresholds (v1 report §5, embedded in
`go_no_go.json.predeclared_threshold`):

- A6 beats A0 with all-positive cluster-bootstrap CI on **both** backbones,
  mean improvement +4.89 pp (≥1.0 pp each) — the PC-CNG augmentation signal
  is real on the remediated manifest.
- Strong-GO additionally requires beating the best non-PC-CNG negative
  baseline by ≥0.5 pp on every backbone. This fails on both
  (Chemformer +0.44 pp n.s. vs A5; GNN −0.58 pp vs A1).
- Therefore **WEAK_GO**: the claim is narrowed to
  *"rule PC-CNG augmentation improves over positive-only training on both
  backbones, and is the best augmentation arm on Chemformer-LoRA; it does
  not significantly exceed the best simple-negative baseline."*

`next_phase_allowed = true`; P4-G5 entry condition (P4-G3 ≥ Weak GO,
P4-G4 completed) is met.

### Verdict-logic fix (auditable)

`compute_go_no_go` previously computed the baseline comparison but did not
gate STRONG_GO on it (the pre-declared `strong_go_min_vs_best_baseline_pp`
was recorded but unenforced). The function now enforces the gate and
records `baseline_comparison` per backbone. Regression tests added
(`test_baseline_gate_blocks_strong_go`, `test_baseline_gate_passes_strong_go`,
`test_baseline_margin_boundary`). On the v1 data this re-judgment is moot
(v1 remains NO_GO via manifest vacuity); on v2 it changes the raw label
from STRONG_GO to WEAK_GO without altering any statistic.

## 5. Limitations

- PC-CNG advantage over the best simple-negative baseline is not
  established (+0.44 pp n.s. on Chemformer, negative on GNN). Any paper
  claim must be the narrowed Weak-GO claim above.
- GNN is an in-repo GAT fallback (named honestly), not
  LocalRetro/Graph2SMILES.
- GNN on v2 shows every negative arm helping (unlike v1); the v2
  corruption/edit candidates are 100% valid molecules, so the GNN now
  receives featurizable structures instead of unparseable strings — arm
  contrasts across manifests are confounded by this validity fix.
- A7 (observed real negatives) reference arm still not run.
- HTE-feasibility ranking remains a development benchmark, not the final
  external conclusion.

## 6. Reproducibility

- v2 builder: `chem_negative_sampling/pc_cng/build_p4_candidate_manifests_v2.py`
- Runner: `chem_negative_sampling/pc_cng/run_p4_augmentation.py`
  (`--manifest data/p4/manifests/hte_feasibility_v2.json`, GPU 6)
- Aggregator: `chem_negative_sampling/pc_cng/aggregate_p4_g3.py`
  (`--manifest data/p4/manifests/hte_feasibility_v2.json`)
- Tests: `test_augmentation_pipeline.py` (44 passed),
  `test_p4_manifest_v2.py` (25 passed)
- Outputs: `results/p4_augmentation_v2/{summary.csv, effect_sizes.csv,
  bootstrap_ci.json, manifest_integrity.json, go_no_go.json,
  run_manifest.json, environment.json, input_hashes.json, commands.log,
  model_manifests/, paired_predictions/}` (140 run dirs), plus per-backbone
  dirs `results/p4_augmentation_v2_{chemformer,gnn}/`
- v2 audit: `results/p4_manifest_v2_audit/`
