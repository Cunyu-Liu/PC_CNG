# P4-G3: PC-CNG Augmentation Main Experiment — Report

**Phase**: P4-G3 (PC-CNG augmentation 主实验)
**Date**: 2026-07-22
**Verdict**: **NO_GO** (see §5 — verdict override with documented root cause)
**Entry conditions**: P4-G1 = GO (commit `0f906a4`), P4-G2 = PARTIAL_GO (commit `0118a32`, C3 LoRA config frozen in `results/p4_lora_ablation/selected_backbone.json`)

---

## 1. Objective

Test the core P4 claim:

```text
the same strong model + PC-CNG augmentation
>
the same strong model without PC-CNG augmentation
```

Seven augmentation arms × two backbones with different inductive biases ×
10 pre-declared seeds, on the frozen v1 candidate benchmark.
Only the augmentation signal varies; generator, scorer, split, seeds, budget,
and candidate manifest are fixed.

## 2. Experimental setup

### 2.1 Benchmark (frozen v1 manifest)

- `data/p4/manifests/hte_feasibility_v1.json`
- 500 groups × 8 candidates (1 gold + 7 negative sources: random_mismatch,
  random_corruption, tanimoto_retrieval, template_perturbation,
  unconstrained_edit, rule_pc_cng, external_beam)
- Split (parent-reaction isolated): train 394 groups / val 51 / test 55;
  each split exactly balanced across all 8 candidate sources.

### 2.2 Backbones

| Backbone | Architecture | Trainable params | Source |
|---|---|---|---|
| Chemformer-LoRA | Frozen pretrained Chemformer + LoRA (C3: attention `out_proj`, rank 8, alpha 16) + BCE ranking head | 180,737 | P4-G2 `selected_backbone.json` |
| GNN | In-repo pure-PyTorch GAT (3 layers, 4 heads, hidden 128, atom/bond RDKit featurization) + ranking head | 187,265 | `pc_cng/gnn_backbone.py` |

External LocalRetro/Graph2SMILES were not installed; per P4-G3 spec the
in-repo real GNN is used and is **not** named after any external model.

### 2.3 Arms

A0 positive-only · A1 +random mismatch · A2 +random corruption ·
A3 +Tanimoto · A4 +template · A5 +unconstrained edit · A6 +rule PC-CNG.
A7 (observed real negatives) not run in this phase.

### 2.4 Protocol (fairness controls)

- Same positives in all arms (394); each treatment arm adds exactly 394
  negatives of one source — identical negative count, identical sampling ratio.
- Same split, same manifest, same seeds `20260721..20260730` (10, pre-declared).
- Same budget: 5 epochs, AdamW lr 1e-4, batch 16, early stopping patience 2
  on val MRR, selection metric = val MRR. No test-set-informed choices.
- Per-sample predictions saved for all 140 runs
  (`paired_predictions/<backbone>_<arm>_seed<seed>/{test,val}_predictions.json`).
- Statistics: paired bootstrap CI (10k resamples) on per-seed test MRR
  deltas; Cohen's d and percentage-point (pp) effect sizes vs A0.
  No significance claim from seed-level means alone.

## 3. Results

### 3.1 Test metrics (mean over 10 seeds)

**Chemformer-LoRA** (A0 baseline MRR 0.2935)

| Arm | MRR | Δ vs A0 (pp) | Top-1 | NDCG | AUPRC | ECE | Brier |
|---|---|---|---|---|---|---|---|
| A0 | 0.2935 | — | 0.0855 | 0.4579 | 0.1269 | 0.8291 | 0.7993 |
| A1 | 0.4153 | +12.18 | 0.1636 | 0.5567 | 0.1853 | 0.3690 | 0.2458 |
| A2 | 0.4509 | +15.74 | 0.2018 | 0.5847 | 0.2060 | 0.3665 | 0.2439 |
| A3 | 0.3717 | +7.82 | 0.1255 | 0.5220 | 0.1624 | 0.3646 | 0.2429 |
| A4 | 0.4008 | +10.73 | 0.1527 | 0.5448 | 0.1755 | 0.3868 | 0.2596 |
| A5 | **0.4694** | **+17.59** | **0.2255** | **0.5990** | **0.2292** | 0.3853 | 0.2577 |
| A6 | 0.4509 | +15.74 | 0.2018 | 0.5847 | 0.2060 | 0.3665 | 0.2439 |

**GNN (GAT)** (A0 baseline MRR 0.4121)

| Arm | MRR | Δ vs A0 (pp) | Top-1 | NDCG | AUPRC | ECE | Brier |
|---|---|---|---|---|---|---|---|
| A0 | 0.4121 | — | 0.1291 | 0.5562 | 0.2087 | 0.6368 | 0.5953 |
| A1 | 0.4540 | +4.19 | 0.2000 | 0.5882 | 0.2901 | 0.3036 | 0.2523 |
| A2 | 0.4120 | −0.01 | 0.1200 | 0.5565 | 0.2031 | 0.6830 | 0.6283 |
| A3 | **0.5237** | **+11.16** | **0.2636** | **0.6423** | 0.2544 | 0.4301 | 0.3827 |
| A4 | 0.4725 | +6.04 | 0.2255 | 0.6023 | 0.2857 | 0.3094 | 0.2574 |
| A5 | 0.4120 | −0.01 | 0.1145 | 0.5567 | 0.2010 | 0.7119 | 0.6696 |
| A6 | 0.4120 | −0.01 | 0.1200 | 0.5565 | 0.2031 | 0.6830 | 0.6283 |

### 3.2 Paired bootstrap CI (A6 − A0, test MRR)

| Backbone | Δ mean | 95% CI | p |
|---|---|---|---|
| Chemformer-LoRA | +0.1574 | [+0.1008, +0.2070] | 0.0000 |
| GNN | −0.0001 | [−0.0127, +0.0134] | 0.5023 |

### 3.3 Observations

1. **Synthetic negatives strongly help Chemformer-LoRA**: every negative
   type improves MRR by +7.8 to +17.6 pp, and calibration improves
   dramatically (ECE 0.83 → ~0.37). Positive-only training is severely
   miscalibrated.
2. **Backbone disagreement**: the best arm on Chemformer is A5
   (unconstrained edit, +17.59 pp); on GNN it is A3 (Tanimoto, +11.16 pp),
   while A2/A5/A6 give the GNN nothing. Augmentation benefit is
   backbone-specific, not universal.
3. **A6 ≡ A2 exactly.** On Chemformer, A6 and A2 produce bit-identical
   scores for all 440 test candidates × all 10 seeds. Root cause below.

## 4. Key finding: A6 arm is vacuous in the frozen v1 manifest

Per-group comparison of the frozen v1 manifest
(`results/p4_augmentation/manifest_integrity.json`):

```text
rule_pc_cng vs random_corruption:
  500 / 500 groups carry byte-identical candidate SMILES
  (differing only in candidate_id / candidate_source /
   candidate_source_rank / edit_type metadata)
```

Consequences:

- The A6 ("rule PC-CNG") training set is the A2 ("random corruption")
  training set relabeled. Deterministic Chemformer training therefore
  reproduces A2 bit-for-bit; the GNN matches on test ranks (its residual
  score differences come from nondeterministic CUDA scatter-add).
- **The v1 manifest provides zero evidence about rule PC-CNG
  augmentation.** This is a P4-G1 manifest-construction issue, discovered
  here via the A2/A6 equivalence. The v1 manifest is immutable per project
  rules; the fix is a new v2 manifest namespace with genuinely
  rule-generated PC-CNG candidates.

## 5. Verdict: NO_GO (override)

Raw statistics alone would yield WEAK_GO (Chemformer A6 CI all-positive,
+15.74 pp ≥ 1.0 pp; GNN null). The verdict is overridden to **NO_GO**
because:

1. **A6 does not test PC-CNG** — it is A2 by construction (§4), so the
   positive Chemformer CI is evidence about *random corruption* negatives,
   not about rule PC-CNG negatives.
2. **A6 does not beat the best non-PC-CNG negative baseline** on either
   backbone: A5 exceeds A6 by +1.85 pp on Chemformer; A3 exceeds A6 by
   +11.17 pp on GNN (pre-declared threshold: +0.5 pp).
3. Spec NO-GO clause triggered: "PC-CNG 不优于简单 negative baseline".

`next_phase_allowed = false` for any PC-CNG augmentation claim; the
required remediation is a v2 candidate manifest (new namespace) with real
rule PC-CNG candidates, then a re-run of the A6 arm (both backbones,
same seeds/protocol). G4 causal decoupling on PC-CNG gains is blocked
until that re-run exists.

## 6. Limitations

- Rule PC-CNG augmentation effect is untestable on the v1 manifest (§4).
- GNN is an in-repo GAT, not LocalRetro/Graph2SMILES (spec-sanctioned
  fallback; named honestly).
- HTE-feasibility ranking is a development benchmark; per spec it is not
  the final external conclusion.
- A7 (observed real negatives) reference arm not run.
- Known-positive collision sensitivity and oracle-coverage deltas were not
  recomputed here; oracle Top-1 coverage of the manifest itself = 1.0
  (P4-G1 audit) and is arm-independent by construction (same test
  candidates in every arm).

## 7. Reproducibility

- Runner: `chem_negative_sampling/pc_cng/run_p4_augmentation.py`
- Aggregator: `chem_negative_sampling/pc_cng/aggregate_p4_g3.py`
- Tests: `chem_negative_sampling/tests/test_augmentation_pipeline.py`
  (41 passed, including manifest-integrity regression tests)
- Commands: `results/p4_augmentation/commands.log`
- Environment: `results/p4_augmentation/environment.json`
- Input hashes: `results/p4_augmentation/input_hashes.json`
- Outputs: `results/p4_augmentation/{summary.csv, effect_sizes.csv,
  bootstrap_ci.json, manifest_integrity.json, go_no_go.json,
  run_manifest.json, model_manifests/, paired_predictions/}`
  (140 run directories = 2 backbones × 7 arms × 10 seeds)
- Hardware: Chemformer on GPU 6, GNN on GPU 7 (single run each arm/seed).
