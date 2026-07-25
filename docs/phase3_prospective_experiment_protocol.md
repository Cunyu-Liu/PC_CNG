# Phase 3 Prospective Experiment Protocol

**Status:** Draft (not yet in `docs/00_当前有效文档/`)
**Owner:** PC-CNG Phase 3 validation
**Created:** 2026-07-25
**Spec reference:** `提示词/pccng 的分阶段提示词.md` (P4 prospective validation)

---

## 1. Purpose

Prospectively test whether PC-CNG-generated negative reactions are
*actually* infeasible in the lab, and whether the model's failure
predictions generalise to reactions never seen during training.  This
is the experimental (wet-lab) counterpart to the retrospective OOD
splits and the G7 expert review.

Per the Phase 3 spec, **OOD results should become the main result, not
supplementary**; the prospective experiment closes the loop from
computational OOD to real-world failure prediction.

## 2. Design overview

| Element | Choice |
|---|---|
| Design | Pre-registered, blinded, prospective |
| Candidate sources | PC-CNG (learned structured), random mismatch, template perturbation |
| Blinding | Experimenters blind to candidate source and to model prediction |
| Conditions | Unified, frozen before experiment |
| Primary metric | Failure enrichment (does PC-CNG predict failures that actually fail?) |
| Secondary metrics | Side-product prediction accuracy, model confidence calibration |
| Testing set | Previously unseen reactions (held out from all training) |

## 3. Pre-registration

Before any wet-lab work begins, the following are frozen and
time-stamped:

1. This protocol (version + SHA-256 hash).
2. The candidate reaction set (the testing set, §4) — generated and
   committed to a sealed repository branch.
3. The failure definition (§5) — written down, signed off.
4. The model checkpoint used for prediction — hash recorded.
5. The analysis plan (§7) — script committed and frozen.
6. The experimental conditions (§6) — standardised reaction template,
   concentrations, temperature, workup, analysis method.

Any deviation from the pre-registered plan is logged in a deviation
register with rationale and timestamp; deviations do not invalidate the
experiment but are reported transparently.

## 4. Testing set

* **Source:** Reactions drawn from the OOD test splits (scaffold,
  reaction_family, author_lab) that were **never** used in any
  training stage of PC-CNG (Stages 1–4 of G8-C) or in the downstream
  classifier training.
* **Size:** 30–60 reactions, balanced across 3 candidate sources
  (10–20 per source).  The exact size is dictated by wet-lab capacity
  and reagent availability.
* **Selection:** Reactions are sampled deterministically (seeded) from
  the OOD test pool, filtered for (a) commercially available
  reactants, (b) compatibility with the unified conditions (§6), and
  (c) no safety hazards (per institutional policy).
* **Ground truth:** Each reaction is run in the lab and the outcome
  (yield, side products) is recorded by an analyst blind to the
  candidate source and to the model's prediction.

## 5. Failure definition (frozen before experiment)

A reaction is classified as **failed** if **any** of the following
hold, assessed by the blind analyst using LC-MS + NMR:

| Criterion | Threshold |
|---|---|
| Isolated yield of the declared product | < 5% |
| LC-MS peak area of the declared product | < 2% of total integrated area |
| Dominant product by LC-MS is **not** the declared product | side product ≥ 50% area |
| Reaction mixture decomposes / precipitates before workup | visual + LC-MS |

The failure label is binary (failed = 1, succeeded = 0).  The analyst
records the reason code (low yield, wrong product, decomposition,
no conversion, other) but the primary metric uses the binary label.

**This definition is frozen before the experiment begins and cannot be
changed after data collection.**

## 6. Unified experimental conditions

All candidate reactions are run under a **single** standardised
condition set, chosen before the experiment:

* **Catalyst loading:** fixed per reaction family (e.g. 5 mol% Pd for
  Suzuki, 10 mol% Ni for Ni-coupling) — declared in the pre-registration.
* **Ligand:** the most common ligand from the HiTEA training set for
  that family (declared per family).
* **Solvent:** the modal solvent from HiTEA for that family.
* **Base / additive:** fixed per family (declared).
* **Concentration:** 0.1 M.
* **Temperature:** 80 °C (or the family modal, declared).
* **Time:** 16 h.
* **Workup:** aqueous extraction + silica plug.
* **Analysis:** LC-MS (UV 254 nm + ESI) and ¹H NMR.

Rationale: by holding conditions constant, any failure is attributable
to the *reaction identity* (substrate + product combination), not to
condition variation.  This isolates the negative-generation quality.

## 7. Metrics

### 7.1 Primary: failure enrichment

For each candidate source *s* (PC-CNG, random, template):

* Let `fail_s` = fraction of source-*s* reactions that failed in the lab.
* Let `fail_baseline` = failure fraction for the random-mismatch source.
* **Failure enrichment** = `fail_s / fail_baseline` (relative risk) and
  the absolute difference `fail_s − fail_baseline`.

The primary endpoint is:

> PC-CNG failure enrichment ≥ 1.5× relative to random mismatch, with
> the 95% CI of the relative risk excluding 1.0.

CIs are computed by the Clopper–Pearson exact method on the
binomial proportions, and by a paired bootstrap on the per-reaction
outcomes when reactions are matched across sources.

### 7.2 Secondary: side-product prediction

For reactions that produce a dominant side product, the analyst records
the side-product SMILES (top-1 by LC-MS area).  PC-CNG is scored on
whether its proposed negative product matches the observed side product:

* **Exact match:** Tanimoto ≥ 0.95 between PC-CNG's negative product
  and the observed side product (Morgan radius 2, 2048 bits).
* **Scaffold match:** Murcko scaffold of PC-CNG's negative product
  equals the Murcko scaffold of the observed side product.
* **Class match:** the side product falls in the same reaction-family
  transformation class as PC-CNG's proposed negative.

Reported as top-1, top-3 accuracy across the failed reactions.

### 7.3 Secondary: model confidence calibration

PC-CNG emits a risk / confidence score for each negative candidate.
We bin the candidates into 5 equal-width risk bins and compute:

* **Expected Calibration Error (ECE)** between predicted risk and
  observed failure rate.
* **Brier score** for the binary failure outcome.

Target: ECE ≤ 0.15 (the model's confidence is actionable); Brier ≤ 0.25.

## 8. Blinding & unblinding

* **Generation:** The PC-CNG / random / template candidates are
  generated by the modelling team and assigned anonymous IDs
  (`R001`, `R002`, …).  The source→ID mapping is sealed in a
  `blinding_key.json` file, committed to a restricted-access branch.
* **Lab:** The wet-lab team receives only the reaction SMILES + unified
  conditions, with no source label and no model prediction.
* **Analysis:** The outcome data is collected and signed off by the
  blind analyst *before* unblinding.
* **Unblinding:** Performed only after the outcome table is finalised
  and hashed; the hash is recorded in the pre-registration.

## 9. Requirements

1. **Testing set previously unseen** — no reaction in the testing set
   appears in any training split (Stages 1–4, downstream classifier,
   or G7 pilot).  Verified by SMILES-canonical deduplication against
   all training CSVs.
2. **Failure definition frozen** — §5 is signed off before the first
   reaction is run.
3. **All results disclosed** — every reaction's outcome is reported,
   including successes.  No reaction is dropped after the experiment
   begins except for documented safety reasons (which are reported as
   exclusions with reason codes).
4. **Reproducible conditions** — the unified conditions (§6) are
   recorded in an electronic lab notebook with reagent lot numbers.
5. **Statistical analysis frozen** — the analysis script is committed
   before unblinding.

## 10. Pass / fail criteria

The prospective experiment **supports the PC-CNG claim** if:

1. **Primary:** PC-CNG failure enrichment ≥ 1.5× vs random mismatch,
   95% CI excludes 1.0.
2. **Calibration:** ECE ≤ 0.15 on the risk score.
3. **No safety incidents** that would invalidate the batch.
4. **Dropout ≤ 20%:** at least 80% of the pre-registered reactions are
   successfully run and analysed.

If criterion 1 is met but criterion 2 fails, the model is reported as
*discriminative but poorly calibrated* — a follow-up calibration
experiment is recommended before deployment.

If criterion 1 fails, the model's prospective utility is not supported;
the negative generation is reported as *retrospectively useful but not
prospectively validated*.

## 11. Limitations & mitigations

| Risk | Mitigation |
|---|---|
| Small n (30–60) underpowered | Pre-register a minimum effect size (1.5×); report CIs even if non-significant |
| Unified conditions may not reflect real-world variability | Acknowledge as a limitation; a follow-up condition-space experiment is recommended |
| Side-product identification limited by LC-MS | Use NMR for ambiguous cases; report identification confidence |
| Reagent availability constraints | Pre-register a substitution rule (top-3 commercial analogues) |
| Experimenter bias despite blinding | Blind analyst signs outcome table before unblinding; hash recorded |

## 12. Artefacts produced

* `docs/phase3_prospective_experiment_protocol.md` (this file)
* `data/phase3/prospective_testing_set.csv` — pre-registered reactions
* `data/phase3/prospective_blinding_key.json` — sealed source mapping
* `data/phase3/prospective_unified_conditions.json` — frozen condition set
* `results/phase3_prospective/outcomes.csv` — per-reaction lab outcomes
* `results/phase3_prospective/failure_enrichment.json` — §7.1
* `results/phase3_prospective/side_product_prediction.json` — §7.2
* `results/phase3_prospective/calibration.json` — §7.3
* `results/phase3_prospective/prospective_report.md` — pass/fail decision

## 13. Relationship to other Phase 3 components

| Component | What it establishes | What it does *not* |
|---|---|---|
| OOD splits (computational) | Generalisation to unseen groups | Real-world feasibility |
| G7 expert review (human) | Perceived feasibility by chemists | Actual wet-lab outcome |
| **Prospective (this protocol)** | **Actual wet-lab failure prediction** | Scale (small n) |

The three components are complementary: OOD splits show *statistical*
generalisation, G7 shows *human-judged* plausibility, and the
prospective experiment shows *empirical* validity.  Only the
prospective experiment can confirm that PC-CNG's negatives actually
fail in the lab.
