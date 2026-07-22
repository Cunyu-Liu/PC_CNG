# P4-G4: Generator × Scorer Causal Decoupling — Failure Diagnostic Report

**Phase**: P4-G4 (Generator × Scorer 因果解耦)
**Date**: 2026-07-23
**Verdict**: **DEFERRED** (failure-diagnostic mode; full 5×4×3 matrix NOT executed)
**Entry conditions**: P4-G3 = **NO_GO** (commit for G3 outputs; see `docs/p4_03_augmentation_story.md`). Per the P4-G4 spec (入口条件 `P4-G3 == Strong GO or Weak GO`; "若 G3 NO-GO，不执行完整矩阵，仅做失败诊断矩阵"), only the failure diagnostic matrix was executed.

---

## 1. Objective

The P4-G4 spec asks five mutually exclusive explanations for the P4-G3
augmentation pattern:

| # | Hypothesis | Verdict |
|---|---|---|
| H1 | PC-CNG negatives 本身有价值 (intrinsically valuable) | **UNTESTABLE** |
| H2 | 只是 Chemformer 更强 (scorer strength) | **REJECTED** |
| H3 | 只是 PC-CNG candidates 更难 (difficulty) | **MOOT_FOR_A6** |
| H4 | 只是候选数量或分布不同 (count/distribution) | **REJECTED_FOR_COUNTS** |
| H5 | negative source 与 scorer 存在强交互 | **CONFIRMED** |

Two structural findings of this diagnostic reshape all interpretation:

1. **A6 ≡ A2 in the frozen v1 manifest.** `rule_pc_cng` candidates are
   byte-identical to `random_corruption` candidates in all 500 groups
   (evidence: `results/p4_augmentation/manifest_integrity.json`). The v1
   benchmark therefore carries **no rule PC-CNG signal at all**; H1 cannot be
   tested until a v2 manifest exists.
2. **Majority-invalid SMILES in three negative sources.** Only 7.4% of
   `random_corruption` / `rule_pc_cng` and 5.0% of `unconstrained_edit`
   candidate SMILES are RDKit-parseable (§4). Structure-based scorers
   (GNN, Morgan MLP) silently drop these candidates, while the text-based
   Chemformer still tokenizes and scores them as strings. Arms A2/A5/A6 are
   therefore **text-artifact arms**: Chemformer's gains on them are partly
   string-level patterns, not chemical signal.

## 2. Experimental setup

### 2.1 Scope actually executed (failure diagnostic matrix)

- **Scorers (3)**: Chemformer-LoRA (frozen C3 config from P4-G2), in-repo GNN
  (GAT), and a **new third independent scorer: Morgan-fingerprint MLP**
  (Morgan radius-2, 2048-bit fingerprint → MLP 2048→512→256→1;
  1,180,673 trainable params; own featurization and training code — no
  representation or training code shared with the other two scorers).
  No external/frozen scorer was available; per spec the Morgan MLP is the
  independent-implementation substitute and is not named after any external
  model.
- **Cells**: 3 scorers × 7 arms (A0–A6) × 10 pre-declared seeds
  (20260721–20260730) = **210 cells**. Chemformer and GNN cells were reused
  from P4-G3 raw outputs (`results/p4_augmentation_{chemformer,gnn}/summary.csv`);
  the 70 Morgan MLP cells were trained in this phase
  (5 epochs, AdamW lr 1e-3, batch 16, early stopping on val MRR — same
  budget protocol as G3; mean wall-clock 3.5 s/run on cuda:0).
- **Protocols**: Protocol A (natural distribution) only. Protocol B
  (count matching) is already satisfied by the G3 design — every treatment
  arm trains on exactly 394 positives + 394 negatives of one source.
  Protocol C (difficulty matching) was **not** executed: it is only
  meaningful for the full matrix after a GO verdict; on v1, A6 ≡ A2 makes
  matched PC-CNG analysis vacuous. Difficulty is instead profiled
  scorer-independently in §4.

### 2.2 Statistics

- Per-scorer arm-vs-A0 effect sizes: paired bootstrap CI (10k resamples) on
  per-seed test-MRR deltas + Cohen's d (`effect_sizes.csv`).
- Two-way ANOVA with interaction on per-seed `delta_vs_A0` (A1–A6):
  `delta ~ C(source) * C(scorer)`, typ2, n = 180 (`interaction_model.json`).
- Mixed-effects `mrr ~ source*scorer + (1|seed)` attempted; the random-effects
  covariance is singular (only 3 scorers, limited scorer diversity) —
  `converged: false` is recorded in `interaction_model.json` and the ANOVA
  interaction is used as the primary interaction evidence.

## 3. Results

### 3.1 A0 (positive-only) baselines — H2 test

| Scorer | A0 test MRR (mean, 10 seeds) |
|---|---|
| Chemformer-LoRA | 0.2935 (weakest) |
| GNN | 0.4121 |
| Morgan MLP | 0.4960 (strongest) |

**H2 REJECTED**: Chemformer has the *weakest* positive-only baseline yet
shows the *largest* augmentation gains (up to +17.6 pp). Scorer strength
alone cannot explain the augmentation pattern.

### 3.2 Arm-vs-A0 effect sizes (test MRR, percentage-point difference)

| Arm (negative source) | Chemformer | GNN | Morgan MLP |
|---|---|---|---|
| A1 random_mismatch | +12.18 [+5.96, +17.65] | +4.19 [+2.46, +6.05] | **+27.47** [+26.40, +28.57] |
| A2 random_corruption † | +15.74 [+10.08, +20.70] | −0.01 [−1.27, +1.34] | +1.65 [−0.21, +3.90] |
| A3 tanimoto_retrieval | +7.82 [+1.70, +13.58] | **+11.16** [+8.63, +13.79] | +9.93 [+9.19, +10.72] |
| A4 template_perturbation | +10.73 [+5.35, +15.80] | +6.04 [+4.30, +7.98] | +22.26 [+21.41, +23.18] |
| A5 unconstrained_edit † | **+17.59** [+11.71, +22.56] | −0.01 [−1.05, +1.20] | +0.87 [−0.50, +2.38] |
| A6 rule_pc_cng † (≡ A2) | +15.74 [+10.08, +20.70] | −0.01 [−1.27, +1.34] | +1.65 [−0.21, +3.90] |

† = text-artifact arm (majority-invalid SMILES, §4). Brackets = 95% paired
bootstrap CI. Full table incl. Cohen's d and p-values: `effect_sizes.csv`.

**Best arm is scorer-dependent**: Chemformer → A5, GNN → A3, Morgan MLP → A1.
No negative source wins across scorers.

### 3.3 Interaction model — H5 test

ANOVA on per-seed deltas (n = 180, R² = 0.671):

| Term | F | df | p-value |
|---|---|---|---|
| C(source) | 13.21 | 5 | 8.6e-11 |
| C(scorer) | 43.74 | 2 | 6.5e-16 |
| **C(source):C(scorer)** | **17.76** | 10 | **1.5e-21** |

**H5 CONFIRMED**: the source × scorer interaction is overwhelmingly
significant (p = 1.47e-21). The value of a negative source cannot be stated
without naming the scorer — exactly the causal ambiguity this phase was
designed to detect.

### 3.4 Count check — H4 test

Every treatment arm trains on exactly 394 positives + 394 negatives with
identical sampling ratio and budget (`counts_matched: true`).
**H4 REJECTED for counts.** Distribution (difficulty) differences remain and
are profiled below.

## 4. Difficulty profiling (scorer-independent)

Tanimoto similarity of each negative candidate to its group's gold,
computed from structures only (`difficulty_profile.json`):

| Source | n_total | n_valid | valid_fraction | mean | p50 | p90 |
|---|---|---|---|---|---|---|
| random_mismatch | 500 | 500 | 1.000 | 0.117 | 0.105 | 0.165 |
| random_corruption | 500 | 37 | **0.074** | 0.816 | 0.813 | 1.000 |
| tanimoto_retrieval | 500 | 500 | 1.000 | 0.149 | 0.131 | 0.235 |
| template_perturbation | 500 | 500 | 1.000 | 0.149 | 0.115 | 0.225 |
| unconstrained_edit | 500 | 25 | **0.050** | 0.166 | 0.046 | 0.429 |
| rule_pc_cng | 500 | 37 | **0.074** | 0.816 | 0.813 | 1.000 |
| external_beam | 500 | 500 | 1.000 | 0.086 | 0.082 | 0.136 |

Interpretation:

- `random_corruption` (and identically `rule_pc_cng`) is **bimodal in the
  worst way**: 92.6% of candidates are unparseable strings, and the 7.4%
  that parse are near-duplicates of the gold (mean similarity 0.816,
  p90 = 1.0). For structure scorers these arms degenerate to ~A0
  (GNN Δ = −0.01 pp; MLP Δ = +1.65 pp, CI crosses 0 for MLP and is ~0 for GNN).
- Chemformer, being text-based, scores the unparseable 92.6% as ordinary
  strings, learns "garbage string → 0", and trivially ranks them below gold
  at test time — inflating A2/A5/A6 gains (all > +15 pp) by a string-level
  artifact. **The G3 headline "A5 unconstrained_edit is the best arm" is
  therefore confounded and must not be cited as chemical signal.**
- `unconstrained_edit` is similar: only 5% parseable; its valid tail is
  genuinely hard (p50 = 0.046 is easy, but p90 = 0.429 is the hardest valid
  tail among sources), yet structure scorers see almost none of it.
- H3 (hardness explains gains) is **moot for A6** (A6 ≡ A2) and descriptively
  unsupported elsewhere: the hardest valid source per scorer never coincides
  with that scorer's best arm.

## 5. Hypothesis verdicts (detail)

- **H1 UNTESTABLE** — A6 ≡ A2 in all 500 v1 groups; no rule PC-CNG signal
  exists on v1. PC-CNG main effect: only 1/3 scorers (Chemformer) shows a
  positive A6 CI, and that trace is the §4 text artifact.
- **H2 REJECTED** — Chemformer is the weakest A0 baseline (§3.1).
- **H3 MOOT_FOR_A6** — PC-CNG hardness undefined on v1; difficulty alone
  does not explain the remaining pattern (§4).
- **H4 REJECTED_FOR_COUNTS** — counts/ratio/budget identical across arms (§3.4);
  distribution differences profiled in §4.
- **H5 CONFIRMED** — interaction p = 1.47e-21; best arm per scorer differs
  (§3.2–3.3).

## 6. Limitations

1. Full 5-source × 4-scorer × 3-protocol matrix not executed (G3 = NO_GO);
   Protocol C difficulty matching deferred.
2. A6 vacuous on v1 (duplicates A2) — all PC-CNG-specific conclusions
   blocked on a v2 manifest.
3. Morgan MLP is the third scorer; no external/frozen scorer was used.
4. Mixed-effects model did not converge (singular random-effects covariance
   with 3 scorers); ANOVA interaction used as primary evidence.
5. Difficulty profile uses train+val+test candidates for descriptive
   statistics only; no test-set information entered any training or
   selection decision.

## 7. Remediation (blocking next phase)

1. **Build v2 candidate manifest** (new namespace, do not modify v1) with
   genuinely rule-generated PC-CNG candidates; enforce SMILES validity
   checks at manifest build time (valid_fraction must be 1.0 for every
   source, or invalid candidates explicitly quarantined and reported).
2. Re-run P4-G3 A6 arm on v2.
3. Only then execute the full P4-G4 matrix (5 sources × 4 scorers ×
   3 protocols), including Protocol C difficulty matching with an
   independent difficulty scorer.

`next_phase_allowed: false` until remediation completes.

## 8. Reproducibility

- Script: `pc_cng/run_p4_g4_diagnostic.py`
- Tests: `tests/test_generator_scorer_matrix.py` — **23 passed, 2 skipped**
  (GPU-dependent) in env `pc_cng_gpu` (Python 3.10.20, torch 2.6.0+cu124,
  statsmodels 0.14.6, RDKit 2025.03.6).
- Command:

```bash
python3 -m chem_negative_sampling.pc_cng.run_p4_g4_diagnostic \
  --manifest data/p4/manifests/hte_feasibility_v1.json \
  --output-dir results/p4_generator_scorer_matrix \
  --stage full --device cuda:0
# diagnostics-only regeneration (reuse existing MLP runs):
#   add --skip-mlp
```

- Outputs (`results/p4_generator_scorer_matrix/`): `summary.csv` (210 cells),
  `mlp_summary.csv` (70 runs), `effect_sizes.csv`, `interaction_model.json`,
  `difficulty_profile.json`, `raw_predictions/` (70 MLP dirs),
  `go_no_go.json`, `run_manifest.json`, `environment.json`,
  `input_hashes.json`, `commands.log`.
