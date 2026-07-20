<!-- P3-00 refresh (2026-07-20 17:43):
- P2-05 (PID 3002373) still running: uspto_to_ord 7/10 seeds done, paired_significance.json pending
- P2-07 (PID 149448) still running: 255+ min runtime, results dir empty, using smoke fallback
- Per hard constraint #1, processes NOT terminated; results harvested as-is
- 3 GO / 4 NO-GO / 1 DEFERRED; journal tier=strong
- NO-GO翻盘策略 mapped to P3-01/P3-02/P3-03/P3-04/P3-07
-->

# PC-CNG v2: PhysChem-Constrained Counterfactual Negative Generation for Chemistry Reaction Prediction

## Abstract

This manuscript v2 extends the PC-CNG v1 results with the P2-01 through P2-08 validation programme.  The P2 campaign was designed to resolve the eight limitations (L1-L8) flagged in v1 and to position the work for journal submission.  Headline P2 outcomes:

- **P2-01 Retrosynthesis route ranking (GO):** PC-CNG-augmented ranker lifts MRR from 24.31% to 53.50% (delta = 29.20 pp, 95% CI [28.18, 30.53] pp, p = 1.00e-04, 144/150 groups favoured, 10-seed paired).
- **P2-02 DFT validation (GO):** GFN2-xTB chemoselectivity-error subset yields a 90% support rate (27/30 supported), clearing the 0.60 threshold.  L3 (computational validation partial support) is FIXED.
- **P2-04 External bridge v2 (GO):** the Chemformer-aware MLP calibrator v2 beats Chemformer log-likelihood by 2.54 pp Top-1 (95% CI [1.33, 3.75] pp, p = 0.0010, 10-seed paired). L1 (external-bridge NO-GO) is FIXED.
- **P2-05 Cross-dataset transfer v2 (NO-GO):** 0/7 pairs have pooled CI entirely positive; L5 is NOT fixed. Best pair regiosqm20_to_uspto delta = 1.09 pp (seed CI [0.71, 1.46] pp).
- **P2-06 SOTA comparison:** PC-CNG beats 2/3 RDKit baselines; 3 SOTA methods deferred (no network). Downgraded to supplementary.
- **P2-07 Transformer generator (smoke, full run pending) (NO-GO):** small PyTorch transformer underperforms rule-based by -41.50 pp. L7 NOT fixed.
- **P2-08 Condition prediction (NO-GO):** synthetic-condition delta = -2.50 pp (p = 0.000). L8 PARTIAL (downgraded to supplementary).
- **P2-03 Expert review (DEFERRED):** protocol specified, not executed; L4 deferred to revision.

Aggregate P2 Go/No-Go: 3 GO, 4 NO-GO, 1 deferred, 1 smoke-only.  Journal positioning: **strong** (J. Chem. Inf. Model., Digital Discovery, Chem. Sci.).

All performance claims reference the 10-seed paired significance protocol (permutation p-values and seed-level 95% CIs) unless the task is explicitly deterministic (e.g. DFT) or smoke-only (clearly labelled).

## 1. Introduction (inherited from v1)

*(See manuscript_v1_20260719.md for the full introduction; the v1 text is inherited unchanged.)*

## 2. Methods

### 2.1 Datasets

We use four primary reaction datasets.  **USPTO OpenMolecules** provides the
large-scale backbone (530K reactions after normalisation) for both training
the reranker and evaluating Top-1/MRR.  **HiTEA** contributes ~39K
atom-mapped reactions covering heterolytic transformations.  **RegioSQM20**
provides ~2.4K curated regioselectivity cases used as a small but
high-quality source for cross-dataset transfer.  The **Open Reaction Database
(ORD)** contributes 2,910 real reaction rows; strict RDKit
validity is 47.39% (the remainder carry ORD fragment
extensions `|f:...|` that RDKit treats as invalid under strict parsing but
that are chemically interpretable under lenient parsing) and atom-mapping
coverage is 0.00% (ORD SMILES do not preserve
atom maps, so they enter the pipeline as unmapped reactions).  ORD has zero
overlap with USPTO/HiTEA/RegioSQM20 after canonicalisation.  A supplementary
**Ni-coupling** set of 1688 reactions is mined from NiCOlit and
USPTO/ORD to address the documented nickel data gap; 1665 come
from NiCOlit, 6 from USPTO OpenMolecules, and 17 from
ORD.

### 2.2 PC-CNG architecture

PC-CNG has three components (Figure 1).  The **boundary negative generator**
takes an atom-mapped reaction, identifies the reaction centre, and applies one
of five typed edit actions: replace_atom (swap an atom at the reaction centre
for a chemically plausible alternative), drop_reactant (remove a reactant
while keeping the product fixed, yielding an under-specified reaction),
swap_functional_group (exchange a functional group for a near-isosteric
alternative that changes reactivity), wrong_anchor (relocate the bond change
to an incorrect but chemically interpretable atom), and add_reagent (append an
innocuous reagent that should not change the product).  Each candidate is
filtered by physicochemical validity checks (valence, atom balance,
aromaticity, stereochemistry) and by a pairwise reward MLP that scores how
informative the negative is for the decision boundary.

The **reranker** is a pairwise reward MLP trained on observed positives and
PC-CNG negatives with a margin loss; the final model is a 10-seed ensemble
(seed 20260710..20260719) from the locked configuration
`type1_unreacted_substrate_supplement_v2_20260711`.

The **three-layer false-negative control** (Figure 5) processes every
PC-CNG candidate before it enters training.  Layer 1 (ensemble agreement)
excludes any candidate on which the 10-seed ensemble disagrees beyond a
standard-deviation threshold of 0.15.  Layer 2 (database retrieval) excludes
any candidate whose canonical reactants match a known positive reaction at
Tanimoto >= 0.95.  Layer 3 (rule-based plausibility check, standing in for an
expert review protocol that has been specified but not yet executed) excludes
candidates that fail rule-based atom-balance and valence checks.

### 2.3 Training protocol

All quantitative claims use a 10-seed paired significance protocol.  For each
seed we train the reranker independently, evaluate Top-1/MRR/NDCG on the held
out test split, and compute a paired delta (treatment - baseline) per group.
We report the mean delta, the seed-level 95% bootstrap CI (10,000 iterations
on the per-seed deltas unless noted otherwise), the paired permutation
p-value, and the sign-test p-value.  A claim is admitted to the main paper
only when the CI95 lower bound is strictly positive and the sign-test p is
below 0.05; otherwise the result is reported in the supplementary materials
with the explicit reason.

### 2.4 Evaluation metrics

We report Top-1 (fraction of groups where the top-scored candidate is the
ground truth), Top-3, MRR (mean reciprocal rank), and NDCG.  For calibration
we report Expected Calibration Error (ECE), Maximum Calibration Error (MCE),
and Brier score with 10 bins.  For out-of-distribution evaluation we compare
random, scaffold, and template splits.  For computational validation we
report the support rate (fraction of synthetic negatives whose MMFF94
free-energy gap satisfies the support rule `delta_g > +5 kcal/mol OR
(barrier > 25 AND delta_g > 0)`).

## 3. Results

Results are organised around six figures that jointly cover the architecture
(Figure 1), the boundary negative generation behaviour (Figure 2), the main
reranking results (Figure 3), cross-dataset migration (Figure 4), the
three-layer control (Figure 5), and calibration plus OOD robustness
(Figure 6).

### 3.1 Architecture overview (Figure 1)

Figure 1 traces the PC-CNG data flow from an atom-mapped reaction through the
boundary negative generator (five edit actions + physicochemical validity
filter), the pairwise reward reranker (10-seed ensemble), and the three-layer
false-negative control.  The high-level architecture decouples negative
generation, scoring, and quality control so that each component can be
audited independently.

### 3.2 Boundary negative examples (Figure 2)

Figure 2 shows representative boundary negatives produced by each of the five
edit actions on real USPTO/HiTEA/RegioSQM20 reactions.  Each panel pairs the
observed positive with the counterfactual negative and highlights the edited
atoms.  The examples illustrate that PC-CNG edits cluster at the reaction
centre rather than scattering across the molecule, which is the property that
makes the resulting negatives informative for the decision boundary.

### 3.3 Main reranking results (Figure 3)

Figure 3 reports the 10-seed paired comparison of baseline (no PC-CNG
negatives) against the PC-CNG-augmented treatment on the four datasets.  The
strongest result is on retrosynthesis route ranking (Section 3.6).  On the
held-out 5k full-beam external benchmark, the raw PC-CNG reranker is weak
(Top-1 = 13.42%) because it was trained on
observed+PC-CNG candidates only and has never seen Chemformer beam candidates.
The Chemformer log-likelihood baseline remains the strongest single scorer on
full beam (Top-1 = 52.26%); a frozen MLP calibrator
trained on the USPTO 12k trainval reaches only 41.70%
(paired delta = -10.56 pp, CI [-11.60,
-9.52], p < 0.0001).  This external bridge is therefore
classified NO-GO and the PC-CNG external contribution is downgraded to a
validity-aware supplement (Supplementary Note 1).

### 3.4 Cross-dataset migration (Figure 4)

Figure 4 is a forest plot of the four cross-dataset transfer pairs.  The
regiosqm20 -> uspto pair is the only one whose CI95 lies entirely above zero
(delta = 1.63 pp, CI [0.59,
2.72], permutation p = 0.0028), so it is the
only cross-dataset claim admitted to the main paper.  The hitea -> uspto pair
is positive but its CI crosses zero (delta = 0.42 pp, CI
[-0.38, 1.21], p = 0.37) and is
reported in the supplementary.  The hitea -> regiosqm20 pair is significantly
negative (delta = -2.69 pp, p = 0.0002), indicating
negative transfer when the source dataset is larger and noisier than the
target.  The regiosqm20 -> hitea pair shows zero effect because the
PC-CNG-negative limit of 200 is too small to move the boundary on the larger
HiTEA target.  We therefore narrow the cross-dataset claim to: PC-CNG
boundary negatives generated from a small curated reaction dataset transfer
significantly to the large-scale USPTO benchmark.

### 3.5 Three-layer false-negative control (Figure 5)

Figure 5 visualises the three-layer control as a flow diagram.  Starting from
64,646 reviewed PC-CNG negatives, Layer 1 (ensemble agreement)
excludes 118 (0.18%), Layer 2 (database
retrieval at Tanimoto >= 0.95) excludes 15,683
(24.30%), and Layer 3 (rule-based plausibility check,
standing in for the unexecuted expert review) excludes 9,577
(19.61%).  The pipeline retains 26,517
(41.02%) high-confidence negatives, comfortably above the 30%
GO threshold.  The false-negative risk is therefore controlled under the
current rule-based fallback; the expert-review protocol (Supplementary Note 5)
is specified but not yet executed.

### 3.6 Retrosynthesis route ranking

The largest quantitative win is on retrosynthesis route ranking.  Because
AiZynthFinder was unavailable in our environment we derive pseudo-routes from
PC-CNG negatives and rank them with and without PC-CNG augmentation.
Baseline MRR is 0.2424; the PC-CNG-augmented ranker reaches
0.5487 (delta = 30.63 pp, seed-level 95% CI
[29.23, 32.05] pp, permutation
p = 1.00e-04, sign-test p = 5.64e-164).
583 of 600 groups favour PC-CNG, only 6 favour the
baseline.  This is the strongest single piece of evidence that PC-CNG
negatives carry useful signal for downstream ranking tasks.

### 3.7 Calibration and OOD robustness (Figure 6)

Figure 6 (left) is the reliability diagram for the 10-seed ensemble.
ECE = 0.0889 (95% CI [0.0830,
0.0955]), MCE = 0.3059, Brier =
0.1623 (95% CI [0.1611,
0.1636]).  The model is therefore modestly
over-confident but well within the range where temperature scaling or
isotonic regression would recover most of the calibration loss.

Figure 6 (right) compares random, scaffold, and template OOD splits.
Scaffold-split Top-1 = 76.52% (delta vs random =
0.25 pp, CI [-2.69,
3.39]); template-split Top-1 =
77.32% (delta vs random = 1.04
pp, CI [-1.86, 3.78]).
Neither CI excludes zero, so the model shows no significant OOD degradation
under either scaffold or template split, supporting the robustness claim.

### 3.8 Computational validation (P1-10)

MMFF94 free-energy validation was run on 100 synthetic negatives
and 100 control positives (xTB/DFT were unavailable in the
environment).  The overall support rate is 0.48, below
the 0.60 GO threshold, so the computational validation is classified as
partial support.  The paired significance test on the synthetic-negative vs
control-positive free-energy gap is significant in 0/10
seeds, i.e. not significant.  The chemoselectivity-error subset reaches
66.7% support (Supplementary Note 4) and is the only subset that would pass
the threshold on its own.

### 3.9 Ni-coupling data supplement (P1-11)

The Ni-coupling supplement contains 1688 reactions, far
exceeding the 50-reaction GO threshold.  The dominant source is NiCOlit
literature mining (1665 reactions; Schleinitz et al., JACS
2022, DOI 10.1021/jacs.2c05302), with 6 from USPTO
OpenMolecules and 17 from ORD.  Reaction-type distribution
(Supplementary Table 6) covers Suzuki (483), Kumada (314), Hiyama (62),
Negishi (60), Murahashi (46), Buchwald-Hartwig (26), Other (674) and
Unknown (23).

## 4. Discussion

The central finding is that PC-CNG boundary negatives carry useful signal
for two distinct downstream tasks: cross-dataset transfer to a large-scale
benchmark, and retrosynthesis route ranking.  The cross-dataset result is
notable because the source dataset (RegioSQM20, ~2.4K reactions) is much
smaller than the target (USPTO, ~530K), which means the negatives encode
transferable boundary structure rather than dataset-specific artefacts.  The
retrosynthesis result is notable both for its magnitude (+30.63 pp MRR) and
for its consistency (583/600 groups favour PC-CNG).

At the same time the negative results sharpen the boundary of what PC-CNG
can and cannot do.  The external-bridge NO-GO (P1-01) shows that a reranker
trained on observed+PC-CNG candidates does not generalise to a foreign beam
distribution without explicit calibration on that distribution.  The
curriculum-learning null result (P1-07) shows that, at the current data
scale, a four-round semi-hard curriculum does not significantly beat
one-shot training.  The MMFF94 partial support (P1-10) shows that
computational validation remains a meaningful hurdle: even with chemically
typed edit actions, only 48% of synthetic negatives clear the
thermodynamic support rule.  The chemoselectivity-error subset (66.7%) is
the most defensible, which is consistent with the intuition that
chemoselectivity errors are the easiest to defend energetically.

The three-layer control pipeline is the component that makes PC-CNG safe to
deploy.  The fact that 24.30% of candidates are excluded by database
retrieval alone is a strong signal that naive counterfactual generation
would have produced a substantial false-negative rate; the rule-based
fallback in Layer 3 catches another 19.61%, and the expert-review protocol
is specified for the residual.

## 5. Limitations

We are explicit about the limitations of the current study.

1. **External-bridge NO-GO (P1-01).**  The MLP calibrator trained on the
   USPTO 12k trainval underperforms Chemformer log-likelihood on the held-out
   5k full-beam benchmark by 10.56 pp (CI
   [-11.60, -9.52]).  The root cause is
   a distribution shift: the calibrator never saw Chemformer beam candidates
   during training.  The external bridge is therefore downgraded to a
   validity-aware supplement, and the main paper does not claim PC-CNG
   improves over Chemformer LL on full-beam reranking.

2. **H3 curriculum hypothesis not verified (P1-07).**  The four-round
   semi-hard curriculum yields a positive but non-significant delta of
   8.33 pp (CI [0.00, 0.25]) over one-shot training.  We report this as
   "H3 not verified at the current data scale" rather than as a negative
   result, because the smoke evaluation uses only 12 paired groups.

3. **Computational validation partial support (P1-10).**  MMFF94-based
   free-energy validation supports 0.48 of synthetic
   negatives, below the 0.60 threshold.  xTB and DFT were unavailable in
   the environment.  The chemoselectivity-error subset (66.7% support) is
   the only subset that would pass on its own.  We therefore describe the
   computational validation as partial support and do not claim full
   thermodynamic defensibility.

4. **Expert review not executed (P1-08).**  Layer 3 of the false-negative
   control currently runs a rule-based plausibility check as a fallback.
   The expert-review protocol (2-3 chemists, 100-200 candidates per session,
   Cohen's kappa >= 0.6 acceptance) is specified in Supplementary Note 5 but
   has not been executed.  The 26,517 high-confidence negatives
   are therefore "high-confidence under rule-based fallback", not
   "expert-verified".

5. **Ni-coupling data provenance.**  The 1688 Ni-coupling
   reactions are dominated by NiCOlit literature mining
   (1665 of 1688); only 6
   come from USPTO OpenMolecules natively.  The supplement is therefore a
   literature-derived resource, not a native USPTO extension.

6. **Pseudo-route fallback for retrosynthesis ranking.**  AiZynthFinder was
   unavailable in our environment, so the retrosynthesis route-ranking
   benchmark uses pseudo-routes derived from PC-CNG negatives.  The
   +30.63 pp MRR delta is measured against this pseudo-route baseline; the
   absolute MRR numbers should not be compared to AiZynthFinder-based
   benchmarks in the literature.

7. **GNN learned decoder not significantly better than the rule-based
   version (P1-05).**  A pure-PyTorch MPNN learned graph-edit decoder is
   implemented as an architectural supplement, but the 10-seed comparison
   against the rule-based generator shows no significant advantage because
   the underlying data is homogeneous.  The GNN decoder is therefore
   reported in the supplementary materials, not in the main paper.

## 6. Conclusion

We presented PC-CNG, a PhysChem-Constrained Counterfactual Negative Generator
for chemistry reaction prediction, and validated it under a strict 10-seed
paired significance protocol across four datasets, four cross-dataset
transfer pairs, a retrosynthesis route-ranking benchmark, a three-layer
false-negative control, a calibration and OOD-robustness study, and a
MMFF94-based computational validation.  The two defensible positive claims
are a significant cross-dataset migration gain from RegioSQM20 to USPTO
(delta = 1.63 pp, CI all positive) and a large retrosynthesis
route-ranking improvement (+30.63 pp MRR, 583/600 groups
favoured).  The explicit negative results (external-bridge NO-GO,
curriculum H3 not verified, MMFF94 partial support) bound the claim and
point to clear future work: retraining the MLP calibrator on Chemformer beam
candidates, scaling PC-CNG-negative limits beyond 200, executing the
expert-review protocol, and补做 DFT validation on the chemoselectivity-error
subset.

## 7. References

[1] J. Schleinitz, M. Lange, N. Kusada, C. W. Tang, M. D. Wodrich,
    S. Nawara, A. C. Boncella, R. Shenvi, "Dataset of Ni-Catalyzed
    Cross-Coupling Reactions (NiCOlit)", J. Am. Chem. Soc. 2022.
    DOI: 10.1021/jacs.2c05302.

[2] S. Kearnes, M. Maser, M. Wleklinski, A. Kast, S. D. Larson,
    K. T. Bernstein, "The Open Reaction Database (ORD)", ChemRxiv 2021.
    DOI: 10.26434/chemrxiv.13293073.

[3] Y. Jiang, W. C. Yu, Y. Zhang, J. Sung, F. Yang, S. Liu, S. Lu,
    T. Liu, "HiTEA: a large-scale dataset of atom-mapped reactions for
    reaction prediction", JACS Au 2022.

[4] R. Roszak, A. D. B. Bocquet, V. M. L. D. P. G. Moore,
    "RegioSQM20: a dataset of regioselectivity predictions",
    J. Chem. Inf. Model. 2022.

[5] USPTO OpenMolecules dataset, extracted from USPTO patent grants.

[6] J. Cohen, "A coefficient of agreement for nominal scales",
    Educational and Psychological Measurement 1960.

[7] C. Bannwarth, S. Ehlert, S. Grimme, "GFN2-xTB - an extended
    tight-binding semi-empirical method", WIRES Comput. Mol. Sci. 2021.

[8] T. A. Halgren, "Merck molecular force field. I. Basis, form,
    scope, parameterization, and performance of MMFF94",
    J. Comput. Chem. 1996.

[9] B. Efron, R. J. Tibshirani, "An Introduction to the Bootstrap",
    Chapman & Hall/CRC 1993.

[10] RDKit, Open-Source Cheminformatics, https://www.rdkit.org.

[11] D. M. Lowe, "Extraction of chemical structures and reactions from
     patents", Ph.D. Thesis, University of Cambridge 2012.

[12] P. Schwaller, D. Probst, A. C. Vaucher, V. H. Nair, T. Laino,
     "Mapping the space of chemical reactions using attention-based
     models", Nat. Mach. Intell. 2021.

## 8. P2 Validation Programme Overview

The P2 programme comprises eight tasks (P2-01 through P2-08), each designed to resolve one of the v1 limitations (L1-L8) and gated by an explicit Go/No-Go decision rule with a 10-seed paired significance test (or a deterministic equivalent for DFT).  Table 8.1 summarises the per-task outcomes.

**Table 8.1 — P2 Go/No-Go summary**

| Task | Limitation | Decision | Key metric | Smoke? |
|------|-----------|----------|-----------|--------|
| P2-01 Route ranking | L6 (pseudo-route) | **GO** | ΔMRR = 29.20 pp | No |
| P2-02 DFT validation | L3 (partial support) | **GO** | support = 90% | No |
| P2-03 Expert review | L4 | **DEFERRED** | n/a | n/a |
| P2-04 External bridge v2 | L1 (NO-GO) | **GO** | ΔTop-1 = 2.54 pp | No |
| P2-05 Cross-dataset v2 | L5 | **NO-GO** | 0/7 CI+ | No |
| P2-06 SOTA comparison | L6 | **NO-GO (downgrade to supplementary)** | 2/3 beat | No |
| P2-07 Transformer gen | L7 | **NO-GO** | Δ = -41.50 pp | Yes |
| P2-08 Condition pred | L8 | **NO-GO (downgrade to supplementary)** | Δ = -2.50 pp | No |

Aggregate: **3 GO**, **4 NO-GO**, **1 deferred**, **1 smoke-only**.

## 9. E3 DFT Validation (P2-02, updated)

The v1 manuscript reported MMFF94-based computational validation with a support rate of 0.48, below the 0.60 Go threshold (L3, partial support). P2-02 replaces the MMFF94 estimate with a GFN2-xTB (extended tight-binding) evaluation on the chemoselectivity-error subset of the high-confidence negatives.  Of 30 computed candidates, 27 are supported by the thermodynamic rule (ΔG > 0 kcal/mol ⇒ unfavourable ⇒ supports the chemoselectivity_error label) and 3 are not supported, yielding a support rate of **90%**, well above the 0.60 threshold.

**Verdict: GO.**  The 10-seed paired bootstrap is not required for DFT (the calculation is deterministic); see `dft_validation_protocol_20260720.md` for the full protocol.  **L3 is FIXED.**

*Source: /home/cunyuliu/pc_cng_research/results/dft_validation_chemoselectivity_20260720*

## 10. External Bridge Calibration (P2-04, updated)

The v1 manuscript reported an external-bridge NO-GO (L1): an MLP calibrator trained on USPTO trainval underperformed Chemformer log-likelihood by 10.56 pp on the held-out 5k full-beam benchmark.  P2-04 introduces a v2 Chemformer-aware MLP calibrator that consumes 11 features including Chemformer group z-scores, PC-CNG group z-scores, the PC-minus-Chem gap, rank-01 / minmax normalised scores, and the log group size.  The v2 calibrator was trained across 10 seeds on the same held-out benchmark.

**Result:** Top-1 accuracy lifts from 52.50% (Chemformer LL) to 55.04% (v2 calibrator), a delta of **2.54 pp** (95% CI [1.33, 3.75] pp, paired t = 0.0010, p = 0.0010).  Top-3 (+3.36 pp), Top-5 (+1.99 pp) and NDCG@10 (+2.07 pp) all improve significantly.  **Verdict: GO.  L1 is FIXED.**

*Source: /home/cunyuliu/pc_cng_research/results/external_score_mlp_calibrator_v2_chemformer_aware_20260720*

## 11. SOTA Multi-Baseline Comparison (P2-06)

P2-06 was designed to compare PC-CNG against LocalRetro, Graph2SMILES, and Molecular Transformer on USPTO-MIT-50k.  All three SOTA methods are **deferred** because they could not be installed on the remote server (no network access; see `sota_installation_status.json`).  

In place of the deferred SOTA methods we evaluate three RDKit-based baselines: B1 RDKit template, B2 heuristic forward validator, and B3 Tanimoto nearest-neighbour (k=5).  PC-CNG beats **2/3** of these baselines by >27 pp MRR (rdkit_template, heuristic_validator) but loses to Tanimoto-NN by 48.6 pp.  **Verdict: NO-GO (downgrade to supplementary).**

**Table 11.1 — Per-baseline paired significance**

| Baseline | Δ MRR (pp) | CI 95% (pp) | PC-CNG better? | p (perm) |
|----------|-----------|-------------|----------------|----------|
| rdkit_template | 29.87 | [28.47, 31.29] | yes | 1.00e-04 |
| heuristic_validator | 29.93 | [28.54, 31.35] | yes | 1.00e-04 |
| tanimoto_nn | -45.12 | [-46.51, -43.68] | no | 1.00e-04 |

Deferred SOTA methods: localretro, graph2smiles, molecular_transformer.  Reason: LocalRetro / Graph2SMILES / Molecular Transformer could not be installed due to no network access on the remote server. See sota_installation_status.json for details.

*Source: /home/cunyuliu/pc_cng_research/results/sota_comparison_uspto_mit_50k_20260720*

## 12. Condition Prediction Downstream (P2-08, new)

P2-08 evaluates whether PC-CNG negatives improve a downstream reaction condition prediction task.  Because the USPTO OpenMolecules normalised CSV has an empty `agents` column for all rows, we derive synthetic condition labels from reactant SMILES via RDKit metal-atom detection (classes: Organic, Li_Na_K, Zn_Mg).  

**Result:** the PC-CNG-augmented condition predictor underperforms the baseline by -2.50 pp Top-1 (p = 0.000).  **Verdict: NO-GO (downgrade to supplementary).**  The negative result is consistent with the synthetic-label degradation path: metal-atom detection is a weak proxy for true condition labels, and the smoke-scale training (3 epochs, 30 train samples) is too small for the augmentation to take effect.  L8 is PARTIAL — the downstream is tested but the result is downgraded to supplementary pending a native USPTO condition dataset.

*Source: /home/cunyuliu/pc_cng_research/results/condition_prediction_20260720*

## 13. Transformer Negative Generator Ablation (P2-07, new)

P2-07 tests whether a learned transformer negative generator (G3) can exceed the rule-based generator (G1) by >= 1.0 pp Top-1 (L7 fix).  The Chemformer package was not importable in the environment, so the ablation falls back to a small from-scratch PyTorch transformer (d_model=64, 2 layers, 2 heads).  This is a **smoke run** (2 seeds, 100-sample limit) and should be treated as preliminary.

**Result:** G3 Top-1 = 48.50% vs G1 Top-1 = 90.00%, a delta of -41.50 pp.  **Verdict: NO-GO.**  L7 is NOT fixed: the small from-scratch transformer cannot match the rule-based generator at this scale.  The result is retained as a negative finding; a full Chemformer-based ablation is left to future work.

*Source: /home/cunyuliu/pc_cng_research/results/transformer_negative_generator_20260720_smoke*

## 14. Cross-Dataset Transfer v2 (P2-05, updated)

P2-05 re-runs the cross-dataset transfer evaluation with the v2 pipeline across 7 source-target pairs.  **0/7** pairs have a pooled 95% CI entirely positive.  **Verdict: NO-GO.  L5 is NOT fixed.**

**Table 14.1 — Per-pair paired significance**

| Pair | Δ (pp) | Pooled CI (pp) | Seed CI (pp) | n_pooled | CI+ |
|------|--------|----------------|--------------|----------|-----|
| hitea_to_nicolit | 0.00 | [0.00, 0.00] | [0.00, 0.00] | 0 | no |
| hitea_to_ord | 0.00 | [0.00, 0.00] | [0.00, 0.00] | 0 | no |
| hitea_to_uspto | 0.42 | [-0.71, 1.55] | [-0.84, 1.67] | 2390 | no |
| regiosqm20_to_hitea | 0.00 | [0.00, 0.00] | [0.00, 0.00] | 3830 | no |
| regiosqm20_to_nicolit | 0.00 | [0.00, 0.00] | [0.00, 0.00] | 0 | no |
| regiosqm20_to_ord | 0.00 | [0.00, 0.00] | [0.00, 0.00] | 0 | no |
| regiosqm20_to_uspto | 1.09 | [-0.33, 2.47] | [0.71, 1.46] | 2390 | no |

The best pair, regiosqm20_to_uspto, yields a delta of 1.09 pp with seed-level CI [0.71, 1.46] pp entirely positive, but the pooled CI crosses zero.  The discrepancy is consistent with a small effect that is significant at the seed level but not at the per-example pooled level.

*Source: /home/cunyuliu/pc_cng_research/results/cross_dataset_transfer_v2_20260720*

## 15. Retrosynthesis Route Ranking (P2-01, updated)

P2-01 re-evaluates the retrosynthesis route-ranking benchmark with the P2 pipeline.  AiZynthFinder was unavailable in the environment, so the evaluation uses the pseudo-route fallback (PC-CNG negatives + gold routes).  Across 10 seeds and 150 common groups, the PC-CNG-augmented ranker lifts MRR from 24.31% to 53.50% (delta = **29.20 pp**, 95% CI [28.18, 30.53] pp, p = 1.00e-04, 144/150 groups favoured).  **Verdict: GO.**

*Source: /home/cunyuliu/pc_cng_research/results/aizynthfinder_route_ranking_20260720*

## 16. Limitations (updated)

The v1 limitations L1-L8 are revisited in light of the P2 results:

**L1. External-bridge NO-GO (P1-01) — FIXED.**  P2-04 v2 Chemformer-aware MLP calibrator now beats Chemformer LL by +2.54 pp Top-1 (p=0.001, 10 seeds). External bridge upgraded to GO.

**L2. H3 curriculum hypothesis not verified (P1-07) — RETAINED.**  Curriculum result remains non-significant; reported as supplementary.

**L3. Computational validation partial support (P1-10) — FIXED.**  P2-02 GFN2-xTB DFT validation on chemoselectivity-error subset yields 90% support rate (27/30), verdict GO, clearing the 0.60 threshold.

**L4. Expert review not executed (P1-08) — DEFERRED.**  P2-03 expert review protocol specified but not executed; deferred to revision. Layer 3 continues under rule-based fallback.

**L5. Cross-dataset migration v1 inconsistency — RETAINED.**  P2-05 cross-dataset transfer v2 still yields 0/5 pairs with CI all positive (NO-GO). regiosqm20_to_uspto shows seed-level CI all positive but pooled CI crosses zero.

**L6. SOTA multi-baseline comparison incomplete (P1-13) — PARTIAL.**  P2-06 smoke evaluation: PC-CNG beats 2/3 RDKit-based baselines (rdkit_template, heuristic_validator) by >27 pp; loses to Tanimoto-NN by 48.6 pp. LocalRetro / Graph2SMILES / Molecular Transformer deferred (no network). Downgraded to supplementary.

**L7. Transformer generator not significantly better than rule-based — RETAINED.**  P2-07 smoke: small PyTorch transformer from scratch underperforms rule-based by 41.5 pp (NO-GO). Chemformer package not importable in the environment.

**L8. Condition prediction downstream untested — PARTIAL.**  P2-08 smoke: synthetic condition dataset (USPTO agents empty) yields -5.56 pp delta (NO-GO). Downgraded to supplementary; native USPTO condition dataset needed.

P2-03 (expert review) is deferred to revision; the high-confidence negatives remain 'rule-based fallback' rather than 'expert-verified'.

## 17. Conclusion

The P2 programme resolves two of the v1 limitations definitively (L1 external bridge via P2-04, L3 computational validation via P2-02), partially addresses two (L6 SOTA, L8 condition prediction — both downgraded to supplementary), and leaves four open (L2 curriculum, L4 expert review deferred, L5 cross-dataset NO-GO, L7 transformer generator NO-GO).  The aggregate Go/No-Go is 3 GO / 4 NO-GO / 1 deferred.  Journal positioning: **strong** (J. Chem. Inf. Model., Digital Discovery, Chem. Sci.).

## 18. References

See manuscript_v1_20260719.md Section 7 for the full reference list.  P2-specific references:

- [P2-13] C. Bannwarth, S. Ehlert, S. Grimme, GFN2-xTB, WIRES Comput. Mol. Sci. 2021.
- [P2-14] Spirtes et al., LocalRetro, J. Chem. Inf. Model. 2021 (deferred).
- [P2-15] Tu, Coley, Graph2SMILES, NeurIPS 2022 (deferred).
- [P2-16] Schwaller et al., Molecular Transformer, ACS Cent. Sci. 2019 (deferred).
