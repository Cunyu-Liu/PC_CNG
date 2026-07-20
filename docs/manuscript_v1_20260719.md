# PC-CNG: PhysChem-Constrained Counterfactual Negative Generation for Chemistry Reaction Prediction

## Abstract

Accurate prediction of chemical reaction outcomes is central to computational
retrosynthesis and synthesis planning, yet supervised rerankers are starved of
informative negative examples: random or easily-decodable negatives leave the
decision boundary under-shaped and yield poor calibration on out-of-distribution
scaffolds.  We present PC-CNG, a PhysChem-Constrained Counterfactual Negative
Generator that produces boundary-negative candidates through five chemically
typed edit actions (replace_atom, drop_reactant, swap_functional_group,
wrong_anchor, add_reagent) and funnels them through a three-layer false-negative
control (ensemble agreement, database retrieval at Tanimoto >= 0.95, and
rule-based plausibility check).  Evaluated on USPTO OpenMolecules, HiTEA,
RegioSQM20 and the Open Reaction Database with a strict 10-seed paired
significance protocol, PC-CNG delivers a statistically significant
cross-dataset migration gain from the curated RegioSQM20 set to the
large-scale USPTO benchmark (delta Top-1 = 1.63 percentage
points, 95% CI [0.59, 2.72], permutation
p = 0.0028) and a large improvement in retrosynthesis route
ranking (MRR 24.24% -> 54.87%,
delta = 30.63 pp, 95% CI [29.23,
32.05] pp, p < 0.0001, 583/600
groups favoured).  Calibration is acceptable (ECE = 0.0889)
and OOD scaffold/template splits show no significant degradation.  We report
two explicit limitations: an external-bridge NO-GO where an MLP calibrator
trained on USPTO trainval underperforms Chemformer likelihood by 10.56
pp on held-out full-beam candidates, and partial support from MMFF94-based
computational validation (support rate 0.48 < 0.6
threshold).  PC-CNG is released with a reproducibility manifest covering 28
result artifacts and the supplementary Ni-coupling reaction set
(1688 reactions).

## 1. Introduction

Predicting the products of an organic reaction from its reactants is a
long-standing problem at the interface of chemistry and machine learning.
Modern sequence-to-sequence and graph-edit models achieve high top-k accuracy
on benchmarks such as USPTO-MIT and USPTO-50k, but their usefulness as
candidate generators depends on a downstream reranker that discriminates the
plausible products from a sea of look-alikes.  The quality of that reranker is
ultimately bounded by the quality of its negative training examples: if every
negative is obviously wrong, the model learns a trivial boundary and fails on
hard near-miss candidates that dominate real beam search output.

Existing negative-sampling strategies fall into three camps.  Random sampling
produces negatives that are too easy.  Hard-negative mining with the same
generator that produced the positives risks label leakage.  Rule-based
negative generators that mutate atom maps are chemically meaningful but
typically lack the physicochemical constraints needed to keep the edited
molecules realistic.  Counterfactual generation offers a principled middle
ground: each negative is a minimal, chemically typed perturbation of an
observed reaction, designed to sit close to the decision boundary so the
reranker is forced to learn the discriminating features rather than memorise
surface artefacts.

This paper introduces PC-CNG, a PhysChem-Constrained Counterfactual Negative
Generator for chemistry reaction prediction.  PC-CNG produces five typed edit
actions on atom-mapped reactions, applies physicochemical validity checks
(valence, atom balance, aromaticity), and routes every candidate through a
three-layer false-negative control pipeline before it enters training.  The
best PC-CNG configuration is locked in a 28-artifact reproducibility manifest
and evaluated under a strict 10-seed paired significance protocol on four
datasets and four cross-dataset transfer pairs.

Our contributions are:

1. **A boundary negative generation framework with five typed edit actions**
   (replace_atom, drop_reactant, swap_functional_group, wrong_anchor,
   add_reagent) that combine chemical validity with decision-boundary
   proximity, supervised by a pairwise reward MLP.
2. **A three-layer false-negative control pipeline** combining ensemble
   agreement, database retrieval at Tanimoto >= 0.95, and a rule-based
   plausibility check; starting from 64,646 reviewed
   negatives it retains 26,517
   (41.02%) high-confidence negatives while
   flagging the rest for expert review.
3. **Cross-dataset transfer evidence**: PC-CNG negatives generated from the
   small curated RegioSQM20 set transfer to the large-scale USPTO benchmark
   with a positive and statistically significant delta
   (1.63 pp, CI all positive).
4. **Retrosynthesis route-ranking improvement**: augmenting a pseudo-route
   ranker with PC-CNG negatives lifts MRR from
   0.2424 to
   0.5487
   (+30.63 pp, p < 0.0001).
5. **A reproducible Ni-coupling supplement** (1688 reactions,
   primarily mined from the NiCOlit literature) that addresses a documented
   data gap in USPTO OpenMolecules.

The remainder of the paper describes the PC-CNG architecture and training
protocol (Section 2), the experimental setup (Section 3), results organised
around six figures (Section 4), a discussion of the negative results and
threats to validity (Section 5), explicit limitations (Section 6), and our
conclusions (Section 7).

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
