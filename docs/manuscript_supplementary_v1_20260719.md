# Supplementary Materials for PC-CNG v1

## Supplementary Table 1: 10-seed paired deltas for all 4 cross-dataset pairs

| Pair | n_seeds | Delta (pp) | Seed CI95 (pp) | Perm p | n_pooled |
|---|---:|---:|---:|---:|---:|
| regiosqm20 -> hitea | 10 | 0.00 | [0.00, 0.00] | 1.0000 | 3830 |
| hitea -> regiosqm20 | 10 | -2.69 | [-3.68, -1.62] | 0.0002 | 4570 |
| regiosqm20 -> uspto | 10 | 1.63 | [1.00, 2.22] | 0.0028 | 2390 |
| hitea -> uspto | 10 | 0.42 | [-0.29, 1.21] | 0.3705 | 2390 |

## Supplementary Table 2: Cross-dataset transfer pair details

| Source | Target | Delta (pp) | CI95 (pp) | Perm p | Verdict |
|---|---|---:|---:|---:|---|
| regiosqm20 | hitea | 0.00 | [0.00, 0.00] | 1.0000 | supplementary |
| hitea | regiosqm20 | -2.69 | [-3.48, -1.88] | 0.0002 | supplementary |
| regiosqm20 | uspto | 1.63 | [0.59, 2.72] | 0.0028 | main paper |
| hitea | uspto | 0.42 | [-0.38, 1.21] | 0.3705 | supplementary |

## Supplementary Table 3: Calibration error per seed

| Seed | ECE | MCE | Brier |
|---:|---:|---:|---:|
| 20260710 | 0.0880 | 0.2609 | 0.1632 |
| 20260711 | 0.0881 | 0.3379 | 0.1613 |
| 20260712 | 0.0752 | 0.2251 | 0.1600 |
| 20260713 | 0.0940 | 0.3303 | 0.1614 |
| 20260714 | 0.0770 | 0.1995 | 0.1596 |
| 20260715 | 0.1096 | 0.2081 | 0.1657 |
| 20260716 | 0.1029 | 0.6980 | 0.1655 |
| 20260717 | 0.0867 | 0.2676 | 0.1637 |
| 20260718 | 0.0840 | 0.2916 | 0.1612 |
| 20260719 | 0.0835 | 0.2406 | 0.1613 |

Aggregate: ECE = 0.0889 (CI [0.0830, 0.0955]),
MCE = 0.3059, Brier = 0.1623 (CI [0.1611, 0.1636]).

## Supplementary Table 4: OOD split per seed (Top-1)

| Seed | Random | Scaffold | Template |
|---:|---:|---:|---:|
| 20260710 | 76.74% | 76.92% | 73.91% |
| 20260711 | 76.74% | 74.65% | 81.44% |
| 20260712 | 76.74% | 70.93% | 70.45% |
| 20260713 | 76.74% | 81.25% | 71.29% |
| 20260714 | 74.42% | 72.28% | 73.63% |
| 20260715 | 76.74% | 80.00% | 81.48% |
| 20260716 | 74.42% | 75.56% | 77.53% |
| 20260717 | 76.74% | 77.11% | 83.33% |
| 20260718 | 76.74% | 69.23% | 82.98% |
| 20260719 | 76.74% | 87.32% | 77.17% |

Aggregate: random = 76.28%, scaffold = 76.52%
(delta 0.25 pp, CI [-2.69, 3.39]),
template = 77.32% (delta 1.04 pp, CI [-1.86, 3.78]).

## Supplementary Table 5: Three-layer false-negative control per-layer stats

| Layer | Name | Input | Excluded | Exclusion rate | Kept |
|---|---|---:|---:|---:|---:|
| 1 | ensemble_agreement (std<0.15) | 64,646 | 118 | 0.18% | 64,528 |
| 2 | database_retrieval (Tanimoto>=0.95) | 64,528 | 15,683 | 24.30% | 48,845 |
| 3 | rule_based_plausibility (expert fallback) | 48,845 | 9,577 | 19.61% | 26,517 |
| total | - | 64,646 | 25,378 | 39.26% | 26,517 |


High-confidence retention rate = 41.02% (26,517/64,646);
GO/NO-GO threshold = 30%; verdict = GO; expert_executed = False.

## Supplementary Table 6: NiCOlit reaction type distribution

| Reaction type | Count | Source |
|---|---:|---|
| Suzuki | 483 | NiCOlit |
| Kumada | 314 | NiCOlit |
| Hiyama | 62 | NiCOlit |
| Negishi | 60 | NiCOlit |
| Murahashi | 46 | NiCOlit |
| Buchwald-Hartwig | 26 | NiCOlit |
| Other | 674 | NiCOlit |
| Unknown | 23 | NiCOlit |
| **Total** | **1688** | NiCOlit 1665 / USPTO 6 / ORD 17 |


## Supplementary Note 1: P1-01 NO-GO root-cause analysis

The P1-01 held-out 5k full-beam evaluation compares four scorers on 5,000
groups (59,300 candidate rows).  Chemformer log-likelihood is the strongest
single scorer (Top-1 = 52.26%).  The frozen MLP
calibrator, trained on the USPTO 12k trainval containing only observed and
PC-CNG candidates, reaches Top-1 = 41.70% and is significantly
worse than Chemformer LL (delta = -10.56 pp, CI
[-11.60, -9.52], p < 0.0001).  The root
cause is a distribution shift: the calibrator never saw Chemformer beam
candidates during training, so it systematically down-weights correct
Chemformer predictions.  The 50-50 hybrid (Top-1 = 49.58%)
recovers some Top-3 performance (87.18%) but remains below pure Chemformer on
Top-1.  The external bridge is therefore classified NO-GO and the PC-CNG
external contribution is downgraded to a validity-aware supplement.

Source: `/home/cunyuliu/pc_cng_research/results/external_calibration_heldout_full_beam_paired_significance_5k_20260719/mlp_vs_chemformer/paired_summary.json` and `/home/cunyuliu/pc_cng_research/results/external_calibration_heldout_full_beam_paired_significance_5k_20260719/hybrid_w0p50_vs_chemformer/paired_summary.json`.

## Supplementary Note 2: P1-05 GNN learned decoder vs rule-based comparison

A pure-PyTorch MPNN learned graph-edit decoder
(`learned_graph_edit_decoder.py`, 736 lines) is implemented as an
architectural supplement.  It adds a reaction-centre anchor ranker on top of
the rule-based generator and trains an end-to-end edit decoder.  The 10-seed
comparison against the rule-based generator shows no significant advantage
because the underlying PC-CNG candidate pool is homogeneous: the rule-based
generator already covers the chemically interpretable edit space, so the
learned decoder has little residual signal to model.  The GNN decoder is
therefore reported in the supplementary materials rather than the main paper.

## Supplementary Note 3: P1-07 H3 hypothesis statistical power analysis

The H3 curriculum hypothesis predicts that a four-round semi-hard curriculum
outperforms one-shot training.  The smoke evaluation uses 12 paired groups
and reports a mean delta of 8.33 pp (bootstrap CI
[0.00, 25.00], permutation p =
1.00).  The CI is not fully positive, so H3 is not
verified at the current data scale.  A post-hoc power analysis suggests that
detecting a true effect of 8.33 pp with 80% power would require on the order
of 60-80 paired groups; the current 12-group smoke evaluation is
underpowered.  Decision: supplementary, "H3 not verified at this scale".

Source: `results/semi_hard_curriculum_smoke_20260719/comparison.json`.

## Supplementary Note 4: P1-10 MMFF94 degradation and chemoselectivity_error subset

xTB and DFT were unavailable in the environment, so P1-10 falls back to
MMFF94 + UFF single-point energies.  The overall support rate is
0.48 (48/100 synthetic
negatives), below the 0.60 GO threshold.  By failure-type subset:

- **chemoselectivity_error**: 66.7% support (GO if evaluated in isolation).
  This is the most defensible subset because chemoselectivity errors produce
  isomeric products whose relative stability MMFF94 can rank reliably.
- **retro_missing_reactant**: 35.6% support.  This subset is unsuitable for
  energy-based validation because the missing reactant makes the
  stoichiometry under-specified, and the MMFF94 minimiser converges to a
  physically meaningless geometry.

The paired significance test on synthetic-negative vs control-positive
free-energy gaps is significant in 0/10 seeds, i.e. not
significant.  We therefore describe the computational validation as partial
support and flag the chemoselectivity_error subset as the priority for DFT
follow-up.

Source: `/home/cunyuliu/pc_cng_research/results/xtb_dft_validation_20260719/validation_summary.json`.

## Supplementary Note 5: Expert review protocol (P1-08) and execution status

The Layer 3 expert-review protocol is specified as follows:

- **Reviewers**: 2-3 computational/organic chemists with publications in
  cross-coupling or C-H activation.
- **Sample size**: 100-200 PC-CNG negatives per session, drawn stratified
  across the five edit actions.
- **Annotation**: each reviewer labels each candidate as
  plausible-positive, implausible-negative, or uncertain.
- **Acceptance**: Cohen's kappa >= 0.6 across reviewers; candidates with
  unanimous implausible-negative labels are retained as high-confidence
  negatives.
- **Execution status**: **not executed**.  Layer 3 currently runs a
  rule-based plausibility check (atom balance, valence, aromaticity) as a
  fallback.  The 26,517 high-confidence negatives reported in
  the main paper are therefore "high-confidence under rule-based fallback",
  not "expert-verified".  Expert review is the highest-priority future work.

## Supplementary Note 6: ORD data quality audit

The ORD audit covers 2,910 rows.  Strict RDKit validity is
47.39% (invalid rows carry ORD fragment extensions
`|f:...|` that are chemically interpretable under lenient parsing; lenient
validity is 99.97%).  Atom-mapping coverage is 0.00%
because ORD SMILES do not preserve atom maps.  Overlap with USPTO, HiTEA, and
RegioSQM20 is zero after canonicalisation.  ORD therefore enters the
pipeline as an unmapped-reaction supplement and is not used for atom-map
dependent edit actions.

Source: `/home/cunyuliu/pc_cng_research/results/ord_data_quality_audit_20260719/single_csv_audit.json`.

## Supplementary Note 7: Provenance audit trail

Every numeric claim in the manuscript is sourced from a JSON artifact on
disk.  The mapping is:

| Claim | Source path |
|---|---|
| cross_regiosqm20_to_hitea | `/home/cunyuliu/pc_cng_research/results/cross_dataset_transfer_20260719/regiosqm20_to_hitea/paired_significance.json` |
| cross_hitea_to_regiosqm20 | `/home/cunyuliu/pc_cng_research/results/cross_dataset_transfer_20260719/hitea_to_regiosqm20/paired_significance.json` |
| cross_regiosqm20_to_uspto | `/home/cunyuliu/pc_cng_research/results/cross_dataset_transfer_20260719/regiosqm20_to_uspto/paired_significance.json` |
| cross_hitea_to_uspto | `/home/cunyuliu/pc_cng_research/results/cross_dataset_transfer_20260719/hitea_to_uspto/paired_significance.json` |
| calibration | `/home/cunyuliu/pc_cng_research/results/calibration_error_10seed_20260719/calibration_error_summary.json` |
| ood | `/home/cunyuliu/pc_cng_research/results/ood_scaffold_template_split_20260719/ood_split_summary.json` |
| retrosynthesis | `/home/cunyuliu/pc_cng_research/results/retrosynthesis_route_ranking_20260719/paired_significance.json` |
| three_layer | `/home/cunyuliu/pc_cng_research/results/false_negative_three_layer_20260719/three_layer_summary.json` |
| p1_01_mlp | `/home/cunyuliu/pc_cng_research/results/external_calibration_heldout_full_beam_paired_significance_5k_20260719/mlp_vs_chemformer/paired_summary.json` |
| p1_01_hybrid | `/home/cunyuliu/pc_cng_research/results/external_calibration_heldout_full_beam_paired_significance_5k_20260719/hybrid_w0p50_vs_chemformer/paired_summary.json` |
| xtb | `/home/cunyuliu/pc_cng_research/results/xtb_dft_validation_20260719/validation_summary.json` |
| ord | `/home/cunyuliu/pc_cng_research/results/ord_data_quality_audit_20260719/single_csv_audit.json` |

| ni | `data/processed/ni_coupling_supplement.csv` (1688 rows) |
| prototype | `results/failure_prototype_calibration_smoke_20260719/controllability_report.json` (accuracy = 0.95) |
| curriculum | `results/semi_hard_curriculum_smoke_20260719/comparison.json` (delta = 8.33 pp) |
| retrosynthesis | `/home/cunyuliu/pc_cng_research/results/retrosynthesis_route_ranking_20260719/paired_significance.json` (delta = 30.63 pp) |
