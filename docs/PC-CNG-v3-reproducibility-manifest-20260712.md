# PC-CNG v3 Reproducibility Manifest

日期：2026-07-14 12:05 CST

服务器：`ssh cunyuliu@36.137.135.49 -p 22`

项目根目录：`/home/cunyuliu/pc_cng_research`

Python：`/home/cunyuliu/miniconda3/envs/pc_cng_gpu/bin/python`

## 1. 固定输入

| Type | Path |
|---|---|
| Real CSV | `/home/cunyuliu/pc_cng_research/data/processed/regiosqm20_normalized.csv` |
| Real CSV | `/home/cunyuliu/pc_cng_research/data/processed/hitea_full_normalized.csv` |
| Synthetic CSV | `/home/cunyuliu/pc_cng_research/results/type1_diverse_anchor_full/diverse_anchor_candidates_reviewed_knownpos_filtered.csv` |
| Synthetic CSV | `/home/cunyuliu/pc_cng_research/results/type1_class_quota_supplement_20260711/class_quota_candidates_reviewed.csv` |
| Synthetic CSV | `/home/cunyuliu/pc_cng_research/results/type1_class_fallback_supplement_20260711/class_fallback_candidates_reviewed.csv` |
| Synthetic CSV | `/home/cunyuliu/pc_cng_research/results/type1_partial_product_supplement_20260711/partial_product_candidates_reviewed.csv` |
| Synthetic CSV | `/home/cunyuliu/pc_cng_research/results/type1_unreacted_substrate_supplement_v2_20260711/unreacted_substrate_candidates_reviewed.csv` |

## 2. Known-positive filter

| Item | Value |
|---|---|
| Source CSV | `results/type1_diverse_anchor_full/diverse_anchor_candidates_reviewed.csv` |
| Filtered CSV | `results/type1_diverse_anchor_full/diverse_anchor_candidates_reviewed_knownpos_filtered.csv` |
| Removed rows | `10` |
| Kept rows | `4535` |
| Removed detail | `results/type1_diverse_anchor_full/diverse_anchor_candidates_reviewed_knownpos_removed.csv` |
| Summary | `results/type1_diverse_anchor_full/diverse_anchor_knownpos_filter_summary.json` |

## 3. Data Quality

| Artifact | Path / Status |
|---|---|
| Benchmark manifest JSON | `/home/cunyuliu/pc_cng_research/docs/PC-CNG-v3-benchmark-manifest-20260712.json` |
| Data-quality audit | `/home/cunyuliu/pc_cng_research/results/benchmark_manifest_data_quality_20260712/benchmark_data_quality_audit.json` |
| Status | `pass_with_warnings` |
| Hard gate failures | `0` |
| Warnings | `7` |
| Unit test | `tests/test_benchmark_data_quality_audit.py`, local and remote passed, 2 tests |

Remaining warnings:

1. RegioSQM20 and HITEA have reactant-contexts appearing in multiple splits, while `source_id` and canonical parent-reaction leakage are zero.
2. Synthetic CSVs contain val/test parent rows for candidate construction; training safety depends on train-parent filtering in the reader.

## 4. Original Benchmark Support

| Scope | Test Groups | Target | Deficit |
|---|---:|---:|---:|
| Real reactants | 43 | 200 | 157 |
| Synthetic source_id | 68 | 200 | 132 |
| Combined real reactants + synthetic source_id | 111 | 200 | 89 |

Artifacts:

| Artifact | Path |
|---|---|
| Support audit JSON | `/home/cunyuliu/pc_cng_research/results/original_benchmark_support_audit_20260712/original_benchmark_support_audit.json` |
| Support summary | `/home/cunyuliu/pc_cng_research/results/original_benchmark_support_audit_20260712/original_benchmark_support_summary.md` |

## 5. Original Test Expansion Candidate Scan

| Item | Value |
|---|---|
| Source CSV | `/home/cunyuliu/pc_cng_research/data/processed/uspto_openmolecules_normalized.csv` |
| Scan JSON | `/home/cunyuliu/pc_cng_research/results/original_test_expansion_uspto_scan_20260712/uspto_original_test_expansion_scan.json` |
| Selected candidates | `/home/cunyuliu/pc_cng_research/results/original_test_expansion_uspto_scan_20260712/uspto_original_test_expansion_candidates.csv` |
| Eligible pool top | `/home/cunyuliu/pc_cng_research/results/original_test_expansion_uspto_scan_20260712/uspto_original_test_expansion_eligible_pool_top.csv` |
| Existing / target / deficit | `111 / 200 / 89` |
| Eligible unique held-out contexts | `51,494` |
| Selected positive parent contexts | `256` |
| Status | `positive_parent_pool_sufficient_for_next_generation_stage` |
| Counting rule | selected parents are not evaluable groups until boundary negatives are generated/reviewed, known-positive filtering is rerun, and support audit passes |
| Unit test | `tests/test_original_test_expansion_scan.py`, local and remote passed, 1 test |

## 6. Original Test Expansion Negatives And M3 Support

| Item | Value |
|---|---|
| Known-positive cache | `/home/cunyuliu/pc_cng_research/results/original_test_expansion_uspto_negatives_20260712/known_positive_cache.json` |
| Cache reactions / products | `524,607 / 486,474` |
| Positive parent CSV | `/home/cunyuliu/pc_cng_research/results/original_test_expansion_uspto_negatives_20260712/uspto_original_test_expansion_positive_parents.csv` |
| Raw candidates | `/home/cunyuliu/pc_cng_research/results/original_test_expansion_uspto_negatives_20260712/uspto_test_expansion_candidates_raw.csv` |
| Reviewed/filtered candidates | `/home/cunyuliu/pc_cng_research/results/original_test_expansion_uspto_negatives_20260712/uspto_test_expansion_candidates_reviewed_knownpos_filtered.csv` |
| Pipeline summary | `/home/cunyuliu/pc_cng_research/results/original_test_expansion_uspto_negatives_20260712/uspto_test_expansion_pipeline_summary.json` |
| Raw / filtered / keep rows | `1,746 / 1,616 / 1,268` |
| Keep source groups | `212` |
| M3 support audit | `/home/cunyuliu/pc_cng_research/results/original_benchmark_support_audit_m3_uspto_20260712/original_benchmark_support_audit.json` |
| Expanded combined test groups | `323 / 200`, deficit `0` |
| M3 data-quality audit | `/home/cunyuliu/pc_cng_research/results/original_test_expansion_uspto_data_quality_20260712/benchmark_data_quality_audit.json` |
| Data-quality status | `pass_with_warnings`, hard gate failures `0` |
| Unit test | `tests/test_known_positive_cache.py`, local and remote passed, 1 test |

## 6a. Ni Atomic Support Audit

| Item | Value |
|---|---|
| Script | `/home/cunyuliu/pc_cng_research/chem_negative_sampling/pc_cng/audit_ni_atomic_support.py` |
| Result dir | `/home/cunyuliu/pc_cng_research/results/ni_atomic_support_audit_20260713` |
| Audit JSON | `/home/cunyuliu/pc_cng_research/results/ni_atomic_support_audit_20260713/ni_atomic_support_audit.json` |
| Examples CSV | `/home/cunyuliu/pc_cng_research/results/ni_atomic_support_audit_20260713/ni_atomic_examples.csv` |
| Summary MD | `/home/cunyuliu/pc_cng_research/results/ni_atomic_support_audit_20260713/ni_atomic_support_audit.md` |
| Detection rule | Text prefilter containing `Ni`, followed by RDKit atomic number 28; remote `rdkit_available=true` |
| HITEA normalized | `39,546` reaction rows, `0` Ni reactions, `0` distinct Ni parent reactants |
| USPTO/OpenMolecules normalized | `530,238` reaction rows, `6` Ni reactions, `6` distinct Ni parent reactants; split examples `train=2`, `val=2`, `test=2` |
| Decision | `ni_remains_external_data_source_gap`; requires external/curated Ni data rather than generator reweighting |
| Unit test | `tests/test_ni_atomic_support_audit.py`, local and remote passed, 1 test |

## 6b. External Product Prediction Support Audit

| Item | Value |
|---|---|
| Script | `/home/cunyuliu/pc_cng_research/chem_negative_sampling/pc_cng/audit_external_product_prediction_support.py` |
| Result dir | `/home/cunyuliu/pc_cng_research/results/external_product_prediction_support_audit_20260713` |
| Audit JSON | `/home/cunyuliu/pc_cng_research/results/external_product_prediction_support_audit_20260713/external_product_prediction_support_audit.json` |
| Summary MD | `/home/cunyuliu/pc_cng_research/results/external_product_prediction_support_audit_20260713/external_product_prediction_support_audit.md` |
| Source contexts | `16,050 / 25,000`, deficit `8,950` |
| Full candidate set | `175,678` rows, `16,050` groups |
| Candidate coverage | Chemformer beam `16,050` groups; PC-CNG candidates `600` groups |
| Strict complete evaluation | `1,197` groups, `81` test groups; best strict Test Top-1 `71.60%` (`pc_cng`) |
| Validity-aware evaluation | `15,973` groups, `1,536` test groups; best validity-aware Test Top-1 `98.50%` (`pc_cng`) |
| Decision flags | `external_context_target_not_met`, `strict_complete_group_target_not_met`, `strict_pc_cng_score_coverage_low` |
| Unit test | `tests/test_external_product_prediction_support_audit.py`, local and remote passed, 1 test |

## 6c. External Product Prediction Context Expansion

| Item | Value |
|---|---|
| Script | `/home/cunyuliu/pc_cng_research/chem_negative_sampling/pc_cng/select_external_product_prediction_contexts.py` |
| Result dir | `/home/cunyuliu/pc_cng_research/results/external_product_prediction_context_expansion_20260713` |
| Summary JSON | `/home/cunyuliu/pc_cng_research/results/external_product_prediction_context_expansion_20260713/external_product_prediction_context_expansion_summary.json` |
| Expansion contexts | `/home/cunyuliu/pc_cng_research/results/external_product_prediction_context_expansion_20260713/external_product_prediction_context_expansion.csv` |
| Merged contexts | `/home/cunyuliu/pc_cng_research/results/external_product_prediction_context_expansion_20260713/external_product_prediction_contexts_merged.csv` |
| Expansion Chemformer input | `/home/cunyuliu/pc_cng_research/results/external_product_prediction_context_expansion_20260713/external_product_prediction_context_expansion_chemformer_input.csv` |
| Merged Chemformer input | `/home/cunyuliu/pc_cng_research/results/external_product_prediction_context_expansion_20260713/external_product_prediction_contexts_merged_chemformer_input.csv` |
| Selection result | `8,950` selected contexts from `517,644` eligible unique source contexts |
| Merged context gate | `25,000/25,000` unique groups, Chemformer input `25,001` lines including header |
| Decision | `context_target_input_prepared_beam_generation_pending` |
| Unit test | `tests/test_external_context_expansion.py`, local and remote passed, 1 test |

## 6d. External Product Prediction 25k Base Prebuild

| Item | Value |
|---|---|
| Runner | `/home/cunyuliu/pc_cng_research/chem_negative_sampling/scripts_run_external_product_prediction_benchmark_25k.sh` |
| Result dir | `/home/cunyuliu/pc_cng_research/results/external_product_prediction_benchmark_25k_20260713` |
| Contexts CSV | `/home/cunyuliu/pc_cng_research/results/external_product_prediction_benchmark_25k_20260713/product_prediction_contexts.csv` |
| Chemformer input | `/home/cunyuliu/pc_cng_research/results/external_product_prediction_benchmark_25k_20260713/chemformer_forward_input.csv` |
| Base candidates | `/home/cunyuliu/pc_cng_research/results/external_product_prediction_benchmark_25k_20260713/base_observed_pc_cng_candidates.csv` |
| Base summary | `/home/cunyuliu/pc_cng_research/results/external_product_prediction_benchmark_25k_20260713/base_observed_pc_cng_candidates_summary.json` |
| Targeted PC-CNG generator | `/home/cunyuliu/pc_cng_research/chem_negative_sampling/pc_cng/generate_external_context_pc_cng_candidates.py` |
| Targeted PC-CNG output | `/home/cunyuliu/pc_cng_research/results/external_product_prediction_25k_pc_cng_targeted_20260713/external_product_prediction_25k_pc_cng_targeted_candidates.csv` |
| Targeted PC-CNG summary | `/home/cunyuliu/pc_cng_research/results/external_product_prediction_25k_pc_cng_targeted_20260713/external_product_prediction_25k_pc_cng_targeted_summary.json` |
| Targeted generation result | processed `24,923` contexts, skipped `77`, generated `47,259` candidates over `24,903` groups |
| Base rows | `76,487` candidates over `25,000` contexts |
| Candidate source counts | `25,000` observed positives + `51,487` PC-CNG rows |
| PC-CNG negative group coverage | `24,903/25,000` groups after targeted generation |
| Base quality audit | `/home/cunyuliu/pc_cng_research/results/external_25k_base_candidate_quality_audit_20260713/external_25k_base_candidate_quality_audit.json` |
| Base quality decision | `pass_with_warnings`; no duplicate products, no same-product PC-CNG negatives, no invalid PC-CNG negatives; warning is `77` invalid observed-positive reactions with blank reactants |
| Decision | `superseded_by_repaired_25k_prebuild` |
| Chemformer chunk manifest | `/home/cunyuliu/pc_cng_research/results/external_product_prediction_benchmark_25k_20260713/chemformer_input_chunks/chemformer_forward_input_25k_chunks_manifest.json` |
| Chemformer chunks | `5` chunks × `5,000` rows |
| Chunked beam runner | `/home/cunyuliu/pc_cng_research/chem_negative_sampling/scripts_run_external_product_prediction_25k_chunked_beams.sh` |
| Beam chunk status | `/home/cunyuliu/pc_cng_research/results/external_product_prediction_benchmark_25k_20260713/chemformer_beam_chunks/chemformer_forward_beam_chunks_status.json`; dry-run `0/5` complete |
| Beam runner safety | waits for GPU memory/utilization thresholds and zero compute apps by default; skips valid chunks and merges only after all chunks validate |
| Old beam watcher | PID `2156024` paused before any chunk generation because the dirty input had `77` blank-reactant contexts |

## 6e. External Product Prediction 25k Repaired Prebuild

| Item | Value |
|---|---|
| Repair dir | `/home/cunyuliu/pc_cng_research/results/external_product_prediction_context_repair_20260713` |
| Repaired result dir | `/home/cunyuliu/pc_cng_research/results/external_product_prediction_benchmark_25k_repaired_20260713` |
| Removed contexts | `77` blank-reactant rows |
| Replacement contexts | `77` USPTO/OpenMolecules test contexts |
| Repaired contexts | `/home/cunyuliu/pc_cng_research/results/external_product_prediction_benchmark_25k_repaired_20260713/product_prediction_contexts.csv`, `25,000` rows |
| Repaired Chemformer input | `/home/cunyuliu/pc_cng_research/results/external_product_prediction_benchmark_25k_repaired_20260713/chemformer_forward_input.csv`, `25,000` rows |
| Repaired base candidates | `/home/cunyuliu/pc_cng_research/results/external_product_prediction_benchmark_25k_repaired_20260713/base_observed_pc_cng_candidates.csv`, `76,672` rows |
| Repaired base summary | `/home/cunyuliu/pc_cng_research/results/external_product_prediction_benchmark_25k_repaired_20260713/base_observed_pc_cng_candidates_summary.json` |
| Repaired quality audit | `/home/cunyuliu/pc_cng_research/results/external_25k_base_candidate_quality_audit_repaired_20260713/external_25k_base_candidate_quality_audit.json`, decision `pass` |
| Repaired PC-CNG negative coverage | `24,980/25,000` groups |
| Repaired Chemformer chunk manifest | `/home/cunyuliu/pc_cng_research/results/external_product_prediction_benchmark_25k_repaired_20260713/chemformer_input_chunks/chemformer_forward_input_25k_chunks_manifest.json`, `5` chunks × `5,000` rows |
| Beam runner header fix | Chemformer now receives temporary no-header chunk inputs; invalid header-included chunk 0 output archived |
| Repaired beam chunks | `5/5` complete; merged beam generated |
| Merged repaired beam TSV | `/home/cunyuliu/pc_cng_research/results/external_product_prediction_benchmark_25k_repaired_20260713/chemformer_forward_beams.tsv`, `25,001` lines |
| Repaired full candidates | `/home/cunyuliu/pc_cng_research/results/external_product_prediction_benchmark_25k_repaired_20260713/full_observed_pc_cng_chemformer_beam_candidates.csv`, `311,150` candidates |
| Repaired full summary | `/home/cunyuliu/pc_cng_research/results/external_product_prediction_benchmark_25k_repaired_20260713/full_observed_pc_cng_chemformer_beam_candidates_summary.json`; observed positive `25,000`, PC-CNG `51,672`, Chemformer beam `235,548` |
| Likelihood smoke | `20` input rows -> `20` score rows, first score row `lm_status=ok` |
| Chemformer likelihood scores | `/home/cunyuliu/pc_cng_research/results/external_product_prediction_benchmark_25k_repaired_20260713/lm_scores_chemformer_log_likelihood.csv`, `311,150` score rows |
| Strict benchmark | `/home/cunyuliu/pc_cng_research/results/external_product_prediction_benchmark_25k_repaired_20260713/benchmark/benchmark_summary.json`; strict complete groups `25,000`, evaluated rows `116,509` |
| Strict test Top-1 | Chemformer likelihood `57.00%`; PC-CNG `13.59%`; selected hybrid weight `0.00` |
| Validity-aware benchmark | `/home/cunyuliu/pc_cng_research/results/external_product_prediction_benchmark_25k_repaired_20260713/benchmark_validity_aware/benchmark_summary.json`; evaluated rows `311,150` |
| Validity-aware test Top-1 | Chemformer likelihood `44.02%`; PC-CNG scored subset `13.59%` |
| Repaired 25k support audit | `/home/cunyuliu/pc_cng_research/results/external_product_prediction_support_audit_25k_repaired_20260714/external_product_prediction_support_audit.json`; contexts `25,000/25,000`, strict groups `25,000/25,000`, decision flags `[]` |
| Evaluator fix | `pc_cng/evaluate_external_product_prediction_benchmark.py` now computes hybrids on the shared scored subset under `--allow-incomplete-groups`; remote unittest passed |
| Score calibration audit | `/home/cunyuliu/pc_cng_research/results/external_score_calibration_25k_repaired_strict_20260715/external_score_calibration_summary.json`; strict shared rows `116,509`; Chemformer Top-1 `57.00%`, best nonzero hybrid `50.87%`, PC-CNG `13.59%`; diagnostic only |
| Held-out calibration protocol | `/home/cunyuliu/pc_cng_research/docs/PC-CNG-v3-external-heldout-calibration-protocol-20260715.{md,json}`; new 5k test contexts outside repaired 25k; base candidate rows `16,873`; quality audit `pass`; beam/scoring not run |
| Frozen calibration recipes | `pc_cng_lr_calibrator_v1` and `pc_cng_pairwise_calibrator_v1`; both trained only on repaired 25k train split and both have val Top-1 `80.62%` vs Chemformer `83.42%`; not held-out primary |
| MLP calibration recipe | `pc_cng_mlp_calibrator_v1`; trained only on repaired 25k train split; val Top-1 `89.26%`, repaired 25k test Top-1 `36.46%`; cross-domain failure, not held-out primary |
| USPTO train/val calibration pool | `results/external_calibration_trainval_contexts_12k_repaired_20260716/`; train `10,000`, val `2,000`; repaired base candidates `40,244`; quality `pass`; Chemformer val Top-1 `89.89%`, PC-CNG val Top-1 `16.42%`, simple hybrid selects `w0p00` |
| USPTO train/val frozen scorer recipes | LR val Top-1 `87.39%`, pairwise val Top-1 `84.68%`, MLP val Top-1 `93.14%`; MLP is validation-positive but not held-out evidence |
| Held-out 5k base-only diagnostic | `results/external_score_mlp_calibrator_v1_uspto12k_apply_heldout_base5k_20260716/`; base candidates only, `4,995` groups; Chemformer Top-1 `91.99%`, PC-CNG Top-1 `17.44%`, MLP Top-1 `94.51%`; diagnostic only |
| Held-out 5k full-beam status | `results/logs/external_calibration_heldout5k_beams_20260716.log`; GPU7 run active, full-beam final scoring pending |

## 6f. Active Execution Status Audit

| Item | Value |
|---|---|
| Script | `/home/cunyuliu/pc_cng_research/chem_negative_sampling/pc_cng/audit_active_execution_status.py` |
| Result dir | `/home/cunyuliu/pc_cng_research/results/active_execution_status_20260713` |
| Status JSON | `/home/cunyuliu/pc_cng_research/results/active_execution_status_20260713/active_execution_status.json` |
| Status MD | `/home/cunyuliu/pc_cng_research/results/active_execution_status_20260713/active_execution_status.md` |
| Generated at | `2026-07-14T10:34:00` |
| Training watcher status | all 5 watcher PIDs alive; no result artifacts yet |
| Beam / benchmark status | repaired likelihood, strict benchmark, validity-aware benchmark, and support audit complete; denominator scale complete but PC-CNG external performance negative |
| Process-tree check | last checked at `2026-07-13 17:53`; watcher shells only had `sleep 300` children; no hidden training/beam subprocess |
| Unit test | `tests/test_active_execution_status.py`, local and remote passed, 1 test |

## 7. Expanded M3 v2 Baseline Evaluation

| Item | Value |
|---|---|
| Script | `/home/cunyuliu/pc_cng_research/run_v2_unreacted_expanded_m3_uspto_eval.sh` |
| Result dir | `/home/cunyuliu/pc_cng_research/results/type1_v2_unreacted_expanded_m3_uspto_eval_20260712` |
| Summary JSON | `/home/cunyuliu/pc_cng_research/results/type1_v2_unreacted_expanded_m3_uspto_eval_20260712/v2_unreacted_expanded_m3_uspto_summary/summary.json` |
| Per-seed CSV | `/home/cunyuliu/pc_cng_research/results/type1_v2_unreacted_expanded_m3_uspto_eval_20260712/v2_unreacted_expanded_m3_uspto_per_seed.csv` |
| Ensemble metrics | `/home/cunyuliu/pc_cng_research/results/type1_v2_unreacted_expanded_m3_uspto_eval_20260712/v2_unreacted_expanded_m3_uspto_ensemble/ranking_metrics.json` |
| Scoring coverage | `/home/cunyuliu/pc_cng_research/results/type1_v2_unreacted_expanded_m3_uspto_eval_20260712/v2_unreacted_expanded_m3_uspto_ensemble/scoring_coverage_summary.json` |
| Support / scored test groups | `323 / 293` |
| 10-seed Test Top-1 | `51.60 ± 0.89%` |
| 10-seed Test MRR / NDCG | `71.61 ± 0.57% / 78.76 ± 0.43%` |
| Ensemble Test Top-1 | `51.88%` |
| USPTO subset Top-1 | `38.02 ± 1.54%`, ensemble `39.15%` |
| M0 retest comparison | `/home/cunyuliu/pc_cng_research/results/type1_expanded_m3_uspto_eval_comparison_20260712/expanded_m3_uspto_model_comparison_with_significance.md` |
| Hidden4096 expanded decision | Test Top-1 `52.12 ± 0.85%`, paired ΔTop-1 `+0.48 pp` CI `[-0.07,+1.03]`, p=`0.143`; reject as main |
| Dropout04 expanded decision | Test Top-1 `51.71 ± 1.06%`, paired ΔTop-1 `-0.14 pp` CI `[-0.41,+0.14]`, p=`0.626`; reject as main |
| Combined expanded decision | Test Top-1 `52.42 ± 0.84%`, paired ΔTop-1 `+0.14 pp` CI `[-0.62,+0.89]`, p=`0.864`; supplement-only |
| Classw050_rc expanded decision | Test Top-1 `52.73 ± 1.13%`, paired ΔTop-1 `+0.07 pp` CI `[-0.89,+1.03]`, p=`1.000`; supplement-only |

## 8. Active Queue Order

| Order | Branch | GPU | PID | Script | Queue log | Result dir | Dependency |
|---:|---|---:|---:|---|---|---|---|
| 1 | `type1_v2_coslr_warm5_20260712` | 4 | complete | `run_v2_coslr_warm5_multiseed.sh` | `results/logs/type1_v2_coslr_warm5_relaxed_20260714.log` | `results/type1_v2_coslr_warm5_20260712/` | 10 seeds complete; Test Top-1 `87.41 ± 1.82%`; paired ΔTop-1 `+0.08 pp`, CI `[-0.24,+0.48]`; not promoted |
| 1b | `type1_v2_nbits8192_10seed_20260714` | 1 | complete | `run_v2_nbits8192_10seed.sh` | `results/logs/type1_v2_nbits8192_10seed_20260714.log` | `results/type1_v2_nbits8192_10seed_20260714/` | reused smoke seeds `20260710-20260712`; 10 seeds complete; Test Top-1 `87.78 ± 1.36%`; paired ΔTop-1 `+0.32 pp`, CI `[-0.32,+0.97]`; not promoted |
| 2 | `type1_v2_gpu5_relaxed_chain_20260714` | 5 | complete | serial chain: filtered baseline -> valtop1 -> representation -> pairwise/margin | `results/logs/type1_v2_gpu5_relaxed_chain_20260714.log` | multiple result dirs below | filtered baseline, valtop1, representation smoke, and pairwise/margin smoke complete |
| 2a | `type1_v2_filtered_baseline_20260712` | 5 | complete | `run_v2_filtered_baseline_multiseed.sh` | `results/logs/type1_v2_filtered_baseline_20260712_seed*.log` | `results/type1_v2_filtered_baseline_20260712/` | Test Top-1 `87.04 ± 2.35%`; not promoted |
| 2b | `type1_v2_valtop1_ckpt_smoke_20260712` | 5 | complete | `run_v2_valtop1_ckpt_smoke.sh` | `results/logs/type1_v2_valtop1_ckpt_smoke_20260712_seed*.log` | `results/type1_v2_valtop1_ckpt_smoke_20260712/` | rejected; Test Top-1 `83.54 ± 1.43%` |
| 2c | `type1_v2_representation_scale_smoke_20260712` | 5 | complete | `run_v2_representation_scale_smoke.sh` | `results/logs/type1_v2_representation_scale_smoke_20260712_*.log` | `results/type1_v2_representation_scale_smoke_20260712/` | nbits8192 3-seed done and promoted; binary_count 3-seed done, Test Top-1 `87.24 ± 0.71%`, paired ΔTop-1 `+0.16 pp`, CI `[-0.16,+0.48]`, p=`0.624`; binary_count not promoted |
| 2d | `type1_v2_pairwise_margin_smoke_20260712` | 5 | complete | `run_v2_pairwise_margin_smoke.sh` | `results/logs/type1_v2_pairwise_margin_smoke_20260712_*.log` | `results/type1_v2_pairwise_margin_smoke_20260712/` | 8 configs complete; selected `pw20_m000` and `pw20_m005` for 10-seed validation |
| 3a | `type1_v2_pairwise_margin_10seed_pw20_m000_20260714` | 0 | complete | `watch_v2_pairwise_margin_10seed_selected.sh` -> `run_v2_pairwise_margin_10seed_selected.sh` | `results/logs/type1_v2_pairwise_margin_10seed_20260714_pw20_m000.watch.log` | `results/type1_v2_pairwise_margin_10seed_20260714/pw20_m000_10seed_summary/summary.json`; `results/type1_v2_pairwise_margin_10seed_20260714/paired_significance_filtered_v2_vs_pw20_m000_10seed/summary.json` | 10 seeds complete; Test Top-1 `88.15 ± 1.95%`; paired ΔTop-1 `+0.08 pp`, CI `[-0.32,+0.48]`, p=`1.000`; not promoted |
| 3b | `type1_v2_pairwise_margin_10seed_pw20_m005_20260714` | 5 | complete | `watch_v2_pairwise_margin_10seed_selected.sh` -> `run_v2_pairwise_margin_10seed_selected.sh` | `results/logs/type1_v2_pairwise_margin_10seed_20260714_pw20_m005.watch.log` | `results/type1_v2_pairwise_margin_10seed_20260714/pw20_m005_10seed_summary/summary.json`; `results/type1_v2_pairwise_margin_10seed_20260714/paired_significance_filtered_v2_vs_pw20_m005_10seed/summary.json` | 10 seeds complete; Test Top-1 `87.41 ± 2.52%`; paired ΔTop-1 `0.00 pp`, CI `[-0.48,+0.48]`, p=`1.000`; not promoted |

## 9. Completed M0 Negative Results

| Branch | Summary | Paired significance | Decision |
|---|---|---|---|
| `type1_v2_hidden4096_20260712` | `results/type1_v2_hidden4096_20260712/hidden4096_multiseed_summary/summary.json` | `results/type1_v2_hidden4096_20260712/paired_significance_v2_vs_hidden4096_same_split/summary.json` | reject as main |
| `type1_v2_dropout04_20260712` | `results/type1_v2_dropout04_20260712/dropout04_multiseed_summary/summary.json` | `results/type1_v2_dropout04_20260712/paired_significance_v2_vs_dropout04_same_split/summary.json` | reject as main |

## 10. Promotion Gates

| Gate | Rule |
|---|---|
| Smoke to 10-seed | 3-seed mean Test Top-1 >=87.7% or predeclared compensating weak-class/external gain |
| Main candidate | 10-seed original held-out Test Top-1 >= v2 + 1.0 pp and group-level ensemble CI entirely positive |
| Final target | Original held-out Test Top-1 >=93.0% or strict external Top-1 >=90.0% with strong statistics |

## 11. SHA256

| Path | SHA256 |
|---|---|
| `/home/cunyuliu/pc_cng_research/run_v2_filtered_baseline_multiseed.sh` | `85040f71a8ebffedeeb20b327e71a0f6f8fa6610ff7c7f6837e81dd5991dc52e` |
| `/home/cunyuliu/pc_cng_research/run_v2_coslr_warm5_multiseed.sh` | `24a50b386fc6e899ff0d3b6d0a8e088d28fb6b3186cee06b77a35ff049ac939f` |
| `/home/cunyuliu/pc_cng_research/run_v2_nbits8192_10seed.sh` | `788084c1159faa5fa20a19f2cd6fd3a82c9ba037ccfc5fd0f33e00df9b4f8e1e` |
| `/home/cunyuliu/pc_cng_research/run_v2_valtop1_ckpt_smoke.sh` | `0790c8300d0988a1a20af630dba869211c491df3c9a198a22a555f6d82e20b85` |
| `/home/cunyuliu/pc_cng_research/run_v2_representation_scale_smoke.sh` | `dd8b71709dfbd63e3c90508844f126e722972baa1985b874b6e27b6fe72ef6be` |
| `/home/cunyuliu/pc_cng_research/run_v2_pairwise_margin_smoke.sh` | `1a2c414e9e37f984d9e13d9c16563ae64ef613f0fb25a11c41f080baebace383` |
| `/home/cunyuliu/pc_cng_research/chem_negative_sampling/scripts_run_external_product_prediction_benchmark_25k.sh` | `035d05bb38e2d5fb5653de68a2363ec9ebd2e70490c9dbb83e7e0a8b39b4a678` |
| `/home/cunyuliu/pc_cng_research/chem_negative_sampling/scripts_run_external_product_prediction_25k_chunked_beams.sh` | `2f8ed330add6e03ed9b815163a41f58eb0f9b5a718bfb53841b1350ecd01c47e` |
| `/home/cunyuliu/pc_cng_research/chem_negative_sampling/pc_cng/audit_benchmark_data_quality.py` | `da1811d4abaf7a1bb6347a35b1035c7007d638b9550da202133cdb32ca9ef5f7` |
| `/home/cunyuliu/pc_cng_research/chem_negative_sampling/pc_cng/audit_original_benchmark_support.py` | `fe3bd598e410331a956ee81e231270ad078cbce4ace95bb16190546172c0324e` |
| `/home/cunyuliu/pc_cng_research/chem_negative_sampling/pc_cng/scan_original_test_expansion_candidates.py` | `4dadc0378cc493be89183ab3f2900a75a4100f887bbd7300a6535d4496bc490e` |
| `/home/cunyuliu/pc_cng_research/chem_negative_sampling/pc_cng/known_positive_cache.py` | `fcfaa8d1649e5d869f9aeae016399e18f8873d8e5ba44e454319657821c08a28` |
| `/home/cunyuliu/pc_cng_research/chem_negative_sampling/pc_cng/build_known_positive_cache.py` | `fc35a903eed0f6b9aef2d18ac4d5dd6e2d2a46287c05859331b8ae026ad06c8c` |
| `/home/cunyuliu/pc_cng_research/chem_negative_sampling/pc_cng/false_negative_review.py` | `632a2b2dd59e1938dd2bbbc51168e1d52d13b5f6749cb4e5598e1a54eb9ac1e3` |
| `/home/cunyuliu/pc_cng_research/chem_negative_sampling/pc_cng/filter_synthetic_known_positives.py` | `563450867d98ac7cbdaec61a11f441ecd27f64f7c8be2d1cbeddd3fd9372467b` |
| `/home/cunyuliu/pc_cng_research/chem_negative_sampling/pc_cng/audit_ni_atomic_support.py` | `41e8dcc9d60da1bbb758a8685a4a7a92875c521ba6c1e790d73ea25e272f5384` |
| `/home/cunyuliu/pc_cng_research/chem_negative_sampling/pc_cng/audit_external_product_prediction_support.py` | `b6f78b5f3d6a6b415889cb4feacc76e99d78aa171ce93f867f15cc4bd54b294a` |
| `/home/cunyuliu/pc_cng_research/chem_negative_sampling/pc_cng/select_external_product_prediction_contexts.py` | `92c2695f6fbba710e5d4d9027255c3f72a62584872d44605fee0377b95f34cfc` |
| `/home/cunyuliu/pc_cng_research/chem_negative_sampling/pc_cng/repair_external_trainval_context_pool.py` | `1c605d3eca778db3a3213f602338b1911c8bb72d281144006181f8e3945fd79a` |
| `/home/cunyuliu/pc_cng_research/chem_negative_sampling/pc_cng/split_chemformer_input.py` | `2aabfcb070a76d40ca3f0eb2183af49f446bf3b74cd285010f5258e2afc418a5` |
| `/home/cunyuliu/pc_cng_research/chem_negative_sampling/pc_cng/generate_external_context_pc_cng_candidates.py` | `914b8718c84ddcd8b5d81872f92e6070ca1bdc82828dc71c02e0650a3e14e430` |
| `/home/cunyuliu/pc_cng_research/chem_negative_sampling/pc_cng/audit_external_25k_base_candidate_quality.py` | `08a500e49421da96b40a3c42b1de768c01ee91ddd7463ffc03b801d0baf1e340` |
| `/home/cunyuliu/pc_cng_research/chem_negative_sampling/pc_cng/audit_active_execution_status.py` | `47c0427e2bc4384785ce64897517925a1b2776006c9edc9fa197e0eaa1b183e1` |
| `/home/cunyuliu/pc_cng_research/chem_negative_sampling/tests/test_benchmark_data_quality_audit.py` | `01d656aa0d1d1dd9bf1cfbf817f559771cc59ff17ff0a96feb9d946426ab146c` |
| `/home/cunyuliu/pc_cng_research/chem_negative_sampling/tests/test_original_test_expansion_scan.py` | `69fda3c4a2a3fc65b7018f4823717524f7593386e7514fd9bda1b337f318d2dd` |
| `/home/cunyuliu/pc_cng_research/chem_negative_sampling/tests/test_known_positive_cache.py` | `ec09db1f7c366d4d0698d2cd2348bada2d7abee55927225805bba8fa2e635fb0` |
| `/home/cunyuliu/pc_cng_research/chem_negative_sampling/tests/test_ni_atomic_support_audit.py` | `76bb80366e0ecf977e8d7e096a5d26999cb339cde65593183c635e782aa2c7fb` |
| `/home/cunyuliu/pc_cng_research/chem_negative_sampling/tests/test_external_product_prediction_support_audit.py` | `13af847e705206188fc08b62e9de0a13d8f0f4991fefe6ecad58d92f8f0e4a32` |
| `/home/cunyuliu/pc_cng_research/chem_negative_sampling/tests/test_external_context_expansion.py` | `2b51cb1562d1fe1e989e388ed9ecb79c4ed5cd3d2848bde8a524feab832b6cc9` |
| `/home/cunyuliu/pc_cng_research/chem_negative_sampling/tests/test_split_chemformer_input.py` | `355f2fdd28b69c7aec88e67ae4610694f8af3b402c8bd41798b090be2e96181c` |
| `/home/cunyuliu/pc_cng_research/chem_negative_sampling/tests/test_active_execution_status.py` | `b25bb1dac5fd61be3119c4cfdb1722277d32ab65a3300bd166a2c5b352f80141` |
| `/home/cunyuliu/pc_cng_research/chem_negative_sampling/tests/test_external_context_pc_cng_candidates.py` | `67866bff034362705f88d93ec7f259fa9dd3b83580d95ad9d7a50b5996c889bf` |
| `/home/cunyuliu/pc_cng_research/chem_negative_sampling/tests/test_external_25k_base_candidate_quality_audit.py` | `ecb538e9a347b568e878e153b16808fc8731e72a2ae94cc6d0c178eaede80b63` |
| `/home/cunyuliu/pc_cng_research/docs/PC-CNG-v3-benchmark-manifest-20260712.json` | `fa51008667cd6e9f483af9adaf2928efba57e8c434a64ed6f6bc5774a86a7e86` |
| `/home/cunyuliu/pc_cng_research/results/type1_diverse_anchor_full/diverse_anchor_candidates_reviewed_knownpos_filtered.csv` | `3e06168ca47f91c8e0ab5ae24a3164b8523935eb700c154c51b64432a6ebbedb` |
| `/home/cunyuliu/pc_cng_research/results/type1_diverse_anchor_full/diverse_anchor_knownpos_filter_summary.json` | `e4566dca10d1d8b46cc18ecba9263cf36d2c49d950c2ecbb138a89b00bae5e66` |
| `/home/cunyuliu/pc_cng_research/results/benchmark_manifest_data_quality_20260712/benchmark_data_quality_audit.json` | `b1c37a9b5eec0f10678a2bc54a393063c804409b5d07c96d90cd61efc96c3f3c` |
| `/home/cunyuliu/pc_cng_research/results/original_benchmark_support_audit_20260712/original_benchmark_support_audit.json` | `41e93e3591108562441db866c8485df2df7f6a36d0a437b8128950c6e6734f7b` |
| `/home/cunyuliu/pc_cng_research/results/original_benchmark_support_audit_20260712/original_benchmark_support_summary.md` | `e063cf940840126f6942bec6acc4014e5d7e9aa3f041f1bbe6374ad54acd5b8c` |
| `/home/cunyuliu/pc_cng_research/results/original_test_expansion_uspto_scan_20260712/uspto_original_test_expansion_scan.json` | `7896b7a41e9133d98a8ef218d2f134505b73422696b8f37ec74d0f78a4e51b0c` |
| `/home/cunyuliu/pc_cng_research/results/original_test_expansion_uspto_scan_20260712/uspto_original_test_expansion_candidates.csv` | `035ee8360c6bca3dc5ac1aada53c16aaf2241ff240419f1ff2a81c351e00944e` |
| `/home/cunyuliu/pc_cng_research/results/original_test_expansion_uspto_negatives_20260712/known_positive_cache.json` | `bdef56a4fdffa66f8b2f74fa8190072d223faf9d5cc7069bbd970fadb9d7260b` |
| `/home/cunyuliu/pc_cng_research/results/original_test_expansion_uspto_negatives_20260712/uspto_original_test_expansion_positive_parents.csv` | `f47c238c9cb08411715049300504530262c7664921b9307de3f6049116d90c05` |
| `/home/cunyuliu/pc_cng_research/results/original_test_expansion_uspto_negatives_20260712/uspto_test_expansion_pipeline_summary.json` | `8a5874f84d127dd993b65ed8848a36b465e1f08f0418eb9646b5905ce8bf4a94` |
| `/home/cunyuliu/pc_cng_research/results/original_test_expansion_uspto_negatives_20260712/uspto_test_expansion_candidates_reviewed_knownpos_filtered.csv` | `53da6c5232ef3828170496fca48f6b1521f4e6a75e24fb61fddc476538dee0c7` |
| `/home/cunyuliu/pc_cng_research/results/original_benchmark_support_audit_m3_uspto_20260712/original_benchmark_support_audit.json` | `136f34ceb806e1ccc6b474a23078b85b01f44512a100c4ef4f2007a47405caa0` |
| `/home/cunyuliu/pc_cng_research/results/original_test_expansion_uspto_data_quality_20260712/benchmark_data_quality_audit.json` | `af6c702edab36fb9c15fd3182be0828007142c9b94613116c7d4621184ce09a9` |
| `/home/cunyuliu/pc_cng_research/results/ni_atomic_support_audit_20260713/ni_atomic_support_audit.json` | `07bafeba5ae1455ba3a48739e598ac6a66c64b326244abd5e2cc27b7db057c64` |
| `/home/cunyuliu/pc_cng_research/results/ni_atomic_support_audit_20260713/ni_atomic_examples.csv` | `a2bdfaaf865b51a550c14c3f0b8c73e6e572c06c93107771af6db977a083b3ee` |
| `/home/cunyuliu/pc_cng_research/results/ni_atomic_support_audit_20260713/ni_atomic_support_audit.md` | `03659e26d58c849e584e47b607b5c2159804372bcec42c4df15a173f2d92eaa3` |
| `/home/cunyuliu/pc_cng_research/results/external_product_prediction_support_audit_20260713/external_product_prediction_support_audit.json` | `797065780b30529736e7ae4dea798a0a40bea056e21ec94722ec2fb9c08e571e` |
| `/home/cunyuliu/pc_cng_research/results/external_product_prediction_support_audit_20260713/external_product_prediction_support_audit.md` | `c6fcbf2ec603e57e34d2eabb5dec8f619273dd8f48f3035593afa9da350e8ce8` |
| `/home/cunyuliu/pc_cng_research/results/external_product_prediction_context_expansion_20260713/external_product_prediction_context_expansion_summary.json` | `93cf2b8969b18ed24f83a0447164b2a93813d69396dfda7a0d529c5a92f63e07` |
| `/home/cunyuliu/pc_cng_research/results/external_product_prediction_context_expansion_20260713/external_product_prediction_context_expansion_summary.md` | `5b6a5c09da7cb84517ae67785fa0b45ec4fa7dc2739906995c62eeaa0dfa79fe` |
| `/home/cunyuliu/pc_cng_research/results/external_product_prediction_context_expansion_20260713/external_product_prediction_context_expansion.csv` | `c0fc52179ccf1c2da8979d690369656a2df52bd573d3681d00c0a33d7de5d5ce` |
| `/home/cunyuliu/pc_cng_research/results/external_product_prediction_context_expansion_20260713/external_product_prediction_contexts_merged.csv` | `532c205fcbb3244d27f918018255c8c16d978d5e50dc4cca6bc1964903548bf1` |
| `/home/cunyuliu/pc_cng_research/results/external_product_prediction_context_expansion_20260713/external_product_prediction_context_expansion_chemformer_input.csv` | `8c9dfc25bf21965055ec70249e0b56af42181630aa0595152fd85bfdb6b74699` |
| `/home/cunyuliu/pc_cng_research/results/external_product_prediction_context_expansion_20260713/external_product_prediction_contexts_merged_chemformer_input.csv` | `db3cb16edf07af7b555628f05863d88df800a0b5c8ffab9a0a1fae5dbe755c28` |
| `/home/cunyuliu/pc_cng_research/results/external_product_prediction_benchmark_25k_20260713/product_prediction_contexts.csv` | `532c205fcbb3244d27f918018255c8c16d978d5e50dc4cca6bc1964903548bf1` |
| `/home/cunyuliu/pc_cng_research/results/external_product_prediction_benchmark_25k_20260713/chemformer_forward_input.csv` | `db3cb16edf07af7b555628f05863d88df800a0b5c8ffab9a0a1fae5dbe755c28` |
| `/home/cunyuliu/pc_cng_research/results/external_product_prediction_25k_pc_cng_targeted_20260713/external_product_prediction_25k_pc_cng_targeted_candidates.csv` | `721adc48fa279224491695397a75deb8edf5142aa7f744a8f095b242a5502da3` |
| `/home/cunyuliu/pc_cng_research/results/external_product_prediction_25k_pc_cng_targeted_20260713/external_product_prediction_25k_pc_cng_targeted_summary.json` | `7b34864d17b7c64caa62eafe8f955a01fc26e37f3fdf66a7a07a21b57b332747` |
| `/home/cunyuliu/pc_cng_research/results/external_product_prediction_25k_pc_cng_targeted_20260713/external_product_prediction_25k_pc_cng_targeted_summary.md` | `5d75abcea9c8af1d5fbdfc781013779560ca81fa2bdfb4192f602b05f52258cc` |
| `/home/cunyuliu/pc_cng_research/results/external_product_prediction_benchmark_25k_20260713/base_observed_pc_cng_candidates.csv` | `aa58206a0acfc3c1976d824a7c411fc41e163b7300248c603193c825d9aa7ff6` |
| `/home/cunyuliu/pc_cng_research/results/external_product_prediction_benchmark_25k_20260713/base_observed_pc_cng_candidates_summary.json` | `f0235876493a51c325f232272fb0199f6d285713609ba9b595cb014121e789ca` |
| `/home/cunyuliu/pc_cng_research/results/external_25k_base_candidate_quality_audit_20260713/external_25k_base_candidate_quality_audit.json` | `8147cc76edbab5f52e0ef54b80690b81c0f47519b2addfb2d9f14268bce9067b` |
| `/home/cunyuliu/pc_cng_research/results/external_25k_base_candidate_quality_audit_20260713/external_25k_base_candidate_quality_audit.md` | `860862cd05ab03bc414fb2d90272989344dc2a64cf06ea2313e8eb059f4a3e96` |
| `/home/cunyuliu/pc_cng_research/results/external_product_prediction_context_repair_20260713/external_product_prediction_context_repair_summary.json` | `220c4ead7fb879f93da3722cc033c547e9235143fa690dc8b4f9108609b5747d` |
| `/home/cunyuliu/pc_cng_research/results/external_product_prediction_context_repair_20260713/external_product_prediction_contexts_blank_reactants_77.csv` | `55a42b2d9c2f2581e9eae37bfd7f7614911c1c3b9c35fa2fe211385db1271e81` |
| `/home/cunyuliu/pc_cng_research/results/external_product_prediction_context_repair_20260713/external_product_prediction_context_expansion.csv` | `d9f7744eefff2b81db9aefe8591d202ed8338b2741fcd729aa015e90b18f1dec` |
| `/home/cunyuliu/pc_cng_research/results/external_product_prediction_context_repair_20260713/external_product_prediction_contexts_merged.csv` | `eb6ac7447accdf5d79e6cf4a3d26f45fc157c1adecbe043f3bd221351689d815` |
| `/home/cunyuliu/pc_cng_research/results/external_product_prediction_context_repair_20260713/external_product_prediction_contexts_merged_chemformer_input.csv` | `9b9f3a605efc33866f81724ba738a54d2498126b5f8c355f6b138831554e265a` |
| `/home/cunyuliu/pc_cng_research/results/external_product_prediction_context_repair_20260713/external_product_prediction_context_repair_pc_cng_candidates.csv` | `60f09a43f850d7f419f30703abc30c365f9a9782f7c9d712358aa917a2cd16dd` |
| `/home/cunyuliu/pc_cng_research/results/external_product_prediction_benchmark_25k_repaired_20260713/product_prediction_contexts.csv` | `eb6ac7447accdf5d79e6cf4a3d26f45fc157c1adecbe043f3bd221351689d815` |
| `/home/cunyuliu/pc_cng_research/results/external_product_prediction_benchmark_25k_repaired_20260713/chemformer_forward_input.csv` | `9b9f3a605efc33866f81724ba738a54d2498126b5f8c355f6b138831554e265a` |
| `/home/cunyuliu/pc_cng_research/results/external_product_prediction_benchmark_25k_repaired_20260713/base_observed_pc_cng_candidates.csv` | `ba7b36d7e30222ef0c95d0fbd0d994d8d0670d4c498e1ff3c386c0d6885ab243` |
| `/home/cunyuliu/pc_cng_research/results/external_product_prediction_benchmark_25k_repaired_20260713/base_observed_pc_cng_candidates_summary.json` | `751f542d469b6d9e9808cc0c5414ec0ac6eb559aa99ae9924333a974ded68d26` |
| `/home/cunyuliu/pc_cng_research/results/external_product_prediction_benchmark_25k_repaired_20260713/chemformer_input_chunks/chemformer_forward_input_25k_chunks_manifest.json` | `f0d66129fc915b1057f096fff2a331c783df203e4656a1b449e000779760fe0a` |
| `/home/cunyuliu/pc_cng_research/results/external_product_prediction_benchmark_25k_repaired_20260713/chemformer_input_chunks/chemformer_forward_input_25k_chunks_manifest.md` | `0640c5a96b1d0f7ceb0f192cd81f010bbfae62453866586a954919f11fc04f12` |
| `/home/cunyuliu/pc_cng_research/results/external_25k_base_candidate_quality_audit_repaired_20260713/external_25k_base_candidate_quality_audit.json` | `63c86f299d6b854807d2ce30d4b5accb2c9d61c89e249ba5b8be4f86b54ceded` |
| `/home/cunyuliu/pc_cng_research/results/external_25k_base_candidate_quality_audit_repaired_20260713/external_25k_base_candidate_quality_audit.md` | `49b38c5636218e07cf3035c89dcccc801645fba3beab4675735b7c15413d003d` |
| `/home/cunyuliu/pc_cng_research/results/external_product_prediction_benchmark_25k_repaired_20260713/chemformer_beam_chunks/chemformer_forward_beams_25k_chunk_0000.tsv` | `747563b5347303513c4ed8f3690a2183abe7f009b7d610ea2be93e16a069aa1b` |
| `/home/cunyuliu/pc_cng_research/results/external_product_prediction_benchmark_25k_repaired_20260713/chemformer_beam_chunks/chemformer_forward_beams_25k_chunk_0000.header_included_invalid.tsv` | `c429ae1773bb3c7ad0940c9d0a68bd03e6b13a97f2950a5e6aefcbc9525fd2bf` |
| `/home/cunyuliu/pc_cng_research/results/external_product_prediction_benchmark_25k_repaired_20260713/chemformer_forward_beams.tsv` | `e6bdb933e81288b5262a20a647ce1116e5be61c31cc43598e448ad23f7fcbcbb` |
| `/home/cunyuliu/pc_cng_research/results/external_product_prediction_benchmark_25k_repaired_20260713/full_observed_pc_cng_chemformer_beam_candidates.csv` | `f7a4ed0450a2a5b4bbf9573ea7c312b83b8893e8f0366b44dbbd9fc3ce38a542` |
| `/home/cunyuliu/pc_cng_research/results/external_product_prediction_benchmark_25k_repaired_20260713/full_observed_pc_cng_chemformer_beam_candidates_summary.json` | `7bbccfa4bfe44fa32416574aa1c18159deb154c240aeb5c13146315b67274460` |
| `/home/cunyuliu/pc_cng_research/results/logs/external_product_prediction_25k_repaired_benchmark_pipeline.pid` | `607a6ff2a84d44c359a0a7cd89b143f5542a28901581057519b020c8a0e0b925` |
| `/home/cunyuliu/pc_cng_research/results/logs/external_product_prediction_25k_repaired_benchmark_pipeline.log` | `10b4c4ad14a2d3cc5ed65b7818bb99e0f4524f9c568827db22ec1f57aac07ef1` |
| `/home/cunyuliu/pc_cng_research/chem_negative_sampling/pc_cng/evaluate_external_product_prediction_benchmark.py` | `d48600ce52dd01bfb0c1c12e58d90514098d6988a269cb4dd844065504e6eb98` |
| `/home/cunyuliu/pc_cng_research/chem_negative_sampling/tests/test_external_product_prediction_benchmark.py` | `8be64caba42eac4b5c84306d46a29bebeda50793ac2246020e8b42b5a8311a72` |
| `/home/cunyuliu/pc_cng_research/results/external_product_prediction_benchmark_25k_repaired_20260713/lm_scores_chemformer_log_likelihood.csv` | `f62e61027017abc3ed61ba82252a2d10dc67c756aa1ffc431c94460ed9f648d2` |
| `/home/cunyuliu/pc_cng_research/results/external_product_prediction_benchmark_25k_repaired_20260713/lm_scores_chemformer_log_likelihood_summary.json` | `746591989868c69d56b2671d9a71eb3658d3cc0d4e9ecb170c37b6e849f7bb40` |
| `/home/cunyuliu/pc_cng_research/results/external_product_prediction_benchmark_25k_repaired_20260713/benchmark/benchmark_summary.json` | `90249fd60c3ba546f664078c03c3ad21436f0126c74a78754d02458e6c2e9e00` |
| `/home/cunyuliu/pc_cng_research/results/external_product_prediction_benchmark_25k_repaired_20260713/benchmark/candidate_scores.csv` | `934ce75d21ad9bdb16fe3d94e6ae81d522700a893fd6d336111bba4bf1e004dd` |
| `/home/cunyuliu/pc_cng_research/results/external_product_prediction_benchmark_25k_repaired_20260713/benchmark/paper_table.md` | `1b476159a03b1d9520751ff3eec65d137c170c8016308802a58b262e8fe38ba9` |
| `/home/cunyuliu/pc_cng_research/results/external_product_prediction_benchmark_25k_repaired_20260713/benchmark_validity_aware/benchmark_summary.json` | `238d7f6cc2135b40d0fa2dce3df55fab060b35b2ed1ef7ef52609ed4d526093a` |
| `/home/cunyuliu/pc_cng_research/results/external_product_prediction_benchmark_25k_repaired_20260713/benchmark_validity_aware/candidate_scores.csv` | `059f512f850691c0ffea5db668ab0c4a5e0e20e80de9c345a0d951859f091a03` |
| `/home/cunyuliu/pc_cng_research/results/external_product_prediction_benchmark_25k_repaired_20260713/benchmark_validity_aware/paper_table.md` | `8edac03aef612e84f43de763f81c51ed6932f1ed4d28f878d7c60f604fb13d48` |
| `/home/cunyuliu/pc_cng_research/chem_negative_sampling/pc_cng/analyze_external_score_calibration.py` | `a2e89bd606da518d0a0c30d626b3ff417733d4ff28f334a3446a7ebc77903449` |
| `/home/cunyuliu/pc_cng_research/results/external_score_calibration_25k_repaired_strict_20260715/external_score_calibration_summary.json` | `ba3ac572bd91d4e2bb8d7f758b36bddf4fab126bd8fe7a4bc34ecc0c39ecaae8` |
| `/home/cunyuliu/pc_cng_research/results/external_score_calibration_25k_repaired_strict_20260715/external_score_calibration_summary.md` | `c0e7854b5476722de6b476480ed9cf49c84728a4faf34f4ab4d823fb66106627` |
| `/home/cunyuliu/pc_cng_research/results/external_score_calibration_25k_repaired_validity_aware_20260715/external_score_calibration_summary.json` | `e87658cfd4c34cfbcb7581c5609ba4fdb946f93228ef7fb0ab7ddc18142499cd` |
| `/home/cunyuliu/pc_cng_research/results/external_score_calibration_25k_repaired_validity_aware_20260715/external_score_calibration_summary.md` | `cb7d2b5a0503dd02a5e6391004d98ecfde18c9bb3424d3bffff6175c0d8ce990` |
| `/home/cunyuliu/pc_cng_research/docs/PC-CNG-v3-external-heldout-calibration-protocol-20260715.md` | `cdbaa0ae2cdffc4baa66ca46810d1d3bdd826e7d8dd3249a733702033f81dbee` |
| `/home/cunyuliu/pc_cng_research/docs/PC-CNG-v3-external-heldout-calibration-protocol-20260715.json` | `c336179168ca27083b0085e0223fa4ab2fc4f41d2a4de378b05484a60ea5677a` |
| `/home/cunyuliu/pc_cng_research/results/external_calibration_heldout_contexts_5k_20260715/external_product_prediction_context_expansion_summary.json` | `0f6e43d4b9cab5274d612e6adc0af250f4bc41c2130b95e6b5771c8d6aeb5525` |
| `/home/cunyuliu/pc_cng_research/results/external_calibration_heldout_contexts_5k_20260715/external_product_prediction_context_expansion.csv` | `1d4e3fa8617acdbef02fb5e6d0a5191352d199c5505b4c8c05b99b12d7b2a19f` |
| `/home/cunyuliu/pc_cng_research/results/external_calibration_heldout_contexts_5k_20260715/external_product_prediction_context_expansion_chemformer_input.csv` | `d099b66bd7d923b1166ad50d8e19675a41eb7886e14642f38f7826a0e9e34518` |
| `/home/cunyuliu/pc_cng_research/results/external_calibration_heldout_benchmark_5k_20260715/base_observed_pc_cng_candidates_summary.json` | `68626c2384451bd17c4d9501099184d4a1874b733ab8e98adf8e9bc139d35d9a` |
| `/home/cunyuliu/pc_cng_research/results/external_calibration_heldout_benchmark_5k_20260715/base_observed_pc_cng_candidates.csv` | `f544ef44ad67a30182d944b056b86f41af294ebb108185e84a9b35467666986e` |
| `/home/cunyuliu/pc_cng_research/results/external_calibration_heldout_base_quality_5k_20260715/external_25k_base_candidate_quality_audit.json` | `e0406b8daca4802354e37195ed1b1e17eed26336595bd84723879956c3a0dada` |
| `/home/cunyuliu/pc_cng_research/results/external_calibration_heldout_base_quality_5k_20260715/external_25k_base_candidate_quality_audit.md` | `aaef8e937f91a3f0485ea1bf1e4856067465bf862d0895bbb169ff631077bd36` |
| `/home/cunyuliu/pc_cng_research/results/external_calibration_heldout_benchmark_5k_20260715/chemformer_input_chunks/chemformer_forward_input_heldout5k_chunk_chunks_manifest.json` | `5d7e85f48f5f2214e684f83db8ea3da0e6d3f0ec85e8a48be716a0726ed3c45a` |
| `/home/cunyuliu/pc_cng_research/results/external_calibration_heldout_benchmark_5k_20260715/chemformer_input_chunks/chemformer_forward_input_heldout5k_chunk_chunks_manifest.md` | `f99d943892125bfccde603e39e5c4a91340ca63119592ef5b6c08031c03dfc08` |
| `/home/cunyuliu/pc_cng_research/chem_negative_sampling/pc_cng/train_external_score_calibrator.py` | `6a86267f63429f7d7d05132cd9a6d8f074c257368b54dbc1252a7b9d00e3d703` |
| `/home/cunyuliu/pc_cng_research/results/external_score_calibrator_lr_v1_repaired25k_train_20260715/external_score_calibrator_model.json` | `e76fd3085ddfb2f474ed4c19b8f311007cd63de8d91df0ee5359b5a112f1b65a` |
| `/home/cunyuliu/pc_cng_research/results/external_score_calibrator_lr_v1_repaired25k_train_20260715/external_score_calibrator_summary.json` | `a37fe44bc88daa13b2906b99f1c40093c4d5c16025feea0af2c538a45e49af9c` |
| `/home/cunyuliu/pc_cng_research/results/external_score_calibrator_lr_v1_repaired25k_train_20260715/external_score_calibrator_summary.md` | `d26e84c8d8166677232737ac0de2c0942cc74855f68c266c1168cf668013b5fe` |
| `/home/cunyuliu/pc_cng_research/results/external_score_calibrator_lr_v1_repaired25k_train_20260715/candidate_scores_with_calibrator.csv` | `d2e7f2a67c1874db34840d848f2fad947d9f1323fdbd438ba937335ce1c5ec63` |
| `/home/cunyuliu/pc_cng_research/results/external_score_calibrator_pairwise_v1_repaired25k_train_20260715/external_score_calibrator_model.json` | `041dd006c5823a5c380db147f61f7a617b8e9bfb456c61fda8c2f033c93fd2c8` |
| `/home/cunyuliu/pc_cng_research/results/external_score_calibrator_pairwise_v1_repaired25k_train_20260715/external_score_calibrator_summary.json` | `0b3941de2ea94cc0620b6209b2412b42c18f8ab27049cf417f72f927bc56d347` |
| `/home/cunyuliu/pc_cng_research/results/external_score_calibrator_pairwise_v1_repaired25k_train_20260715/external_score_calibrator_summary.md` | `43eaa43e9d68267990fd73ead2fbb391a342eee12df1eb157cafae57e055467c` |
| `/home/cunyuliu/pc_cng_research/results/external_score_calibrator_pairwise_v1_repaired25k_train_20260715/candidate_scores_with_calibrator.csv` | `f7ed98227e9e13ad3a52d28a8e073edaae519d137e38b78892275d7721d6e19f` |
| `/home/cunyuliu/pc_cng_research/chem_negative_sampling/pc_cng/train_external_score_mlp_calibrator.py` | `104777af886fd1b93ae038e19eeee9a23034f6fd20f3aaba60939e121da0f8a1` |
| `/home/cunyuliu/pc_cng_research/results/external_score_mlp_calibrator_v1_repaired25k_train_20260715/external_score_mlp_calibrator_model.json` | `5086058b9532d4d53db156c1041a4108ab5afae1d3771198a7a39302f5b25922` |
| `/home/cunyuliu/pc_cng_research/results/external_score_mlp_calibrator_v1_repaired25k_train_20260715/external_score_mlp_calibrator_summary.json` | `a853b9d34976bd148079c114a969bff62641effa20cf80ae9ea7544853d4ef07` |
| `/home/cunyuliu/pc_cng_research/results/external_score_mlp_calibrator_v1_repaired25k_train_20260715/external_score_mlp_calibrator_summary.md` | `0bd6ae8855c7f10858b5d7d4fe820d1721fda2b0d3490eb2fffa06a476a3adaf` |
| `/home/cunyuliu/pc_cng_research/results/external_score_mlp_calibrator_v1_repaired25k_train_20260715/candidate_scores_with_mlp_calibrator.csv` | `f6662f17d17e90f654cc91f26269da303e8fe0acc1a7df5465df2ad28d1917ec` |
| `/home/cunyuliu/pc_cng_research/results/external_calibration_train_contexts_10k_20260715/external_product_prediction_context_expansion_summary.json` | `3f6486ddd0793020d2a22acae519133d512192517daaf1a0052dd03862ac02ee` |
| `/home/cunyuliu/pc_cng_research/results/external_calibration_train_contexts_10k_20260715/external_product_prediction_context_expansion.csv` | `049a6a25d7d44dbbdbda5d17ddf82a4f5484c6825530c12361a4312d63f07f77` |
| `/home/cunyuliu/pc_cng_research/results/external_calibration_val_contexts_2k_20260715/external_product_prediction_context_expansion_summary.json` | `d4191a49681a42aea53ba958cfdc7b80bcff62f05b5b156224f3715033af392e` |
| `/home/cunyuliu/pc_cng_research/results/external_calibration_val_contexts_2k_20260715/external_product_prediction_context_expansion.csv` | `3717a148448a50c837289dac5aefe131fb10ee1f6daac8ee14d4b480997cb221` |
| `/home/cunyuliu/pc_cng_research/results/external_calibration_trainval_contexts_12k_20260716/external_calibration_trainval_contexts_12k_summary.json` | `ffe39b065330bdf7b55edc86d1960f6610511ba6354693da201beff891da256f` |
| `/home/cunyuliu/pc_cng_research/results/external_calibration_trainval_contexts_12k_20260716/external_calibration_trainval_contexts_12k.csv` | `cf5618c383b91b178f0117c5fbbf3938b338deb1bc62d66b5164feaec3bd6981` |
| `/home/cunyuliu/pc_cng_research/results/external_calibration_trainval_benchmark_12k_20260716/base_observed_pc_cng_candidates_summary.json` | `edf60698047363fb38a6af4e0c8a78424de62b98b14fa5f60be247507cac25d9` |
| `/home/cunyuliu/pc_cng_research/results/external_calibration_trainval_benchmark_12k_20260716/base_observed_pc_cng_candidates.csv` | `e65bc6897946231a0c488089760baee9a0cdbbb2e3ed8cb0688e636cf146f55b` |
| `/home/cunyuliu/pc_cng_research/results/external_calibration_trainval_base_quality_12k_20260716/external_25k_base_candidate_quality_audit.json` | `46350db362c7c2aa63451a5617329aa220d6cfe53d641308b700a9c0cdac6654` |
| `/home/cunyuliu/pc_cng_research/results/external_calibration_trainval_base_quality_12k_20260716/external_25k_base_candidate_quality_audit.md` | `970e7b48c2aad7a5d2b65cb9485ca44facd82119080e5290a8d54c4cf8ee1c74` |
| `/home/cunyuliu/pc_cng_research/results/external_calibration_train_context_repair_2ctx_20260716/external_product_prediction_context_expansion_summary.json` | `b7f72ad7d30b06bf2710de21d7377a4ec2cd6cd49ce322729b040f86c1cfc62c` |
| `/home/cunyuliu/pc_cng_research/results/external_calibration_train_context_repair_2ctx_20260716/external_product_prediction_context_expansion.csv` | `b76cf83dfb0e00a28a5b17aa764cccb7fb0346488bd5c9567144a92c6f2a5c4e` |
| `/home/cunyuliu/pc_cng_research/results/external_calibration_trainval_contexts_12k_repaired_20260716/external_calibration_trainval_contexts_12k_repaired_summary.json` | `3aa1cb5e7e07b661a5ce4377dfaa363f221402aaa65a325e70bb9f79faa0397b` |
| `/home/cunyuliu/pc_cng_research/results/external_calibration_trainval_contexts_12k_repaired_20260716/external_calibration_trainval_contexts_12k_repaired.csv` | `47553df09911a4a06f2690173ecc6dde602dc708886c0c15e1a44f1d4c36e0b6` |
| `/home/cunyuliu/pc_cng_research/results/external_calibration_trainval_pc_cng_targeted_12k_repaired_20260716/pc_cng_candidates_summary.json` | `32f34a1e56ffc83152df7aeec13f57c7ab6838064a230b4432b07961971f5e9c` |
| `/home/cunyuliu/pc_cng_research/results/external_calibration_trainval_pc_cng_targeted_12k_repaired_20260716/pc_cng_candidates.csv` | `f22d877a6dbb8dc064c0eeea560afda67fe5010f0c9ee8fa472f2a3db69adcde` |
| `/home/cunyuliu/pc_cng_research/results/external_calibration_trainval_benchmark_12k_repaired_20260716/base_observed_pc_cng_candidates_summary.json` | `07c726357fb6d44c8744f2bd1a5fee26a1ec1efbdd44fc0505b4101f05609e75` |
| `/home/cunyuliu/pc_cng_research/results/external_calibration_trainval_benchmark_12k_repaired_20260716/base_observed_pc_cng_candidates.csv` | `61f4534bbb41573e05c97e9bf6d0296af50858203bce6ee204c4054073f0b270` |
| `/home/cunyuliu/pc_cng_research/results/external_calibration_trainval_base_quality_12k_repaired_20260716/external_25k_base_candidate_quality_audit.json` | `d14ca002ec9a1006838aea81340cb1452d0572d96a19bf3d0390c07e2cf32ec2` |
| `/home/cunyuliu/pc_cng_research/results/external_calibration_trainval_base_quality_12k_repaired_20260716/external_25k_base_candidate_quality_audit.md` | `f27c8f380f59d13b23825761cd644c92844a698863c5f3479089869a273e3d42` |
| `/home/cunyuliu/pc_cng_research/results/external_calibration_trainval_pc_cng_scored_12k_repaired_20260716/benchmark_summary.json` | `57e57150892387da37883b76202c4a0819a5e787244fc05b4541945bb333d06b` |
| `/home/cunyuliu/pc_cng_research/results/external_calibration_trainval_pc_cng_scored_12k_repaired_20260716/candidate_scores.csv` | `e14f117a0eb2eb59eb27cfae551d1eced7183c972d48dc4ea09087450fa2fb5e` |
| `/home/cunyuliu/pc_cng_research/results/external_calibration_trainval_pc_cng_scored_12k_repaired_20260716/paper_table.csv` | `47d0850e59e2082445713393bf9cb1d4b1004496a2aeb68bca34f27a374da4b1` |
| `/home/cunyuliu/pc_cng_research/results/external_calibration_trainval_pc_cng_scored_12k_repaired_20260716/paper_table.md` | `6e1e4ea958bc289819c58bae0c1d2da353368624d3086519da762ae1387a4544` |
| `/home/cunyuliu/pc_cng_research/results/external_calibration_trainval_chemformer_ll_12k_repaired_20260716/lm_scores_chemformer_log_likelihood_summary.json` | `6119b1f6d92fdb12c4d13d75cf1d24df81c54499c07035f48ef001dc791527af` |
| `/home/cunyuliu/pc_cng_research/results/external_calibration_trainval_chemformer_ll_12k_repaired_20260716/lm_scores_chemformer_log_likelihood.csv` | `f84d33dd80b944b81147031a153c13b8efe1184d1aab12469d1fa50cd28ace85` |
| `/home/cunyuliu/pc_cng_research/results/external_calibration_trainval_scored_12k_repaired_20260716/benchmark_summary.json` | `013ffd82881b1edbfa99541f47e64f41c452e65c233c3e793a010c370c0a01b2` |
| `/home/cunyuliu/pc_cng_research/results/external_calibration_trainval_scored_12k_repaired_20260716/candidate_scores.csv` | `7c899400fd087da8fda3577be7d364ae86884d6054f2ca6a3ca3b2e6c1b6cb04` |
| `/home/cunyuliu/pc_cng_research/results/external_score_calibrator_lr_v1_uspto_trainval12k_repaired_20260716/external_score_calibrator_summary.json` | `621c77fde0de038463a81b864bf462c2b60a8c3fe226882f2faefbd6773df06b` |
| `/home/cunyuliu/pc_cng_research/results/external_score_calibrator_pairwise_v1_uspto_trainval12k_repaired_20260716/external_score_calibrator_summary.json` | `500337cfc2484c4f0d8f2c60235942ad3d6036871955bd0a7201a0b86784383e` |
| `/home/cunyuliu/pc_cng_research/results/external_score_mlp_calibrator_v1_uspto_trainval12k_repaired_20260716/external_score_mlp_calibrator_summary.json` | `de5dedd41fda81afb447f419230ef8cc854606b6f94fbee826935ba96dbe65ad` |
| `/home/cunyuliu/pc_cng_research/results/external_calibration_heldout_pc_cng_scored_base_5k_20260716/benchmark_summary.json` | `238836c4f921b59c4a9eac3a4bc56cdf5a1239d576bf0df1811c7cf62e9f0f3a` |
| `/home/cunyuliu/pc_cng_research/results/external_calibration_heldout_pc_cng_scored_base_5k_20260716/candidate_scores.csv` | `e45891c460822a0fe54e9acdc684fe549f9c48a68be5f2d3847d8b53bba34a37` |
| `/home/cunyuliu/pc_cng_research/results/external_calibration_heldout_chemformer_ll_base_5k_20260716/lm_scores_chemformer_log_likelihood_summary.json` | `aae5b5a1d4ee28f58b10864e4fb8faab817200b02c5dbe2ee111b784495ad35f` |
| `/home/cunyuliu/pc_cng_research/results/external_calibration_heldout_chemformer_ll_base_5k_20260716/lm_scores_chemformer_log_likelihood.csv` | `dd2841c92510daeeb38354da2e4335dc96d35e38540925ba6181819b34e6fc80` |
| `/home/cunyuliu/pc_cng_research/results/external_calibration_heldout_base_scored_5k_20260716/benchmark_summary.json` | `7c21e9f5bfc40a3376fefe96410f75ed2d3887bcb2c58501bb3c36ee905e19e6` |
| `/home/cunyuliu/pc_cng_research/results/external_score_mlp_calibrator_v1_uspto12k_apply_heldout_base5k_20260716/external_score_mlp_calibrator_apply_summary.json` | `e99e90188ba391c2092f254ad1210434e1b4010534a7c6a0ca02ed643acd0873` |
| `/home/cunyuliu/pc_cng_research/results/external_score_mlp_calibrator_v1_uspto12k_apply_heldout_base5k_20260716/candidate_scores_with_mlp_calibrator.csv` | `55151782cf72a4e6cf29305fb33f3536b78e9c27d073f778ceea9f3c01b86d9e` |
| `/home/cunyuliu/pc_cng_research/results/external_product_prediction_support_audit_25k_repaired_20260714/external_product_prediction_support_audit.json` | `001f81f30541a964331e086b48223312b4506a23976989a223cee2f6a2b7c0b9` |
| `/home/cunyuliu/pc_cng_research/results/external_product_prediction_support_audit_25k_repaired_20260714/external_product_prediction_support_audit.md` | `468cfc3c09bb5e5eb5382d820014417c410afc5ecf17e8cf93548e6764a4cfa6` |
| `/home/cunyuliu/pc_cng_research/results/logs/type1_v2_gpu5_relaxed_chain_20260714.pid` | `0b47fa66d647eee070483d63e25966075e943a0f6606601b7a505de9f6425a90` |
| `/home/cunyuliu/pc_cng_research/results/logs/type1_v2_gpu5_relaxed_chain_20260714.log` | `5d649367a2cf7a6332e0776a355be3fd33ceb1693497c796f076410673628640` |
| `/home/cunyuliu/pc_cng_research/results/type1_v2_filtered_baseline_20260712/v2_filtered_pairwise_seed20260710/metrics.json` | `c00ece3673513ed0206c0bfe1fd67c5fd57a9bbf20a634f27474737712fcb4c4` |
| `/home/cunyuliu/pc_cng_research/results/type1_v2_filtered_baseline_20260712/v2_filtered_pairwise_seed20260710/rerank_same_split/ranking_metrics.json` | `7c82cd1911be9bada91ee7a86bb3951122a8c0c7aa8e8d798a3c977825e3a573` |
| `/home/cunyuliu/pc_cng_research/results/type1_v2_filtered_baseline_20260712/v2_filtered_pairwise_seed20260711/metrics.json` | `86ab97e651a68ee2f5e92cb092c99de11911a3454ce2ee1e09d5b2181ae63e94` |
| `/home/cunyuliu/pc_cng_research/results/type1_v2_filtered_baseline_20260712/v2_filtered_pairwise_seed20260711/rerank_same_split/ranking_metrics.json` | `80eec1a144fe56318d4ea13bd236bdf0160a47b7b9349a230387f50d12c74f55` |
| `/home/cunyuliu/pc_cng_research/results/type1_v2_filtered_baseline_20260712/filtered_baseline_multiseed_summary/summary.json` | `5d4216f43ab37c8d0b2bbdd960e0779e1ff81bda9b47e43e54905c03a69f27db` |
| `/home/cunyuliu/pc_cng_research/results/type1_v2_filtered_baseline_20260712/paired_significance_original_v2_vs_filtered_v2/summary.json` | `11da920ac505ae397aeeb27d73e35f752fe3721206261dc8e07c3fd2fe19bb14` |
| `/home/cunyuliu/pc_cng_research/results/type1_v2_valtop1_ckpt_smoke_20260712/valtop1_ckpt_smoke_summary/summary.json` | `859df2de150a5b4d8f59a41a8aba8e8ed14b7867d558df744313ff4d3d05501f` |
| `/home/cunyuliu/pc_cng_research/results/type1_v2_valtop1_ckpt_smoke_20260712/paired_significance_v2_vs_valtop1_ckpt_smoke/summary.json` | `83e026080118015367fd413d94af414888603894018dba5dd5b24b61f552a779` |
| `/home/cunyuliu/pc_cng_research/results/logs/type1_v2_coslr_warm5_relaxed_20260714.pid` | `925fc0b6d06e3bfc116f3114bdf8c86fcdef3e85a7c53bbdef15eb5ee69cdf88` |
| `/home/cunyuliu/pc_cng_research/results/logs/type1_v2_coslr_warm5_relaxed_20260714.log` | `91195710698de1f70f2e0e8fef1526932b6d48159b980b3ef0b34e6df06fbc87` |
| `/home/cunyuliu/pc_cng_research/results/type1_v2_coslr_warm5_20260712/v2_coslr_warm5_pairwise_seed20260710/metrics.json` | `15c641ba39dc73fd11209e713994369a45aa752a3fa3415dc7a551e2bcbadd28` |
| `/home/cunyuliu/pc_cng_research/results/type1_v2_coslr_warm5_20260712/v2_coslr_warm5_pairwise_seed20260710/rerank_same_split/ranking_metrics.json` | `f9308ddfe159f2fbe5ea3f060c89f4123d06d918ffc741f4bc812d4b0167481b` |
| `/home/cunyuliu/pc_cng_research/results/type1_v2_representation_scale_smoke_20260712/v2_nbits8192_pairwise_seed20260710/metrics.json` | `d78ee823e3a325ea164954dfde8beba02e67936598c51288c65a45066ded1708` |
| `/home/cunyuliu/pc_cng_research/results/type1_v2_representation_scale_smoke_20260712/v2_nbits8192_pairwise_seed20260710/rerank_same_split/ranking_metrics.json` | `9b08347495fcb140a6f736218ea503de1222ad7c0a51a6535cbcb7b5b21a24e6` |
| `/home/cunyuliu/pc_cng_research/results/type1_v2_representation_scale_smoke_20260712/v2_nbits8192_pairwise_seed20260711/metrics.json` | `dd6121335143f96590be1459f805750267e0f2df763cf402404c8a942cdf737c` |
| `/home/cunyuliu/pc_cng_research/results/type1_v2_representation_scale_smoke_20260712/v2_nbits8192_pairwise_seed20260711/rerank_same_split/ranking_metrics.json` | `ceec2e2be41fb2b7eeaf91f05302bffb9b1019bda4e9503cab5d53161b0307ff` |
| `/home/cunyuliu/pc_cng_research/results/type1_v2_representation_scale_smoke_20260712/v2_nbits8192_pairwise_seed20260712/rerank_same_split/ranking_metrics.json` | `52adbc75c5bfe5d72b825ae615d08b90b4c145e9eee83421496972066ffc7e56` |
| `/home/cunyuliu/pc_cng_research/results/type1_v2_representation_scale_smoke_20260712/nbits8192_smoke_summary/summary.json` | `b0154bfdf9e37e5e6f1b1f38647710daf06d1220b1a52e5db92f1b2168e7c1ca` |
| `/home/cunyuliu/pc_cng_research/results/type1_v2_representation_scale_smoke_20260712/paired_significance_v2_vs_nbits8192_smoke/summary.json` | `5e39881c2eb6d162c5848bb27c0bd3b3a98577ad46d3a322d036c99e384e1a99` |
| `/home/cunyuliu/pc_cng_research/results/logs/type1_v2_nbits8192_10seed_20260714.pid` | `089d9f133166acccf6900189cda89a29a8341705ade252ceb5779fdda9482273` |
| `/home/cunyuliu/pc_cng_research/results/logs/type1_v2_nbits8192_10seed_20260714.log` | `87455c2baf79eb4c6d788e7e12e1d1947bc59123ac36eab08aa5975b1f289cd2` |
| `/home/cunyuliu/pc_cng_research/results/external_product_prediction_benchmark_25k_20260713/chemformer_input_chunks/chemformer_forward_input_25k_chunks_manifest.json` | `76bd5162c915c4acac580981a7592f269945edc5abda6ee972669872f8a0be5e` |
| `/home/cunyuliu/pc_cng_research/results/external_product_prediction_benchmark_25k_20260713/chemformer_input_chunks/chemformer_forward_input_25k_chunks_manifest.md` | `6e9e66784b3cc1b3d10086eb6e2f030911d138291702ccadd0a58fa0da3d37b4` |
| `/home/cunyuliu/pc_cng_research/results/external_product_prediction_benchmark_25k_20260713/chemformer_input_chunks/chemformer_forward_input_25k_chunk_0000.tsv` | `7da6eee64a2c4bc1b0547e79e11a3a9f965a83080f723954af409d31e0f4a70b` |
| `/home/cunyuliu/pc_cng_research/results/external_product_prediction_benchmark_25k_20260713/chemformer_input_chunks/chemformer_forward_input_25k_chunk_0001.tsv` | `0ee1e559ee002c630b2f80770627ae54452ed4504f7f1548bb67279b119a358a` |
| `/home/cunyuliu/pc_cng_research/results/external_product_prediction_benchmark_25k_20260713/chemformer_input_chunks/chemformer_forward_input_25k_chunk_0002.tsv` | `841c7939e6fde1b513625ac4a92f842087ae79eac063ac8c7027c5b249d2fb2c` |
| `/home/cunyuliu/pc_cng_research/results/external_product_prediction_benchmark_25k_20260713/chemformer_input_chunks/chemformer_forward_input_25k_chunk_0003.tsv` | `33bb6bdf4b806b2fc79fb710f7fed9e8af4fb3bae585816f80afca22add4ff8b` |
| `/home/cunyuliu/pc_cng_research/results/external_product_prediction_benchmark_25k_20260713/chemformer_input_chunks/chemformer_forward_input_25k_chunk_0004.tsv` | `3becdd974b0be5233b81375a133e70ab34c21fbbc170d232fcb7f250f4d79394` |
| `/home/cunyuliu/pc_cng_research/results/external_product_prediction_benchmark_25k_20260713/chemformer_beam_chunks/chemformer_forward_beam_chunks_status.json` | `3ae8fbd5a8fbbc5ccafbc9c46df90b671448d6aedea2a4e89db81fd6de4e3ab7` |
| `/home/cunyuliu/pc_cng_research/results/active_execution_status_20260713/active_execution_status.json` | `6db0448a7997f02d9fdf8c68bb6588e757046117457f1df866c53e14fe34eb12` |
| `/home/cunyuliu/pc_cng_research/results/active_execution_status_20260713/active_execution_status.md` | `9f247d21614a26272940a124152a133943cb46d2b2d0a7c5cb5a860e40a4e40a` |
| `/home/cunyuliu/pc_cng_research/run_v2_unreacted_expanded_m3_uspto_eval.sh` | `b77d890b08c510893c3ac4945531ac4ba96da430ae05d66e9aee25f3ea065015` |
| `/home/cunyuliu/pc_cng_research/run_expanded_m3_uspto_multiseed_eval.sh` | `4f196e58374e218a7e581e75d5458ffb8bc5ab9235098cd1469c636d029e6e17` |
| `/home/cunyuliu/pc_cng_research/results/type1_v2_unreacted_expanded_m3_uspto_eval_20260712/v2_unreacted_expanded_m3_uspto_summary/summary.json` | `c87691324dbc3641528162a8113ee4756ce2759a7ac4c8dfaffc7cd4f203c5d3` |
| `/home/cunyuliu/pc_cng_research/results/type1_v2_unreacted_expanded_m3_uspto_eval_20260712/v2_unreacted_expanded_m3_uspto_per_seed.csv` | `690e5e7b86b1a54796a8544204aa6215c51d70fccd0c1ac70e5b7ab2da86d086` |
| `/home/cunyuliu/pc_cng_research/results/type1_v2_unreacted_expanded_m3_uspto_eval_20260712/v2_unreacted_expanded_m3_uspto_ensemble/ranking_metrics.json` | `b9c6cd61af6df999e497037c090bdd3e32df5c8aad49cb2d34e9ce4e9f47e8b1` |
| `/home/cunyuliu/pc_cng_research/results/type1_v2_unreacted_expanded_m3_uspto_eval_20260712/v2_unreacted_expanded_m3_uspto_ensemble/scoring_coverage_summary.json` | `a072d46ef7d7b321cbdbcc8e3fdc6aead3d5df35c0187273dedf08568f867a16` |
| `/home/cunyuliu/pc_cng_research/results/type1_v2_hidden4096_expanded_m3_uspto_eval_20260712/hidden4096_expanded_m3_uspto_summary/summary.json` | `619474e27d386753d237af2fa10d43ce2facb526658e66ff71fd540607b2a6c7` |
| `/home/cunyuliu/pc_cng_research/results/type1_v2_hidden4096_expanded_m3_uspto_eval_20260712/hidden4096_expanded_m3_uspto_summary_ensemble/ranking_metrics.json` | `05dfc9069862f6934dee915e40f4e60b6be72ecb4de190e76d3ecab4f56e6e72` |
| `/home/cunyuliu/pc_cng_research/results/type1_v2_dropout04_expanded_m3_uspto_eval_20260712/dropout04_expanded_m3_uspto_summary/summary.json` | `a9bafa32a0e5bd0f0558c6aab1c01e7d141f60393593fc4ce0e52ee7f813d974` |
| `/home/cunyuliu/pc_cng_research/results/type1_v2_dropout04_expanded_m3_uspto_eval_20260712/dropout04_expanded_m3_uspto_summary_ensemble/ranking_metrics.json` | `c899a5c18b45dbd7be4b61c80537a72924d09e4a7ba9d7a1fd03252a89b3da05` |
| `/home/cunyuliu/pc_cng_research/results/type1_expanded_m3_uspto_eval_comparison_20260712/paired_v2_vs_hidden4096/summary.json` | `d29e02c7481f422332aef9fedb1d317246de21cbf58bd94950593aa9c2c69a62` |
| `/home/cunyuliu/pc_cng_research/results/type1_expanded_m3_uspto_eval_comparison_20260712/paired_v2_vs_dropout04/summary.json` | `6a6783af0382a1aa2e7d10637b2ac62506af3af11fcc47d4e845303ba3e0166a` |
| `/home/cunyuliu/pc_cng_research/results/type1_combined_feature_expanded_m3_uspto_eval_20260713/combined_expanded_m3_uspto_summary/summary.json` | `57f3aace2a11fc8e33e25a25c44748147f3bf7262f924fa2010bcf60848b8236` |
| `/home/cunyuliu/pc_cng_research/results/type1_combined_feature_expanded_m3_uspto_eval_20260713/combined_expanded_m3_uspto_summary_ensemble/ranking_metrics.json` | `f08ccd68d450ef06c02e7fa12f4bd2c833fa780768abe39997c5fe77ee238db9` |
| `/home/cunyuliu/pc_cng_research/results/type1_expanded_m3_uspto_eval_comparison_20260712/paired_v2_vs_combined/summary.json` | `d19a8821e27f13c206c7c81870ab43480f03cbe3a86ecf1fe563d6573643f989` |
| `/home/cunyuliu/pc_cng_research/results/type1_classw050_rc_expanded_m3_uspto_eval_20260713/classw050_rc_expanded_m3_uspto_summary/summary.json` | `e5ff95c50ab164ef3664445d7ac3dc1f51ad8b42167af31b3e33491aba69418b` |
| `/home/cunyuliu/pc_cng_research/results/type1_classw050_rc_expanded_m3_uspto_eval_20260713/classw050_rc_expanded_m3_uspto_summary_ensemble/ranking_metrics.json` | `12b16fc97e0936347234853d79f12e46e624dc99e2356262215dd71b86ce68cd` |
| `/home/cunyuliu/pc_cng_research/results/type1_expanded_m3_uspto_eval_comparison_20260712/paired_v2_vs_classw050_rc/summary.json` | `5ea9510f44a06d2ed3b8fd84662ee7b555c339be8229a951cb5f54e190c6bcc0` |
| `/home/cunyuliu/pc_cng_research/results/type1_expanded_m3_uspto_eval_comparison_20260712/expanded_m3_uspto_model_comparison_with_significance.json` | `2108531aa7c331b246a5f1b2f74559b5de86d6ded6909bd51ec977e64e67800e` |
| `/home/cunyuliu/pc_cng_research/results/type1_expanded_m3_uspto_eval_comparison_20260712/expanded_m3_uspto_model_comparison_with_significance.md` | `2c5c2032b27e0d06f486f05e058ebcf94e185deb86d64e20065f6880344562e2` |
