# PC-CNG v3 Benchmark Manifest

日期：2026-07-12

Manifest ID: `pc_cng_v3_benchmark_manifest_20260712`

服务器根目录：`/home/cunyuliu/pc_cng_research`

Python 环境：`/home/cunyuliu/miniconda3/envs/pc_cng_gpu/bin/python`

## 1. 固定 seed 与统计规则

| 用途 | Rule |
|---|---|
| Main-claim 训练 | 10 seeds: `20260710-20260719` |
| Smoke triage | 3 seeds: `20260710-20260712`; 只用于是否升 10-seed，不用于主张 |
| Main promotion | Original held-out Test Top-1 >= v2 + 1.0 pp，且 group-level ensemble CI 全正 |
| Group-level statistics | 10-seed ensemble scores, paired bootstrap CI, paired permutation p, sign-test p |
| Seed-level statistics | seed-index bootstrap CI |
| Test usage | Test set 只用于最终报告，不用于 checkpoint/hyperparameter 选择 |
| Checkpoint selection | 默认 `val_roc_auc`; 对齐分支可预声明 `val_top1`; 必须写入 `metrics.json/config` |

## 2. 数据集与 split

| Dataset | Role | Path | Split rule | Current use |
|---|---|---|---|---|
| RegioSQM20 normalized | Original same-context benchmark | `data/processed/regiosqm20_normalized.csv` | CSV `split`; parent/context inherited | Train/val/test, original held-out |
| HITEA full normalized | Original same-context benchmark | `data/processed/hitea_full_normalized.csv` | CSV `split`; parent/context inherited | Train/val/test, original held-out |
| PC-CNG diverse-anchor reviewed | Boundary negatives | `results/type1_diverse_anchor_full/diverse_anchor_candidates_reviewed_knownpos_filtered.csv` | Synthetic train only via parent source split; 10 known-positive-overlap keep rows removed from original reviewed CSV | Pairwise training and reranking candidates |
| Class quota supplement | Weak-class stress negatives | `results/type1_class_quota_supplement_20260711/class_quota_candidates_reviewed.csv` | Synthetic train only via parent source split | Pairwise training |
| Class fallback supplement | Weak-class stress negatives | `results/type1_class_fallback_supplement_20260711/class_fallback_candidates_reviewed.csv` | Synthetic train only via parent source split | Pairwise training |
| Partial-product supplement | Negative ablation | `results/type1_partial_product_supplement_20260711/partial_product_candidates_reviewed.csv` | Synthetic train only via parent source split | Pairwise training, not main claim |
| Unreacted-substrate v2 supplement | Hydrogenation/Rh support | `results/type1_unreacted_substrate_supplement_v2_20260711/unreacted_substrate_candidates_reviewed.csv` | Synthetic train only via parent source split | Current v2 baseline |
| Curated weak-class contexts | Amide/Cu support | `results/type1_curated_weak_class_contexts_20260711/` | Curated context split must be source-audited | Supplement weak-class benchmark |
| Chemformer beam benchmark | External product-selection bridge | `results/external_product_prediction_benchmark_20260711/` | External context split in benchmark artifacts | Strict shared and validity-aware bridge |

## 3. Task A: original same-context candidate reranking

Scientific question: given one reaction context, can the scorer rank the observed product above real and PC-CNG boundary-negative candidates?

| Field | Requirement |
|---|---|
| `source_id` | Parent reaction id |
| `reaction_smiles` | Reactants/reagents to candidate product |
| `label` | `1` observed positive, `0` real/synthetic negative |
| `split` | Train/val/test inherited from parent positive |
| `dataset` | `regiosqm20`, `hitea_full`, or curated/external dataset id |
| `reaction_class` | Required for weak-class audit |
| `review_status` | Synthetic negatives must be reviewed-status aware |

Evaluation command pattern:

```bash
PYTHONPATH=. python -m pc_cng.evaluate_candidate_reranking \
  --real-csv data/processed/regiosqm20_normalized.csv \
  --real-csv data/processed/hitea_full_normalized.csv \
  --synthetic-csv <reviewed_csv> \
  --model-dir <seed_model_dir> \
  --output-dir <seed_model_dir>/rerank_same_split \
  --group-by reactants \
  --candidate-scope same_split \
  --batch-size 4096
```

Primary metrics:

| Metric | Level | Promotion use |
|---|---|---|
| Top-1 | overall, split, test, dataset, candidate source | Headline same-context metric |
| Top-3 | same as above | Shortlist robustness |
| MRR | same as above | Ranking quality beyond Top-1 |
| NDCG | same as above | Discounted ranking quality |
| ROC-AUC/AUPRC/F1 | binary val/test | Auxiliary training health only |

Required artifacts per seed:

| Artifact | Required path pattern |
|---|---|
| Training metrics | `<seed_dir>/metrics.json` |
| Checkpoint | `<seed_dir>/best_pairwise_reward_mlp.pt` |
| Candidate scores | `<seed_dir>/rerank_same_split/candidate_scores.csv` |
| Ranking metrics | `<seed_dir>/rerank_same_split/ranking_metrics.json` |

Required artifacts per 10-seed branch:

| Artifact | Required path pattern |
|---|---|
| Multiseed summary | `<branch_dir>/<branch>_multiseed_summary/summary.json` |
| Paired significance | `<branch_dir>/paired_significance_v2_vs_<branch>_same_split/summary.json` |
| Group deltas | `<branch_dir>/paired_significance_v2_vs_<branch>_same_split/paired_group_deltas.csv` |

## 4. Task B: external product-selection bridge

Scientific question: in Chemformer/Molecular Transformer-style beam candidates, can PC-CNG choose the observed product better than frozen likelihood?

| Variant | Candidate scope | Required reporting |
|---|---|---|
| Strict shared intersection | Candidates scored by all compared methods | Strongest apples-to-apples bridge table |
| Validity-aware full beam | Valid/featurizable generated candidates | Product-selection bridge; not pure generation |
| Hybrid beam + PC-CNG candidates | External model beams plus PC-CNG candidates | Exploratory bridge only |

Current artifact roots:

| Artifact | Path |
|---|---|
| Strict benchmark summary | `results/external_product_prediction_benchmark_20260711/benchmark/benchmark_summary.json` |
| Strict candidate scores | `results/external_product_prediction_benchmark_20260711/benchmark/candidate_scores.csv` |
| Validity-aware summary | `results/external_product_prediction_benchmark_20260711/benchmark_validity_aware/benchmark_summary.json` |
| Validity-aware candidate scores | `results/external_product_prediction_benchmark_20260711/benchmark_validity_aware/candidate_scores.csv` |
| Support/coverage audit | `results/external_product_prediction_support_audit_20260713/external_product_prediction_support_audit.json` |

Legacy 16k denominator audit:

| Denominator | Current | Target / note |
|---|---:|---|
| Source contexts | `16,050` | target `25,000`, deficit `8,950` |
| Full candidate rows | `175,678` | `16,050` groups |
| Groups with PC-CNG candidates | `600` | strict coverage bottleneck |
| Strict complete groups | `1,197` | target not met |
| Strict test groups | `81` | PC-CNG Test Top-1 `71.60%` |
| Validity-aware groups | `15,973` | PC-CNG Test Top-1 `98.50%` |

Current repaired 25k denominator audit:

| Denominator | Current | Target / note |
|---|---:|---|
| Source contexts | `25,000` | target met |
| Full candidate rows | `311,150` | `25,000` groups |
| Groups with PC-CNG candidates | `24,980` base negative groups; `25,000` strict complete scored groups after positive/external overlap | strict denominator target met |
| Strict complete groups | `25,000` | target met |
| Strict test groups | `10,563` | Chemformer likelihood Test Top-1 `57.00%`; PC-CNG `13.59%` |
| Validity-aware rows | `311,150` | Chemformer likelihood Test Top-1 `44.02%`; PC-CNG scored subset `13.59%` |

Claim boundary: repaired 25k denominator scale is complete, but PC-CNG
underperforms frozen Chemformer likelihood; external bridge is negative
performance evidence, not a SOTA success claim.

Context expansion input:

| Artifact | Value |
|---|---|
| Expansion summary | `results/external_product_prediction_context_expansion_20260713/external_product_prediction_context_expansion_summary.json` |
| Expansion contexts | `results/external_product_prediction_context_expansion_20260713/external_product_prediction_context_expansion.csv` |
| Merged 25k contexts | `results/external_product_prediction_context_expansion_20260713/external_product_prediction_contexts_merged.csv` |
| Expansion Chemformer input | `results/external_product_prediction_context_expansion_20260713/external_product_prediction_context_expansion_chemformer_input.csv` |
| Merged Chemformer input | `results/external_product_prediction_context_expansion_20260713/external_product_prediction_contexts_merged_chemformer_input.csv` |
| Selection result | selected `8,950` USPTO/OpenMolecules contexts; merged contexts `25,000/25,000` |
| Next gate | generate external beams and PC-CNG candidates/scores, then rerun strict/validity-aware support audit |

25k benchmark prebuild:

| Artifact | Value |
|---|---|
| Runner | `chem_negative_sampling/scripts_run_external_product_prediction_benchmark_25k.sh` |
| Result dir | `results/external_product_prediction_benchmark_25k_20260713/` |
| Contexts / Chemformer input | `25,000` rows each, copied from merged expansion inputs |
| Base candidates | `results/external_product_prediction_benchmark_25k_20260713/base_observed_pc_cng_candidates.csv` |
| Base summary | `results/external_product_prediction_benchmark_25k_20260713/base_observed_pc_cng_candidates_summary.json` |
| Targeted PC-CNG generator | `chem_negative_sampling/pc_cng/generate_external_context_pc_cng_candidates.py`; generated `47,259` forward-outcome candidates over `24,903` groups |
| Targeted PC-CNG output | `results/external_product_prediction_25k_pc_cng_targeted_20260713/external_product_prediction_25k_pc_cng_targeted_candidates.csv` |
| Base candidate rows | `76,487` rows: `25,000` observed positives + `51,487` PC-CNG rows |
| PC-CNG negative group coverage | `24,903/25,000` groups after targeted context generation |
| Base quality audit | `results/external_25k_base_candidate_quality_audit_20260713/external_25k_base_candidate_quality_audit.json`; `pass_with_warnings`, warning=`77` blank-reactant observed positives |
| Next gate | repair/replace/filter `77` blank-reactant contexts, then generate 25k Chemformer beams, build full candidates, score/evaluate strict and validity-aware variants |
| Chemformer input chunks | `results/external_product_prediction_benchmark_25k_20260713/chemformer_input_chunks/chemformer_forward_input_25k_chunks_manifest.json`; 5 chunks × 5,000 rows |
| Chunked beam runner | `chem_negative_sampling/scripts_run_external_product_prediction_25k_chunked_beams.sh`; dry-run status at `results/external_product_prediction_benchmark_25k_20260713/chemformer_beam_chunks/chemformer_forward_beam_chunks_status.json` |
| Beam runner safety | waits for GPU memory/util thresholds and zero compute apps by default; skips already valid chunks and merges only after all chunks are valid |
| Active beam watcher | PID `2156024`, log `results/logs/external_product_prediction_25k_chunked_beams.queue.log`; GPU1, mem<=`2500MiB`, util<=`10%`, compute apps=`0`; currently waiting on chunk 0 |

25k repaired benchmark prebuild:

| Artifact | Value |
|---|---|
| Repair dir | `results/external_product_prediction_context_repair_20260713/` |
| Repaired result dir | `results/external_product_prediction_benchmark_25k_repaired_20260713/` |
| Blank-reactant contexts removed | `77` |
| Replacement contexts | `77` USPTO/OpenMolecules test contexts selected with the same expansion filters |
| Repaired contexts / Chemformer input | `25,000` rows each |
| Repaired base candidates | `76,672` rows: `25,000` observed positives + `51,672` PC-CNG rows |
| Repaired PC-CNG negative coverage | `24,980/25,000` groups |
| Repaired quality audit | `results/external_25k_base_candidate_quality_audit_repaired_20260713/external_25k_base_candidate_quality_audit.json`; `pass`, invalid candidate reactions=`0` |
| Repaired Chemformer input chunks | `results/external_product_prediction_benchmark_25k_repaired_20260713/chemformer_input_chunks/chemformer_forward_input_25k_chunks_manifest.json`; 5 chunks × 5,000 rows |
| Beam runner header fix | `scripts_run_external_product_prediction_25k_chunked_beams.sh` now feeds Chemformer temporary no-header chunk inputs; header-included invalid chunk 0 output archived |
| Repaired beam chunks | `5/5` complete; each valid chunk has `5,001` lines |
| Merged repaired beam TSV | `results/external_product_prediction_benchmark_25k_repaired_20260713/chemformer_forward_beams.tsv`; `25,001` lines, SHA256 `e6bdb933e81288b5262a20a647ce1116e5be61c31cc43598e448ad23f7fcbcbb` |
| Repaired full candidate set | `results/external_product_prediction_benchmark_25k_repaired_20260713/full_observed_pc_cng_chemformer_beam_candidates.csv`; `311,150` candidates over `25,000` contexts |
| Repaired full candidate summary | `results/external_product_prediction_benchmark_25k_repaired_20260713/full_observed_pc_cng_chemformer_beam_candidates_summary.json`; observed positive `25,000`, PC-CNG `51,672`, Chemformer beam `235,548` |
| Chemformer likelihood scores | `results/external_product_prediction_benchmark_25k_repaired_20260713/lm_scores_chemformer_log_likelihood.csv`; `311,150` rows scored |
| Strict benchmark | `results/external_product_prediction_benchmark_25k_repaired_20260713/benchmark/benchmark_summary.json`; `25,000` complete groups, `116,509` evaluated rows |
| Strict test Top-1 | Chemformer likelihood `57.00%`; PC-CNG `13.59%`; best hybrid weight `0.00` |
| Validity-aware benchmark | `results/external_product_prediction_benchmark_25k_repaired_20260713/benchmark_validity_aware/benchmark_summary.json`; `311,150` evaluated rows |
| Validity-aware test Top-1 | Chemformer likelihood `44.02%`; PC-CNG scored subset `13.59%`; hybrid computed only on shared scored subset |
| Repaired support audit | `results/external_product_prediction_support_audit_25k_repaired_20260714/external_product_prediction_support_audit.json`; contexts `25,000/25,000`, strict complete groups `25,000/25,000`, decision flags `[]` |
| Score calibration audit | `results/external_score_calibration_25k_repaired_strict_20260715/external_score_calibration_summary.json`; strict shared rows `116,509`; Chemformer Top-1 `57.00%`, best nonzero hybrid `50.87%`, PC-CNG `13.59%` |
| Held-out calibration protocol | `docs/PC-CNG-v3-external-heldout-calibration-protocol-20260715.{md,json}`; selected new `5,000` test contexts outside repaired 25k; base candidate quality audit `pass`; scoring not run |
| Frozen calibration recipes | `results/external_score_calibrator_lr_v1_repaired25k_train_20260715/` and `results/external_score_calibrator_pairwise_v1_repaired25k_train_20260715/`; both val Top-1 `80.62%` < Chemformer `83.42%`; not primary |
| MLP calibration recipe | `results/external_score_mlp_calibrator_v1_repaired25k_train_20260715/`; val Top-1 `89.26%` > Chemformer `83.42%`, but repaired 25k test Top-1 `36.46%` < Chemformer `57.00%`; cross-domain failure |
| USPTO train/val calibration pool | `results/external_calibration_trainval_contexts_12k_repaired_20260716/`; split counts train `10,000`, val `2,000`; repaired base candidates `40,244`; quality `pass`; Chemformer val Top-1 `89.89%`, PC-CNG val Top-1 `16.42%`, simple hybrid selects `w0p00` |
| USPTO train/val frozen scorer recipes | LR val Top-1 `87.39%`, pairwise val Top-1 `84.68%`, MLP val Top-1 `93.14%`; MLP is validation-positive but not held-out evidence |
| Held-out 5k base-only diagnostic | `results/external_score_mlp_calibrator_v1_uspto12k_apply_heldout_base5k_20260716/`; base candidates only, `4,995` groups; Chemformer Top-1 `91.99%`, PC-CNG Top-1 `17.44%`, MLP Top-1 `94.51%`; diagnostic only, not full-beam protocol |
| Held-out 5k full-beam status | `results/logs/external_calibration_heldout5k_beams_20260716.log`; GPU6 run stopped after no TSV output, GPU7 run active; full-beam final scoring pending |
| Current gate | denominator scale complete; PC-CNG external performance negative, so no external SOTA claim |

Metrics:

| Metric | Target |
|---|---:|
| Strict shared Test Top-1 | recover to >=Chemformer `57.00%`; stretch >=85.0% |
| Strict shared MRR | recover to >=Chemformer `73.99%`; stretch >=90.0% |
| Strict shared NDCG | recover to >=Chemformer `80.48%`; stretch >=92.0% |
| Validity-aware Test Top-1 | recover to >=Chemformer `44.02%`; stretch >=85.0% |
| Coverage | Report groups and rows for every variant |

## 5. Task C: weak-class robustness

Classes:

```text
Amide coupling
Cu coupling
Hydrogenation
Rh coupling
Ni coupling
```

Support and performance gates:

| Gate | Requirement |
|---|---|
| Molecular parent support | >=20 distinct parent reactions per class |
| Evaluable candidate groups | >=20 groups per supported class |
| Per-class Top-1 | >=95% where support gate passes |
| Statistical test | paired CI positive and p < 0.05 vs v2 |
| Limitation | Ni must be solved or explicitly documented as data-source gap |

Current artifact roots:

| Artifact | Path |
|---|---|
| Reaction-class benchmark | `results/reaction_class_benchmark_20260711/` |
| classw050_rc branch | `results/type1_curated_weak_class_contexts_20260711/` |
| 10-seed paired significance | `results/type1_curated_weak_class_contexts_20260711/paired_significance_10seed/` |

## 6. Task D: Type-2 low-yield feasibility

Scientific question: can PC-CNG model low-yield/failed-reaction tendency, separate from candidate reranking?

| Metric | Phase-1 target | Paper-ready target |
|---|---:|---:|
| ROC-AUC | >=88.0% | >=90.0% |
| AUPRC | >=82.0% | >=85.0% |
| F1 | >=75.0% | >=78.0% |
| Calibration ECE | report | <=0.05 |

Claim boundary: Type-2 remains auxiliary unless it passes 10-seed stability and external-baseline alignment.

## 7. Fixed baselines and active branches

| Name | Role | Path / status |
|---|---|---|
| v2/unreacted | Main original-scope baseline | `results/type1_unreacted_substrate_supplement_v2_20260711/` |
| combined feature v2 | Architecture supplement | `results/type1_combined_feature_v2_20260712/` |
| classw050_rc | Weak-class supplement | `results/type1_curated_weak_class_contexts_20260711/` |
| hidden4096 | M0 scale-up negative result | `results/type1_v2_hidden4096_20260712/`; rejected as main |
| dropout04 | M0 regularization null result | `results/type1_v2_dropout04_20260712/`; rejected as main |
| v2 filtered known-positive-safe baseline | Same-data baseline for filtered-input branches | `results/type1_v2_filtered_baseline_20260712/`; queued first on GPU5 |
| cosine LR warmup | Complete, not promoted | `results/type1_v2_coslr_warm5_20260712/`; Test Top-1 `87.41 ± 1.82%`, paired CI crosses 0 |
| val_top1 checkpoint | Active metric-alignment smoke | `results/type1_v2_valtop1_ckpt_smoke_20260712/`; queued on GPU 5 |
| n_bits=8192 / binary_count | Representation smoke complete | `results/type1_v2_representation_scale_smoke_20260712/`; nbits8192 promoted to 10-seed validation; binary_count4096 not promoted |
| pairwise_weight / margin matrix | Selected 10-seed complete; not promoted | `results/type1_v2_pairwise_margin_10seed_20260714/`; `pw20_m000` Test Top-1 `88.15 ± 1.95%`, paired CI crosses 0; `pw20_m005` Test Top-1 `87.41 ± 2.52%`, paired CI crosses 0 |

## 8. Model-selection and promotion rules

1. Checkpoint selection must be declared before training and stored in `metrics.json/config`.
2. Historical baseline uses `val_roc_auc`; metric-alignment branch uses `val_top1`.
3. Hyperparameter/branch promotion can use val metrics and 3-seed smoke only.
4. Test metrics cannot be used to tune checkpoints; test is final reporting only.
5. A smoke branch is promoted to 10-seed only if 3-seed mean Test Top-1 is at least 87.7% or there is a compensating, predeclared weak-class/external gain.
6. A main branch is promoted only if 10-seed Test Top-1 improves by >=1.0 pp over v2 and paired CI is entirely positive.
7. Supplement branches must state task boundary and cannot replace main model unless main gates pass.
8. The pairwise objective matrix treats `pairwise_weight=1.0, margin=0.0` as the v2 baseline reference and runs only the 8 non-default cells before selecting at most two configs for 10-seed confirmation.

## 9. Paper table schema

Main and supplement tables must include these fields:

| Field | Meaning |
|---|---|
| `table_id` | Main/Supp table id |
| `task` | Task A/B/C/D |
| `scope` | original, expanded curated, strict external, validity-aware, weak-class, Type-2 |
| `dataset` | Dataset or benchmark artifact root |
| `candidate_scope` | same_split, all_group, strict_shared, validity_aware, hybrid |
| `model_name` | Branch/model id |
| `checkpoint_rule` | val_roc_auc, val_top1, ensemble, frozen likelihood |
| `seeds` | seed list or single/ensemble id |
| `metric` | Top-1, Top-3, MRR, NDCG, ROC-AUC, AUPRC, F1, calibration |
| `mean` | Metric mean |
| `std_or_ci` | Seed std or bootstrap CI |
| `baseline_name` | v2/unreacted, Chemformer, RegioSQM20, etc. |
| `delta_vs_baseline` | Absolute delta |
| `paired_ci` | Paired bootstrap CI where applicable |
| `permutation_p` | Paired permutation p where applicable |
| `sign_test_p` | Sign-test p where applicable |
| `n_groups` | Evaluable group count |
| `n_rows` | Candidate or row count |
| `artifact_path` | Reproducible result path |
| `claim_boundary` | main, supplement, bridge, auxiliary, limitation |

## 10. Data-quality gates

Every new benchmark/data expansion must report:

1. Parent/context leakage audit across train/val/test.
2. Known-positive filtering against all real positive products.
3. RDKit parse success/featurization success count.
4. Duplicate parent/context audit.
5. Per-class molecular support audit.
6. Candidate label provenance.
7. Negative difficulty audit: margin, hard-negative win rate, family distribution.

Current audit artifact:

| Artifact | Path / Status |
|---|---|
| Manifest JSON | `docs/PC-CNG-v3-benchmark-manifest-20260712.json` |
| Data-quality summary | `results/benchmark_manifest_data_quality_20260712/benchmark_data_quality_audit.json` |
| Real dataset table | `results/benchmark_manifest_data_quality_20260712/real_dataset_audit.md` |
| Synthetic dataset table | `results/benchmark_manifest_data_quality_20260712/synthetic_dataset_audit.md` |
| Status | `pass_with_warnings`; hard gate failures cleared after filtering diverse-anchor known-positive overlaps |
| Filtered diverse-anchor CSV | `results/type1_diverse_anchor_full/diverse_anchor_candidates_reviewed_knownpos_filtered.csv` |
| Removed diverse-anchor rows | `results/type1_diverse_anchor_full/diverse_anchor_candidates_reviewed_knownpos_removed.csv` |
| Unit tests | `chem_negative_sampling/tests/test_benchmark_data_quality_audit.py`; local and remote unittest discover passed, 2 tests |
| Original test expansion scan | `results/original_test_expansion_uspto_scan_20260712/uspto_original_test_expansion_scan.json`; 51,494 eligible held-out USPTO contexts, 256 selected positive parents |
| Original test expansion scan test | `chem_negative_sampling/tests/test_original_test_expansion_scan.py`; local and remote unittest discover passed, 1 test |
| Original test expansion negatives | `results/original_test_expansion_uspto_negatives_20260712/uspto_test_expansion_candidates_reviewed_knownpos_filtered.csv`; 1,616 reviewed/filtered rows, 1,268 keep negatives |
| M3 expanded support audit | `results/original_benchmark_support_audit_m3_uspto_20260712/original_benchmark_support_audit.json`; combined test groups 323/200, deficit 0 |
| M3 expanded data-quality audit | `results/original_test_expansion_uspto_data_quality_20260712/benchmark_data_quality_audit.json`; `pass_with_warnings`, hard gate failures 0 |
| M3 expanded v2/unreacted baseline | `results/type1_v2_unreacted_expanded_m3_uspto_eval_20260712/v2_unreacted_expanded_m3_uspto_summary/summary.json`; Test Top-1 51.60 ± 0.89% over 293 scored/evaluable test groups |
| M3 expanded ensemble baseline | `results/type1_v2_unreacted_expanded_m3_uspto_eval_20260712/v2_unreacted_expanded_m3_uspto_ensemble/ranking_metrics.json`; ensemble Test Top-1 51.88%, USPTO Top-1 39.15% |
| M3 expanded branch retest | `results/type1_expanded_m3_uspto_eval_comparison_20260712/expanded_m3_uspto_model_comparison_with_significance.md`; hidden4096 ΔTop-1 +0.48 pp CI crosses zero, dropout04 null, combined p=0.864, classw050_rc p=1.000 |

Remaining warnings to disclose:

1. RegioSQM20 and HITEA contain reactant-contexts that appear in multiple splits, although `source_id` and canonical parent-reaction leakage are zero.
2. Synthetic CSVs include val/test parent rows for reranking/evaluation candidate construction; training safety depends on the reader filtering synthetic training pairs to train-parent rows only.
3. Existing historical v2/unreacted results used the original diverse-anchor reviewed CSV. If a filtered-data branch is promoted, rerun v2 with the filtered CSV for final same-data comparison.
4. Current GPU5 smoke scripts are configured to use the filtered v2 baseline paths for paired significance once that queued baseline completes.

## 11. Current M1 status

| M1 item | Status |
|---|---|
| Benchmark manifest | This document |
| Machine-readable manifest | `PC-CNG-v3-benchmark-manifest-20260712.json` |
| Data-quality audit | `pass_with_warnings`; no hard gate failures after known-positive filtering |
| Data-quality tests | `test_benchmark_data_quality_audit.py` passed locally and remotely |
| Original support audit | `results/original_benchmark_support_audit_20260712/original_benchmark_support_audit.json`; pre-expansion combined test groups 111/200, deficit 89 |
| Original test expansion | support-audited: `results/original_benchmark_support_audit_m3_uspto_20260712/original_benchmark_support_audit.json`; expanded combined test groups 323/200 |
| Expanded v2 baseline | 10-seed mean Test Top-1 51.60 ± 0.89%; ensemble Test Top-1 51.88%; scorer evaluable test groups 293/323 support groups |
| Expanded branch retest | hidden4096, dropout04, combined, and classw050_rc do not pass paired Top-1 promotion gate; combined/classw050_rc remain supplement-only |
| Fixed model-selection rule | Declared in sections 1 and 8 |
| Paper table schema | Declared in section 9 |
| Baseline list | Declared in section 7 |
| Statistics rule | Declared in section 1 |
