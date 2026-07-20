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

Metrics:

| Metric | Target |
|---|---:|
| Strict shared Test Top-1 | >=85.0%, stretch >=90.0% |
| Strict shared MRR | >=90.0% |
| Strict shared NDCG | >=92.0% |
| Validity-aware Test Top-1 | maintain >=98.50%, stretch >=99.00% |
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
| cosine LR warmup | Active optimization branch | `results/type1_v2_coslr_warm5_20260712/`; queued on GPU 4 |
| val_top1 checkpoint | Active metric-alignment smoke | `results/type1_v2_valtop1_ckpt_smoke_20260712/`; queued on GPU 5 |
| n_bits=8192 / binary_count | Active representation smoke | `results/type1_v2_representation_scale_smoke_20260712/`; queued behind val_top1 |
| pairwise_weight / margin matrix | Active objective smoke | `results/type1_v2_pairwise_margin_smoke_20260712/`; queued behind representation-scale |

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

## 11. Current M1 status

| M1 item | Status |
|---|---|
| Benchmark manifest | This document |
| Fixed model-selection rule | Declared in sections 1 and 8 |
| Paper table schema | Declared in section 9 |
| Baseline list | Declared in section 7 |
| Statistics rule | Declared in section 1 |
