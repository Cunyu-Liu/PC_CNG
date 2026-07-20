# PC-CNG v3 SOTA追赶目标与Scale-up执行计划

日期：2026-07-12

适用项目：未知化学反应推理 / 负样本生成模型 / PC-CNG v3

## 一、核心目标

本阶段核心目标是将 PC-CNG v3 从“方法学论文级候选重排序框架”推进到“顶刊投稿级、可与同领域强模型公平比较的反应候选选择与负样本边界学习框架”。

必须同时满足三类要求：

1. **下游任务对齐**：所有结果必须在预声明的 benchmark、输入输出格式、split、metric、seed、统计检验协议下比较，避免把 candidate reranking、validity-aware product selection 和 end-to-end forward product generation 混为同一任务。
2. **SOTA追赶与超越**：对每个可公平比较的指标设定明确目标值；对尚不可 apples-to-apples 比较的 SOTA，需要先完成任务桥接 benchmark，再进行追赶。
3. **scale up闭环**：模型参数量、训练数据规模、外部候选集、弱类数据支持、训练资源和统计验收必须同步扩展，不能只扩 hidden_dim。

最终论文目标：

```text
PC-CNG v3 should be positioned as a boundary-negative generation,
candidate-reranking, and validity-aware product-selection framework.
It should only claim SOTA where the downstream task and benchmark are aligned.
```

## 二、当前基线与主要差距

### 2.1 当前核心结果

| Task / Scope | Current best | Evidence | Current status |
|---|---:|---|---|
| Original Regio/HiTEA overall Top-1 | 97.40 ± 0.11% | combined 10-seed | strong, but not external SOTA task |
| Original Regio/HiTEA held-out test Top-1 | 87.16 ± 1.58% | v2/unreacted 10-seed | main remaining gap vs RegioSQM20 |
| Expanded curated overall Top-1 | 97.16 ± 0.30% | classw050_rc 10-seed | weak-class supplement strong |
| Expanded curated Test Top-1 | 83.98 ± 1.28% | classw050_rc 10-seed | improved, still should be pushed higher |
| Expanded curated Top-1 paired gain | +13.27 pp | 10-seed ensemble, p < 0.0001 | top-journal-grade weak-class evidence |
| Combined feature expanded Top-1 gain | +6.30 pp | 10-seed ensemble, p < 0.0001 | architecture ablation strong |
| External validity-aware Test Top-1 | 98.50% | PC-CNG vs Chemformer likelihood | already far above frozen Chemformer |
| External strict shared Test Top-1 | 71.60% | PC-CNG vs Chemformer likelihood | strong vs Chemformer, but target should rise |
| Type-2 low-yield ROC-AUC | 85.56 ± 0.21% | low_yield_synth05 | auxiliary branch, not main claim |
| Type-2 low-yield AUPRC | 79.93 ± 0.08% | low_yield_synth05 | needs improvement if included prominently |
| Type-2 low-yield F1 | 72.30 ± 0.08% | low_yield_synth10 | needs improvement |

### 2.2 Main SOTA gaps

| Gap | Current | SOTA / strong reference | Gap | Required action |
|---|---:|---:|---:|---|
| Original held-out candidate Top-1 vs RegioSQM20 with tautomers | 87.16% | 92.7% | -5.54 pp | raise held-out Top-1 to >=93.0% |
| Forward product prediction alignment vs Molecular Transformer | candidate reranking only | >90% top-1 on forward benchmarks | not apples-to-apples | build end-to-end or beam reranking bridge |
| Strict external shared-candidate benchmark | 71.60% | Chemformer likelihood 4.94% | already ahead | push strict Top-1 to >=85%, stretch >=90% |
| Weak-class complete support | Amide/Cu/H/Rh mostly solved; Ni gap | >=20 molecular parent reactions per class | Ni missing | acquire or curate >=20 Ni molecular contexts |
| Type-2 feasibility | ROC-AUC 85.56 / AUPRC 79.93 / F1 72.30 | no fixed external SOTA yet | internal gap | first align external baseline, then push to 90/85/78 |

## 三、统一下游任务、输入输出与评估协议

### 3.1 Task A: same-context candidate reranking

**Scientific question**：在同一 reactant/context 下，模型能否从真实产物和 PC-CNG 生成的边界负样本中把真实产物排到最前？

**Input**：

| Field | Requirement |
|---|---|
| `source_id` | parent reaction id; all candidates in one group share the same id |
| `reaction_smiles` | reactants > reagents > product candidate |
| `label` | 1 for observed positive, 0 for synthetic/failed/counterfactual negative |
| `split` | train / val / test, inherited from parent positive reaction |
| `dataset` | RegioSQM20 / HITEA / curated USPTO / external beam |
| `reaction_class` | reaction class label for weak-class audit |
| `review_status` | keep/exclude status for reviewed synthetic negatives |

**Output**：

| Artifact | Requirement |
|---|---|
| `candidate_scores.csv` | one score per candidate row |
| `ranking_metrics.json` | overall and split-wise Top-1 / Top-3 / MRR / NDCG |
| `metrics.json` | training config, binary metrics, counts, pair family/class counts |
| `summary.csv` | multi-seed mean/std/min/max |
| paired significance report | group-level ensemble CI + permutation p + sign test; seed-level bootstrap CI |

**Metrics**：

| Metric | Primary use | Success threshold |
|---|---|---:|
| Top-1 | headline ranking metric | main target >=93.0% on original held-out test |
| Top-3 | robustness / candidate shortlisting | >=98.0% on original held-out test |
| MRR | ranking quality beyond Top-1 | >=95.0% on original held-out test |
| NDCG | graded ranking quality | >=96.5% on original held-out test |
| Mean regret / score margin | error severity audit | decreasing vs v2 by >=10% |

### 3.2 Task B: external product-selection bridge

**Scientific question**：在 Chemformer / Molecular Transformer 生成的 beam candidates 中，PC-CNG scorer 能否比 frozen likelihood 更好地选择真实产物？

**Benchmark variants**：

| Variant | Definition | Required reporting |
|---|---|---|
| Strict shared intersection | only candidates scored by all models are included | strongest apples-to-apples table |
| Validity-aware full beam | full generated candidate pool with featurizability/validity filtering | product-selection bridge, not pure generation |
| Hybrid beam + PC-CNG candidates | external model beams plus PC-CNG boundary negatives | bridge toward forward prediction |

**Metrics and targets**：

| Metric | Current | Minimum target | Stretch target |
|---|---:|---:|---:|
| Strict shared Test Top-1 | 71.60% | >=85.0% | >=90.0% |
| Strict shared MRR | to be refreshed | >=90.0% | >=94.0% |
| Strict shared NDCG | to be refreshed | >=92.0% | >=96.0% |
| Validity-aware Test Top-1 | 98.50% | maintain >=98.50% | >=99.00% |
| Full beam coverage | 15,973 groups / 174,908 rows | maintain or expand | >=25,000 groups |
| Inference speed | not yet fully tabled | report CPU/GPU throughput | >=100x faster than DFT-style reranking if wall-clock reference available |

### 3.3 Task C: weak-class robustness benchmark

**Scientific question**：PC-CNG 是否只在主分布强，还是在弱反应类上也可靠？

**Classes**：

```text
Amide coupling
Cu coupling
Hydrogenation
Rh coupling
Ni coupling
```

**Support gate**：

| Requirement | Success standard |
|---|---|
| Molecular support | each class >=20 distinct molecular parent reactions |
| Candidate support | each class >=20 evaluable candidate groups |
| Performance | per-class Top-1 >=95% where support is sufficient |
| Statistical evidence | group-level paired p < 0.05 and positive bootstrap CI vs v2 |
| Limitation handling | if data source lacks class support, explicitly mark as data-source gap |

Current status:

| Class | Current status | Next target |
|---|---|---|
| Amide coupling | solved by curated contexts | keep >=95% Top-1 |
| Cu coupling | solved by curated contexts | keep >=95% Top-1 |
| Hydrogenation | support solved by unreacted-substrate v2 | raise strict Top-1 >=90%, tie-aware >=95% |
| Rh coupling | support solved, Top-1 100% | maintain |
| Ni coupling | hard data-source gap | collect/curate >=20 molecular contexts |

### 3.4 Task D: Type-2 low-yield feasibility

**Scientific question**：模型能否判断低产率/失败倾向，而不仅是候选重排序？

**Metrics and targets**：

| Metric | Current best | Phase-1 target | Paper-ready target |
|---|---:|---:|---:|
| Test ROC-AUC | 85.56 ± 0.21% | >=88.0% | >=90.0% |
| Test AUPRC | 79.93 ± 0.08% | >=82.0% | >=85.0% |
| Test F1 | 72.30 ± 0.08% | >=75.0% | >=78.0% |
| Calibration ECE | not yet tabled | report | <=0.05 |
| Class-wise F1 | incomplete | report for major classes | no major class below 65% |

Type-2 is not the main contribution unless it reaches the paper-ready targets above.

## 四、SOTA追赶目标矩阵

### 4.1 Non-negotiable headline targets

| Priority | Metric | Current | Must exceed | Final target | Stretch target | Promotion rule |
|---|---|---:|---:|---:|---:|---|
| P0 | Original held-out Test Top-1 | 87.16 ± 1.58% | RegioSQM20 no-tautomer 90.7% | >=93.0% | >=94.0% | 10-seed mean, CI positive vs v2 |
| P0 | Original held-out MRR | ~92-93% range in recent runs | internal v2 | >=95.0% | >=96.0% | no Top-1 regression |
| P0 | Original held-out NDCG | ~94-95% range in recent runs | internal v2 | >=96.5% | >=97.5% | no Top-1 regression |
| P0 | Strict external Test Top-1 | 71.60% | Chemformer 4.94% | >=85.0% | >=90.0% | strict shared benchmark |
| P0 | Validity-aware Test Top-1 | 98.50% | Chemformer 0.07% | >=98.50% | >=99.00% | maintain coverage |
| P0 | Expanded curated Top-1 | 97.16 ± 0.30% | v2 84.66% | >=97.50% | >=98.00% | no original-scope loss >0.5 pp |
| P0 | Weak-class per-class Top-1 | mixed | class-specific v2 | >=95.0% | >=97.0% | support >=20 groups |
| P1 | Type-2 ROC-AUC | 85.56 ± 0.21% | internal best | >=90.0% | >=92.0% | 10-seed stable |
| P1 | Type-2 AUPRC | 79.93 ± 0.08% | internal best | >=85.0% | >=88.0% | 10-seed stable |
| P1 | Type-2 F1 | 72.30 ± 0.08% | internal best | >=78.0% | >=80.0% | threshold tuned on val only |

### 4.2 Statistical promotion gates

A model can be promoted to “main candidate” only if all conditions hold:

1. 10 seeds: `20260710-20260719`.
2. Original held-out Test Top-1 improves by at least +1.0 pp over v2/unreacted.
3. Group-level ensemble paired bootstrap 95% CI is entirely positive.
4. Paired permutation p < 0.05 and sign-test p < 0.05.
5. No material regression on RegioSQM20, HITEA, synthetic candidate Top-1, MRR, or NDCG.
6. Inference cost and training cost are recorded.
7. All scripts, logs, checkpoints, ranking metrics, and tables are reproducible from a single manifest.

For a “supplement-only” model:

1. It may specialize in weak classes or external bridge tasks.
2. It must not be presented as a universal main-model replacement unless it passes the main promotion gates.
3. It must include a clear claim boundary in manuscript tables.

## 五、模型scale-up扩展计划

### 5.1 Current architecture baseline

Current MLP:

```text
input -> Linear(input_dim, hidden_dim) -> ReLU -> Dropout
      -> Linear(hidden_dim, hidden_dim / 2) -> ReLU -> Dropout
      -> Linear(hidden_dim / 2, 1)
```

Approximate parameter scale:

| Configuration | Input dim | Hidden dim | Approx params | Status |
|---|---:|---:|---:|---|
| Morgan n_bits=4096, hidden=2048 | 12,288 | 2,048 | ~27.3M | current v2 baseline |
| Combined 4096 + graph_stats, hidden=2048 | 12,452 | 2,048 | ~27.6M | completed, strong on expanded curated |
| Morgan n_bits=4096, hidden=4096 | 12,288 | 4,096 | ~58.7M | running; early trend negative on test |
| Morgan n_bits=8192, hidden=2048 | 24,576 | 2,048 | ~52.4M | planned |
| Combined n_bits=8192, hidden=2048 | 24,740 | 2,048 | ~52.8M | planned after n_bits smoke |
| Morgan n_bits=4096, hidden=8192 | 12,288 | 8,192 | ~134M | only if 4096 scale shows benefit |
| Binary+count Morgan 4096, hidden=2048 | 24,576 | 2,048 | ~52.4M | planned feature-scale branch |

Important interpretation:

```text
The 4096 hidden-dim branch currently suggests capacity alone is not enough.
Future scale-up should prioritize representation scale and data scale before
blindly increasing hidden_dim.
```

### 5.2 Model expansion stages

| Stage | Dates | Model change | Run size | Success criterion | Decision |
|---|---|---|---|---|---|
| S0 | 2026-07-12 | Finish hidden4096 and dropout04 | 10 seeds each | paired test complete | decide whether capacity/reg helps |
| S1 | 2026-07-12 to 2026-07-13 | Cosine LR + 5-epoch warmup | 10 seeds if GPU available | Test Top-1 >=88.0 or positive 3-seed trend | promote to 10-seed/pair test |
| S2 | 2026-07-13 | Early stopping / checkpoint selection by val Top-1 | 3 seeds smoke, then 10 seeds | val/test ranking alignment improves | keep if test not hurt |
| S3 | 2026-07-13 to 2026-07-14 | n_bits=8192 and binary_count features | 3 seeds per config | >=+0.5 pp test Top-1 over v2 in smoke | run selected config 10 seeds |
| S4 | 2026-07-14 to 2026-07-16 | combined + original-scope mitigation | 10 seeds | retain expanded gain while original test >=87.16 | candidate architecture branch |
| S5 | 2026-07-15 to 2026-07-17 | Chemformer reference score feature | 3 seeds smoke then 10 seeds | strict external Top-1 >=85; original test no regression | main bridge candidate |
| S6 | 2026-07-16 to 2026-07-19 | lightweight graph-pair / reaction-difference encoder | 5 seeds then 10 seeds | original test >=90 or weak-class gain significant | architecture upgrade |
| S7 | 2026-07-19 to 2026-07-21 | model-family ensemble | 10-seed family ensemble | original test >=93 or strict external >=90 | final SOTA candidate |

### 5.3 Training objective expansion

| Objective component | Current | Next experiment | Success criterion |
|---|---|---|---|
| BCE anchor | enabled | keep | binary ROC-AUC not collapsed |
| Pairwise loss | weight=1.0 | sweep 0.5 / 1.0 / 2.0 | ranking Top-1 improves |
| Pairwise margin | 0.0 | sweep 0.05 / 0.10 / class-aware margins | margin distribution improves |
| Class weights | classw050 validated | extend to combined/Chemformer-feature | weak-class gain without original loss |
| LR schedule | fixed lr=1e-3 | cosine to 1e-5, warmup=5 | smoother val/test curves |
| Checkpoint selection | val ROC-AUC | val Top-1 or composite metric | Top-1 alignment improves |
| Ensemble scoring | seed average only | family average Morgan/combined/graph/Chemformer-feature | significant Top-1 gain |

## 六、训练数据规模扩展方案

### 6.1 Current data sources

| Source | Role | Current use |
|---|---|---|
| RegioSQM20 | regioselectivity / original benchmark | real train/val/test |
| HITEA full normalized | broad reaction contexts | real train/val/test |
| PC-CNG diverse-anchor candidates | type-1 boundary negatives | pairwise preference training |
| class quota / class fallback candidates | weak-class stress tests | training + supplement |
| partial-product negatives | negative ablation | not selected as main |
| unreacted-substrate v2 candidates | Hydrogenation/Rh support | selected supplement |
| curated USPTO weak-class contexts | Amide/Cu support | selected supplement |
| Chemformer beam candidates | external bridge | benchmark/scoring |

### 6.2 Data expansion targets

| Priority | Data target | Current | Target size | Deadline | Acceptance |
|---|---|---:|---:|---|---|
| P0 | Original train groups | ~1,060 train groups | >=2,000 groups | 2026-07-15 | split-stable, no leakage |
| P0 | Original test groups | 81 groups | >=200 groups | 2026-07-16 | held-out molecular contexts |
| P0 | External beam benchmark | 15,973 groups | >=25,000 groups | 2026-07-16 | Chemformer/MolTrans comparable |
| P0 | Ni molecular contexts | 0 HITEA / 6 USPTO | >=20 distinct parent reactions | 2026-07-18 | manually or source-verified |
| P1 | Hard negative candidates per group | variable | 32 to 64 candidates/group | 2026-07-15 | reviewed-status-aware |
| P1 | Weak-class candidate groups | mixed | >=50 per solved class | 2026-07-17 | per-class Top-1 table |
| P1 | Type-2 low-yield labels | limited | +2x labeled/weak-labeled rows | 2026-07-18 | class-balanced audit |

### 6.3 Data quality gates

Every new data expansion must pass:

1. No parent leakage across train/val/test.
2. Known-positive filtering against all real positive products.
3. RDKit parse success rate reported.
4. Duplicate parent/context audit.
5. Per-class molecular support audit.
6. Candidate label provenance recorded.
7. Negative difficulty audit: score margin, hard negatives beating positive, family distribution.

## 七、计算资源分配与调度计划

### 7.1 Server constraint

Training and evaluation must run on:

```text
ssh cunyuliu@36.137.135.49 -p 22
```

Recommended project root:

```text
/home/cunyuliu/pc_cng_research
```

Python environment:

```text
/home/cunyuliu/miniconda3/envs/pc_cng_gpu/bin/python
```

### 7.2 GPU scheduling policy

| Resource | Use | Policy |
|---|---|---|
| GPU 4 | main 10-seed MLP/feature experiments | one training process at a time |
| GPU 5 | parallel regularization/data experiments | one training process at a time |
| Other free GPUs | reranking/evaluation/smoke tests | use only after checking `nvidia-smi` |
| CPU | featurization, summaries, manuscript tables | avoid competing with active GPU dataloading |

Operational rules:

1. Run at most two full 10-seed training jobs in parallel.
2. Run reranking immediately after each seed.
3. Write logs to `results/logs/`.
4. Store each experiment under a unique date-stamped directory.
5. After every 10-seed run, immediately generate:
   - multi-seed summary;
   - paired significance vs v2;
   - manuscript-ready table row;
   - experiment note in docs.

### 7.3 Expected runtime

| Experiment type | Per seed estimate | 10-seed estimate | Notes |
|---|---:|---:|---|
| Morgan hidden=2048 | 8-12 min | 1.5-2.5 h | baseline scale |
| Morgan hidden=4096 | 12-18 min | 2.5-4 h | current running branch |
| Combined hidden=2048 | 10-15 min | 2-3 h | modest feature overhead |
| n_bits=8192 hidden=2048 | 15-25 min | 3-5 h | larger input dimension |
| binary_count 4096 | 15-25 min | 3-5 h | input roughly doubles |
| graph-pair encoder | TBD | smoke first | depends on implementation |
| external beam evaluation | 10-60 min | per checkpoint/ensemble | depends on candidate count |

## 八、阶段性里程碑、交付物与验收标准

### Milestone M0: current in-flight experiments closed

Deadline: 2026-07-12

Deliverables:

1. hidden4096 10-seed complete.
2. dropout04 10-seed complete.
3. cosine LR scheduler experiment launched or queued.
4. Summary CSV and paired significance reports generated.
5. Progress document updated.

Acceptance:

| Item | Pass criterion |
|---|---|
| hidden4096 | 10/10 seeds, ranking metrics present |
| dropout04 | 10/10 seeds, ranking metrics present |
| paired tests | v2 vs each branch, Top-1/MRR/NDCG |
| decision | branch promoted, rejected, or kept as supplement with reason |

### Milestone M1: metric alignment and benchmark freeze

Deadline: 2026-07-13

Deliverables:

1. A benchmark manifest listing datasets, splits, candidate scopes, metrics, and baselines.
2. A fixed model-selection rule.
3. A paper table schema for all downstream tasks.

Acceptance:

| Item | Pass criterion |
|---|---|
| Input schema | all CSV columns documented and validated |
| Metrics | Top-1/Top-3/MRR/NDCG/ROC-AUC/AUPRC/F1/calibration defined |
| Baselines | v2, combined, classw050_rc, Chemformer, RegioSQM20 references included |
| Statistics | 10-seed + paired tests mandatory for main claims |

### Milestone M2: fast optimization sweep

Deadline: 2026-07-14

Deliverables:

1. Cosine LR 10-seed.
2. Early-stopping-by-val-Top-1 branch.
3. n_bits=8192 smoke and selected 10-seed branch.
4. pairwise-weight/margin smoke matrix.

Acceptance:

| Item | Pass criterion |
|---|---|
| Smoke promotion | >=+0.5 pp test Top-1 over v2 in 3-seed smoke |
| 10-seed promotion | >=+1.0 pp test Top-1 and positive paired CI |
| Rejection | clear negative/null result documented |

### Milestone M3: data scale-up and weak-class closure

Deadline: 2026-07-16 to 2026-07-18

Deliverables:

1. Expanded original held-out benchmark with >=200 test groups.
2. External beam benchmark expanded to >=25,000 groups.
3. Ni coupling curated/source-mined contexts.
4. Weak-class per-class table refreshed.

Acceptance:

| Item | Pass criterion |
|---|---|
| Expanded test | no leakage, stable split, support audit passed |
| Ni support | >=20 distinct molecular parent reactions or documented hard limitation |
| Weak classes | >=95% Top-1 for supported classes |
| External bridge | strict and validity-aware tables regenerated |

### Milestone M4: architecture scale-up

Deadline: 2026-07-17 to 2026-07-19

Deliverables:

1. Chemformer reference-score feature branch.
2. graph-aware reaction-difference branch.
3. combined + class-weighted mitigation branch.
4. family ensemble branch.

Acceptance:

| Item | Pass criterion |
|---|---|
| Original test | target >=90% at minimum, stretch >=93% |
| Strict external | target >=85%, stretch >=90% |
| Expanded curated | maintain >=97.5% overall |
| Statistics | 10-seed paired significance passes |

### Milestone M5: paper-ready evidence package

Deadline: 2026-07-20 to 2026-07-21

Deliverables:

1. Final manuscript tables.
2. Final SOTA gap table with absolute and relative deltas.
3. Full reproducibility manifest.
4. Training logs and result paths.
5. Limitations section, especially Ni data-source gap and task-scope boundaries.

Acceptance:

| Item | Pass criterion |
|---|---|
| Main table | includes current model, SOTA references, deltas, CI |
| Supplement | all ablations, negative results, weak-class audits |
| Reproducibility | scripts + seeds + environment + result paths complete |
| Claim boundary | no end-to-end SOTA overclaim unless benchmark supports it |

## 九、Prioritized todo list

### P0: today / immediate

1. Finish hidden4096 10-seed training and reranking.
   - Deliverable: `results/type1_v2_hidden4096_20260712/*/ranking_metrics.json`
   - Acceptance: 10/10 seeds complete.
2. Finish dropout04 10-seed training and reranking.
   - Deliverable: `results/type1_v2_dropout04_20260712/*/ranking_metrics.json`
   - Acceptance: 10/10 seeds complete.
3. Run `multiseed_summary.py` for hidden4096 and dropout04.
   - Acceptance: mean/std/min/max for overall/train/val/test Top-1/MRR/NDCG.
4. Run paired significance vs v2 for hidden4096 and dropout04.
   - Acceptance: group-level ensemble + seed-level bootstrap.
5. Start cosine LR + warmup 10-seed experiment as soon as GPU 4 is free.
   - Deliverable: `results/type1_v2_coslr_warm5_20260712/`
   - Acceptance: first seed completes training + rerank without error.
6. Update progress report and SOTA gap document with completed experimental evidence.
   - Acceptance: no stale “early positive” language if 10-seed result contradicts it.

### P0: next 24 hours

1. Implement or run checkpoint-selection branch based on val Top-1 instead of val ROC-AUC.
   - Acceptance: val/test Top-1 selection comparison table.
2. Run n_bits=8192 Morgan smoke test on 3 seeds.
   - Acceptance: if mean test Top-1 >=87.7%, promote to 10 seeds.
3. Run binary_count Morgan smoke test on 3 seeds.
   - Acceptance: if no featurization/memory issue and test improves, promote.
4. Run pairwise-weight / margin smoke matrix:
   - `pairwise_weight`: 0.5, 1.0, 2.0
   - `margin`: 0.0, 0.05, 0.10
   - Acceptance: select at most two configs for 10-seed confirmation.
5. Freeze benchmark manifest.
   - Acceptance: every table has dataset, split, scope, metric, baseline, seed rule.

### P1: 2-4 days

1. Expand original held-out benchmark to reduce 81-group variance.
   - Acceptance: >=200 test groups, no leakage.
2. Expand external beam benchmark.
   - Acceptance: >=25,000 groups or documented source limit.
3. Add Chemformer reference score as scalar feature.
   - Acceptance: strict external Top-1 >=85% in 3-seed smoke.
4. Test combined + original-scope mitigation:
   - class weights;
   - lower graph_stats scaling;
   - gated ensemble between Morgan and graph_stats/combined.
   - Acceptance: expanded curated gain retained, original test not below v2 by >0.5 pp.
5. Start Ni data acquisition or curation.
   - Acceptance: >=20 distinct Ni parent reactions or clear written limitation.

### P1: 5-7 days

1. Build lightweight graph-pair / reaction-difference encoder.
   - Acceptance: 5-seed smoke improves original test or weak-class metrics.
2. Run final 10-seed candidates:
   - best optimization branch;
   - best data-scale branch;
   - best architecture branch;
   - best ensemble branch.
3. Run full external product-selection bridge.
   - Acceptance: strict shared and validity-aware metrics for all selected models.
4. Run Type-2 feasibility refresh only if resources allow.
   - Acceptance: ROC-AUC >=88% or keep Type-2 as auxiliary.

### P2: manuscript consolidation

1. Generate final manuscript tables.
2. Generate final SOTA delta table:
   - absolute gap;
   - relative gap;
   - confidence interval;
   - significance test.
3. Write limitations:
   - PC-CNG is not pure end-to-end generation unless bridge benchmark supports it;
   - Ni remains data-source gap if not solved;
   - original held-out test size and variance must be disclosed.
4. Prepare reproducibility checklist:
   - scripts;
   - seeds;
   - data manifests;
   - environment;
   - result paths;
   - unit tests.

## 十、Decision rules

### 10.1 When to stop an experiment branch

Stop or deprioritize a branch if:

1. 3-seed smoke mean test Top-1 is <= v2 - 1.0 pp and there is no compensating weak-class/external gain.
2. 10-seed paired CI crosses zero and effect size is <+0.5 pp.
3. The branch improves val but hurts test repeatedly, unless it reveals a useful distribution-shift insight.
4. Runtime or memory cost doubles without measurable metric gain.

### 10.2 When to promote a branch

Promote to main candidate if:

1. Original held-out Test Top-1 >=90% in 10 seeds for Phase-1; final target >=93%.
2. Paired significance vs v2 is positive.
3. No major regression on expanded curated, weak-class, external bridge, or Type-2 auxiliary metrics.
4. The model can be explained scientifically, not just as an opaque hyperparameter win.

Promote to supplement if:

1. It improves a specific task substantially and significantly.
2. It has a clear claim boundary.
3. It does not replace the main model in manuscript wording.

## 十一、Expected final paper package

The final submission package should include:

1. Main Table 1: PC-CNG vs Chemformer / Molecular Transformer bridge / RegioSQM20-aligned targets.
2. Main Table 2: Original same-context 10-seed ranking metrics.
3. Main Table 3: External strict and validity-aware product-selection benchmark.
4. Supplement Table S1: Combined / Morgan / graph_stats architecture ablation.
5. Supplement Table S2: classw050_rc weak-class robustness and paired significance.
6. Supplement Table S3: hidden_dim, dropout, LR scheduler, n_bits, pairwise-weight sweeps.
7. Supplement Table S4: Type-2 low-yield feasibility.
8. Supplement Table S5: source/molecular support audit including Ni limitation.
9. Reproducibility manifest: all paths, seeds, scripts, configs, checksums where possible.
10. Claim-boundary statement: what is SOTA, what is bridge evidence, what is supplement.

## 十二、Final success definition

The project reaches the requested target only when the following are true:

1. **Metric alignment complete**：all downstream tasks have fixed input/output formats, metrics, baselines, and split rules.
2. **SOTA pursuit complete**：for each comparable SOTA metric, PC-CNG either exceeds the target or has a documented reason why the task is not apples-to-apples.
3. **Main performance target met**：original held-out Test Top-1 reaches >=93.0% or an equivalent external product-selection benchmark reaches >=90.0% strict shared Top-1 with strong statistical evidence.
4. **Scale-up evidence complete**：model-size, data-size, and feature/architecture scale-up have been tested with 10-seed statistics.
5. **Weak-class robustness complete**：Amide/Cu/Hydrogenation/Rh pass support and performance gates; Ni is either solved with >=20 contexts or honestly documented as a hard data-source gap.
6. **Paper package complete**：all final tables, logs, scripts, and reproducibility artifacts are present and internally consistent.

Until these conditions are satisfied, the project should remain in active optimization and evidence-building mode rather than being marked complete.
