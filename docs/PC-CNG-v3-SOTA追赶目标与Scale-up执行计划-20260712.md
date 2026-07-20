# PC-CNG v3 SOTA追赶目标与Scale-up执行计划

日期：2026-07-12

适用项目：未知化学反应推理 / 负样本生成模型 / PC-CNG v3

## 0、执行更新：2026-07-12 20:10

本轮已在服务器 `/home/cunyuliu/pc_cng_research` 完成 M0 关键闭环：

| Branch | 10-seed rerank | Test Top-1 mean ± std | Group-ensemble Top-1 delta vs v2 | Seed-bootstrap Top-1 CI95 | Decision |
|---|---:|---:|---:|---:|---|
| hidden4096 | 10/10 | 86.30 ± 1.36% | +0.16 pp, CI [-0.16, +0.48] pp, p=0.629 | [-0.13, +0.24] pp | reject as main; capacity alone not enough |
| dropout04 | 10/10 | 86.30 ± 1.69% | +0.00 pp, CI [0.00, 0.00] pp, p=1.000 | [-0.06, +0.08] pp | reject as main; regularization does not recover test gap |

Artifacts:

| Artifact | Path |
|---|---|
| hidden4096 multiseed summary | `/home/cunyuliu/pc_cng_research/results/type1_v2_hidden4096_20260712/hidden4096_multiseed_summary/summary.json` |
| hidden4096 paired significance | `/home/cunyuliu/pc_cng_research/results/type1_v2_hidden4096_20260712/paired_significance_v2_vs_hidden4096_same_split/summary.json` |
| dropout04 multiseed summary | `/home/cunyuliu/pc_cng_research/results/type1_v2_dropout04_20260712/dropout04_multiseed_summary/summary.json` |
| dropout04 paired significance | `/home/cunyuliu/pc_cng_research/results/type1_v2_dropout04_20260712/paired_significance_v2_vs_dropout04_same_split/summary.json` |
| cosine LR + warmup queue | `/home/cunyuliu/pc_cng_research/results/logs/type1_v2_coslr_warm5_20260712.queue.log`, PID `1512519` |
| val Top-1 checkpoint-selection code | `pc_cng/train_pairwise_reward_mlp.py`, args `--checkpoint-metric val_top1 --checkpoint-group-by reactants` |
| val Top-1 checkpoint-selection smoke queue | superseded by GPU5 relaxed chain PID `2468629`; waits for filtered baseline stage inside the chain |
| representation-scale smoke script | `/home/cunyuliu/pc_cng_research/run_v2_representation_scale_smoke.sh` |
| representation-scale smoke queue | superseded by GPU5 relaxed chain PID `2468629`; runs after val_top1 smoke inside the chain |
| pairwise-weight/margin smoke script | `/home/cunyuliu/pc_cng_research/run_v2_pairwise_margin_smoke.sh` |
| pairwise-weight/margin smoke queue | superseded by GPU5 relaxed chain PID `2468629`; runs after representation-scale smoke inside the chain |
| benchmark manifest | `00_当前有效文档/PC-CNG-v3-benchmark-manifest-20260712.md`; remote copy `/home/cunyuliu/pc_cng_research/docs/PC-CNG-v3-benchmark-manifest-20260712.md` |
| benchmark data-quality audit | `/home/cunyuliu/pc_cng_research/results/benchmark_manifest_data_quality_20260712/benchmark_data_quality_audit.json`; status `pass_with_warnings` |
| benchmark data-quality tests | `chem_negative_sampling/tests/test_benchmark_data_quality_audit.py`; remote unittest discover passed, 2 tests |
| filtered v2 same-data baseline script | `/home/cunyuliu/pc_cng_research/run_v2_filtered_baseline_multiseed.sh` |
| filtered v2 same-data baseline queue | GPU5 relaxed chain `/home/cunyuliu/pc_cng_research/results/logs/type1_v2_gpu5_relaxed_chain_20260714.log`, PID `2468629`; old sleep-only watcher PID `2312365` stopped |
| reproducibility manifest | `00_当前有效文档/PC-CNG-v3-reproducibility-manifest-20260712.{md,json}`; remote copy under `/home/cunyuliu/pc_cng_research/docs/` |
| original benchmark support audit | `/home/cunyuliu/pc_cng_research/results/original_benchmark_support_audit_20260712/original_benchmark_support_audit.json`; pre-expansion combined test groups `111/200`, deficit `89` |
| M3 original test expansion scan | `/home/cunyuliu/pc_cng_research/results/original_test_expansion_uspto_scan_20260712/uspto_original_test_expansion_scan.json`; selected `256` held-out USPTO/OpenMolecules positive parents from `51,494` eligible unique contexts |
| M3 original test expansion negatives | `/home/cunyuliu/pc_cng_research/results/original_test_expansion_uspto_negatives_20260712/uspto_test_expansion_pipeline_summary.json`; generated `1,746` raw candidates, filtered to `1,616` reviewed rows with `1,268` keep negatives |
| M3 expanded support audit | `/home/cunyuliu/pc_cng_research/results/original_benchmark_support_audit_m3_uspto_20260712/original_benchmark_support_audit.json`; expanded combined test groups `323/200`, deficit `0` |
| M3 expanded v2/unreacted baseline eval | `/home/cunyuliu/pc_cng_research/results/type1_v2_unreacted_expanded_m3_uspto_eval_20260712/v2_unreacted_expanded_m3_uspto_summary/summary.json`; scored/evaluable test groups `293`, Test Top-1 `51.60 ± 0.89%` |
| M3 expanded hidden/dropout eval | `/home/cunyuliu/pc_cng_research/results/type1_expanded_m3_uspto_eval_comparison_20260712/expanded_m3_uspto_model_comparison_with_significance.md`; hidden4096 ΔTop-1 `+0.48 pp` CI `[-0.07,+1.03]`, p=0.143; dropout04 ΔTop-1 `-0.14 pp`, p=0.626 |
| M3 expanded combined eval | `/home/cunyuliu/pc_cng_research/results/type1_expanded_m3_uspto_eval_comparison_20260712/expanded_m3_uspto_model_comparison_with_significance.md`; combined Test Top-1 `52.42 ± 0.84%`, paired ΔTop-1 `+0.14 pp` CI `[-0.62,+0.89]`, p=0.864 |
| M3 expanded classw050_rc eval | `/home/cunyuliu/pc_cng_research/results/type1_expanded_m3_uspto_eval_comparison_20260712/expanded_m3_uspto_model_comparison_with_significance.md`; classw050_rc Test Top-1 `52.73 ± 1.13%`, paired ΔTop-1 `+0.07 pp` CI `[-0.89,+1.03]`, p=1.000 |
| Ni atomic support audit | `/home/cunyuliu/pc_cng_research/results/ni_atomic_support_audit_20260713/ni_atomic_support_audit.json`; HITEA normalized `0/39,546`, USPTO/OpenMolecules `6/530,238` Ni atom reactions |
| External bridge legacy support audit | `/home/cunyuliu/pc_cng_research/results/external_product_prediction_support_audit_20260713/external_product_prediction_support_audit.json`; old 16k audit, superseded by repaired 25k audit below |
| External bridge context expansion | `/home/cunyuliu/pc_cng_research/results/external_product_prediction_context_expansion_20260713/external_product_prediction_context_expansion_summary.json`; selected `8,950` USPTO contexts; repaired final context input `25,000/25,000` |
| External bridge 25k base prebuild | `/home/cunyuliu/pc_cng_research/results/external_product_prediction_benchmark_25k_20260713/base_observed_pc_cng_candidates_summary.json`; targeted PC-CNG generation rebuilt base to `76,487` rows; PC-CNG negative coverage `24,903/25,000` groups |
| External bridge 25k base quality audit | `/home/cunyuliu/pc_cng_research/results/external_25k_base_candidate_quality_audit_20260713/external_25k_base_candidate_quality_audit.json`; `pass_with_warnings`: no duplicate products, no same-product PC-CNG negatives, but `77` invalid observed reactions with blank reactants |
| External bridge 25k repaired prebuild | `/home/cunyuliu/pc_cng_research/results/external_product_prediction_benchmark_25k_repaired_20260713/base_observed_pc_cng_candidates_summary.json`; replaced `77` blank-reactant contexts, base quality audit now `pass`, repaired beams completed `5/5` and merged to `25,001`-line TSV |
| External bridge 25k repaired full candidates | `/home/cunyuliu/pc_cng_research/results/external_product_prediction_benchmark_25k_repaired_20260713/full_observed_pc_cng_chemformer_beam_candidates_summary.json`; `25,000` contexts, `311,150` candidates (`25,000` observed positives + `51,672` PC-CNG + `235,548` Chemformer beam rows) |
| External bridge 25k repaired benchmark | `/home/cunyuliu/pc_cng_research/results/external_product_prediction_benchmark_25k_repaired_20260713/benchmark/benchmark_summary.json`; strict complete groups `25,000`, strict test Top-1: Chemformer likelihood `57.00%`, PC-CNG `13.59%`, best hybrid weight `0.00` |
| External bridge 25k validity-aware benchmark | `/home/cunyuliu/pc_cng_research/results/external_product_prediction_benchmark_25k_repaired_20260713/benchmark_validity_aware/benchmark_summary.json`; full rows `311,150`, validity-aware test Top-1: Chemformer likelihood `44.02%`, PC-CNG scored subset `13.59%` |
| External bridge 25k support audit | `/home/cunyuliu/pc_cng_research/results/external_product_prediction_support_audit_25k_repaired_20260714/external_product_prediction_support_audit.json`; contexts `25,000/25,000`, strict complete groups `25,000/25,000`, decision flags `[]`; denominator scale complete but performance is negative for PC-CNG |
| External score calibration audit | `/home/cunyuliu/pc_cng_research/results/external_score_calibration_25k_repaired_strict_20260715/external_score_calibration_summary.json`; strict shared rows `116,509`, test Top-1 Chemformer `57.00%`, best nonzero hybrid `50.87%`, PC-CNG `13.59%`; diagnostic only, no external SOTA claim |
| External held-out calibration protocol | `00_当前有效文档/PC-CNG-v3-external-heldout-calibration-protocol-20260715.{md,json}`; selected new `5,000` USPTO/OpenMolecules test contexts excluding repaired 25k; base candidate quality audit `pass`; base-only diagnostic scored, full-beam generation running |
| External frozen calibration recipes | `/home/cunyuliu/pc_cng_research/results/external_score_calibrator_lr_v1_repaired25k_train_20260715/` and `/home/cunyuliu/pc_cng_research/results/external_score_calibrator_pairwise_v1_repaired25k_train_20260715/`; both train-only recipes have val Top-1 `80.62%` vs Chemformer `83.42%`; not promoted to held-out primary |
| External MLP calibration recipe | `/home/cunyuliu/pc_cng_research/results/external_score_mlp_calibrator_v1_repaired25k_train_20260715/`; val Top-1 `89.26%` but repaired 25k test Top-1 `36.46%` vs Chemformer `57.00%`; cross-domain failure, not promoted to USPTO held-out primary |
| External USPTO train/val calibration pool | `/home/cunyuliu/pc_cng_research/results/external_calibration_trainval_contexts_12k_repaired_20260716/external_calibration_trainval_contexts_12k_repaired_summary.json`; repaired `10,000` train + `2,000` val contexts outside repaired25k/heldout5k; base quality `pass`; Chemformer val Top-1 `89.89%`, MLP calibrator val Top-1 `93.14%` |
| Filtered v2 same-data baseline | `/home/cunyuliu/pc_cng_research/results/type1_v2_filtered_baseline_20260712/filtered_baseline_multiseed_summary/summary.json`; 10 seeds complete, Test Top-1 `87.04 ± 2.35%`; paired ΔTop-1 vs original v2 `+0.08 pp`, CI `[-0.24,+0.48]`, p=`1.000` |
| Val Top-1 checkpoint smoke | `/home/cunyuliu/pc_cng_research/results/type1_v2_valtop1_ckpt_smoke_20260712/valtop1_ckpt_smoke_summary/summary.json`; 3 seeds complete, Test Top-1 `83.54 ± 1.43%`; paired ΔTop-1 vs filtered baseline `-0.97 pp`, CI `[-1.77,-0.24]`, p=`0.0168`; reject |
| nbits8192 representation smoke | `/home/cunyuliu/pc_cng_research/results/type1_v2_representation_scale_smoke_20260712/nbits8192_smoke_summary/summary.json`; 3 seeds complete, Test Top-1 `88.48 ± 1.43%`; paired ΔTop-1 vs filtered baseline `+0.16 pp`, CI `[-0.56,+0.89]`; promoted to 10-seed validation |
| nbits8192 10-seed validation | `/home/cunyuliu/pc_cng_research/results/type1_v2_nbits8192_10seed_20260714/nbits8192_10seed_summary/summary.json`; 10 seeds complete, Test Top-1 `87.78 ± 1.36%`; paired ΔTop-1 `+0.32 pp`, CI `[-0.32,+0.97]`, p=`0.489`; do not promote |
| Cosine LR + warmup 10-seed | `/home/cunyuliu/pc_cng_research/results/type1_v2_coslr_warm5_20260712/coslr_warm5_multiseed_summary/summary.json`; 10 seeds complete, Test Top-1 `87.41 ± 1.82%`; paired ΔTop-1 `+0.08 pp`, CI `[-0.24,+0.48]`, p=`1.000`; do not promote |
| binary_count4096 representation smoke | `/home/cunyuliu/pc_cng_research/results/type1_v2_representation_scale_smoke_20260712/binary_count4096_smoke_summary/summary.json`; 3 seeds complete, Test Top-1 `87.24 ± 0.71%`; paired ΔTop-1 vs filtered baseline `+0.16 pp`, CI `[-0.16,+0.48]`, p=`0.624`; below smoke-to-10seed threshold, do not promote |
| pairwise/margin smoke matrix | `/home/cunyuliu/pc_cng_research/results/type1_v2_pairwise_margin_smoke_20260712/`; 8 configs complete; selected `pw20_m000` (Test Top-1 `88.48 ± 0.71%`) and `pw20_m005` (`88.07 ± 0.71%`) for 10-seed validation, no significance claim |
| pairwise/margin 10-seed `pw20_m000` | `/home/cunyuliu/pc_cng_research/results/type1_v2_pairwise_margin_10seed_20260714/pw20_m000_10seed_summary/summary.json`; 10 seeds complete, Test Top-1 `88.15 ± 1.95%`; paired ΔTop-1 `+0.08 pp`, CI `[-0.32,+0.48]`, p=`1.000`; do not promote |
| pairwise/margin 10-seed `pw20_m005` | `/home/cunyuliu/pc_cng_research/results/type1_v2_pairwise_margin_10seed_20260714/pw20_m005_10seed_summary/summary.json`; 10 seeds complete, Test Top-1 `87.41 ± 2.52%`; paired ΔTop-1 `0.00 pp`, CI `[-0.48,+0.48]`, p=`1.000`; do not promote |

Interpretation:

```text
Both hidden4096 and dropout04 fail the promotion gate because the 10-seed
test mean is below v2/unreacted and paired CIs do not show a positive effect.
The next active branch is LR scheduling / representation-scale rather than
additional hidden_dim or stronger dropout.
```

2026-07-12 20:28 update:

```text
Checkpoint selection is now configurable. Default remains val_roc_auc for
backward compatibility, and the new val_top1 path uses real validation rows
grouped by reactants. The val_top1 smoke run is queued on GPU 5 and will run
3 seeds (20260710-20260712) before any 10-seed promotion decision.

Representation-scale smoke is also prepared and queued behind val_top1 smoke:
Morgan n_bits=8192 and Morgan binary_count n_bits=4096 will each run 3 seeds
with same-split reranking, multiseed summary, and paired significance vs v2.

M1 benchmark manifest draft is created, including fixed task boundaries,
dataset/split/candidate-scope definitions, metrics, baselines, statistics,
model-selection rules, and paper table schema.

Pairwise objective smoke matrix is prepared and queued behind representation
scale. The default cell (pairwise_weight=1.0, margin=0.0) is the current v2
baseline; the queue runs the remaining 8 cells for 3 seeds each and writes
summary/significance per config.

Data-quality audit found 10 keep-synthetic diverse-anchor rows whose products
overlapped known positives. A filtered diverse-anchor CSV has been generated
and all still-queued scripts now use it. Hard data gate failures are cleared;
remaining warnings are reactant-context split overlap and synthetic CSVs that
contain val/test parent rows but rely on the training reader's train-parent
filter. If a filtered-data branch is promoted, v2 should be rerun on the same
filtered input for final same-data fairness.

To make smoke comparisons fair immediately, a filtered v2 baseline 10-seed
rerun has been queued before GPU5 smoke jobs. The downstream smoke scripts now
use `results/type1_v2_filtered_baseline_20260712/*/candidate_scores.csv` for
paired significance.

The data-quality audit and known-positive filter are covered by unit tests:
`tests/test_benchmark_data_quality_audit.py` verifies hard-failure detection,
filtering, and pass_with_warnings status after filtering.

A reproducibility manifest has been created with fixed inputs, queue order,
result directories, tests, data-audit status, promotion gates, and SHA256 hashes.

Original benchmark support audit is complete. Under current same-context
candidate construction, real reactant groups contribute 43 test groups and
synthetic source groups contribute 68 test groups, for 111 combined test groups.
Before the USPTO expansion, the M3 >=200 held-out test group target therefore
had a quantified deficit of 89 groups.

M3 original-test expansion is now support-audited. From
`data/processed/uspto_openmolecules_normalized.csv`, after excluding source
contexts/reactions that cross train/val/test splits and excluding contexts or
canonical reactions already present in the current original benchmark, the scan
found 51,494 eligible held-out test reactant contexts. A conservative top-256
positive-parent candidate list was converted into reviewed/filtered boundary
negative candidates. The integrated support audit now reports 323 combined
test groups against the target 200, so the M3 original held-out test-group
size gate is closed. Remaining model-performance SOTA gates are still open.

The first variance-reduced expanded-benchmark baseline is complete for the
existing v2/unreacted 10-seed models. The support audit has 323 test groups;
model-scored reranking has 293 evaluable test groups after featurization and
positive/negative group filtering. The 10-seed mean Test Top-1 is 51.60 ±
0.89%, with MRR 71.61 ± 0.57% and NDCG 78.76 ± 0.43%. The USPTO expansion
subset is intentionally hard: Top-1 38.02 ± 1.54% over 212 scored groups.
This exposes the new data-scale gap that future branches must close.

Expanded-benchmark retest for M0 negative branches is complete. hidden4096
shows a small positive trend on expanded Test Top-1 (52.12 ± 0.85% vs v2
51.60 ± 0.89%; ensemble ΔTop-1 +0.48 pp), but the paired Top-1 CI crosses zero
[-0.07, +1.03] pp and permutation p=0.143, so it still fails the main Top-1
promotion gate. dropout04 remains null/negative (ensemble ΔTop-1 -0.14 pp,
p=0.626). The M0 decision remains: do not promote capacity/dropout alone.

The combined Morgan+graph_stats branch was also retested on the expanded M3
benchmark. It has a slightly higher 10-seed mean Test Top-1 (52.42 ± 0.84%),
but the paired ensemble Top-1 delta is only +0.14 pp with CI [-0.62,+0.89] pp
and p=0.864. Therefore combined remains valuable as an expanded-curated /
weak-class architecture supplement, but it is not an expanded main-candidate
replacement for v2/unreacted.

The classw050_rc weak-class branch was retested on the same expanded M3
benchmark. It slightly improves mean Test Top-1 to 52.73 ± 1.13% and USPTO
subset mean Top-1 to 40.38 ± 1.67%, but paired ensemble Top-1 delta is only
+0.07 pp with CI [-0.89,+1.03] pp and p=1.000. It therefore remains a
weak-class supplement rather than an expanded main-candidate replacement.

2026-07-13 10:31 audit: classw050_rc documentation and manifest updates were
synced to `/home/cunyuliu/pc_cng_research/docs/` and revalidated remotely. Both
benchmark/reproducibility JSON files parse on the server; the remote benchmark
manifest SHA256 is `bfcffa4c8ec5222fc7e0e0a521bf6a7d73f96fa0763000ef0e59f57b92258f7c`,
matching the reproducibility manifest record. Queue audit shows all five
watchers alive (`1512519`, `2312365`, `2312371`, `2312379`, `2312385`) but no
new summary artifacts yet. GPU4 latest watcher sample was `4265MiB / 24%`, and
GPU5 was `11692MiB / 99%`; therefore queued jobs remain correctly waiting for
safe launch rather than consuming shared GPU resources.

2026-07-13 10:45 audit: a reproducible Ni atomic support audit was added and
run on the remote RDKit environment. `audit_ni_atomic_support.py` first applies
a text prefilter containing `Ni`, then confirms atomic number 28 with RDKit.
The current processed HITEA normalized file has `39,546` reaction rows and
`0` Ni atom reactions; USPTO/OpenMolecules has `530,238` reaction rows and only
`6` Ni atom reactions / `6` distinct Ni parent reactants. This keeps Ni as a
hard external data-source gap, not a generator or reweighting failure.

2026-07-13 10:50 audit: external product-selection bridge support was audited
with `audit_external_product_prediction_support.py`. The current benchmark has
`16,050` source contexts against the `25,000` target, leaving a deficit of
`8,950`. The full candidate set has `175,678` rows and Chemformer beam coverage
for all `16,050` groups, but only `600` groups currently contain PC-CNG
candidates; strict complete evaluation therefore keeps only `1,197` groups
(`81` test groups). Validity-aware evaluation remains broad (`15,973` groups,
`1,536` test groups, PC-CNG Test Top-1 `98.50%`), but strict bridge scale-up
still requires more contexts and better PC-CNG score/candidate coverage.

2026-07-13 10:55 prep: external bridge context expansion input is prepared.
`select_external_product_prediction_contexts.py` selected `8,950` safe
USPTO/OpenMolecules contexts from `517,644` eligible unique source contexts,
excluding existing external contexts and source-internal cross-split reactant
contexts. The merged context file now has `25,000` unique groups and the
Chemformer input has `25,001` lines including the header. This closes the
context-input size gate, but not the strict benchmark gate: external beams,
PC-CNG candidates/scores, and strict/validity-aware support audit must still be
rerun on the merged 25k context set.

2026-07-13 11:00 prep: the 25k external bridge runner
`scripts_run_external_product_prediction_benchmark_25k.sh` was added with safe
defaults (`GENERATE_BEAMS=0`, `RUN_BENCHMARK=0`) so it does not start GPU-heavy
Chemformer generation unless explicitly requested. The CPU-only base prebuild
completed under `results/external_product_prediction_benchmark_25k_20260713/`:
`25,000` contexts, `25,000` Chemformer input rows, and `29,265` base candidate
rows (`25,000` observed positives + `4,265` existing PC-CNG rows). Existing
PC-CNG candidate coverage is `764/25,000` groups, so the next strict-bridge
step remains 25k beam generation plus additional PC-CNG candidate/score
coverage before rerunning strict/validity-aware benchmarks.

2026-07-13 11:05 prep: the 25k Chemformer input was split into resumable
chunks with `split_chemformer_input.py`. The output manifest is
`results/external_product_prediction_benchmark_25k_20260713/chemformer_input_chunks/chemformer_forward_input_25k_chunks_manifest.json`.
It contains 5 chunks, each with `5,000` rows (plus header in the TSV file).
This makes the next Chemformer beam generation step chunkable and resumable
instead of a single monolithic 25k run.

2026-07-13 11:10 prep: a safe chunk-level Chemformer beam runner was added:
`scripts_run_external_product_prediction_25k_chunked_beams.sh`. A dry-run wrote
`results/external_product_prediction_benchmark_25k_20260713/chemformer_beam_chunks/chemformer_forward_beam_chunks_status.json`,
showing 5 expected chunks and 0 completed beam chunks. The runner waits for
GPU memory/utilization thresholds and zero compute apps by default, skips valid
completed chunks, and merges outputs only after all chunks pass line-count
validation. No GPU beam generation was launched in this step.

2026-07-13 11:19 queue: the 25k chunked Chemformer beam watcher is now active
as PID `2156024`, log
`/home/cunyuliu/pc_cng_research/results/logs/external_product_prediction_25k_chunked_beams.queue.log`.
It targets GPU1 with conservative gates: memory <= `2500MiB`, utilization <=
`10%`, and compute apps = `0`. The latest sample was GPU1 `839MiB / 100% /
compute_apps=1`, so the watcher is correctly waiting on chunk 0 and has not
started beam generation.

2026-07-13 11:25 status: the 25k chunked beam watcher remains alive (PID
`2156024`) and is still waiting on chunk 0. Status JSON reports `0/5` complete
chunks and no merged beam file. Latest GPU1 sample remains `839MiB / 100% /
compute_apps=1`, so the no-steal gate is working. The main training queues also
remain in watcher mode with no new summary artifacts: GPU4 latest sample
`4167MiB / 48%`, GPU5 latest sample `11668MiB / 37%`.

2026-07-13 11:30 audit: a unified active execution status audit was added and
run at `/home/cunyuliu/pc_cng_research/results/active_execution_status_20260713/`.
The audit records GPU status, queue PID liveness, queue log tails, discovered
artifacts, and 25k beam watcher status in both JSON and Markdown. Current audit
summary: all five training watcher PIDs are alive but have `0` result artifacts;
25k beam watcher PID `2156024` is alive with `0/5` chunks complete and no merged
beam file. Latest sampled logs remain in waiting state.

2026-07-13 11:37 status refresh: active-status audit was rerun. It still shows
all five training watchers alive with `0` result artifacts. The 25k beam watcher
PID `2156024` is alive with `0/5` chunks complete and no merged beam file.
Latest samples: GPU1 `839MiB / 80%`, beam log last line
`gpu1 mem=839MiB util=97% compute_apps=1`; GPU4 queue last line
`4455MiB / 56%`; GPU5 queue last line `11638MiB / 97%`.

2026-07-13 11:41 status refresh: active-status audit was rerun at
`2026-07-13T11:41:59`. It still shows all five training watchers alive with `0`
result artifacts. The 25k beam watcher PID `2156024` is alive with `0/5` chunks
complete and no merged beam file. Latest no-steal evidence: beam log last line
`gpu1 mem=839MiB util=66% compute_apps=1`; GPU4 queue last line
`4195MiB / 21%`; GPU5 queue last line `11638MiB / 97%`. Updated active-status
SHA: JSON `6a0ff561a2e17565104b398b10227ec549014a34dde05003286d0a174524dbff`,
MD `dc90fee44b5471fdcb9f1c6917d51989ef046b2990b12818fc0a654fc98788a8`.

2026-07-13 12:14 targeted PC-CNG coverage: added CPU-only targeted generation
for the merged 25k external contexts via
`generate_external_context_pc_cng_candidates.py`, then rebuilt the base candidate
set with `GENERATE_TARGETED_PC_CNG=1 FORCE_REBUILD_BASE=1 BUILD_BASE_ONLY=1`.
The targeted generator processed `24,923` usable contexts, skipped `77`, and
generated `47,259` forward-outcome PC-CNG candidates covering `24,903` groups.
The rebuilt base set now has `76,487` rows: `25,000` observed positives plus
`51,487` PC-CNG rows, with PC-CNG negative group coverage `24,903/25,000`.
This closes the candidate-coverage gate for the 25k input; beam generation,
LM scoring, PC-CNG scoring, and strict/validity-aware audit remain pending.

2026-07-13 18:15 base-quality audit: added CPU-only
`audit_external_25k_base_candidate_quality.py` and validated it locally/remotely.
The real 25k base candidate audit reports `25,000/25,000` observed-positive
groups, `24,903/25,000` PC-CNG-negative groups, `0` duplicate candidate-product
groups, `0` same-product PC-CNG negatives, and `0` invalid PC-CNG negatives.
Decision is `pass_with_warnings` because `77` observed-positive candidate
reactions have blank reactants (`>>product`). Before accepting 25k beam outputs
for strict claims, these `77` contexts must be replaced, repaired, or explicitly
filtered in the strict/validity-aware audit.

2026-07-13 12:19 status refresh: active-status audit was rerun at
`2026-07-13T12:19:57`. It still shows all five training watchers alive with `0`
result artifacts. The 25k beam watcher PID `2156024` is alive with `0/5` chunks
complete and no merged beam file. Latest no-steal evidence: beam log last line
`gpu1 mem=7427MiB util=99% compute_apps=2`; GPU4 queue last line
`9782MiB / 97%`; GPU5 queue last line `4801MiB / 37%`.

2026-07-13 17:53 status refresh: active-status audit was rerun at
`2026-07-13T17:53:16`. All five training watchers remain alive with `0` result
artifacts. A direct process-tree check found only watcher shells plus `sleep 300`
children, so there is no hidden training/beam subprocess that the artifact scan
missed. The 25k beam watcher PID `2156024` remains alive with `0/5` chunks
complete and no merged beam file. Latest no-steal evidence: beam log last line
`gpu1 mem=12155MiB util=91% compute_apps=2`; GPU4 queue last line
`15675MiB / 38%`; GPU5 queue last line `16117MiB / 94%`.

2026-07-13 18:03 status refresh: active-status audit was rerun at
`2026-07-13T18:03:54`. All five training watchers remain alive with `0` result
artifacts. The 25k beam watcher PID `2156024` remains alive with `0/5` chunks
complete and no merged beam file. Latest no-steal evidence: beam log last line
`gpu1 mem=12155MiB util=93% compute_apps=2`; GPU4 queue last line
`4143MiB / 34%`; GPU5 queue last line `5691MiB / 39%`.

2026-07-13 18:09 status refresh: active-status audit was rerun at
`2026-07-13T18:09:04`. State is unchanged: all five training watchers are alive
with `0` result artifacts, and the 25k beam watcher PID `2156024` is alive with
`0/5` chunks complete and no merged beam file. Latest no-steal evidence:
`gpu1 mem=12155MiB util=98% compute_apps=2`.

2026-07-13 18:12 status refresh: active-status audit was rerun at
`2026-07-13T18:12:34`. State is unchanged: all five training watchers are alive
with `0` result artifacts, and the 25k beam watcher PID `2156024` is alive with
`0/5` chunks complete and no merged beam file. Latest no-steal evidence:
`gpu1 mem=12155MiB util=92% compute_apps=2`.

2026-07-13 18:15 status refresh: active-status audit was rerun at
`2026-07-13T18:15:09`. State is unchanged: all five training watchers are alive
with `0` result artifacts, and the 25k beam watcher PID `2156024` is alive with
`0/5` chunks complete and no merged beam file. Latest no-steal evidence:
`gpu1 mem=12155MiB util=97% compute_apps=2`.

2026-07-13 18:35 repair complete: the old dirty-input beam watcher was paused
before any chunk was generated. The `77` blank-reactant contexts were removed
and replaced by `77` safe USPTO/OpenMolecules test contexts selected with the
same context-expansion filters. A repaired benchmark directory was built at
`results/external_product_prediction_benchmark_25k_repaired_20260713/` with
`25,000` contexts, `76,672` base candidate rows, `51,672` PC-CNG rows, and
`24,980/25,000` PC-CNG-negative groups. The repaired base-quality audit reports
`decision=pass`, `invalid_candidate_reaction_rows=0`, `duplicate_candidate_product_groups=0`,
and `same_product_pc_cng_negative_rows=0`. A new beam watcher PID `3770322`
is active on the repaired chunk manifest and is waiting on repaired chunk 0;
no repaired beam chunk has been generated yet because GPU1 still has
`compute_apps=2`.

2026-07-13 19:26 GPU4 beam execution: per user approval, the repaired beam
watcher was moved from GPU1 to GPU4 because GPU4 had sufficient free memory.
The first GPU4 run exposed a Chemformer input-format issue: header-preserving
chunks caused Chemformer to predict the TSV header as an extra sample, yielding
`5002` lines for chunk 0. The invalid output was archived as
`chemformer_forward_beams_25k_chunk_0000.header_included_invalid.tsv`, and
`scripts_run_external_product_prediction_25k_chunked_beams.sh` was patched to
feed Chemformer temporary no-header chunk inputs. The repaired rerun produced a
valid chunk 0 beam file with `5001` lines; active watcher PID `4113889` is now
running chunk 1 on GPU4. Active audit reports repaired beam progress `1/5`,
merged beam not yet available.

2026-07-14 10:34 repaired 25k bridge progress: the GPU4 repaired Chemformer
beam run completed all `5/5` chunks and merged
`results/external_product_prediction_benchmark_25k_repaired_20260713/chemformer_forward_beams.tsv`
with `25,001` lines (header + `25,000` contexts), SHA256
`e6bdb933e81288b5262a20a647ce1116e5be61c31cc43598e448ad23f7fcbcbb`.
The full observed + PC-CNG + Chemformer-beam candidate set was built at
`results/external_product_prediction_benchmark_25k_repaired_20260713/full_observed_pc_cng_chemformer_beam_candidates.csv`
with `311,150` candidates: `25,000` observed positives, `51,672` PC-CNG rows,
and `235,548` Chemformer beam rows; CSV SHA256
`f7a4ed0450a2a5b4bbf9573ea7c312b83b8893e8f0366b44dbbd9fc3ce38a542`.
A 20-row Chemformer conditional-likelihood smoke test produced exactly `20`
output rows with `lm_status=ok`, so the full likelihood + strict/validity-aware
benchmark pipeline was launched as PID `1659359` on GPU4. At the latest check it
was still actively running `chemformer_log_likelihood`; no external bridge
scale-complete claim is made until likelihood scores, PC-CNG evaluation, and
support audit are complete.

2026-07-14 11:20 GPU5 queue repair: the old GPU5 watchers
`2312365/2312371/2312379/2312385` were confirmed to be sleep-only wrappers
blocked by the previous conservative utilization gate. They were stopped to
avoid duplicate writes, and a new relaxed serial chain was launched on GPU5 as
PID `2468629` using
`results/logs/type1_v2_gpu5_relaxed_chain_20260714.log`. The chain runs
filtered v2 baseline -> val_top1 checkpoint smoke -> representation-scale smoke
-> pairwise/margin smoke. At the latest check, filtered baseline seeds
`20260710` and `20260711` had completed training plus same-split rerank, and
seed `20260712` was running. This keeps the fairness dependency chain intact
while using the user-approved memory-sufficient GPU policy.

2026-07-14 12:05 repaired 25k benchmark result: Chemformer conditional
likelihood scoring completed for all `311,150` candidates, producing
`lm_scores_chemformer_log_likelihood.csv` with `311,151` lines and SHA256
`f62e61027017abc3ed61ba82252a2d10dc67c756aa1ffc431c94460ed9f648d2`.
The strict shared evaluation now has `25,000` complete groups and `116,509`
evaluated rows. Strict test Top-1 is Chemformer likelihood `57.00%` vs PC-CNG
`13.59%`; MRR/NDCG are `73.99/80.48%` vs `44.61/58.30%`. The selected hybrid
is `hybrid_pc_cng_w0p00`, i.e. no PC-CNG contribution. After fixing
`evaluate_external_product_prediction_benchmark.py` for missing PC-CNG scores
under `--allow-incomplete-groups`, validity-aware evaluation also completed on
all `311,150` rows: Chemformer likelihood test Top-1 `44.02%`, PC-CNG scored
subset test Top-1 `13.59%`. The repaired support audit reports contexts
`25,000/25,000`, strict complete groups `25,000/25,000`, and decision flags
`[]`. This closes the external bridge denominator scale gate, but it is a
negative performance result for PC-CNG and does not support an external SOTA
claim.

2026-07-15 10:20 external score calibration audit: added
`chem_negative_sampling/pc_cng/analyze_external_score_calibration.py` and ran
it on both strict shared and validity-aware repaired 25k candidate scores. Under
the strict shared candidate set (`116,509` rows, `25,000` groups), test Top-1 is
Chemformer likelihood `57.00%`, PC-CNG `13.59%`, and the best nonzero hybrid
(`hybrid_pc_cng_w0p25`) is only `50.87%`; paired shared-row ΔTop-1 vs
Chemformer is `-6.13 pp` for `w0p25` and `-43.41 pp` for PC-CNG. The
validity-aware audit uses shared scored rows for paired deltas, so full-beam
rows without PC-CNG scores cannot inflate hybrid comparisons. Conclusion:
Chemformer-reference calibration is diagnostic-only at this stage; it does not
rescue the external bridge into a SOTA claim.

2026-07-15 10:34 held-out calibration protocol freeze: created
`PC-CNG-v3-external-heldout-calibration-protocol-20260715.{md,json}` and
selected a new external 5k test-only context set that excludes the repaired 25k
contexts by source ID and canonical reactant context. Selection summary:
existing contexts `25,000`, selected new contexts `5,000`, eligible unique
source contexts after exclusions `508,617`, selected split counts
`{"test": 5000}`. CPU-only base candidate construction completed with `16,873`
rows (`5,000` observed positives + `11,873` PC-CNG negatives), and base quality
audit passed with no hard failures or warnings. Chemformer input is chunked as a
single `5,000`-row chunk. Beam generation and held-out scoring are explicitly
not run yet; this freezes the denominator for a future predeclared calibration
or scorer-architecture evaluation.

2026-07-15 10:43 frozen recipe audit: added
`chem_negative_sampling/pc_cng/train_external_score_calibrator.py` and trained
two recipes on the repaired 25k strict `train` split only. The pointwise
balanced logistic recipe `pc_cng_lr_calibrator_v1` reaches validation Top-1
`80.62%`, Top-3 `98.75%`, MRR `89.00%`; the pairwise preference recipe
`pc_cng_pairwise_calibrator_v1` reaches validation Top-1 `80.62%`, Top-3
`98.83%`, MRR `89.08%`. Both are below the Chemformer validation Top-1 baseline
`83.42%`, so neither is promoted as the primary held-out 5k candidate. This
keeps the held-out 5k unscored and preserves it for a stronger frozen scorer
architecture rather than spending it on recipes that already failed the Top-1
sanity gate.

2026-07-15 10:52 MLP calibration audit: added
`chem_negative_sampling/pc_cng/train_external_score_mlp_calibrator.py` and
trained `pc_cng_mlp_calibrator_v1` on the repaired 25k strict `train` split
only. It improves validation Top-1 to `89.26%` vs Chemformer `83.42%`, but it
fails on the repaired 25k test split (`36.46%` vs Chemformer `57.00%`). Because
the frozen held-out 5k consists of USPTO/OpenMolecules test contexts, this is
treated as cross-domain overfitting rather than a promotable external scorer.
Held-out 5k remains unscored.

2026-07-16 11:06 USPTO train/val calibration pool: added split-filtering support
to `select_external_product_prediction_contexts.py` and selected same-domain
USPTO/OpenMolecules calibration contexts outside repaired25k and heldout5k:
`10,000` train contexts and `2,000` val contexts. The initial combined 12k pool
had `40,242` base candidate rows and audit `pass_with_warnings` due to `2`
invalid observed-positive candidate reactions (`uspto_openmol_000019302`,
`uspto_openmol_000022432`). A repaired pool replaced those two train sources
with `uspto_openmol_000301515` and `uspto_openmol_000301516`, rebuilt targeted
PC-CNG negatives and base candidates, and upgraded the base quality audit to
`pass`: `40,244` candidate rows (`12,000` observed positives + `28,244` PC-CNG
negatives), PC-CNG negative group coverage `0.99817`, warnings `[]`, invalid
candidate reactions `0`. PC-CNG-only scoring with the 10-model filtered-v2
ensemble attached scores to all `40,244` rows; val Top-1 is only `16.42%`
(`11,978` groups with ranking decisions), so this is diagnostic/readiness
evidence, not a positive scorer result. Chemformer likelihood scoring then
completed on the repaired pool: Chemformer val Top-1 `89.89%`, PC-CNG val
Top-1 `16.42%`, and simple hybrid selection remains `w0p00`. Three frozen
train-only recipes were trained: LR val Top-1 `87.39%`, pairwise val Top-1
`84.68%`, and MLP val Top-1 `93.14%`. Decision: the MLP recipe is
validation-positive and may be frozen as the next held-out candidate, but this
is not held-out evidence. The frozen MLP has been applied to held-out 5k
base-only candidates as a diagnostic: Chemformer Top-1 `91.99%`, PC-CNG Top-1
`17.44%`, MLP Top-1 `94.51%` over `4,995` groups. This is not full-beam
held-out evidence because Chemformer beam candidates are absent. Full-beam
generation is running on GPU7 after a stalled GPU6 attempt; final held-out
scoring remains pending and forbidden for recipe selection.

2026-07-14 12:20 filtered v2 same-data baseline complete: GPU5 relaxed chain
finished all `10/10` filtered baseline seeds and wrote
`results/type1_v2_filtered_baseline_20260712/filtered_baseline_multiseed_summary/summary.json`.
Test Top-1 is `87.04 ± 2.35%`, MRR `93.14 ± 1.19%`, NDCG `94.92 ± 0.88%`.
Paired significance vs the original v2/unreacted CSV baseline gives ensemble
ΔTop-1 `+0.08 pp`, CI `[-0.24,+0.48]`, permutation p=`1.000`, sign-test
p=`1.000`; seed-bootstrap ΔTop-1 CI is `[-0.17,+0.28]` pp. Therefore this
branch is a fairness/control baseline, not a promoted model. The same GPU5
chain has moved on to the val_top1 checkpoint smoke stage.

2026-07-14 12:40 val_top1 checkpoint smoke complete: the 3-seed smoke wrote
`results/type1_v2_valtop1_ckpt_smoke_20260712/valtop1_ckpt_smoke_summary/summary.json`.
Test Top-1 is `83.54 ± 1.43%`, MRR `91.29 ± 0.78%`, NDCG `93.55 ± 0.58%`.
Paired significance vs the filtered 3-seed baseline gives ensemble ΔTop-1
`-0.97 pp`, CI `[-1.77,-0.24]`, permutation p=`0.0168`, sign-test p=`0.0169`;
seed-bootstrap ΔTop-1 CI is `[-1.05,-0.89]` pp. The val_top1 checkpoint
selection branch is therefore rejected and should not be promoted to 10 seeds.
GPU5 relaxed chain has moved on to representation-scale smoke.

2026-07-14 13:10 cosine-LR queue repair and representation progress: the old
coslr watcher PID `1512519` was confirmed sleep-only and stopped. The
`run_v2_coslr_warm5_multiseed.sh` script had a real `$!` bug because the
training command was not backgrounded; it failed after seed `20260710` training
before reranking. The script was fixed to run synchronously, skip existing
checkpoints, rerank missing seeds, and write multiseed summary/significance.
The restarted relaxed GPU4 chain PID is `3855829`; seed `20260710` rerank now
exists with Test Top-1 `88.89%`, MRR `94.14%`, NDCG `95.65%`.
Meanwhile GPU5 representation-scale smoke has completed nbits8192 seeds
`20260710-20260711` with Test Top-1 `87.65%` and `90.12%`; seed `20260712` is
running. No representation promotion decision is made until the 3-seed summary
and paired significance are available.

2026-07-14 13:21 nbits8192 smoke promotion: representation-scale nbits8192
finished all `3/3` smoke seeds with Test Top-1 `88.48 ± 1.43%`, MRR
`93.45 ± 1.04%`, NDCG `95.13 ± 0.79%`. This passes the predeclared
smoke-to-10seed threshold (`>=87.7%`), although paired ensemble ΔTop-1 vs the
filtered 3-seed baseline is only `+0.16 pp` with CI `[-0.56,+0.89]` and
p=`0.828`, so it is not a significance claim. A dedicated 10-seed validation
script `/home/cunyuliu/pc_cng_research/run_v2_nbits8192_10seed.sh` was launched
on GPU1 as PID `3969228`; it reused completed smoke seeds `20260710-20260712`
and is running seed `20260713`. The same run will write 10-seed summary and
paired significance vs the filtered baseline before any promotion decision.

2026-07-14 13:40 active training progress: nbits8192 10-seed validation has
`4/10` rerank outputs complete (seeds `20260710-20260713`), with current
partial Test Top-1 mean `88.27%`; this is still not a decision because the
10-seed summary/significance are pending. Cosine LR + warmup has `5/10` rerank
outputs complete (seeds `20260710-20260714`) with partial Test Top-1 mean
`87.41%`; summary/significance pending. GPU5 binary_count4096 smoke has `2/3`
rerank outputs complete with partial Test Top-1 mean `87.04%`; third seed and
paired significance pending. All three chains remain active.

2026-07-14 13:48 active training progress refresh: nbits8192 10-seed validation
has `5/10` rerank outputs complete (seeds `20260710-20260714`) and is running
seed `20260715`; partial Test Top-1 mean is now `87.65%`. Cosine LR + warmup has
`6/10` rerank outputs complete (seeds `20260710-20260715`) and is running seed
`20260716`; partial Test Top-1 mean is also `87.65%`. GPU5 binary_count4096
smoke remains `2/3` rerank outputs complete, running seed `20260712`, partial
Test Top-1 mean `87.04%`. These are progress-only numbers; promotion/rejection
still requires the configured summary and paired-significance artifacts.

2026-07-14 13:52 representation-scale update: binary_count4096 smoke completed
all `3/3` rerank outputs. The smoke Test Top-1 is `87.24 ± 0.71%`, below the
predefined smoke-to-10seed threshold `87.7%`; paired ensemble ΔTop-1 vs filtered
3-seed baseline is only `+0.16 pp` with CI `[-0.16,+0.48]` and p=`0.624`.
Decision: do not promote binary_count4096 to 10-seed. GPU5 relaxed chain has
advanced to pairwise/margin smoke, starting config `pw05_m000` seed `20260710`.

2026-07-14 13:55 active training progress refresh: nbits8192 10-seed validation
has `6/10` rerank outputs complete (seeds `20260710-20260715`) and is running
seed `20260716`; partial Test Top-1 mean is `87.45%`. Cosine LR + warmup has
`7/10` rerank outputs complete (seeds `20260710-20260716`) and is running seed
`20260717`; partial Test Top-1 mean is `87.48%`. No 10-seed summary or paired
significance exists yet for either branch.

2026-07-14 18:59 completion and next validation update: nbits8192 10-seed and
cosine LR + warmup 10-seed are complete but not promoted. nbits8192 Test Top-1
is `87.78 ± 1.36%`, paired ΔTop-1 `+0.32 pp`, CI `[-0.32,+0.97]`, p=`0.489`;
coslr Test Top-1 is `87.41 ± 1.82%`, paired ΔTop-1 `+0.08 pp`, CI
`[-0.24,+0.48]`, p=`1.000`. Pairwise/margin smoke completed all 8 non-default
cells; only `pw20_m000` (`88.48 ± 0.71%`) and `pw20_m005` (`88.07 ± 0.71%`)
passed the `87.7%` smoke-to-10seed mean threshold, but both paired CIs cross 0,
so they are promoted only to validation, not to a claim. New load-gated watchers
were launched for 10-seed validation: `pw20_m000` PID `3361815` on GPU0 and
`pw20_m005` PID `3362023` on GPU5. Both are waiting because current load1
`97.99` exceeds `MAX_LOADAVG=80`; no training has started yet.

2026-07-14 19:04 watcher heartbeat: both pairwise/margin 10-seed watchers are
still alive and waiting. `pw20_m000` PID `3361815` has child `sleep 300` and
targets GPU0; `pw20_m005` PID `3362023` has child `sleep 300` and targets GPU5.
Latest load1 is `93.45`, still above `MAX_LOADAVG=80`; GPU free memory gates
pass (`pw20_m000`: `39551 MiB` free on GPU0; `pw20_m005`: `35328 MiB` free on
GPU5). The 10-seed result directory does not exist yet, so no training has
started.

2026-07-14 19:09 watcher heartbeat: both selected pairwise/margin 10-seed
watchers remain alive and waiting with `sleep 300` children. Latest heartbeat
load1 is `100.31`, still above `MAX_LOADAVG=80`; GPU memory gates still pass
(`pw20_m000`: `39551 MiB` free on GPU0; `pw20_m005`: `35328 MiB` free on GPU5).
The 10-seed result directory still does not exist, so no training has started.

2026-07-14 19:14 watcher heartbeat: both selected pairwise/margin 10-seed
watchers remain alive and waiting with `sleep 300` children. Latest heartbeat
load1 is `98.93`, still above `MAX_LOADAVG=80`; GPU memory gates still pass
(`pw20_m000`: `39551 MiB` free on GPU0; `pw20_m005`: `35404 MiB` free on GPU5).
The 10-seed result directory still does not exist, so no training has started.

2026-07-14 19:19 watcher heartbeat + 19:20 conditional start check: both
selected pairwise/margin 10-seed watchers remain alive and waiting with
`sleep 300` children. Latest watcher heartbeat load1 is `87.62`, still above
`MAX_LOADAVG=80`; GPU memory gates still pass (`pw20_m000`: `39551 MiB` free on
GPU0; `pw20_m005`: `35404 MiB` free on GPU5). A direct-start guard check at
19:20 saw load1 `83.05`, still above the gate, so the watcher processes were
left intact and no direct training was launched. The 10-seed result directory
still does not exist.

2026-07-14 19:24 scheduler adjustment: to avoid missing short compliant load
windows, both selected pairwise/margin 10-seed watchers were restarted with the
same hard gate (`MAX_LOADAVG=80`, `MIN_GPU_FREE_MIB=8192`) but a shorter
`POLL_SECONDS=60`. Old watcher PIDs `3361815`/`3362023` were stopped; new
watcher PIDs are `3625840` for `pw20_m000` on GPU0 and `3626039` for
`pw20_m005` on GPU5. Relaunch heartbeat load1 was `92.92`, so no training was
started and the 10-seed result directory still does not exist.

2026-07-14 19:26 watcher heartbeat: the new 60-second watchers remain alive and
waiting. Latest heartbeat load1 is `115.23`, still above `MAX_LOADAVG=80`; GPU
memory gates still pass (`pw20_m000`: `39551 MiB` free on GPU0; `pw20_m005`:
`35404 MiB` free on GPU5). The 10-seed result directory still does not exist,
so no training has started.

2026-07-14 19:33 watcher heartbeat: the 60-second watchers remain alive and
waiting. Latest heartbeat load1 is `154.42`, still above `MAX_LOADAVG=80`; GPU
memory gates still pass (`pw20_m000`: `39551 MiB` free on GPU0; `pw20_m005`:
`35254 MiB` free on GPU5). The 10-seed result directory still does not exist,
so no training has started.

2026-07-14 19:37 watcher heartbeat: the 60-second watchers remain alive and
waiting. Latest heartbeat load1 is `105.81`, still above `MAX_LOADAVG=80`; GPU
memory gates still pass (`pw20_m000`: `39551 MiB` free on GPU0; `pw20_m005`:
`35254 MiB` free on GPU5). The 10-seed result directory still does not exist,
so no training has started.

2026-07-14 19:41 watcher heartbeat: the 60-second watchers remain alive and
waiting. Latest heartbeat load1 is `92.27`, still above `MAX_LOADAVG=80`; GPU
memory gates still pass (`pw20_m000`: `39551 MiB` free on GPU0; `pw20_m005`:
`35254 MiB` free on GPU5). The 10-seed result directory still does not exist,
so no training has started.

2026-07-15 10:02 completion update: selected pairwise/margin 10-seed validation
has completed and does not pass the main promotion gate. `pw20_m000` completed
10/10 rerank seeds with Test Top-1 `88.15 ± 1.95%`; paired ensemble ΔTop-1 vs
filtered v2 is `+0.08 pp`, CI `[-0.32,+0.48]`, p=`1.000`. `pw20_m005`
completed 10/10 rerank seeds with Test Top-1 `87.41 ± 2.52%`; paired ensemble
ΔTop-1 is `0.00 pp`, CI `[-0.48,+0.48]`, p=`1.000`. Both paired CIs cross 0 and
neither branch reaches the `>= v2 +1.0 pp` promotion requirement, so both are
recorded as complete negative/neutral scale-up evidence rather than SOTA claims.
```



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
| External validity-aware Test Top-1 | Chemformer `44.02%`, PC-CNG scored subset `13.59%` | repaired 25k bridge | denominator complete; PC-CNG negative result |
| External strict shared Test Top-1 | Chemformer `57.00%`, PC-CNG `13.59%` | repaired 25k bridge | denominator complete; no external SOTA claim |
| Type-2 low-yield ROC-AUC | 85.56 ± 0.21% | low_yield_synth05 | auxiliary branch, not main claim |
| Type-2 low-yield AUPRC | 79.93 ± 0.08% | low_yield_synth05 | needs improvement if included prominently |
| Type-2 low-yield F1 | 72.30 ± 0.08% | low_yield_synth10 | needs improvement |

### 2.2 Main SOTA gaps

| Gap | Current | SOTA / strong reference | Gap | Required action |
|---|---:|---:|---:|---|
| Original held-out candidate Top-1 vs RegioSQM20 with tautomers | 87.16% | 92.7% | -5.54 pp | raise held-out Top-1 to >=93.0% |
| Forward product prediction alignment vs Molecular Transformer | candidate reranking only | >90% top-1 on forward benchmarks | not apples-to-apples | build end-to-end or beam reranking bridge |
| Strict external shared-candidate benchmark | PC-CNG 13.59%, Chemformer likelihood 57.00% | repaired 25k bridge | -43.41 pp vs Chemformer | add Chemformer-reference feature / scorer calibration before further external claims |
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
| Strict shared Test Top-1 | PC-CNG `13.59%`; Chemformer `57.00%` | recover to >=Chemformer | >=85.0% |
| Strict shared MRR | PC-CNG `44.61%`; Chemformer `73.99%` | recover to >=Chemformer | >=90.0% |
| Strict shared NDCG | PC-CNG `58.30%`; Chemformer `80.48%` | recover to >=Chemformer | >=92.0% |
| Validity-aware Test Top-1 | PC-CNG scored subset `13.59%`; Chemformer `44.02%` | recover to >=Chemformer | >=85.0% |
| Full beam coverage | `25,000` groups / `311,150` rows | achieved | maintain `>=25,000` groups |
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
| P0 | Strict external Test Top-1 | PC-CNG 13.59% | Chemformer 57.00% | recover to >=57.00% | >=85.0% | repaired 25k strict shared benchmark |
| P0 | Validity-aware Test Top-1 | PC-CNG scored subset 13.59% | Chemformer 44.02% | recover to >=44.02% | >=85.0% | repaired 25k validity-aware benchmark |
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
| Morgan n_bits=4096, hidden=4096 | 12,288 | 4,096 | ~58.7M | completed; rejected as main, Test Top-1 86.30 ± 1.36% |
| Morgan n_bits=4096, hidden=2048, dropout=0.40 | 12,288 | 2,048 | ~27.3M | completed; rejected as main, Test Top-1 86.30 ± 1.69% |
| Morgan n_bits=8192, hidden=2048 | 24,576 | 2,048 | ~52.4M | planned |
| Combined n_bits=8192, hidden=2048 | 24,740 | 2,048 | ~52.8M | planned after n_bits smoke |
| Morgan n_bits=4096, hidden=8192 | 12,288 | 8,192 | ~134M | deprioritized unless representation/data scale first shows benefit |
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
| S0 | 2026-07-12 | Finish hidden4096 and dropout04 | 10 seeds each | paired test complete | completed; neither capacity nor dropout is promoted |
| S1 | 2026-07-12 to 2026-07-13 | Cosine LR + 5-epoch warmup | 10 seeds if GPU available | Test Top-1 >=88.0 or positive 3-seed trend | queued on GPU 4 watcher, PID `1512519` |
| S2 | 2026-07-13 | Early stopping / checkpoint selection by val Top-1 | 3 seeds smoke, then 10 seeds | val/test ranking alignment improves | running inside GPU5 relaxed chain PID `2468629` after filtered baseline completion |
| S3 | 2026-07-13 to 2026-07-14 | n_bits=8192 and binary_count features | 3 seeds per config | >=+0.5 pp test Top-1 over v2 in smoke | script ready; queued behind val_top1 smoke inside PID `2468629` |
| S4 | 2026-07-14 to 2026-07-16 | combined + original-scope mitigation | 10 seeds | retain expanded gain while original test >=87.16 | candidate architecture branch |
| S5 | 2026-07-15 to 2026-07-17 | Chemformer reference score feature | 3 seeds smoke then 10 seeds | strict external Top-1 >=85; original test no regression | main bridge candidate |
| S6 | 2026-07-16 to 2026-07-19 | lightweight graph-pair / reaction-difference encoder | 5 seeds then 10 seeds | original test >=90 or weak-class gain significant | architecture upgrade |
| S7 | 2026-07-19 to 2026-07-21 | model-family ensemble | 10-seed family ensemble | original test >=93 or strict external >=90 | final SOTA candidate |

### 5.3 Training objective expansion

| Objective component | Current | Next experiment | Success criterion |
|---|---|---|---|
| BCE anchor | enabled | keep | binary ROC-AUC not collapsed |
| Pairwise loss | weight=1.0 | sweep 0.5 / 1.0 / 2.0 | ranking Top-1 improves |
| Pairwise margin | 0.0 | sweep 0.05 / 0.10 / class-aware margins | smoke script ready; queued behind representation-scale |
| Class weights | classw050 validated | extend to combined/Chemformer-feature | weak-class gain without original loss |
| LR schedule | fixed lr=1e-3 | cosine to 1e-5, warmup=5 | smoother val/test curves |
| Checkpoint selection | val ROC-AUC | val Top-1 or composite metric | implemented: `--checkpoint-metric val_top1`; smoke queued |
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
| USPTO/OpenMolecules normalized | M3 held-out parent pool | scan completed; 256 positive parents selected for negative generation/review |
| Chemformer beam candidates | external bridge | benchmark/scoring |

### 6.2 Data expansion targets

| Priority | Data target | Current | Target size | Deadline | Acceptance |
|---|---|---:|---:|---|---|
| P0 | Original train groups | ~1,060 train groups | >=2,000 groups | 2026-07-15 | split-stable, no leakage |
| P0 | Original test groups | 323 combined evaluable groups after USPTO expansion support audit; pre-expansion baseline was 111 groups | >=200 groups | 2026-07-16 | PASS for support size; use expanded benchmark for variance-reduced evaluation |
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
| Morgan hidden=4096 | 12-18 min | 2.5-4 h | completed; no main-branch gain |
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
| hidden4096 | PASS: 10/10 seeds, ranking metrics present |
| dropout04 | PASS: 10/10 seeds, ranking metrics present |
| paired tests | PASS: v2 vs each branch, Top-1/MRR/NDCG generated |
| decision | PASS: both rejected as main; proceed to LR scheduling and representation/data scale |

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
| Manifest draft | PASS: `PC-CNG-v3-benchmark-manifest-20260712.md` created |

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

Current M3 data-expansion evidence:

| Item | Status |
|---|---|
| Original held-out parent scan | DONE: USPTO/OpenMolecules scan found 51,494 eligible unique held-out contexts after split/context/reaction leakage filters |
| Expansion candidate list | DONE: 256 positive parents selected at `results/original_test_expansion_uspto_scan_20260712/uspto_original_test_expansion_candidates.csv` |
| Boundary-negative generation/review | DONE: 1,746 raw candidates generated; reviewed/filter output has 1,616 rows and 1,268 `keep_synthetic_negative` rows |
| Evaluable group closure | DONE: integrated support audit reports 323 combined test groups, target 200, deficit 0 |
| Expanded v2 baseline evaluation | DONE: v2/unreacted 10-seed mean Test Top-1 51.60 ± 0.89% over 293 scored/evaluable test groups; ensemble Test Top-1 51.88% |
| Expanded M0 branch retest | DONE: hidden4096 and dropout04 evaluated; neither passes paired Top-1 promotion gate |
| Expanded combined retest | DONE: combined evaluated; no significant paired Top-1 gain on expanded M3 benchmark |
| Expanded classw050_rc retest | DONE: classw050_rc evaluated; no significant paired Top-1 gain on expanded M3 benchmark |
| External bridge support audit | COMPLETE on repaired 25k: contexts `25,000/25,000`, strict complete groups `25,000/25,000`, decision flags `[]`; performance negative for PC-CNG |
| External bridge context expansion | COMPLETE: selected `8,950` USPTO/OpenMolecules contexts, repaired `77` blank-reactant contexts, final repaired context set `25,000/25,000` |
| External bridge 25k benchmark | COMPLETE: `311,150` candidates; strict Top-1 Chemformer `57.00%` vs PC-CNG `13.59%`; validity-aware Top-1 Chemformer `44.02%` vs PC-CNG scored subset `13.59%` |

Acceptance:

| Item | Pass criterion |
|---|---|
| Expanded test | PASS: scan leakage filters applied, data-quality audit `pass_with_warnings` with 0 hard failures, support audit passed at 323/200 test groups |
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

1. Finish hidden4096 10-seed training and reranking. **DONE**
   - Deliverable: `results/type1_v2_hidden4096_20260712/*/ranking_metrics.json`
   - Acceptance: 10/10 seeds complete.
2. Finish dropout04 10-seed training and reranking. **DONE**
   - Deliverable: `results/type1_v2_dropout04_20260712/*/ranking_metrics.json`
   - Acceptance: 10/10 seeds complete.
3. Run `multiseed_summary.py` for hidden4096 and dropout04. **DONE**
   - Acceptance: mean/std/min/max for overall/train/val/test Top-1/MRR/NDCG.
4. Run paired significance vs v2 for hidden4096 and dropout04. **DONE**
   - Acceptance: group-level ensemble + seed-level bootstrap.
5. Start cosine LR + warmup 10-seed experiment as soon as GPU 4 is free. **DONE + NOT PROMOTED**
   - Deliverable: `results/type1_v2_coslr_warm5_20260712/`
   - Acceptance: first seed completes training + rerank without error.
   - Result: Test Top-1 `87.41 ± 1.82%`; paired ΔTop-1 `+0.08 pp`, CI `[-0.24,+0.48]`, p=`1.000`; not promoted.
6. Update progress report and SOTA gap document with completed experimental evidence. **DONE**
   - Acceptance: no stale “early positive” language if 10-seed result contradicts it.

### P0: next 24 hours

1. Implement or run checkpoint-selection branch based on val Top-1 instead of val ROC-AUC. **IMPLEMENTED + QUEUED**
   - Acceptance: val/test Top-1 selection comparison table.
   - Current smoke: `results/type1_v2_valtop1_ckpt_smoke_20260712/`; completed 3 seeds and rejected (Test Top-1 `83.54 ± 1.43%`, paired ΔTop-1 `-0.97 pp`, CI fully negative).
2. Run n_bits=8192 Morgan smoke test on 3 seeds. **10-SEED DONE + NOT PROMOTED**
   - Acceptance: if mean test Top-1 >=87.7%, promote to 10 seeds.
   - Result: Test Top-1 `87.78 ± 1.36%`; paired ΔTop-1 `+0.32 pp`, CI `[-0.32,+0.97]`, p=`0.489`; not promoted.
3. Run binary_count Morgan smoke test on 3 seeds. **SMOKE DONE + NOT PROMOTED**
   - Acceptance: if no featurization/memory issue and test improves, promote.
   - Result: Test Top-1 `87.24 ± 0.71%`; paired ΔTop-1 `+0.16 pp`, CI `[-0.16,+0.48]`, p=`0.624`; below smoke-to-10seed threshold, not promoted.
4. Run pairwise-weight / margin smoke matrix: **10-SEED DONE + NOT PROMOTED**
   - `pairwise_weight`: 0.5, 1.0, 2.0
   - `margin`: 0.0, 0.05, 0.10
   - Acceptance: select at most two configs for 10-seed confirmation.
   - Result: `pw20_m000` Test Top-1 `88.15 ± 1.95%`, paired ΔTop-1 `+0.08 pp`, CI `[-0.32,+0.48]`, p=`1.000`; `pw20_m005` Test Top-1 `87.41 ± 2.52%`, paired ΔTop-1 `0.00 pp`, CI `[-0.48,+0.48]`, p=`1.000`. Both not promoted.
5. Freeze benchmark manifest. **DRAFT CREATED**
   - Acceptance: every table has dataset, split, scope, metric, baseline, seed rule.
   - Draft: `PC-CNG-v3-benchmark-manifest-20260712.md`.

### P1: 2-4 days

1. Expand original held-out benchmark to reduce 81-group variance.
   - Acceptance: >=200 test groups, no leakage.
2. Expand external beam benchmark.
   - Acceptance: >=25,000 groups or documented source limit.
   - Current evidence: repaired `25,000/25,000` contexts pass base-quality audit; repaired Chemformer beams completed `5/5`; full candidate set has `311,150` rows; strict complete groups are `25,000/25,000`; support-audit decision flags are empty.
   - Result boundary: denominator scale is complete, but PC-CNG underperforms Chemformer likelihood on strict and validity-aware external product selection; no SOTA claim.
   - Next action: use this negative bridge evidence to motivate Chemformer-reference features or scorer calibration, rather than claiming external success.
3. Add Chemformer reference score as scalar feature.
   - Acceptance: strict external Top-1 >=85% in 3-seed smoke.
4. Test combined + original-scope mitigation:
   - class weights;
   - lower graph_stats scaling;
   - gated ensemble between Morgan and graph_stats/combined.
   - Acceptance: expanded curated gain retained, original test not below v2 by >0.5 pp.
5. Start Ni data acquisition or curation. **AUDITED GAP**
   - Acceptance: >=20 distinct Ni parent reactions or clear written limitation.
   - Current evidence: RDKit atomic-number audit reports HITEA normalized `0` Ni atom reactions and USPTO/OpenMolecules `6` distinct Ni parent reactants.
   - Next action: acquire external/curated Ni molecular contexts or state the limitation explicitly.

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
