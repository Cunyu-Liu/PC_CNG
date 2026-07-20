# PC-CNG v3 External Held-out Calibration Protocol

日期：2026-07-15

适用任务：external product-selection bridge / Chemformer-reference calibration / PC-CNG scorer calibration

## 1. 目的

上一轮 repaired 25k external bridge 已关闭 denominator scale gate，但结果为负：strict shared test Top-1 为 Chemformer likelihood `57.00%`、PC-CNG `13.59%`、best nonzero hybrid `50.87%`，不能形成 external SOTA 主张。

本协议冻结一个新的 5k held-out evaluation 输入集，用于后续验证任何 Chemformer-reference feature、PC-CNG scorer calibration 或新 scorer architecture。该 5k 不用于调参、选权重、选择 reaction-class rule，也不回填上一轮 repaired 25k 结论。

## 2. 数据冻结

| Item | Value |
|---|---|
| Existing excluded context set | `/home/cunyuliu/pc_cng_research/results/external_product_prediction_benchmark_25k_repaired_20260713/product_prediction_contexts.csv` |
| Source pool | `/home/cunyuliu/pc_cng_research/data/processed/uspto_openmolecules_normalized.csv` |
| Selection output dir | `/home/cunyuliu/pc_cng_research/results/external_calibration_heldout_contexts_5k_20260715/` |
| Selected held-out contexts | `5,000` |
| Existing repaired contexts excluded | `25,000` |
| Eligible unique source contexts after exclusions | `508,617` |
| Selected split counts | `{"test": 5000}` |
| Selected dataset counts | `{"uspto_openmolecules_yield25to150": 5000}` |
| Context selection summary SHA256 | `0f6e43d4b9cab5274d612e6adc0af250f4bc41c2130b95e6b5771c8d6aeb5525` |
| Held-out contexts CSV SHA256 | `1d4e3fa8617acdbef02fb5e6d0a5191352d199c5505b4c8c05b99b12d7b2a19f` |
| Held-out Chemformer input SHA256 | `d099b66bd7d923b1166ad50d8e19675a41eb7886e14642f38f7826a0e9e34518` |

Selection rules inherited from `select_external_product_prediction_contexts.py`:

1. Exclude source IDs already present in the repaired 25k external benchmark.
2. Exclude canonical reactant contexts already present in the repaired 25k external benchmark.
3. Exclude source contexts that appear across multiple source splits.
4. Keep only positive source reactions with valid reactants/products and canonicalizable reactants.
5. Select deterministically by split, yield-descending representative score, then source ID.

## 3. Base Candidate Readiness

CPU-only base candidate construction has been completed; no Chemformer beam generation or GPU scoring has been run for this held-out set yet.

| Artifact | Value |
|---|---|
| Base benchmark dir | `/home/cunyuliu/pc_cng_research/results/external_calibration_heldout_benchmark_5k_20260715/` |
| Targeted PC-CNG dir | `/home/cunyuliu/pc_cng_research/results/external_calibration_heldout_pc_cng_targeted_5k_20260715/` |
| Base candidate rows | `16,873` |
| Observed positive rows | `5,000` |
| PC-CNG negative rows | `11,873` |
| PC-CNG negative groups | `4,995 / 5,000` |
| PC-CNG negative group coverage | `0.999` |
| Base quality decision | `pass` |
| Base candidate summary SHA256 | `68626c2384451bd17c4d9501099184d4a1874b733ab8e98adf8e9bc139d35d9a` |
| Base candidate CSV SHA256 | `f544ef44ad67a30182d944b056b86f41af294ebb108185e84a9b35467666986e` |
| Base quality audit SHA256 | `e0406b8daca4802354e37195ed1b1e17eed26336595bd84723879956c3a0dada` |

Quality gate details:

| Gate | Result |
|---|---|
| Missing observed-positive groups | `0` |
| Bad observed-positive multiplicity groups | `0` |
| Duplicate candidate-product groups | `0` |
| Same-product PC-CNG negatives | `0` |
| Invalid candidate reactions | `0` |
| Invalid PC-CNG negative reactions | `0` |
| Hard failures | `[]` |
| Warnings | `[]` |

## 4. Beam / Scoring Input

| Item | Value |
|---|---|
| Chemformer input chunks dir | `/home/cunyuliu/pc_cng_research/results/external_calibration_heldout_benchmark_5k_20260715/chemformer_input_chunks/` |
| Chunk size | `5,000` |
| Chunk count | `1` |
| Total rows | `5,000` |
| Chunk manifest SHA256 | `5d7e85f48f5f2214e684f83db8ea3da0e6d3f0ec85e8a48be716a0726ed3c45a` |
| Chunk manifest MD SHA256 | `f99d943892125bfccde603e39e5c4a91340ca63119592ef5b6c08031c03dfc08` |

## 5. Allowed Calibration Inputs

Allowed before final held-out scoring:

1. Existing repaired 25k train/val rows and their existing score artifacts.
2. Existing repaired 25k strict/shared calibration audit for diagnostic model design.
3. Training-set metadata that does not include the new 5k held-out labels or scores.

Forbidden before freezing a candidate calibration recipe:

1. Using the new 5k held-out Chemformer/PC-CNG scores to choose weights, features, thresholds, reaction-class routing, or checkpoints.
2. Selecting among multiple calibration recipes by the new 5k held-out Top-1.
3. Reporting subgroup-selected gains from this 5k as SOTA unless the subgroup rule was fixed before scoring.

## 6. Final Evaluation Protocol

When GPU resources are available:

1. Generate Chemformer beams for the single 5k input chunk.
2. Build full observed + PC-CNG + Chemformer beam candidate set.
3. Score Chemformer conditional likelihood over the full candidate set.
4. Score PC-CNG or any frozen new scorer over the same candidates.
5. Evaluate strict shared groups first; validity-aware results are secondary.
6. Report Top-1, Top-3, MRR, NDCG, coverage, candidate rows, complete groups, and SHA256 for every score/candidate artifact.

## 7. Promotion Gate

A calibration/scorer branch can be promoted only if all of the following are true on this frozen held-out 5k:

1. Strict shared test Top-1 improves over Chemformer likelihood.
2. Paired group-level CI95 for Top-1 delta is entirely positive.
3. Coverage and candidate denominator are not reduced relative to the strict shared evaluation.
4. The branch was frozen before held-out scoring.
5. The result is documented as external bridge evidence; it is not a full SOTA success unless it also satisfies the project-level external target and task-alignment gates.

Current status: held-out 5k context selection, base candidate construction, base quality audit, and Chemformer input chunking are complete. Beam generation and final scoring have not been run.

## 8. Frozen Pre-heldout Recipe Audit

Two lightweight calibration recipes were trained before any held-out 5k beam/scoring. Both use only the repaired 25k strict shared `train` split and the same fixed feature family:

`[bias, Chemformer group-z, PC-CNG group-z, PC-CNG minus Chemformer group-z, interaction, PC-CNG group-z squared]`.

| Recipe | Objective | Train split rows/pairs | Val Top-1 | Repaired 25k Test Top-1 | Decision |
|---|---|---:|---:|---:|---|
| Chemformer likelihood baseline | none | n/a | `83.42%` | `57.00%` | reference baseline |
| `pc_cng_lr_calibrator_v1` | pointwise balanced logistic | `39,753` rows | `80.62%` | `45.04%` | frozen negative pre-heldout evidence; not primary |
| `pc_cng_pairwise_calibrator_v1` | pairwise logistic preference | `26,601` pairs | `80.62%` | `44.99%` | frozen negative pre-heldout evidence; not primary |
| `pc_cng_mlp_calibrator_v1` | fixed-feature pairwise MLP, hidden_dim=16 | `26,601` pairs | `89.26%` | `36.46%` | val-positive but cross-domain test-negative; not primary for USPTO held-out |

Artifacts:

| Artifact | SHA256 |
|---|---|
| `/home/cunyuliu/pc_cng_research/chem_negative_sampling/pc_cng/train_external_score_calibrator.py` | `6a86267f63429f7d7d05132cd9a6d8f074c257368b54dbc1252a7b9d00e3d703` |
| `/home/cunyuliu/pc_cng_research/results/external_score_calibrator_lr_v1_repaired25k_train_20260715/external_score_calibrator_model.json` | `e76fd3085ddfb2f474ed4c19b8f311007cd63de8d91df0ee5359b5a112f1b65a` |
| `/home/cunyuliu/pc_cng_research/results/external_score_calibrator_lr_v1_repaired25k_train_20260715/external_score_calibrator_summary.json` | `a37fe44bc88daa13b2906b99f1c40093c4d5c16025feea0af2c538a45e49af9c` |
| `/home/cunyuliu/pc_cng_research/results/external_score_calibrator_pairwise_v1_repaired25k_train_20260715/external_score_calibrator_model.json` | `041dd006c5823a5c380db147f61f7a617b8e9bfb456c61fda8c2f033c93fd2c8` |
| `/home/cunyuliu/pc_cng_research/results/external_score_calibrator_pairwise_v1_repaired25k_train_20260715/external_score_calibrator_summary.json` | `0b3941de2ea94cc0620b6209b2412b42c18f8ab27049cf417f72f927bc56d347` |
| `/home/cunyuliu/pc_cng_research/chem_negative_sampling/pc_cng/train_external_score_mlp_calibrator.py` | `104777af886fd1b93ae038e19eeee9a23034f6fd20f3aaba60939e121da0f8a1` |
| `/home/cunyuliu/pc_cng_research/results/external_score_mlp_calibrator_v1_repaired25k_train_20260715/external_score_mlp_calibrator_model.json` | `5086058b9532d4d53db156c1041a4108ab5afae1d3771198a7a39302f5b25922` |
| `/home/cunyuliu/pc_cng_research/results/external_score_mlp_calibrator_v1_repaired25k_train_20260715/external_score_mlp_calibrator_summary.json` | `a853b9d34976bd148079c114a969bff62641effa20cf80ae9ea7544853d4ef07` |

Pre-heldout decision: linear recipes do not improve validation Top-1. The MLP recipe improves validation Top-1, but fails badly on the repaired 25k test split (`36.46%` vs Chemformer `57.00%`), which is dominated by USPTO/OpenMolecules-like contexts. Because the frozen held-out 5k is also USPTO/OpenMolecules test-only, `pc_cng_mlp_calibrator_v1` is not promoted as the primary held-out 5k candidate. Held-out 5k beam/scoring should wait for a scorer with stronger cross-domain evidence or be run only as benchmark-readiness/baseline evidence.

## 9. USPTO Train/Val Calibration Pool

To avoid training future scorers only on HITEA/Regio-like repaired 25k train/val rows, a same-domain USPTO/OpenMolecules train/val calibration pool was prepared outside both repaired 25k and held-out 5k.

| Item | Value |
|---|---|
| Train context selection | `/home/cunyuliu/pc_cng_research/results/external_calibration_train_contexts_10k_20260715/external_product_prediction_context_expansion_summary.json`; selected `10,000` train contexts |
| Val context selection | `/home/cunyuliu/pc_cng_research/results/external_calibration_val_contexts_2k_20260715/external_product_prediction_context_expansion_summary.json`; selected `2,000` val contexts |
| Original train/val context pool | `/home/cunyuliu/pc_cng_research/results/external_calibration_trainval_contexts_12k_20260716/external_calibration_trainval_contexts_12k_summary.json` |
| Repaired train/val context pool | `/home/cunyuliu/pc_cng_research/results/external_calibration_trainval_contexts_12k_repaired_20260716/external_calibration_trainval_contexts_12k_repaired_summary.json` |
| Split counts | `{"train": 10000, "val": 2000}` |
| Removed invalid observed-positive sources | `uspto_openmol_000019302`, `uspto_openmol_000022432` |
| Added replacement train sources | `uspto_openmol_000301515`, `uspto_openmol_000301516` |
| Repaired base candidate rows | `40,244` (`12,000` observed positives + `28,244` PC-CNG negatives) |
| PC-CNG negative group coverage | `0.99817` |
| Repaired base quality audit | `pass` |
| Warning / hard failures | `[] / []` |
| Invalid candidate reactions | `0`; PC-CNG negative invalid rows `0` |

SHA256:

| Artifact | SHA256 |
|---|---|
| `select_external_product_prediction_contexts.py` | `92c2695f6fbba710e5d4d9027255c3f72a62584872d44605fee0377b95f34cfc` |
| `repair_external_trainval_context_pool.py` | `1c605d3eca778db3a3213f602338b1911c8bb72d281144006181f8e3945fd79a` |
| train10k summary | `3f6486ddd0793020d2a22acae519133d512192517daaf1a0052dd03862ac02ee` |
| val2k summary | `d4191a49681a42aea53ba958cfdc7b80bcff62f05b5b156224f3715033af392e` |
| original trainval12k summary | `ffe39b065330bdf7b55edc86d1960f6610511ba6354693da201beff891da256f` |
| repaired trainval12k summary | `3aa1cb5e7e07b661a5ce4377dfaa363f221402aaa65a325e70bb9f79faa0397b` |
| repaired trainval12k contexts | `47553df09911a4a06f2690173ecc6dde602dc708886c0c15e1a44f1d4c36e0b6` |
| repaired targeted PC-CNG candidates | `f22d877a6dbb8dc064c0eeea560afda67fe5010f0c9ee8fa472f2a3db69adcde` |
| repaired base candidate summary | `07c726357fb6d44c8744f2bd1a5fee26a1ec1efbdd44fc0505b4101f05609e75` |
| repaired base candidate CSV | `61f4534bbb41573e05c97e9bf6d0296af50858203bce6ee204c4054073f0b270` |
| repaired base quality audit | `d14ca002ec9a1006838aea81340cb1452d0572d96a19bf3d0390c07e2cf32ec2` |
| repaired PC-CNG scoring summary | `57e57150892387da37883b76202c4a0819a5e787244fc05b4541945bb333d06b` |
| repaired PC-CNG candidate scores | `e14f117a0eb2eb59eb27cfae551d1eced7183c972d48dc4ea09087450fa2fb5e` |

PC-CNG-only scoring has also been completed on the repaired pool using the 10-model filtered-v2 ensemble (`cuda:6`): all `40,244` candidate rows received PC-CNG scores, no invalid-negative penalty was filled, and the ranking-decision subset covers `11,978` groups. PC-CNG-only Top-1 is low (`14.94%` train, `16.42%` val, `15.19%` overall), so this is readiness/diagnostic evidence rather than a positive scorer result.

Chemformer likelihood scoring is now complete on the same repaired pool. Chemformer val Top-1 is `89.89%`, PC-CNG val Top-1 is `16.42%`, and the validation-selected simple hybrid remains `w0p00` (Chemformer-only). Three frozen train-only recipes were then trained on the repaired train split: LR val Top-1 `87.39%`, pairwise val Top-1 `84.68%`, and MLP val Top-1 `93.14%`. The MLP recipe is validation-positive and can be frozen as the next held-out candidate, but it is not held-out evidence.

Decision: the repaired pool is now suitable as a same-domain calibration source candidate for the next frozen scorer. The MLP recipe is the current validation-positive candidate. The scorer recipe must be frozen before any held-out 5k scoring, and held-out 5k labels/scores remain forbidden for recipe selection.

## 10. Held-out 5k Execution Status

The frozen USPTO12k MLP recipe has been applied to the held-out 5k base candidate set only. This diagnostic uses observed positives plus PC-CNG negatives, not Chemformer beam candidates, so it is not the final full-beam held-out protocol result.

| Item | Value |
|---|---|
| Base-only scored rows | `16,868 / 16,873` |
| Base-only groups | `4,995` |
| Chemformer Top-1 | `91.99%` |
| PC-CNG Top-1 | `17.44%` |
| MLP Top-1 | `94.51%` |
| Diagnostic decision | positive on base-only candidates, pending full-beam validation |

Full-beam generation is still running via `results/logs/external_calibration_heldout5k_beams_20260716.sh`. The initial GPU6 run was stopped after no completed TSV output; the GPU7 run is active. No full-beam held-out SOTA claim is valid until the merged beam TSV, full candidate set, Chemformer likelihood, PC-CNG scores, frozen MLP scores, and strict/shared metrics are all produced and audited.
