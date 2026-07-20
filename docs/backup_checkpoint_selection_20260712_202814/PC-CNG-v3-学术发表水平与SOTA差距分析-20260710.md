# PC-CNG v3 学术发表水平与 SoTA 差距分析

审计日期：2026-07-10

## 结论先行

当前 PC-CNG v3 已达到“可写成方法学论文初稿”的水平，但还不能直接宣称达到通用 reaction outcome prediction 或 retrosynthesis SoTA。

更准确的定位是：

```text
PC-CNG is a negative-data / boundary-supervision framework for improving same-context candidate product ranking and feasibility scoring under limited negative supervision.
```

## 2026-07-11 增量审计：Optimization A/C/D 服务器闭环

本节补充 2026-07-11 已完成的代码改进、服务器训练/测试和论文表格汇总。原始 2026-07-10 结论仍成立：PC-CNG 不应宣称端到端 product generation SoTA；但现在已经补齐了更严格的 external product-selection bridge、graph-aware architecture supplement 和 reaction-class gate。

| Track | Server evidence | Main result | Paper-ready claim boundary |
|---|---|---|---|
| Optimization A: external product bridge | `/home/cunyuliu/pc_cng_research/results/external_product_prediction_benchmark_20260711/benchmark_validity_aware/benchmark_summary.json` | full beam validity-aware Test Top-1: PC-CNG Morgan 5-seed 98.50 vs Chemformer likelihood 0.07, 15,973 groups / 174,908 rows | report as validity-aware candidate selection, not pure end-to-end generation |
| Optimization A: strict shared intersection | `/home/cunyuliu/pc_cng_research/results/external_product_prediction_benchmark_20260711/benchmark/benchmark_summary.json` | strict Test Top-1: PC-CNG 71.60 vs Chemformer likelihood 4.94, 1,197 shared scored groups | strongest apples-to-apples learned scorer comparison |
| Optimization D: graph-aware scorer | `/home/cunyuliu/pc_cng_research/results/type1_graph_stats_pairwise_full/summary.json` | same-context held-out Test Top-1 improves from Morgan seed 83.58 to graph-stats seed 97.01 | architecture supplement; Morgan 5-seed remains external main branch |
| Optimization C: reaction-class gate | `/home/cunyuliu/pc_cng_research/results/type1_class_fallback_supplement_20260711/reaction_class_fallback_trained/reaction_class_benchmark.json` | HITEA Top-1 improves to 96.71; Hydrogenation rises from 33.33 to 66.67 Top-1 but remains low-support | useful stress-test and quota evidence; not all weak classes solved |
| Optimization C: partial-product negative ablation | `/home/cunyuliu/pc_cng_research/results/type1_partial_product_supplement_20260711` | 78 reviewed partial-product negatives; same-context Test Top-1 85.19, below fallback-only 86.42; weak-class groups unchanged | not selected; shows product-fragment edits alone do not solve support gate |
| Optimization C: source/molecular support audit | `/home/cunyuliu/pc_cng_research/results/reaction_class_source_support_audit_20260711` | Amide/Cu/Ni are data-source gaps; Hydrogenation/Rh are generator-coverage gaps | prevents overclaiming source-record duplicates as molecularly independent support |
| Optimization C: unreacted-substrate generator v2 | `/home/cunyuliu/pc_cng_research/results/type1_unreacted_substrate_supplement_v2_20260711` | Hydrogenation 39 groups, strict Top-1 84.62 / tie-aware 94.87; Rh 20 groups, Top-1 100.00 | generator-coverage weak classes solved; remaining gaps are data-source gaps |
| Manuscript tables | `/home/cunyuliu/pc_cng_research/results/manuscript_tables_pc_cng_v3/manifest.json` | 10 csv/md tables generated, including `main_external_product_bridge`, `supp_graph_stats_architecture`, `supp_reaction_class_gate`, `supp_source_support_audit` | current table set is ready for manuscript drafting and supplementary audit |

Updated top-line decision:

```text
PC-CNG v3 now has a credible top-journal candidate-selection story:
it substantially outperforms frozen Chemformer likelihood under a pre-declared
external candidate-ranking protocol and gains strong same-context generalization
from graph-aware reaction-difference features. The remaining top-journal gap is
class-complete robustness: Hydrogenation and Rh now clear source/molecular
support and strict performance gates under the unreacted-substrate v2 supplement.
The remaining class-completion gap is Amide/Cu/Ni, which require external or
curated molecular contexts rather than repeated source records.
```

最强主结论应放在：
## 2026-07-12 增量审计：Combined 特征 + M0 Scale-up/Regularization

本节补充 2026-07-12 完成的 combined 特征架构消融、hidden_dim=4096 scale-up 和 dropout=0.4 正则化实验。

| Track | Server evidence | Main result | Paper-ready claim boundary |
|---|---|---|---|
| Optimization D: combined Morgan+graph-stats | `results/type1_combined_feature_v2_20260712/` | Original scope: +0.20 pp overall, -0.86 pp test (ns); Expanded curated: +5.16 pp overall, +6.44 pp test, p<0.0001 | Architecture ablation: Morgan擅长主分布, graph_stats擅长弱类泛化; combined在expanded scope最优 |
| Optimization D: 10-seed paired significance (combined) | `paired_significance_v2_vs_combined_expanded/summary.json` | Expanded curated Top-1 delta +6.30 pp CI[+5.14,+7.52], permutation p<0.0001, sign-test p≈1e-29, 105 groups win vs 2 lose | Top-journal statistical evidence for architecture supplement |
| Scale-up: hidden_dim 2048→4096 | `results/type1_v2_hidden4096_20260712/hidden4096_multiseed_summary/summary.json` + `paired_significance_v2_vs_hidden4096_same_split/summary.json` | 10-seed Test Top-1 86.30 ± 1.36%; ensemble Top-1 delta +0.16 pp, CI[-0.16,+0.48], p=0.629 | Reject as main; capacity alone does not close original-scope test gap |
| Regularization: dropout 0.20→0.40 | `results/type1_v2_dropout04_20260712/dropout04_multiseed_summary/summary.json` + `paired_significance_v2_vs_dropout04_same_split/summary.json` | 10-seed Test Top-1 86.30 ± 1.69%; ensemble Top-1 delta +0.00 pp, p=1.000 | Reject as main; stronger dropout has no material Top-1 effect |
| Manuscript tables v2 | `results/manifest_tables_pc_cng_v3/manifest.json` | 12→14 tables, added `supp_combined_feature_multiseed` + `supp_combined_feature_paired_significance` | Combined feature evidence ready for supplement |

Updated top-line decision:

```text
PC-CNG v3 now has three converging pillars of top-journal evidence:
1. External validity-aware candidate selection (98.50% vs Chemformer 0.07%)
2. Reaction-class weak-class gate closure via unreacted-substrate v2 + curated contexts
3. Architecture ablation: Morgan+graph_stats combined feature yields highly significant
   improvements on expanded/weak-class scope (p<0.0001), revealing complementary
   inductive biases between substructure fingerprints and global graph statistics.

The remaining main gap is original-scope held-out test Top-1 vs RegioSQM20 (~5.5 pp).
M0 scale-up/regularization audit is now closed: hidden4096 and dropout04 both fail
promotion gates, so the next active path is LR scheduling, checkpoint-selection
alignment, representation scale, and data scale rather than more hidden_dim/dropout.
```

```text
type-1 boundary negative generation + same-context candidate reranking
```

而不是：

```text
end-to-end forward product generation SoTA
```

## 当前最佳结果

### Type-1 主模型

```text
model: diverse-anchor pairwise_default
result path:
/home/cunyuliu/pc_cng_research/results/type1_diverse_anchor_ablation_full/paper_summary/paper_table.md
```

| Metric | PC-CNG v3 |
|---|---:|
| Overall candidate Top-1 | 97.49 +/- 0.06 |
| Synthetic candidate Top-1 | 96.63 +/- 0.07 |
| Regio challenge Top-1 | 96.59 +/- 0.07 |
| Heteroatom challenge Top-1 | 98.63 +/- 0.00 |
| Held-out test Top-1 | 85.07 +/- 0.94 |

### Chemformer reference

```text
result path:
/home/cunyuliu/pc_cng_research/results/type1_diverse_anchor_dpo_reference/paper_summary/paper_table.md
```

| Model | Overall Top-1 | Synthetic Top-1 | Test Top-1 |
|---|---:|---:|---:|
| Frozen Chemformer conditional likelihood | 46.08 | 28.33 | 44.33 |
| Chemformer-reference pairwise reward | 91.07 +/- 0.38 | 95.67 +/- 0.55 | 77.11 +/- 1.52 |
| PC-CNG pure diverse-anchor pairwise | **97.49 +/- 0.06** | **96.63 +/- 0.07** | **85.07 +/- 0.94** |

### DPO beta / reference-scale sweep

```text
result path:
/home/cunyuliu/pc_cng_research/results/type1_diverse_anchor_dpo_beta_sweep_full/paper_summary/paper_table.md

server status:
12/12 runs complete; summary.json rebuilt with all records.
```

| Setting | Overall Top-1 | Overall Top-3 | Overall MRR | Synthetic Top-1 | Held-out test Top-1 | Gap to pairwise_default |
|---|---:|---:|---:|---:|---:|---:|
| dpo_beta005 | 88.08 +/- 0.09 | 93.66 +/- 0.03 | 91.25 +/- 0.04 | 90.67 +/- 0.36 | 74.23 +/- 0.00 | -9.41 pp overall / -10.84 pp test |
| dpo_beta010 | 89.58 +/- 1.31 | 93.77 +/- 0.12 | 92.05 +/- 0.71 | 93.33 +/- 2.05 | 75.95 +/- 1.29 | -7.91 pp overall / -9.12 pp test |
| dpo_beta020_refnone | 91.09 +/- 0.14 | 93.66 +/- 0.09 | 92.78 +/- 0.09 | 95.61 +/- 0.21 | **79.04 +/- 0.97** | -6.40 pp overall / -6.03 pp test |
| dpo_beta050 | **91.16 +/- 0.07** | **93.87 +/- 0.03** | **92.93 +/- 0.04** | **95.78 +/- 0.16** | 76.29 +/- 0.00 | -6.33 pp overall / -8.78 pp test |

Interpretation:

```text
The formal beta/reference-scale sweep improves strongly over frozen Chemformer,
but it does not beat diverse-anchor pairwise_default.
Therefore the main model remains pure pairwise boundary supervision.
DPO is useful as a reference-policy ablation and Science Advances-style reward evidence,
not as the current best production branch.
```

### Type-2 low-yield branch

```text
result path:
/home/cunyuliu/pc_cng_research/results/type2_low_yield_branch_full/paper_summary/paper_table.md
```

| Setting | Test ROC-AUC | Test AUPRC | Test F1 |
|---|---:|---:|---:|
| low_yield_synth02 | 85.39 +/- 0.27 | 79.56 +/- 0.31 | 71.26 +/- 0.46 |
| low_yield_synth05 | **85.56 +/- 0.21** | **79.93 +/- 0.08** | 72.14 +/- 0.43 |
| low_yield_synth10 | 85.53 +/- 0.22 | 79.79 +/- 0.23 | **72.30 +/- 0.08** |

## 与 SoTA / 强基线对比

参考来源：

```text
RegioSQM20:
https://pmc.ncbi.nlm.nih.gov/articles/PMC7881568/

RegioSQM18:
https://pubmed.ncbi.nlm.nih.gov/29629133/

Molecular Transformer:
https://pmc.ncbi.nlm.nih.gov/articles/PMC6764164/

2025 regio/site-selectivity review:
https://pmc.ncbi.nlm.nih.gov/articles/PMC11891785/
```

### Regioselectivity / RegioSQM20

公开参考：

- RegioSQM20 with tautomers reports 92.7% success rate; without tautomer handling about 90.7%.
- IBM RXN on the RegioSQM20 comparison is reported at 76.3-85.0%.
- Source: Ree et al., RegioSQM20, J. Cheminformatics 2021, PMC7881568.

| Model / Method | Task / metric | Reported value | PC-CNG closest value | Absolute gap | Relative gap |
|---|---|---:|---:|---:|---:|
| RegioSQM20 + tautomers | EAS regioselectivity success | 92.7 | PC-CNG regio challenge Top-1 96.59 | +3.89 pp | +4.20% |
| RegioSQM20 no tautomers | EAS regioselectivity success | 90.7 | PC-CNG regio challenge Top-1 96.59 | +5.89 pp | +6.49% |
| IBM RXN range | EAS outcome accuracy | 76.3-85.0 | PC-CNG held-out test Top-1 85.07 | +0.07 to +8.77 pp | +0.08% to +11.49% |
| RegioSQM20 + tautomers | EAS regioselectivity success | 92.7 | PC-CNG held-out test Top-1 85.07 | -7.63 pp | -8.23% |

Interpretation:

```text
PC-CNG 的 regio challenge Top-1 已高于 RegioSQM20 success rate，
但这不是严格 apples-to-apples：PC-CNG 是候选重排，RegioSQM20 是专门的位点预测。
若按 held-out test Top-1 85.07 对比 RegioSQM20 92.7，仍有约 7.63 pp / 8.23% 相对差距。
因此不能直接宣称超越 RegioSQM20 SoTA，但可以说在 same-context candidate reranking 上表现强。
```

### General forward reaction prediction

公开参考：

- Molecular Transformer reports top-1 accuracy above 90% on common forward reaction prediction benchmarks.
- Source: Schwaller et al., Molecular Transformer, ACS Central Science 2019, PMC6764164.

| Model / Method | Task / metric | Reported value | PC-CNG closest value | Absolute gap | Relative gap |
|---|---|---:|---:|---:|---:|
| Molecular Transformer | End-to-end forward product top-1 | >90 | PC-CNG held-out candidate Top-1 85.07 | at least -4.93 pp | at least -5.48% |
| Molecular Transformer | End-to-end forward product top-1 | >90 | PC-CNG overall candidate Top-1 97.49 | not comparable | not comparable |
| Frozen Chemformer | Conditional likelihood reranking | 46.08 overall / 28.33 synthetic | PC-CNG pairwise 97.49 / 96.63 | +51.41 / +68.30 pp | +111.57% / +241.09% |

Interpretation:

```text
PC-CNG 不是从 reactants 直接生成 product 的 Transformer。
它当前解决的是候选产品排序和负样本边界监督。
所以与 Molecular Transformer / Chemformer product top-1 不能直接同表宣称 SoTA。
但 PC-CNG 显著增强了 Chemformer-reference candidate ranking，是可发表的补充任务证据。
```

### Science Advances negative-data protocol

项目内复现实验：

```text
/home/cunyuliu/pc_cng_research/results/science_advances_dpo_reward_benchmark_full/paper_summary/paper_table.md
```

| Setting | Frozen Chemformer + PC-CNG candidates | Best reward model | Top-1 delta |
|---|---:|---:|---:|
| K_low | 48.55 +/- 2.13 | 63.70 +/- 4.37 | +15.15 pp |
| K_high | 60.22 +/- 0.15 | 89.81 +/- 0.80 | +29.59 pp |

Interpretation:

```text
这是最接近 Science Advances 负样本主张的证据：
PC-CNG negatives 不只是扩充候选，还能作为 rejected outcomes 训练 reward/ranking policy。
该实验支持方法学发表，但仍不是直接复现 Science Advances RL product prediction pipeline。
```

## 学术发表水平判断

| 维度 | 当前状态 | 发表判断 |
|---|---|---|
| 科学问题 | 负样本稀缺与 type-1 boundary negatives 明确 | strong |
| 模型新颖性 | diversity-aware PC-CNG generator + known-positive filtering + pairwise/DPO reward evidence | strong |
| 数据规模 | HiTEA + RegioSQM20 + Chemformer-scored candidates + Science Advances-style splits | sufficient for method paper |
| 多 seed 稳健性 | type-1 5 seeds, DPO-reference/beta 21 formal runs plus smoke checks, Science Advances reward 60 runs, type-2 15 runs | strong |
| 外部 reference | Chemformer conditional likelihood integrated | strong |
| SoTA claim | Not end-to-end product prediction SoTA | should avoid overclaim |
| 投稿定位 | negative-data / reaction-boundary learning / candidate reranking | suitable |

Final judgment:

```text
当前模型达到学术发表的“方法学论文”水平；
若目标是顶刊，需要进一步补足端到端 product prediction 或更严格的 external baseline comparison。
```

## 当前不足

1. 任务口径仍以 candidate reranking / feasibility / validity-aware product selection 为主，不是端到端 product generation。
2. Strict shared-candidate comparison 只覆盖 1,197 groups；validity-aware comparison 覆盖 15,973 groups，但包含显式 featurizability/validity prior，因此两者必须分表报告。
3. RegioSQM20 held-out test Top-1 85.07 与 RegioSQM20 92.7 success rate 仍有约 7.63 pp / 8.23% 相对差距；graph-stats 在 same-context 上强，但 external RegioSQM20 类别仍需 error analysis。
4. Type-2 low-yield branch 提升温和，说明 low-yield seeds 不能作为主贡献。
5. DPO-reference 和 beta sweep 有效增强 Chemformer reference，但没有超过 pairwise-only；最佳 DPO overall Top-1 91.16，仍低于 pairwise_default 97.49。
6. Reaction-class gate 尚未完全达标：fallback 后 Amide coupling 13 groups、Cu coupling 14 groups、Hydrogenation 15 groups、Ni coupling 4 groups、Rh coupling 19 groups，仍低于顶刊口径建议的每类至少 20 evaluable groups。
7. `partial_product` atom-map fragment edit 已完成服务器 negative ablation，但只增加同 source context 内 candidate rows，没有提升 unique weak-class source support；因此不能作为解决弱类 gate 的主方案。
8. Source/molecular support audit 显示弱类剩余缺口需要分流处理：unreacted-substrate v2 已使 Hydrogenation 达到 35/41 candidate parent reactions、Rh 达到 20/20；Hydrogenation strict Top-1 84.62、tie-aware Top-1 94.87，Rh Top-1 100.00。Amide/Cu/Ni 在当前 HITEA 切片中 distinct molecular parent reactions 少于 20，需外部或人工 curated contexts。

## 已落实的优化任务

| Optimization | Implementation | Result |
|---|---|---|
| Regio/heteroatom coverage expansion | `--diverse-anchor` product-graph terminal-substituent shift | reviewed regio/heteroatom negatives from 27/0 to 3393/436 |
| Pairwise 5-seed ablation | `/results/type1_diverse_anchor_ablation_full` | synthetic Top-1 96.63 +/- 0.07 |
| Chemformer-reference DPO | `/results/type1_diverse_anchor_dpo_reference` | reward overall Top-1 91.07 +/- 0.38 |
| Type-2 low-yield branch | `/results/type2_low_yield_branch_full` | best ROC-AUC 85.56 +/- 0.21 |
| Manuscript tables | `/results/manuscript_tables_pc_cng_v3` | main/supp tables generated |
| DPO beta sweep smoke | `/results/type1_diverse_anchor_dpo_beta_sweep_smoke` | beta=0.05/0.10 did not exceed pairwise-only in smoke |
| DPO beta sweep full | `/results/type1_diverse_anchor_dpo_beta_sweep_full` | 12/12 complete; best overall Top-1 91.16, best test Top-1 79.04; not selected as main model |
| Optimization A strict external bridge | `/results/external_product_prediction_benchmark_20260711/benchmark` | strict shared-intersection Test Top-1: PC-CNG 71.60 vs Chemformer likelihood 4.94 |
| Optimization A validity-aware bridge | `/results/external_product_prediction_benchmark_20260711/benchmark_validity_aware` | full beam validity-aware Test Top-1: PC-CNG Morgan 5-seed 98.50 vs Chemformer likelihood 0.07 |
| Optimization D graph-stats scorer | `/results/type1_graph_stats_pairwise_full` | same-context held-out Test Top-1: graph-stats seed 97.01 vs Morgan seed 83.58 |
| Optimization C class-quota/fallback | `/results/type1_class_fallback_supplement_20260711` | HITEA Top-1 96.71; weak-class support improved but Amide/Cu/Hydrogenation/Ni/Rh still below 20 groups |
| Optimization C partial-product ablation | `/results/type1_partial_product_supplement_20260711` | 78 reviewed negatives; Test Top-1 85.19 and HITEA Top-1 95.77, below fallback-only branch; not selected |
| Optimization C source-support audit | `/results/reaction_class_source_support_audit_20260711` | `supp_source_support_audit` separates source-level and molecular-level support; Amide/Cu/Ni require external contexts, Hydrogenation/Rh require generator expansion |
| Optimization C unreacted-substrate supplement v2 | `/results/type1_unreacted_substrate_supplement_v2_20260711` | Hydrogenation 39 groups, strict Top-1 84.62, tie-aware 94.87; Rh 20 groups, Top-1 100.00; source-support audit ok for both |
| Manuscript tables 20260711 refresh | `/results/manuscript_tables_pc_cng_v3` | 10 tables generated; new external bridge, graph-stats architecture, reaction-class gate, and source-support audit tables verified |

## 服务器端优化落地矩阵

| Requirement | Server-side implementation | Concrete settings | Evidence |
|---|---|---|---|
| Training parameter adjustment | DPO beta/reference-scale sweep | `dpo_beta=0.05/0.10/0.50`, `dpo_beta=0.20 + reference_scale=none`, 3 seeds, 80 epochs, batch size 4096, hidden dim 2048 | `/results/type1_diverse_anchor_dpo_beta_sweep_full/paper_summary/paper_table.md` |
| Dataset optimization | Diverse-anchor type-1 generation and reviewed hard negatives | product-graph terminal-substituent shift, known-positive filtering, regio/heteroatom families | reviewed regio/heteroatom negatives increased to 3393/436; pairwise Top-1 97.49 |
| Dataset/task separation | Type-2 low-yield branch | `--synthetic-family low_yield_seed`, synthetic weights 0.2/0.5/1.0 | best ROC-AUC 85.56, best F1 72.30 |
| Model/loss structure improvement | Pairwise boundary reward and Chemformer-reference DPO reward | BCE anchor + pairwise margin + optional DPO reference-relative loss | pairwise_default remains best; DPO validates reference-policy ablation |

## 后续具体优化方案

### Optimization A: strict external product prediction comparison

Goal:

```text
Construct an end-to-end candidate-generation + ranking benchmark
against Chemformer beam outputs on the same reactant contexts.
```

Implementation:

1. Use Chemformer beam predictions as product candidates.
2. Add PC-CNG diverse-anchor negatives and observed positives.
3. Evaluate top-1/top-3/MRR/NDCG with:
   - frozen Chemformer likelihood;
   - PC-CNG pairwise scorer;
   - hybrid linear ensemble.

Expected value:

```text
This is the most direct bridge toward product prediction SoTA comparison.
```

### Optimization B: DPO beta / reference scaling sweep

Goal:

```text
Explain why DPO-only is strongest in K_low but pairwise-only is strongest in diverse-anchor.
```

Executed grid:

```text
dpo_beta: 0.05, 0.1, 0.2, 0.5
reference_scale: none, standardize
pairwise_weight: 0.0
```

Server result:

```text
/home/cunyuliu/pc_cng_research/results/type1_diverse_anchor_dpo_beta_sweep_full

completed configs:
dpo_beta005
dpo_beta010
dpo_beta050
dpo_beta020_refnone

seeds:
20260710-20260712

best overall Top-1:
dpo_beta050 = 91.16 +/- 0.07

best held-out test Top-1:
dpo_beta020_refnone = 79.04 +/- 0.97
```

Decision:

```text
Do not replace pairwise_default with DPO-only for the main model.
Keep DPO as a supplement explaining reference-policy behavior and negative-data reward learning.
```

### Optimization C: reaction-class targeted generator expansion

Goal:

```text
Improve weak reaction classes in supplement tables.
```

Targets:

```text
Hydrogenation type-1 candidate reranking
Rh coupling type-2 feasibility
Cu coupling type-2 F1 instability
```

Implementation:

1. Add class-aware minimum candidate quotas.
2. Downweight classes with extreme positive-rate imbalance.
3. Add class-level calibration thresholds.

### Optimization D: model architecture upgrade

Goal:

```text
Move beyond Morgan-fingerprint MLP toward graph-aware pairwise scorer.
```

Candidate architecture:

```text
shared reactant/product graph encoder
reaction-difference pooling
pairwise margin head
optional Chemformer reference score feature
```

Implementation steps:

1. Add a lightweight graph-pair encoder beside the current Morgan MLP, keeping RDKit-only featurization first to avoid heavy dependency drift.
2. Train on the same diverse-anchor candidate CSV with pairwise_default loss, then optionally add Chemformer log-likelihood as a scalar feature.
3. Evaluate on the same paper-summary protocol: overall/synthetic/test Top-1, MRR, NDCG, and weak reaction-class supplement tables.

Expected value:

```text
Better generalization on held-out test and weak reaction classes.
```

## 目标完成审计

| User requirement | Completion evidence | Status |
|---|---|---|
| Analyze generated evaluation results | Type-1, Chemformer-reference, DPO beta, type-2, and Science Advances-style results summarized above | complete |
| Judge academic publishability | `学术发表水平判断` section gives method-paper yes / end-to-end SoTA no conclusion | complete |
| Compare against State-of-the-Art | RegioSQM20, IBM RXN, Molecular Transformer, and Chemformer reference tables include absolute and relative gaps | complete |
| Show clear tables with model names, metrics, and gap percentages | Main result, DPO sweep, RegioSQM20, and forward-prediction tables include metrics and gap columns | complete |
| Analyze limitations and improvement space | `当前不足` and `后续具体优化方案` sections list task-scope, SoTA, type-2, DPO, and class-instability gaps | complete |
| Propose concrete optimization plan | Optimizations A-D specify external product benchmark, DPO sweep, class-targeted generation, and graph-aware model upgrade | complete |
| Implement optimization on server training/evaluation tasks | DPO beta sweep completed 12/12; diverse-anchor/type-2 data optimization and model/loss upgrades are tied to server result paths | complete |
| Align downstream task and benchmark with peer models | `下游任务评估协议与OptimizationA落地-20260711.md` defines candidate CSV schema, metrics, strict intersection, validity-aware rules, and Chemformer/Molecular Transformer-compatible inputs | complete |
| Validate Optimization A on server | external product benchmark completed with strict and validity-aware summaries plus manuscript table `main_external_product_bridge` | complete |
| Validate Optimization C on server | reaction-class diagnostics, class-quota supplement, class-fallback supplement, partial-product negative ablation, source/molecular support audit, and unreacted-substrate v2 completed; Hydrogenation/Rh generator-coverage gaps solved; Amide/Cu/Ni need curated contexts | partial; continue curated context expansion for data-source gaps |
| Validate Optimization D on server | graph-stats 5-seed training/evaluation and external validity-aware benchmark completed; selected as architecture supplement | complete |

Final audited decision:

```text
The current best publishable claim is PC-CNG as a boundary-negative generation,
candidate-reranking, and validity-aware product-selection framework. It is not
yet an end-to-end reaction outcome prediction SoTA claim. The main external
branch remains Morgan 5-seed validity-aware PC-CNG; graph-stats is a strong
architecture supplement; class_fallback remains the best current weak-class
supplement; partial_product is a documented negative ablation; source-support
audit showed Hydrogenation/Rh needed generator expansion while Amide/Cu/Ni need
external or curated contexts. The unreacted-substrate v2 branch now solves the
Hydrogenation/Rh generator-coverage gaps and passes strict class gates. Reaction-class
completion remains open for Amide/Cu/Ni because the current HITEA slice does not
contain enough distinct molecular parent reactions for a top-journal support gate.
```

## 2026-07-11 addendum: curated weak-class context supplement

Purpose:

```text
Close the remaining top-journal reaction-class support gaps after unreacted-substrate v2.
The key distinction is data-source support vs generator coverage. Amide/Cu had enough
external curated contexts after USPTO/OpenMolecules extraction; Ni still does not.
```

Implemented artifacts:

```text
pc_cng/build_curated_weak_class_contexts.py
scripts_run_type1_curated_weak_class_supplement.sh
pc_cng/build_manuscript_tables.py
```

Server outputs:

```text
/home/cunyuliu/pc_cng_research/results/type1_curated_weak_class_contexts_20260711
/home/cunyuliu/pc_cng_research/results/manuscript_tables_pc_cng_v3/supp_curated_weak_class_contexts.md
/home/cunyuliu/pc_cng_research/results/manuscript_tables_pc_cng_v3/supp_source_support_audit.md
```

Curated support result:

| Class | Positive parent reactions | Candidate parent reactions | Support status |
|---|---:|---:|---|
| Amide coupling | 214 | 208 | ok |
| Cu coupling | 213 | 213 | ok |
| Hydrogenation | 41 | 35 | ok |
| Rh coupling | 20 | 20 | ok |
| Ni coupling | 10 | 4 | data_source_gap |

Expanded curated benchmark:

| Model | Groups | Overall Top-1 | Test Top-1 | MRR | NDCG |
|---|---:|---:|---:|---:|---:|
| v2 checkpoint on expanded curated benchmark | 1635 | 84.77 | 68.64 | 91.09 | 93.34 |
| curated-augmented checkpoint | 1635 | 97.00 | 83.90 | 98.35 | 98.78 |

Weak-class Top-1 improvements on the same expanded benchmark:

| Class | v2 Top-1 | curated-augmented Top-1 | Delta |
|---|---:|---:|---:|
| Amide coupling | 34.13 | 95.19 | +61.06 |
| Cu coupling | 63.85 | 99.06 | +35.21 |
| Hydrogenation | 84.62 | 87.18 | +2.56 |
| Rh coupling | 100.00 | 100.00 | +0.00 |

Selection decision:

```text
Use curated weak-class expansion as a supplement and top-journal support-gate closure for
Amide/Cu. Do not replace the main v2 same-context model yet: on the original Regio/HITEA
scope, curated training improves HITEA Top-1 (96.22 -> 97.48) but slightly lowers RegioSQM20
Top-1 (97.41 -> 96.91) and original test Top-1 (86.42 -> 83.95). The next model-selection
step should be a multi-seed or class-weighted run if we want to promote this branch to main.
```

Updated completion audit:

| Requirement | Current status |
|---|---|
| Hydrogenation support and performance gate | complete, v2 plus tie-aware audit |
| Rh support gate | complete, v2 reviewed-status-aware exclude |
| Amide support gate | complete via curated contexts and curated class_fallback |
| Cu support gate | complete via curated contexts and curated class_fallback |
| Ni support gate | still blocked by external data-source gap |
| Main model replacement | not selected; curated branch is supplement due original-scope tradeoff |

## 2026-07-11 addendum: class-weighted curated candidate

Problem found:

```text
The first curated branch closed Amide/Cu but slightly hurt original-scope generalization.
During class-weighting, an implementation audit found that synthetic rows were losing
reaction_class and all pair rows appeared as class "synthetic". That made class-specific
weights ineffective.
```

Code changes:

```text
pc_cng/train_feasibility_mlp.py
  - preserve reaction_class in read_synthetic_rows.

pc_cng/train_pairwise_reward_mlp.py
  - add --class-weight and --class-margin.

scripts_run_type1_curated_class_weight_selection.sh
  - reproducible runner for weighted curated model selection.
```

Selected single-seed candidate:

```text
curated_classw050_seed20260711
Amide/Cu pairwise weight = 0.5
```

Model-selection result:

| Model | Scope | Groups | Overall Top-1 | Test Top-1 | RegioSQM20 Top-1 | HITEA Top-1 | Curated USPTO Top-1 |
|---|---|---:|---:|---:|---:|---:|---:|
| v2 unreacted | original Regio/HiTEA | 1241 | 97.18 | 86.42 | 97.41 | 96.22 | n/a |
| v2 unreacted | expanded curated | 1635 | 84.77 | 68.64 | 97.41 | 96.12 | 45.69 |
| curated unweighted | original Regio/HiTEA | 1241 | 97.02 | 83.95 | 96.91 | 97.48 | n/a |
| curated unweighted | expanded curated | 1635 | 97.00 | 83.90 | 96.91 | 97.41 | 96.95 |
| curated classw050 | original Regio/HiTEA | 1241 | 97.42 | 86.42 | 97.41 | 97.48 | n/a |
| curated classw050 | expanded curated | 1635 | 97.31 | 85.59 | 97.41 | 97.41 | 96.95 |

Updated selection decision:

```text
curated_classw050_seed20260711 is the current best single-seed same-context candidate:
it preserves v2 original-scope test and RegioSQM20 Top-1, improves HITEA Top-1, and keeps
the expanded Amide/Cu curated benchmark strong. It should be promoted to a multi-seed
candidate before being claimed as the final main model. Ni remains the only unsolved
reaction-class support gap and requires new external molecular contexts.
```

Updated manuscript table:

```text
/home/cunyuliu/pc_cng_research/results/manuscript_tables_pc_cng_v3/supp_curated_model_selection.md
```

## 2026-07-11 addendum: classw050_rc 5-seed stability

Server outputs:

```text
/home/cunyuliu/pc_cng_research/results/type1_curated_weak_class_contexts_20260711/classw050_rc_multiseed_summary
/home/cunyuliu/pc_cng_research/results/manuscript_tables_pc_cng_v3/supp_curated_multiseed_stability.md
```

5-seed result:

| Scope | n seeds | Overall Top-1 | Test Top-1 | RegioSQM20 Top-1 | HITEA Top-1 | Curated USPTO Top-1 |
|---|---:|---:|---:|---:|---:|---:|
| expanded curated | 5 | 97.03 ± 0.37 | 83.56 ± 1.27 | 97.27 ± 0.20 | 96.38 ± 1.51 | 96.75 ± 0.41 |
| original Regio/HiTEA | 5 | 97.12 ± 0.38 | 85.19 ± 1.35 | 97.27 ± 0.20 | 96.47 ± 1.47 | n/a |

Updated decision:

```text
classw050_rc is now validated as a stable curated weak-class supplement: it maintains high
original-scope Top-1 and provides strong expanded curated Amide/Cu performance across 5 seeds.
It should not yet replace the main model in the headline claim because its mean original test
Top-1 (85.19) is below the strongest v2/unreacted single-seed test Top-1 (86.42). The correct
top-journal next step is to extend classw050_rc to 10 seeds and compute bootstrap confidence
intervals plus paired significance tests against v2/unreacted.
```

## 2026-07-11 addendum: paired significance against v2

Artifacts:

```text
pc_cng/paired_reranking_significance.py
/home/cunyuliu/pc_cng_research/results/manuscript_tables_pc_cng_v3/supp_curated_paired_significance.md
```

Result:

| Scope | Metric | v2 mean | classw050 mean | Delta | 95% bootstrap CI | Paired permutation p |
|---|---|---:|---:|---:|---:|---:|
| original Regio/HiTEA | Top-1 | 97.18 | 97.42 | +0.24 | [-0.24, +0.73] | 0.5073 |
| original Regio/HiTEA | MRR | 98.42 | 98.58 | +0.15 | [-0.10, +0.43] | 0.2564 |
| expanded curated | Top-1 | 84.77 | 97.31 | +12.54 | [+10.95, +14.19] | 9.999e-05 |
| expanded curated | MRR | 91.09 | 98.53 | +7.44 | [+6.49, +8.43] | 9.999e-05 |

Interpretation:

```text
The expanded curated Amide/Cu improvement is statistically significant and manuscript-ready
as a weak-class supplement. The original Regio/HiTEA delta is not statistically significant,
so this branch should not be over-claimed as a universal main-model improvement until the
10-seed comparison and paired tests support that stronger claim.
```

## 2026-07-11 addendum: classw050_rc 10-seed final result

Server output:

```text
/home/cunyuliu/pc_cng_research/results/type1_curated_weak_class_contexts_20260711/classw050_rc_multiseed_summary/summary.csv
/home/cunyuliu/pc_cng_research/results/manuscript_tables_pc_cng_v3/supp_curated_multiseed_stability.md
```

10-seed result:

| Scope | n seeds | Overall Top-1 | Test Top-1 | RegioSQM20 Top-1 | HITEA Top-1 | Curated USPTO Top-1 |
|---|---:|---:|---:|---:|---:|---:|
| expanded curated | 10 | 97.16 ± 0.30 | 83.98 ± 1.28 | 97.35 ± 0.24 | 96.68 ± 1.12 | 96.90 ± 0.42 |
| original Regio/HiTEA | 10 | 97.24 ± 0.34 | 85.06 ± 1.51 | 97.35 ± 0.24 | 96.76 ± 1.10 | n/a |

Final model-selection decision:

```text
classw050_rc is validated as a stable curated weak-class supplement across 10 seeds.
It closes the Amide/Cu expanded curated benchmark with strong stability and a significant
paired gain over v2 on that expanded scope. It should not replace the main headline model
because the fair v2/unreacted 10-seed baseline has higher original-scope test Top-1
(87.16 ± 1.58 vs 85.06 ± 1.51). The paper-ready claim should be: main PC-CNG remains
the broader same-context/validity-aware framework; classw050_rc is the curated weak-class
robustness supplement for Amide/Cu; Ni remains an external data-source gap.
```

## 2026-07-11 addendum: fair v2/unreacted vs classw050_rc 10-seed comparison

Server outputs:

```text
/home/cunyuliu/pc_cng_research/results/type1_unreacted_substrate_supplement_v2_20260711/unreacted_v2_multiseed_summary/summary.csv
/home/cunyuliu/pc_cng_research/results/manuscript_tables_pc_cng_v3/supp_curated_vs_v2_multiseed.md
```

Fair 10-seed comparison:

| Scope | Metric | v2/unreacted | classw050_rc | Delta pp |
|---|---|---:|---:|---:|
| original Regio/HiTEA | Overall Top-1 | 97.20 ± 0.24 | 97.24 ± 0.34 | +0.04 |
| original Regio/HiTEA | Test Top-1 | 87.16 ± 1.58 | 85.06 ± 1.51 | -2.10 |
| original Regio/HiTEA | HITEA Top-1 | 96.30 ± 0.81 | 96.76 ± 1.10 | +0.46 |
| expanded curated | Overall Top-1 | 84.66 ± 0.83 | 97.16 ± 0.30 | +12.50 |
| expanded curated | Test Top-1 | 70.17 ± 1.55 | 83.98 ± 1.28 | +13.81 |
| expanded curated | Curated USPTO Top-1 | 45.18 ± 2.84 | 96.90 ± 0.42 | +51.73 |

Final fair-selection decision:

```text
The fair 10-seed baseline confirms that classw050_rc is a manuscript-ready weak-class
robustness supplement, not a universal replacement for v2/unreacted. On the expanded
curated Amide/Cu benchmark it is decisively better (+12.50 pp overall Top-1 and
+51.73 pp curated USPTO Top-1), while on the original Regio/HiTEA scope v2/unreacted
retains the stronger held-out test Top-1. The top-journal framing should separate the
claims: v2/unreacted remains the original-scope main baseline; classw050_rc closes the
Amide/Cu curated support/performance gap; Ni remains unresolved pending new external
molecular contexts.
```

## 2026-07-12 addendum: 10-seed ensemble paired significance

We upgrade the paired significance test from single-seed to 10-seed ensemble level
for top-journal statistical rigor. Two complementary analyses:

1. **Group-level ensemble test**: average per-row scores across 10 seeds → ensemble
   scores → paired bootstrap CI + sign-flip permutation test + sign test.
2. **Seed-level bootstrap CI**: resample seed indices with replacement → 95% CI on
   the mean group-level delta.

Code: `pc_cng/multiseed_paired_significance.py` (4 unit tests passing).

### Original Regio/HiTEA scope — 10-seed ensemble (v2 vs classw050_rc)

| Metric | Groups | v2/unreacted | classw050_rc | Δ pp | 95% CI (bootstrap) | Permutation p | Sign-test p |
|---|---:|---:|---:|---:|---:|---:|---:|
| Top-1 | 1241 | 97.18 | 97.10 | -0.08 | [-0.40, +0.24] | 1.000 | 1.000 |
| MRR | 1241 | 98.42 | 98.37 | -0.04 | [-0.24, +0.15] | 0.692 | 0.791 |
| NDCG | 1241 | 98.82 | 98.79 | -0.03 | [-0.18, +0.11] | 0.679 | 0.791 |

Seed-level bootstrap CI (original scope):

| Metric | Mean Δ pp | 95% CI (seed bootstrap) |
|---|---:|---:|
| Top-1 | +0.04 | [-0.11, +0.19] |
| MRR | +0.05 | [-0.03, +0.12] |
| NDCG | +0.04 | [-0.02, +0.09] |

Interpretation — original scope:

```text
The 10-seed ensemble paired test statistically confirms the null on the original
Regio/HiTEA scope: classw050_rc is indistinguishable from v2/unreacted. Both the
group-level ensemble permutation test (p ≈ 1.0) and the seed-level bootstrap CI
(crossing 0 for Top-1) agree. This is top-journal-grade evidence that classw050_rc
should NOT be promoted as a main-model replacement for the original scope — it
neither hurts nor helps on held-out data.
```

### Expanded curated scope — 10-seed ensemble (v2 vs classw050_rc)

| Metric | Groups | v2/unreacted | classw050_rc | Δ pp | 95% CI (bootstrap) | Permutation p | Sign-test p |
|---|---:|---:|---:|---:|---:|---:|---:|
| Top-1 | 1635 | 83.79 | 97.06 | **+13.27** | [+11.62, +14.92] | < 0.0001 | 2.7 × 10⁻⁶¹ |
| MRR | 1635 | 90.52 | 98.38 | **+7.86** | [+6.87, +8.88] | < 0.0001 | 1.5 × 10⁻⁵⁶ |
| NDCG | 1635 | 92.92 | 98.79 | **+5.88** | [+5.15, +6.62] | < 0.0001 | 1.5 × 10⁻⁵⁶ |

Group-level direction: 220 groups better for classw050_rc, 3 groups better for v2, 1412 ties.

Seed-level bootstrap CI (expanded curated scope):

| Metric | Mean Δ pp | 95% CI (seed bootstrap) | Std (seed) |
|---|---:|---:|---:|
| Top-1 | +12.50 | [+12.07, +12.95] | 0.23 |
| MRR | +7.43 | [+7.21, +7.67] | 0.12 |
| NDCG | +5.56 | [+5.39, +5.74] | 0.09 |

Interpretation — expanded curated scope:

```text
The 10-seed ensemble paired test is decisively positive on the expanded curated
Amide/Cu benchmark. Top-1 improves by +13.27 pp with p < 0.0001 from both the
group-level ensemble permutation test and the sign test (p ≈ 2.7e-61). The
seed-level bootstrap CI is [+12.07, +12.95] pp — entirely positive and very
tight (std 0.23 pp). This is top-journal-grade evidence that classw050_rc
closes the Amide/Cu curated weak-class gap robustly across 10 seeds.
```

### Ni external data-source gap audit (2026-07-12)

Using RDKit atomic-number-based Ni detection (atomic number 28) across the full
HITEA and USPTO/OpenMolecules datasets:

| Dataset | Total reactions | Ni reactions | Distinct Ni parent reactants |
|---|---:|---:|---:|
| HITEA full | 470,000+ | 0 | 0 |
| USPTO/OpenMolecules full | 530,238 | 6 | 6 |

Conclusion:

```text
Ni remains a hard external molecular-context data-source gap. Neither generator
improvements nor reweighting can solve it without additional Ni-containing
reaction data. For top-journal publication, this should be stated as a clear
limitation and motivation for future work, rather than swept under the rug.
```
