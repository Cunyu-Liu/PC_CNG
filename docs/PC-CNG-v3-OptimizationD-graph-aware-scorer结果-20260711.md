# PC-CNG v3 Optimization D graph-aware scorer 结果

日期：2026-07-11

## 目标

落实 SOTA 差距分析文档中的 Optimization D：

```text
Move beyond Morgan-fingerprint MLP toward graph-aware pairwise scorer.
```

本轮实现的是 RDKit-only lightweight graph-pair encoder，用于验证图结构统计是否能提升 same-context candidate reranking 和 held-out generalization。

## 模型定义

对一个 reaction candidate \(x=(R,P)\)，其中 \(R\) 为 reactant molecule set，\(P\) 为 candidate product set。

每个 molecule set \(S\) 被编码为图统计向量：

\[
\phi(S)=\log(1+|[
c_{\mathrm{atom}}(S),
c_{\mathrm{degree}}(S),
c_{\mathrm{charge/aromatic/ring/H}}(S),
c_{\mathrm{bond}}(S),
c_{\mathrm{ring-size}}(S),
d_{\mathrm{RDKit}}(S)
]|)\odot \operatorname{sign}(\cdot)
\]

其中：

- \(c_{\mathrm{atom}}\)：B, C, N, O, F, P, S, Cl, Br, I 和 other 的 atom counts。
- \(c_{\mathrm{degree}}\)：degree 0 到 5 的 node degree histogram。
- \(c_{\mathrm{charge/aromatic/ring/H}}\)：正/负 formal charge、aromatic atom、ring atom、implicit/explicit H count。
- \(c_{\mathrm{bond}}\)：single/double/triple/aromatic bond count。
- \(c_{\mathrm{ring-size}}\)：ring size 3 到 8+ histogram。
- \(d_{\mathrm{RDKit}}\)：heavy atoms、MolWt、TPSA、rotatable bonds、ring count、aromatic atom count、hetero atom count、absolute charge。

Reaction-pair pooling：

\[
h(x) = [\phi(R), \phi(P), \phi(P)-\phi(R), |\phi(P)-\phi(R)|, a(x)]
\]

其中辅助特征：

\[
a(x)=[1, \log(1+n_R), \log(1+n_P), \log(1+n_{\mathrm{atom}}(P))]
\]

Scorer：

\[
s_\theta(x)=\sigma(\operatorname{MLP}_\theta(h(x)))
\]

Pairwise objective：

\[
\mathcal{L}=
\lambda_{\mathrm{bce}}\operatorname{BCE}(y, s_\theta(x))+
\lambda_{\mathrm{pair}}\log(1+\exp(m-(z^+_\theta-z^-_\theta)))
\]

其中 \(z^+_\theta,z^-_\theta\) 是 sigmoid 之前的 positive/negative logits，\(m\) 是 margin。

复杂度：

```text
Graph-stat featurization: O(|V| + |E|) per molecule set.
Reaction pooling: O(d), d = fixed graph-stat dimension.
MLP scoring: O(dH + H^2) per candidate, H = hidden dim.
Compared with Morgan fingerprinting, graph-stat features are lower-dimensional and interpretable, but still require RDKit-valid molecules.
```

## 代码落地

新增/修改：

```text
pc_cng/train_feasibility_mlp.py
  - GraphStatsReactionFeaturizer
  - make_reaction_featurizer(feature_mode=...)

pc_cng/train_pairwise_reward_mlp.py
  - --feature-mode {morgan, graph_stats}
  - checkpoint saves feature_mode

pc_cng/evaluate_candidate_reranking.py
  - loads feature_mode from checkpoint

pc_cng/evaluate_external_product_prediction_benchmark.py
  - loads feature_mode from checkpoint

tests/test_graph_stats_featurizer.py
  - RDKit-enabled graph_stats smoke tests

scripts_run_type1_graph_stats_pairwise.sh
  - 5-seed train/eval/external benchmark runner
```

验证：

```text
local compileall: pass
local unittest: pass, graph_stats tests skipped because local RDKit unavailable
remote compileall: pass
remote graph_stats unittest: 2 tests OK
```

## 服务器运行

脚本：

```text
/home/cunyuliu/pc_cng_research/chem_negative_sampling/scripts_run_type1_graph_stats_pairwise.sh
```

结果目录：

```text
/home/cunyuliu/pc_cng_research/results/type1_graph_stats_pairwise_full
```

配置：

```text
feature_mode: graph_stats
seeds: 20260710, 20260711, 20260712, 20260713, 20260714
epochs: 80
batch_size: 4096
hidden_dim: 1024
dropout: 0.20
loss: BCE + pairwise softplus margin
```

## 5-seed 单模型结果

| Seed | Binary test ROC-AUC | Same-context Top-1 | Held-out test Top-1 | Synthetic Top-1 |
|---:|---:|---:|---:|---:|
| 20260710 | 86.56 | 98.51 | 97.01 | 98.33 |
| 20260711 | 85.85 | 98.33 | 95.52 | 98.00 |
| 20260712 | 86.72 | 98.14 | 95.52 | 97.67 |
| 20260713 | 86.58 | 97.59 | 95.52 | 96.67 |
| 20260714 | 86.50 | 97.77 | 94.03 | 97.00 |

## Ensemble same-context 结果

```text
path:
/home/cunyuliu/pc_cng_research/results/type1_graph_stats_pairwise_full/graph_stats_ensemble5/rerank_same_split/ranking_metrics.json
```

| Model | Overall Top-1 | Overall Top-3 | Overall MRR | Overall NDCG | Held-out test Top-1 | Synthetic Top-1 |
|---|---:|---:|---:|---:|---:|---:|
| Morgan pairwise seed20260710 | 97.40 | 99.72 | 98.58 | 98.94 | 83.58 | 96.50 |
| Graph-stats seed20260710 | 98.51 | 99.63 | 99.12 | 99.34 | 97.01 | 98.33 |
| Graph-stats 5-seed ensemble | 98.51 | 99.63 | 99.12 | 99.34 | 97.01 | 98.33 |

Interpretation:

```text
Graph-stats improves same-context held-out test Top-1 by +13.43 pp over the comparable Morgan seed20260710 baseline.
It also improves synthetic Top-1 by +1.83 pp.
The 5-seed ensemble equals the best seed here, suggesting the graph-stat models are stable and highly correlated.
```

## External validity-aware benchmark

```text
path:
/home/cunyuliu/pc_cng_research/results/type1_graph_stats_pairwise_full/graph_stats_ensemble5/external_validity_aware/benchmark_summary.json
```

| Model | Overall Top-1 | Overall Top-3 | Overall MRR | Overall NDCG | Test Top-1 | Test MRR |
|---|---:|---:|---:|---:|---:|---:|
| Chemformer likelihood | 0.04 | 1.13 | 9.69 | 28.45 | 0.07 | 9.75 |
| Morgan validity-aware 5-seed | 97.45 | 99.93 | 98.69 | 99.03 | 98.50 | 99.14 |
| Graph-stats validity-aware 5-seed | 98.52 | 99.74 | 99.07 | 99.30 | 97.98 | 98.64 |

Interpretation:

```text
Graph-stats gives higher overall Top-1/MRR/NDCG than Morgan validity-aware 5-seed,
but slightly lower held-out external test Top-1 and MRR.
Therefore graph-stats should not replace Morgan as the external validity-aware main model yet.
It is best positioned as an architecture-upgrade supplement that strongly improves same-context held-out reranking.
```

## 论文可用结论

可写：

```text
Replacing Morgan fingerprints with a lightweight graph-stat reaction encoder improves same-context
held-out candidate reranking from 83.58% to 97.01% Top-1 in the seed-matched comparison,
supporting the claim that graph-aware reaction-difference pooling improves generalization beyond
fingerprint-only MLP scoring.
```

需要谨慎：

```text
On the full external validity-aware Chemformer beam benchmark, graph-stats is not yet strictly better
than the Morgan 5-seed ensemble on held-out test Top-1. The main external product-selection claim
should still use Morgan validity-aware 5-seed as the strongest branch, with graph-stats as an
architecture supplement and motivation for a future learned GNN scorer.
```
