# PC-CNG v3 下游任务评估协议与 Optimization A 落地

日期：2026-07-11

## 结论

本轮将 PC-CNG v3 的下游评估口径统一为两层：

1. **主任务：same-context candidate product reranking**
   在同一个反应上下文内，对 observed product、PC-CNG diverse-anchor negatives、Chemformer/Molecular Transformer beam products 组成的候选集排序。

2. **桥接任务：strict external product prediction comparison**
   不再把 PC-CNG 的 candidate reranking 结果直接和端到端生成模型硬比，而是先让所有模型面对同一批 reactant contexts 和同一候选集合，再比较 Chemformer likelihood、PC-CNG pairwise reward、hybrid ensemble 的排序能力。

这对应 SOTA 差距文档中的 **Optimization A**，是向顶刊口径补齐 external baseline comparison 的优先落地项。

## 对齐的同领域模型与任务口径

| Model / method | 对齐任务 | 输入 | 输出 | 本项目可比口径 | 备注 |
|---|---|---|---|---|---|
| Chemformer | forward product prediction / conditional likelihood | reactants 或 reactants + agents | beam products 或 candidate log-likelihood | beam candidates + likelihood reranking | 用作 frozen reference policy |
| Molecular Transformer / IBM RXN | forward product prediction | reactants | top-k generated products | beam candidates + exact-match / rank score | 可接入同一 CSV schema |
| ReactionT5 / Uni-Mol3 | reaction foundation model product prediction | reactants / prompted reaction context | product top-k | 作为外部 generation baseline | 需另接 adapter 或导入 beam CSV |
| RegioSQM20 / RegioML | regio/site-selectivity | substrate + reaction type | predicted site/product class | 仅用于 regio subset，不与 overall candidate ranking 混表 | RegioSQM20 报告 90.7/92.7% success rate，IBM RXN 约 76.3-85.0% |
| PC-CNG v3 pairwise_default | boundary negative candidate reranking | reactants + candidate product | feasibility / reward score | same-context Top-1/Top-3/MRR/NDCG | 主贡献口径 |
| PC-CNG + Chemformer hybrid | external bridge reranking | same candidate CSV + two model scores | linear ensemble score | validation-selected hybrid Top-k/MRR/NDCG | 本轮新增 |

参考文献/基准来源：

- RegioSQM20: https://pmc.ncbi.nlm.nih.gov/articles/PMC7881568/
- RegioML: https://pubs.rsc.org/en/content/articlehtml/2022/dd/d1dd00032b
- 2025 regio/site-selectivity review: https://pmc.ncbi.nlm.nih.gov/articles/PMC11891785/
- ReactionT5 2025: https://pmc.ncbi.nlm.nih.gov/articles/PMC12366004/
- Uni-Mol3 2025: https://arxiv.org/html/2508.00920

## 统一输入输出格式

标准候选 CSV 字段：

| Field | Meaning |
|---|---|
| `group_id` | 一个 reactant context 的唯一排序组 |
| `source_id` | 原始正例反应 ID |
| `reactants` | 反应物 SMILES |
| `agents` | 试剂/条件，可为空 |
| `candidate_product` | 待排序候选产物 |
| `candidate_reaction` | `reactants>agents>candidate_product` 或 `reactants>>candidate_product` |
| `label` | observed product 为 1，其余 beam/PC-CNG candidates 为 0，若 beam 命中 observed product 则去重保留 label=1 |
| `split` | train / val / test |
| `dataset` | HiTEA / RegioSQM20 / 其他数据源 |
| `candidate_source` | `observed_positive`、`pc_cng`、`chemformer_beam` 等 |
| `candidate_family` | diverse-anchor action family 或 external beam family |
| `reaction_class` | reaction-class supplement 用于弱类分析 |

候选集合构造规则：

1. 每个 observed positive 生成一个 `group_id`。
2. PC-CNG diverse-anchor negatives 只加入其 parent positive 所在 group。
3. Chemformer / Molecular Transformer beam products 按 context row index、`source_id` 或 reactants context 对齐到同一 group。
4. 同一 group 内按 canonical product 去重，若任何重复项是 observed product，则该候选 label 记为 1。
5. 所有模型必须在同一候选行交集上报告指标：先保留同时拥有主模型分数的 candidate rows，再仅评估仍包含正负候选的 groups。不能因为某个 group 内存在无法被某模型打分的额外候选而丢弃整个 group。

Validity-aware 扩展规则：

```text
For full Chemformer beam candidate coverage, PC-CNG may additionally report a validity-aware protocol:
label=0 candidates that cannot be featurized/scored by PC-CNG receive a pre-declared low score, e.g. 0.0.
label=1 candidates must not receive a fake score; groups with unscored positives remain filtered unless
another positive candidate row is available.
```

该规则必须单独命名为 `validity-aware PC-CNG reranking`，不能与纯 learned scorer 交集结果混为一谈。

## 评价指标与公式

设共有 \(G\) 个有效 group，第 \(g\) 个 group 的排序结果为 \(\pi_g\)，第一个正例排名为 \(r_g\)。

Top-k:

\[
\operatorname{TopK}=\frac{1}{G}\sum_{g=1}^{G}\mathbf{1}[r_g \le k]
\]

MRR:

\[
\operatorname{MRR}=\frac{1}{G}\sum_{g=1}^{G}\frac{1}{r_g}
\]

DCG / NDCG:

\[
\operatorname{DCG}_g=\sum_{i=1}^{n_g}\frac{y_{\pi_g(i)}}{\log_2(i+1)},\quad
\operatorname{NDCG}=\frac{1}{G}\sum_g \frac{\operatorname{DCG}_g}{\operatorname{IDCG}_g}
\]

Hybrid score：

\[
s_{\lambda}(x)=\lambda z_g(s_{\mathrm{PC-CNG}}(x))+(1-\lambda)z_g(s_{\mathrm{LM}}(x))
\]

其中 \(z_g\) 默认是 group-level z-score：

\[
z_g(s_i)=\frac{s_i-\mu_g}{\sigma_g+\epsilon}
\]

\(\lambda\) 在验证集上按照 Top-1 选择，MRR、NDCG、Top-3 作为并列时的 tie-breaker；若没有 val split，则记录 fallback 到 overall selection。

复杂度：

- 候选构造：\(O(N)\)，其中 \(N\) 为候选行数。
- 去重：\(O(N)\)，canonicalization 视 RDKit 可用性为 \(O(L)\) 到分子解析开销。
- 排序评估：\(\sum_g O(n_g\log n_g)\)。
- PC-CNG MLP scoring：约 \(O(N \cdot D \cdot H)\)，其中 \(D\) 为 Morgan reaction feature 维度，\(H\) 为 hidden dim。
- Hybrid grid：\(O(N \cdot |\Lambda|)\)。

## 本轮代码落地

新增：

- `pc_cng/ranking_metrics.py`
  轻量级 Top-k/MRR/NDCG 计算，避免 LM-only evaluation 导入完整训练栈。

- `pc_cng/build_external_product_prediction_candidate_set.py`
  构建 observed positives + PC-CNG negatives + external beam products 的严格候选集。

- `pc_cng/evaluate_external_product_prediction_benchmark.py`
  在同一 complete candidate groups 上评估 external LM、PC-CNG、hybrid ensemble，并输出 `benchmark_summary.json`、`paper_table.csv`、`paper_table.md`。

- `scripts_run_external_product_prediction_benchmark.sh`
  服务器端一键流程：候选构建 -> Chemformer beam generation -> Chemformer likelihood scoring -> PC-CNG scoring -> hybrid benchmark。

- `tests/test_external_product_prediction_benchmark.py`
  覆盖候选集合并、分数读取、complete-group 过滤、hybrid 选择和 paper table 输出。

扩展：

- `pc_cng/reaction_lm_scorer.py` 的标准 schema 增加 `reaction_class`。
- `pc_cng/build_reaction_lm_candidate_set.py` 透传 `reaction_class`。
- `pc_cng/evaluate_reaction_lm_scores.py` 改用轻量 ranking metrics。

## 服务器运行命令

默认全量运行：

```bash
cd /home/cunyuliu/pc_cng_research/chem_negative_sampling
bash scripts_run_external_product_prediction_benchmark.sh
```

推荐正式记录配置：

```bash
ROOT=/home/cunyuliu/pc_cng_research \
GPU_EVAL=0 \
RESULTS_DIR=/home/cunyuliu/pc_cng_research/results/external_product_prediction_benchmark_20260711 \
PC_CNG_MODEL_DIRS="/home/cunyuliu/pc_cng_research/results/type1_diverse_anchor_ablation_full/pairwise_default_seed20260710 /home/cunyuliu/pc_cng_research/results/type1_diverse_anchor_ablation_full/pairwise_default_seed20260711 /home/cunyuliu/pc_cng_research/results/type1_diverse_anchor_ablation_full/pairwise_default_seed20260712" \
SCORER=chemformer_log_likelihood \
N_BEAMS=10 \
bash scripts_run_external_product_prediction_benchmark.sh
```

输出路径：

```text
$RESULTS_DIR/base_observed_pc_cng_candidates_summary.json
$RESULTS_DIR/full_observed_pc_cng_chemformer_beam_candidates_summary.json
$RESULTS_DIR/lm_scores_chemformer_log_likelihood_summary.json
$RESULTS_DIR/benchmark/benchmark_summary.json
$RESULTS_DIR/benchmark/paper_table.md
```

## 可复现实验记录要求

每次正式运行必须记录：

1. 代码版本或文件 hash。
2. `REGIO_ALIGNMENT`、`HITEA_ALIGNMENT`、`SYNTHETIC_CSV` 的绝对路径与行数。
3. Chemformer checkpoint、vocabulary、beam size、batch size、GPU id。
4. PC-CNG model dirs 与 seed 列表。
5. 候选行交集过滤前后的 group 数和候选行数，以及各模型 scorer coverage。
6. external LM、PC-CNG、selected hybrid 的 overall / val / test Top-1、Top-3、MRR、NDCG。
7. by-dataset、by-candidate-source、by-reaction-class 补充表。
8. 若启用 validity-aware 规则，必须额外记录 filled negative rows、missing positive rows 和 score value。

## 顶刊口径判断

若 selected hybrid 在 held-out test 上相对 frozen Chemformer likelihood 显著提升，并且 PC-CNG 对 weak reaction classes 的补充表不出现大面积退化，则可以将 Optimization A 写成：

```text
PC-CNG boundary negatives improve forward-model candidate selection under a strict shared-candidate external product-prediction benchmark.
```

这比直接宣称 end-to-end product generation SoTA 更严谨，也更符合顶刊审稿对 fair comparison 的要求。
