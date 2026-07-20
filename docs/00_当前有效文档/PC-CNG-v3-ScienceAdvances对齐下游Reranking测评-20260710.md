# PC-CNG v3 与 Science Advances 数据体系对齐的下游 Reranking 测评

## 1. Science Advances 论文使用的数据与证明方式

论文：

```text
Negative chemical data boosts language models in reaction outcome prediction
Science Advances 2025
DOI: 10.1126/sciadv.adt5578
```

论文核心不是把负样本直接混入 BCE 分类，而是：

```text
USPTO pretrained forward Transformer
-> RegioSQM20 / HiTEA positive-negative feedback
-> reward model
-> RL / REINFORCE policy optimization
-> forward product prediction positive accuracy
```

### RegioSQM20 受控数据

论文用 RegioSQM20 作为受控低样本场景，其中负样本主要是 type-1 negative：

```text
same / similar reaction context
-> unexpected but chemically meaningful alternative product
```

论文主表数据量：

| Setting | Train positives | Train negatives | Valid positives | Test positives |
|---|---:|---:|---:|---:|
| K_high | 220 | 748 | 165 | 164 |
| K_low | 22 | 748 | 165 | 164 |

论文报告的 positive accuracy：

| Setting | FT RegioSQM20 | RL RegioSQM20 |
|---|---:|---:|
| K_high | 68.48 ± 1.38 | 63.15 ± 1.64 |
| K_low | 54.91 ± 1.04 | 58.55 ± 1.75 |

解释：

在 low-positive regime 下，负样本通过 reward/RL 提升了 forward product prediction 的 positive accuracy。

### HiTEA 真实 HTE 数据

论文还用真实 high-throughput experimentation 数据 HiTEA。其标签来自实验 yield：

```text
UV area yield > 1 -> positive
otherwise -> negative
```

HiTEA 在论文中用于证明真实 failed / low-yield experimental reactions 也能提供有效负反馈，但论文指出 HiTEA 存在 domain shift，需要额外 tuning。

## 2. 我们本轮做的对齐测评

我们没有复现论文的 transformer policy/RL，而是把当前 PC-CNG 模型接到更贴近下游的 candidate product reranking：

```text
same reactants / same context
-> observed positive product + real/synthetic negative candidate products
-> feasibility model scoring
-> top-1 / top-3 / MRR / NDCG
```

新增代码：

```text
pc_cng/evaluate_candidate_reranking.py
scripts_run_candidate_reranking_eval.sh
```

远端结果：

```text
/home/cunyuliu/pc_cng_research/results/candidate_reranking_eval/
/home/cunyuliu/pc_cng_research/results/candidate_reranking_eval_all_group/
```

### Candidate set 构造

两类 candidate set：

1. Real candidate set：
   - 来自 RegioSQM20 / HiTEA normalized CSV；
   - 同一 `reactants` 组内，positive 与 real_negative 共同排序。

2. Synthetic candidate set：
   - observed positive reaction；
   - 加入 PC-CNG expanded hard negatives；
   - 只保留 `review_status=keep_synthetic_negative`。

两个 scope：

1. `same_split`
   - 同一 split 内构造候选组；
   - 更严格，适合 held-out reporting；
   - test groups 较少。

2. `all_group`
   - 同一 reactants 下所有候选一起排序；
   - 更接近 RegioSQM20 完整候选集；
   - 但不作为严格 held-out。

## 3. Same-split Reranking 结果

候选组：

```text
groups: 752
candidate rows: 2,534
random top1 expected: 0.3494
```

总体结果：

| Model | Top-1 | MRR | NDCG |
|---|---:|---:|---:|
| real-only | 0.9628 | 0.9807 | 0.9857 |
| v2 direct BCE | 0.9747 | 0.9870 | 0.9904 |
| rule-hard direct BCE | 0.9721 | 0.9857 | 0.9894 |
| expanded pairwise | **0.9920** | **0.9960** | **0.9971** |
| expanded BCE synth=0.5 | 0.9907 | 0.9953 | 0.9966 |
| mean core PC-CNG | 0.9827 | 0.9910 | 0.9934 |

结论：

PC-CNG negatives 对 candidate product reranking 有明确增益，尤其 expanded pairwise 从 real-only 的 0.9628 top1 提升到 0.9920。

## 4. Held-out Test Split 结果

test split：

```text
groups: 36
candidate rows: 96
random top1 expected: 0.4236
```

| Model | Test Top-1 | Test MRR | Test NDCG |
|---|---:|---:|---:|
| real-only | 0.8889 | 0.9398 | 0.9554 |
| v2 direct BCE | 0.8889 | 0.9444 | 0.9590 |
| rule-hard direct BCE | 0.9444 | 0.9722 | 0.9795 |
| expanded pairwise | **1.0000** | **1.0000** | **1.0000** |
| expanded BCE synth=0.5 | 0.9722 | 0.9861 | 0.9897 |
| mean core PC-CNG | 0.8889 | 0.9444 | 0.9590 |

按 candidate source 拆分：

| Source | real-only Top-1 | expanded pairwise Top-1 | Delta |
|---|---:|---:|---:|
| real candidates | 1.0000 | 1.0000 | +0.0000 |
| synthetic candidates | 0.8261 | 1.0000 | +0.1739 |

按 dataset 拆分：

| Dataset | real-only Top-1 | expanded pairwise Top-1 | Delta |
|---|---:|---:|---:|
| HiTEA | 1.0000 | 1.0000 | +0.0000 |
| RegioSQM20 | 0.8788 | 1.0000 | +0.1212 |

解释：

held-out test 上主要增益来自 synthetic candidate reranking 与 RegioSQM20 子集。这与 Science Advances 对 type-1 negative 的强调一致。

限制：

test 组数只有 36，因此这是正向证据，但还不能作为最终 SOTA 结论。

## 5. All-group 完整候选集结果

all-group 不按 split 切断同一 reactants 下的候选：

```text
groups: 846
candidate rows: 3,261
random top1 expected: 0.3133
```

总体：

| Model | Top-1 | MRR | NDCG |
|---|---:|---:|---:|
| real-only | 0.9551 | 0.9764 | 0.9825 |
| expanded pairwise | 0.9764 | 0.9874 | 0.9907 |
| expanded BCE synth=0.5 | **0.9787** | **0.9886** | **0.9915** |

HiTEA:

| Model | Top-1 | MRR | NDCG |
|---|---:|---:|---:|
| real-only | 0.8065 | 0.8978 | 0.9243 |
| expanded pairwise | **0.9194** | **0.9597** | **0.9702** |
| expanded BCE synth=0.5 | **0.9194** | **0.9597** | **0.9702** |

RegioSQM20:

| Model | Top-1 | MRR | NDCG |
|---|---:|---:|---:|
| real-only | 0.9668 | 0.9826 | 0.9871 |
| expanded pairwise | 0.9809 | 0.9896 | 0.9923 |
| expanded BCE synth=0.5 | **0.9834** | **0.9909** | **0.9932** |

结论：

在 Science Advances 使用的两个同源数据体系上，PC-CNG 生成/筛选的负样本对 candidate reranking 有稳定正向增益。

## 6. 当前是否证明“生成负样本有效”

可以说：

```text
在 RegioSQM20 / HiTEA 同源数据体系上，
PC-CNG-generated hard negatives 提升了下游 candidate product reranking，
尤其提升了 synthetic boundary candidate 与 RegioSQM20 type-1-like candidate 的排序。
```

暂时不能说：

```text
已经复现或超过 Science Advances 的 RL forward Transformer。
```

原因：

1. Science Advances 是 product-generation policy tuning；
2. 我们当前是 feasibility scorer / reranker；
3. 我们使用的是同源数据集，但不是论文作者 exact random split；
4. 我们还没有接入 USPTO-pretrained Molecular Transformer / Chemformer 作为 policy。

## 7. 下一步

为了把证据链升级到论文级别，需要继续：

1. 构造严格的 RegioSQM20 K_low / K_high split：
   - K_low: 22 positives + 748 negatives；
   - K_high: 220 positives + 748 negatives；
   - valid/test positives 按论文规模对齐。

2. 用 PC-CNG negatives 训练 reward/reranker：
   - 直接对比 no-negative / real-negative / generated-negative；
   - 指标使用 positive top-1、MRR、NDCG。

3. 接 pretrained reaction LM：
   - Molecular Transformer / Chemformer frozen scorer；
   - 再做 DPO / RL-style tuning。

4. 把 candidate reranking 结果与 binary feasibility 结果并列表述：
   - binary feasibility 证明分类判别；
   - candidate reranking 证明对下游产品排序有用；
   - policy tuning 才能证明 forward synthesis prediction 层面的 SOTA 潜力。
