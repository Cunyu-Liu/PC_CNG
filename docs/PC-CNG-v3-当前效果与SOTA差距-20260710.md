# PC-CNG v3 当前效果与 SOTA 差距

## 当前最佳模型

当前 validation-selected 模型为 stacked ensemble：

```text
method: logreg_C3.0
used_runs: 27
```

远端结果：

```text
/home/cunyuliu/pc_cng_research/results/current_best_ensemble_detailed_metrics.json
/home/cunyuliu/pc_cng_research/results/stacked_ensemble_summary.json
```

## 总体测试集指标

测试集规模：

```text
n = 3,710
positive rate = 41.4%
```

总体效果：

| Metric | Current PC-CNG v3 ensemble |
|---|---:|
| Accuracy | 0.7973 |
| F1 | 0.7508 |
| Precision | 0.7645 |
| Positive recall | 0.7376 |
| Negative specificity | 0.8395 |
| ROC-AUC | 0.8822 |
| AUPRC | 0.8242 |

Confusion matrix：

```text
TN = 1825
FP = 349
FN = 403
TP = 1133
```

## 分数据集效果

### HiTEA

```text
n = 3,468
```

| Metric | HiTEA test |
|---|---:|
| Accuracy | 0.7941 |
| F1 | 0.7553 |
| Precision | 0.7690 |
| Positive recall | 0.7421 |
| Negative specificity | 0.8331 |
| ROC-AUC | 0.8810 |
| AUPRC | 0.8302 |

### RegioSQM20

```text
n = 242
```

| Metric | RegioSQM20 test |
|---|---:|
| Accuracy | 0.8430 |
| F1 | 0.6200 |
| Precision | 0.6327 |
| Positive recall | 0.6078 |
| Negative specificity | 0.9058 |
| ROC-AUC | 0.8633 |
| AUPRC | 0.6361 |

RegioSQM20 子集 specificity 高但 positive recall/AUPRC 低，说明模型更擅长排除负反应，不够擅长召回真实 regio-positive。

## 和内部强基线差距

当前最强 real-only baseline：

```text
full_feasibility_mlp_real_only_h2048_n4096_e80
```

| Metric | Real-only | Current ensemble | Delta |
|---|---:|---:|---:|
| Accuracy | 0.7620 | 0.7973 | +0.0353 |
| F1 | 0.7236 | 0.7508 | +0.0272 |
| ROC-AUC | 0.8582 | 0.8822 | +0.0240 |
| AUPRC | 0.8009 | 0.8242 | +0.0233 |

结论：

PC-CNG 的 synthetic/boundary signal 对 ensemble 是有效的，已经显著超过本项目 real-only 基线。

## Expanded actions 消融结论

| Run | Test ROC-AUC | Test AUPRC | Test F1 |
|---|---:|---:|---:|
| expanded direct BCE | 0.8525 | 0.7967 | 0.7151 |
| expanded BCE, synthetic=0.2 | 0.8560 | 0.7989 | 0.7122 |
| expanded BCE, synthetic=0.5 | 0.8577 | 0.7983 | 0.7240 |
| expanded pairwise reward | 0.8590 | 0.8015 | 0.7219 |
| expanded family-aware pairwise | 0.8583 | 0.7998 | 0.7237 |

结论：

1. 降低 synthetic BCE 权重可以缓解 test 下滑，但不能成为最优单模型。
2. family-aware pairwise 没有超过普通 expanded pairwise，原因是 pairable family 分布极不均衡：

```text
regio: 27
tautomer: 390
low_yield_seed: 0/1 usable pair
```

3. 当前瓶颈仍是高质量 type-1 regio/heteroatom boundary negatives 不足。

## 与外部 SOTA/强基线的差距

需要强调：下面不是严格 apples-to-apples。公开文献多报告 product top-1 / regioselectivity success rate / positive accuracy，而我们当前报告的是 binary feasibility ROC-AUC/AUPRC/F1。严格 SOTA 结论需要把 IBM RXN / Molecular Transformer / Science Advances RL reward model 跑在我们的同一 split 上。

### RegioSQM20 / regioselectivity domain

RegioSQM20 论文报告：

```text
RegioSQM20 with tautomers success rate: 92.7%
WLN / RegioSQM20 without tautomers: about 89.1% / 90.7%
IBM RXN: 76.3%-85.0%
```

当前 PC-CNG 在 RegioSQM20 子集：

```text
binary accuracy: 84.3%
positive recall: 60.8%
ROC-AUC: 86.3%
AUPRC: 63.6%
```

粗略差距：

| Reference | External metric | Current closest metric | Gap |
|---|---:|---:|---:|
| RegioSQM20 with tautomers | success 92.7% | binary accuracy 84.3% | -8.4 pp |
| IBM RXN lower-upper range | 76.3%-85.0% | binary accuracy 84.3% | within range |
| RegioSQM20 success | 92.7% | positive recall 60.8% | -31.9 pp |

解释：

RegioSQM20 专门做 EAS 位点预测，PC-CNG 当前是通用 binary feasibility classifier。真正的差距集中在 regio-positive recall，而不是 negative filtering。

### Science Advances negative-data RL

Science Advances 2025 报告：

```text
RegioSQM20 K_low:
FT positive accuracy: 54.91%
RL positive accuracy: 58.55%

RegioSQM20 K_high:
FT positive accuracy: 68.48%
```

当前 PC-CNG 在 RegioSQM20 子集：

```text
positive recall: 60.78%
binary accuracy: 84.30%
```

粗略差距：

| Reference | External metric | Current closest metric | Gap |
|---|---:|---:|---:|
| RL K_low | positive accuracy 58.55% | positive recall 60.78% | +2.23 pp |
| FT K_high | positive accuracy 68.48% | positive recall 60.78% | -7.70 pp |

解释：

这个比较只能说明“量级”，不能说明我们超过或落后该论文，因为该论文是 transformer product prediction / positive accuracy，我们是 binary feasibility / recall。

### General USPTO product prediction

Molecular Transformer / Chemformer 等 forward product prediction 模型常见 USPTO top-1 accuracy 在 90% 以上，相关工作还报告 product prediction top-1 可到约 0.90-0.92。

当前 PC-CNG：

```text
binary accuracy: 79.7%
```

粗略差距：

```text
about -10 to -13 percentage points in accuracy-like metric
```

解释：

这个差距不能作为结论，因为 USPTO product top-1 和 binary feasibility 是不同任务；但它说明如果论文目标要冲 SOTA，需要补 product/reranking benchmark，而不能只做二分类。

## 当前结论

1. 在我们自己的 HiTEA + RegioSQM20 binary feasibility benchmark 上，PC-CNG ensemble 已明显超过 real-only 强基线。
2. 与外部 regioselectivity SOTA 相比，最大短板是 RegioSQM20 positive recall / AUPRC。
3. 与 Science Advances 方法学相比，我们已经验证了 pairwise/reward-style 信号比 naive BCE 更合理，但还没有实现真正的 transformer/RL policy-level tuning。
4. 与 USPTO product prediction SOTA 相比，当前任务不对齐；若要顶刊，需要增加 product/reranking top-k 评估。

## 下一步优先级

1. 针对 RegioSQM20 设计 regio-positive recall 优化：更多 type-1 regio hard negatives + calibrated reranking。
2. 将 PC-CNG 从 binary classifier 推进到 candidate product reranker，报告 top-1/MRR/NDCG。
3. 引入 pretrained reaction LM / Molecular Transformer / Chemformer 作为 frozen scorer 或 policy，再用 PC-CNG negatives 做 DPO/RL-style tuning。
4. 在统一 split 上复现 IBM RXN/Molecular Transformer/Science Advances RL-style baseline，形成严格 SOTA comparison。
