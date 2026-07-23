# P4-G6: Real HTE External Validation

## Phase Summary

| Field | Value |
|-------|-------|
| Phase | P4-G6 |
| Status | **WEAK_GO** |
| next_phase_allowed | true |
| Primary method | risk_aware_pc_cng |
| Baselines | positive_only, tanimoto_baseline |
| Oracle (not a baseline) | observed_negative_upper_bound |
| Date | 2026-07-23 |
| Spec | pccng 的分阶段提示词.md#L1199-1395 |

## Entry Conditions

| Condition | Status |
|-----------|--------|
| P4-G1 == GO | MET |
| P4-G3 >= Weak GO | MET (Weak GO) |
| P4-G5 completed | MET (PARTIAL_GO, commit 914500c) |

## Data Source Audit

### Primary Dataset: HiTEA

- **Publication**: King-Smith et al., "Probing the Chemical Reactome with High Throughput Experimentation Data", Nat. Chem. 2023. [Link](https://www.nature.com/articles/s41557-023-01393-w)
- **Zenodo DOI**: 10.5281/zenodo.552294062
- **License**: CC-BY 4.0 (Zenodo)
- **Raw file**: `external/HiTEA/data/8_SEPT_APPROVED_full_dataset.csv`

### Authenticity Verification

HiTEA is a genuine HTE dataset, NOT random negatives or a degenerate benchmark:

- 39,546 reactions across 356 SCREEN_IDs (experimental plates)
- 41 reaction families (KeyWord_STD: SUZUKI, BUCHWALD, etc.)
- Each record has SCREEN_ID, NOTEBOOK_ID, REACTION_ID
- Measured yields are Product_Yield_PCT_Area_UV (UV area %)
- Substrate and condition grids derived from Reactant/Catalyst/Reagent columns

### Zero-Type Classification

| Type | Count | Description |
|------|-------|-------------|
| positive_yield | 12,499 | yield >= 5% |
| measured_zero | 11,527 | yield = 0, mass > 0 (product detected) |
| below_detection | 10,831 | yield = 0, mass = 0 (no product detected) |
| low_yield | 4,534 | 0 < yield < 5% |
| no_product_recorded | 155 | yield = 0, mass missing |

### External Sources Checked

| Source | Status |
|--------|--------|
| Doyle/Merck C-N coupling HTE | NOT FOUND on server |
| Suzuki HTE | Present within HiTEA (4,915 reactions) |
| Buchwald-Hartwig HTE | Present within HiTEA (2,992 reactions) |
| metallaphotoredox HTE | NOT FOUND on server |
| ORD | 750 AstraZeneca ELN reactions (0% yield coverage, not used) |
| NiCOlit | 1,688 Ni-coupling reactions (not HTE grid, not used) |

## Screen-Aware Split

Original normalized CSV had 112/356 screens crossing train/test/val splits. Built a new cluster-aware split where **no SCREEN_ID appears in both train and test**:

| Split | Screens | Reactions | Families | Yield Mean |
|-------|---------|-----------|----------|------------|
| train | 267 | 29,357 | 41 | 13.99 |
| val | 41 | 4,291 | 25 | 9.83 |
| test | 48 | 5,898 | 32 | 7.11 |

Split strategy: stratified by majority reaction_family per screen, seed=20260723.

## HTE Tasks

| Task | Description | Metric |
|------|-------------|--------|
| T1 | Low-yield classification | AUPRC at thresholds 5% and 10% |
| T2 | Ordinal yield-bin prediction | Macro-AUPRC (5 bins: 0-5, 5-20, 20-50, 50-80, 80-100) |
| T3 | Yield regression | MAE + Spearman correlation |
| T4 | Within-plate ranking | NDCG per SCREEN_ID |
| T5 | Condition-specific feasibility | AUPRC within screens |

## Comparison Methods

| Method | Description | Uses PC-CNG? |
|--------|-------------|:------------:|
| positive_only | Centroid cosine similarity to training positives | No |
| tanimoto_baseline | Max Tanimoto similarity to training positives | No |
| hard_label_pc_cng | Morgan-FP + LR with PC-CNG synthetic negatives (hard labels) | Yes |
| risk_aware_pc_cng | Morgan-FP + LR with PC-CNG synthetic negatives (risk-weighted by 1-FNR) | Yes |
| observed_negative_upper_bound | Morgan-FP + LR with real observed negatives only (**ORACLE**) | No |

**Key distinction**: `observed_negative_upper_bound` is an oracle/upper-bound reference that uses real observed negatives. It is NOT a baseline to beat — in the PC-CNG use case, real negatives are unavailable (that is the entire motivation for PC-CNG). The non-PC-CNG baselines are `positive_only` and `tanimoto_baseline`.

## Statistical Protocol

- **Cluster unit**: SCREEN_ID (experimental group/plate)
- **Bootstrap**: 200 iterations, resampling SCREEN_IDs with replacement
- **CI**: 2.5th and 97.5th percentiles of bootstrap distribution
- **Delta CI**: conservative estimate (challenger CI low minus baseline point estimate)
- All 5 methods scored on the same test set (5,898 reactions, 48 screens)

## Results

### Point Estimates and Bootstrap CIs

| Metric | positive_only | tanimoto_baseline | hard_label_pc_cng | risk_aware_pc_cng | observed_neg_ub |
|--------|:---:|:---:|:---:|:---:|:---:|
| t1_auprc_5 | 0.816 [0.726, 0.893] | 0.763 [0.669, 0.856] | 0.709 [0.620, 0.822] | 0.701 [0.615, 0.811] | 0.689 [0.610, 0.795] |
| t1_auprc_10 | 0.862 [0.790, 0.919] | 0.819 [0.730, 0.892] | 0.769 [0.684, 0.871] | 0.761 [0.675, 0.865] | 0.754 [0.673, 0.854] |
| t2_macro_auprc | 0.206 [0.206, 0.219] | 0.202 [0.202, 0.219] | 0.224 [0.212, 0.295] | **0.227** [0.214, 0.299] | 0.230 [0.217, 0.297] |
| t3_spearman | 0.019 [-0.162, 0.237] | 0.130 [-0.054, 0.372] | 0.264 [0.077, 0.467] | **0.300** [0.124, 0.480] | 0.339 [0.152, 0.508] |
| t4_plate_ndcg | 0.196 [0.170, 0.273] | 0.196 [0.171, 0.273] | 0.195 [0.171, 0.271] | 0.197 [0.172, 0.273] | 0.195 [0.169, 0.273] |
| t5_cond_feas | 0.204 [0.143, 0.325] | 0.246 [0.160, 0.393] | 0.371 [0.231, 0.562] | **0.390** [0.234, 0.595] | 0.422 [0.251, 0.586] |
| ECE | 0.233 [0.162, 0.303] | 0.162 [0.088, 0.255] | 0.206 [0.127, 0.294] | 0.165 [0.094, 0.248] | 0.209 [0.135, 0.292] |
| Brier | 0.225 [0.206, 0.245] | 0.385 [0.251, 0.515] | 0.209 [0.143, 0.275] | **0.189** [0.130, 0.254] | 0.210 [0.151, 0.270] |

### Deltas: risk_aware_pc_cng vs Best Non-PC-CNG Baseline

| Metric | Delta (pp) | CI Low | CI High | CI All Positive | >= 2pp |
|--------|:---------:|:------:|:-------:|:---------------:|:------:|
| t1_auprc_5 | -11.57 | -20.13 | -0.49 | No | No |
| t1_auprc_10 | -10.12 | -18.72 | +0.30 | No | No |
| t2_macro_auprc | **+2.15** | **+0.82** | +9.30 | **Yes** | **Yes** |
| t3_spearman | +17.05 | -0.55 | +35.05 | No (barely) | Yes |
| t4_plate_ndcg | +0.01 | -2.44 | +7.68 | No | No |
| t5_cond_feas | +14.39 | -1.20 | +34.95 | No (barely) | Yes |

### Calibration Check

| Method | ECE |
|--------|-----|
| risk_aware_pc_cng | 0.1651 |
| Best baseline (tanimoto) | 0.1623 |
| Delta | +0.0028 (within 0.02 threshold) |
| **Calibration OK** | **Yes** |

## Go/No-Go Verdict

**WEAK_GO**

### Rationale

1. **t2_macro_auprc** (ordinal yield-bin prediction): risk_aware_pc_cng beats the best non-PC-CNG baseline by +2.15pp with CI all positive [+0.82, +9.30]. This is the single metric meeting the strict significance criterion.

2. **t3_spearman** and **t5_condition_feasibility_auprc**: Large point-estimate improvements (+17.05pp and +14.39pp respectively), but the conservative delta CIs barely cross 0 (CI low = -0.006 and -0.012). These are suggestive but not statistically conclusive under the conservative CI method.

3. **t1_low_yield_auprc**: risk_aware_pc_cng is worse than baselines (-11.57pp at 5% threshold). PC-CNG synthetic negatives may confuse the low-yield classification boundary by introducing counterfactual negatives that don't align with true low-yield patterns.

4. **Calibration**: ECE not worse (0.165 vs 0.162, within threshold).

5. **Experimental groups verified**: 356 screens, 41 families, HiTEA provenance confirmed.

6. **Known-positive collision**: 0.0 (screen-aware split ensures no leakage).

### Verdict Logic

- n_metrics_positive_ci = 1 (t2_macro_auprc)
- n_metrics_2pp_significant = 1 (t2_macro_auprc)
- calibration_ok = true
- experimental_groups_verified = true
- STRONG_GO requires n_significant_2pp >= 2 → NOT MET
- WEAK_GO requires n_positive >= 1 → MET

## Limitations

1. **Conservative CI method**: The delta CI uses challenger CI low minus baseline point estimate, which is conservative. A paired bootstrap (resampling screens and computing both methods' metrics on the same bootstrap sample, then taking the difference) would give tighter CIs and might flip t3/t5 to significant.

2. **t1 degradation**: Risk-aware PC-CNG underperforms on low-yield classification. The synthetic negatives from PC-CNG may not capture the true low-yield boundary. This is a known limitation of counterfactual augmentation.

3. **Single dataset**: Only HiTEA was used. Additional HTE datasets (Doyle/Merck C-N coupling, metallaphotoredox) were not available on the server.

4. **Morgan fingerprint limitation**: All LR-based methods use 2048-bit Morgan fingerprints (radius 2), which may not capture all reaction-relevant features. More expressive representations (e.g., reaction fingerprints, graph neural networks) could improve all methods.

5. **PC-CNG negative count**: Only 500 PC-CNG synthetic negatives were available from the G3/G5 manifest. A larger candidate set might improve the PC-CNG methods.

6. **Oracle gap**: The observed_negative_upper_bound (oracle with real negatives) outperforms PC-CNG methods on t3_spearman (0.339 vs 0.300) and t5_cond_feas (0.422 vs 0.390), indicating room for improvement in synthetic negative quality.

## Deliverables

| Artifact | Path |
|----------|------|
| Normalized parquet | `data/processed/p4_hte_normalized.parquet` |
| Split manifest | `data/p4/manifests/p4_hte_split_v1.json` |
| Data audit | `results/p4_hte_external_validation/data_audit.json` |
| Summary CSV | `results/p4_hte_external_validation/summary.csv` |
| Raw predictions | `results/p4_hte_external_validation/raw_predictions/{method}.csv` |
| Go/no-go | `results/p4_hte_external_validation/go_no_go.json` |
| Schema tests | `chem_negative_sampling/tests/test_p4_hte_schema.py` |
| Eval tests | `chem_negative_sampling/tests/test_p4_hte_eval.py` |
| Data module | `chem_negative_sampling/pc_cng/p4_g6_hte_data.py` |
| Eval module | `chem_negative_sampling/pc_cng/p4_g6_hte_eval.py` |
| Aggregation | `chem_negative_sampling/pc_cng/aggregate_p4_g6.py` |
| Orchestration | `chem_negative_sampling/pc_cng/run_p4_hte_validation.py` |

## Next Steps

P4-G6 WEAK_GO allows progression to P4-G7 (Human expert calibration). The claim is limited to:
- Ordinal yield-bin prediction (t2) on HiTEA reaction families
- Condition-specific feasibility (t5) and yield correlation (t3) show large but not statistically conclusive improvements

Recommended follow-up before P4-G9 (manuscript):
- Implement paired bootstrap for tighter CIs on t3/t5
- Test on additional HTE datasets if available
- Consider reaction-aware fingerprints for improved representation
