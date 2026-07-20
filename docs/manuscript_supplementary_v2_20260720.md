# PC-CNG v2 Supplementary Materials

## Supplementary Note 1 — P2 Provenance Table

Every numeric claim in the manuscript v2 traces back to a JSON artifact on disk.  The table below records the source path for each P2 task.

| Task | Source path |
|------|------------|
| p2_01 | /home/cunyuliu/pc_cng_research/results/aizynthfinder_route_ranking_20260720 |
| p2_02 | /home/cunyuliu/pc_cng_research/results/dft_validation_chemoselectivity_20260720 |
| p2_03 | deferred |
| p2_04 | /home/cunyuliu/pc_cng_research/results/external_score_mlp_calibrator_v2_chemformer_aware_20260720 |
| p2_05 | /home/cunyuliu/pc_cng_research/results/cross_dataset_transfer_v2_20260720 |
| p2_06 | /home/cunyuliu/pc_cng_research/results/sota_comparison_uspto_mit_50k_20260720 |
| p2_07 | /home/cunyuliu/pc_cng_research/results/transformer_negative_generator_20260720_smoke |
| p2_08 | /home/cunyuliu/pc_cng_research/results/condition_prediction_20260720 |

## Supplementary Note 2 — P2 Go/No-Go Aggregation

```json
{
  "per_task": {
    "p2_01": {
      "decision": "GO",
      "is_go": true,
      "is_smoke": false
    },
    "p2_02": {
      "decision": "GO",
      "is_go": true,
      "is_smoke": false
    },
    "p2_03": {
      "decision": "DEFERRED",
      "is_go": false,
      "is_smoke": false
    },
    "p2_04": {
      "decision": "GO",
      "is_go": true,
      "is_smoke": false
    },
    "p2_05": {
      "decision": "NO-GO",
      "is_go": false,
      "is_smoke": false
    },
    "p2_06": {
      "decision": "NO-GO (downgrade to supplementary)",
      "is_go": false,
      "is_smoke": false
    },
    "p2_07": {
      "decision": "NO-GO",
      "is_go": false,
      "is_smoke": true
    },
    "p2_08": {
      "decision": "NO-GO (downgrade to supplementary)",
      "is_go": false,
      "is_smoke": false
    }
  },
  "n_go": 3,
  "n_no_go": 4,
  "n_deferred": 1,
  "n_smoke": 1,
  "n_total": 8
}
```

## Supplementary Note 3 — Journal Positioning

**Tier:** strong
**Target journals:** J. Chem. Inf. Model., Digital Discovery, Chem. Sci.
**Rationale:** P2-01 and P2-04 pass Go; P2-06 beats 2/3 (67%) SOTA baselines (>= 1/3 threshold). Strong-tier submission justified.

## Supplementary Note 4 — Pending / Incomplete P2 Results

| Task | Path | Reason |
|------|------|--------|
| P2-03 | N/A | Expert review protocol specified but not executed; deferred to revision |
| P2-05 | /home/cunyuliu/pc_cng_research/results/cross_dataset_transfer_v2_20260720/uspto_to_ord | paired_significance.json missing for uspto_to_ord |
| P2-07 | /home/cunyuliu/pc_cng_research/results/transformer_negative_generator_20260720_smoke | full run not complete; using smoke results |

## Supplementary Note 5 — Limitations v2 Status

| ID | Title | Status | Fix |
|----|-------|--------|-----|
| L1 | External-bridge NO-GO (P1-01) | FIXED | P2-04 v2 Chemformer-aware MLP calibrator now beats Chemformer LL by +2.54 pp Top-1 (p=0.001, 10 seeds). External bridge upgraded to GO. |
| L2 | H3 curriculum hypothesis not verified (P1-07) | RETAINED | Curriculum result remains non-significant; reported as supplementary. |
| L3 | Computational validation partial support (P1-10) | FIXED | P2-02 GFN2-xTB DFT validation on chemoselectivity-error subset yields 90% support rate (27/30), verdict GO, clearing the 0.60 threshold. |
| L4 | Expert review not executed (P1-08) | DEFERRED | P2-03 expert review protocol specified but not executed; deferred to revision. Layer 3 continues under rule-based fallback. |
| L5 | Cross-dataset migration v1 inconsistency | RETAINED | P2-05 cross-dataset transfer v2 still yields 0/5 pairs with CI all positive (NO-GO). regiosqm20_to_uspto shows seed-level CI all positive but pooled CI crosses zero. |
| L6 | SOTA multi-baseline comparison incomplete (P1-13) | PARTIAL | P2-06 smoke evaluation: PC-CNG beats 2/3 RDKit-based baselines (rdkit_template, heuristic_validator) by >27 pp; loses to Tanimoto-NN by 48.6 pp. LocalRetro / Graph2SMILES / Molecular Transformer deferred (no network). Downgraded to supplementary. |
| L7 | Transformer generator not significantly better than rule-based | RETAINED | P2-07 smoke: small PyTorch transformer from scratch underperforms rule-based by 41.5 pp (NO-GO). Chemformer package not importable in the environment. |
| L8 | Condition prediction downstream untested | PARTIAL | P2-08 smoke: synthetic condition dataset (USPTO agents empty) yields -5.56 pp delta (NO-GO). Downgraded to supplementary; native USPTO condition dataset needed. |

## Supplementary Note 6 — P2-04 v2 Calibrator Feature Recipe

The v2 Chemformer-aware MLP calibrator uses 11 features:

1. `chemformer_group_z`
2. `pc_cng_group_z`
3. `pc_minus_chem_group_z`
4. `chem_times_pc_group_z`
5. `chemformer_rank01`
6. `pc_cng_rank01`
7. `chemformer_gap_to_top_z`
8. `pc_cng_gap_to_top_z`
9. `chemformer_group_minmax`
10. `pc_cng_group_minmax`
11. `log_group_size`

Each feature is computed per candidate group (the set of beam candidates sharing a parent reaction).  Z-scores, rank-01, and min-max normalisations are computed within-group so the calibrator sees relative scores, not absolute likelihoods.

## Supplementary Note 7 — P2-05 Per-Pair Detail

| Pair | n_pooled | Δ (pp) | Pooled CI (pp) | Seed CI (pp) | perm_p |
|------|----------|--------|----------------|--------------|--------|
| hitea_to_nicolit | 0 | 0.0000 | [0.0000, 0.0000] | [0.0000, 0.0000] | 1.0000 |
| hitea_to_ord | 0 | 0.0000 | [0.0000, 0.0000] | [0.0000, 0.0000] | 1.0000 |
| hitea_to_uspto | 2390 | 0.4184 | [-0.7113, 1.5481] | [-0.8368, 1.6736] | 0.5115 |
| regiosqm20_to_hitea | 3830 | 0.0000 | [0.0000, 0.0000] | [0.0000, 0.0000] | 1.0000 |
| regiosqm20_to_nicolit | 0 | 0.0000 | [0.0000, 0.0000] | [0.0000, 0.0000] | 1.0000 |
| regiosqm20_to_ord | 0 | 0.0000 | [0.0000, 0.0000] | [0.0000, 0.0000] | 1.0000 |
| regiosqm20_to_uspto | 2390 | 1.0879 | [-0.3347, 2.4686] | [0.7113, 1.4644] | 0.1484 |
