# P2 Baseline Metrics — Locked from P1 Final State

**Generated**: 2026-07-20
**Source**: P1 final state (manuscript_v1_20260719.md, P1_initial_state_manifest_20260719.json)
**Purpose**: Establish P2 starting baseline for Go/No-Go comparisons.

## 1. P1 Main Claims (Locked)

These are the headline numerical claims PC-CNG v3 carries into P2. Each claim
must be either reaffirmed, upgraded, or downgraded by P2 task outputs.

| ID  | Claim                                                                                                              | Metric                                                  | 95% CI              | p-value  | Seeds | Status         |
| --- | ------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------- | ------------------- | -------- | -----: | -------------- |
| C1  | Statistically significant cross-dataset migration gain (RegioSQM20 → USPTO)                                       | delta Top-1 = +1.63 pp                                 | [0.59, 2.72]        | 0.0028   |     10 | P1 main claim  |
| C2  | PC-CNG improves retrosynthesis route ranking                                                                       | MRR 24.24% → 54.87% (delta = +30.63 pp)                | [29.23, 32.05] pp   | < 0.0001 |     10 | P1 main claim  |
| C3  | Calibration acceptable                                                                                            | ECE = 0.0889                                            | n/a                 | n/a      |     10 | P1 main claim  |
| C4  | OOD scaffold/template splits show no significant degradation                                                      | no significant drop                                     | n/a                 | n/a      |     10 | P1 main claim  |
| C5  | Reproducibility manifest covers 28 artifacts + Ni-coupling supplement (1688 reactions)                             | n/a                                                     | n/a                 | n/a      |      — | P1 main claim  |

## 2. P1 Limitations (L1–L8) Targeted by P2

| ID  | Name                                 | Issue                                                                                  | P2 Task | Go/No-Go Target                                                                                  |
| --- | ----------------------------------- | -------------------------------------------------------------------------------------- | ------- | ------------------------------------------------------------------------------------------------ |
| L1  | External bridge NO-GO               | MLP calibrator underperforms Chemformer LL by 10.56 pp on held-out full-beam          | P2-04   | v2 calibrator Top-1 ≥ Chemformer + 1.0 pp, 10-seed CI all positive                              |
| L2  | Retrosynthesis pseudo-route fallback| P1-04 used pseudo-route instead of AiZynthFinder real routes                          | P2-01   | PC-CNG MRR > AiZynthFinder baseline + 1.0 pp, CI all positive                                     |
| L3  | DFT partial support                 | MMFF94 support rate 0.48 < 0.6 threshold                                              | P2-02   | DFT support ≥ 60% of chemoselectivity_error subset                                               |
| L4  | Expert review not executed          | P1-08 Layer 3 used rule-based fallback                                                | P2-03   | Cohen's κ ≥ 0.6, approval rate ≥ 70% (or document deferred-to-revision if experts unavailable)    |
| L5  | Cross-dataset transfer weak         | Only 1/4 migration pairs significant                                                  | P2-05   | ≥ 3/10 migration pairs with CI all positive                                                      |
| L6  | No SOTA direct comparison           | Missing LocalRetro / Graph2SMILES / Molecular Transformer                             | P2-06   | PC-CNG Top-1 ≥ 3/3 SOTA + 1.0 pp, CI all positive                                                 |
| L7  | GNN decoder not better than rules   | GNN learned decoder did not exceed rule-based                                          | P2-07   | Transformer generator Test Top-1 ≥ rule-based + 1.0 pp, CI all positive                           |
| L8  | Downstream coverage insufficient    | No condition prediction evaluation                                                     | P2-08   | PC-CNG augmented Top-1 > baseline + 1.0 pp, CI all positive                                       |

## 3. P2 Journal-Positioning Rules

| Tier          | Journals                                                    | Required Go/No-Go Pass                                          |
| ------------- | ---------------------------------------------------------- | --------------------------------------------------------------- |
| Top tier      | Nature Chemistry / JACS Au / Nature Machine Intelligence | P2-01, P2-02, P2-03, P2-06 all pass Go                          |
| Strong tier   | J. Chem. Inf. Model. / Digital Discovery / Chem. Sci.    | P2-01, P2-04 pass Go AND P2-06 ≥ 1/3 SOTA pass                 |
| Fallback      | (paper rewrite, target deferred)                          | Above criteria not met                                          |

## 4. P2 Constraints (Hard Rules)

1. **No delete/rename** of existing `results/` subdirectories.
2. **No modify** existing `.md` files under `docs/00_当前有效文档/` (only append new docs).
3. **No GPU 4** usage — `calibrate` PID 2544995 still running.
4. **All new code** must ship with unit tests.
5. **All performance claims** must use 10-seed paired significance test.
6. **AiZynthFinder / DFT / SOTA tools** must be installed in isolated venv
   (`/home/cunyuliu/venvs/{aizynthfinder,dft,sota}/`) to avoid polluting pc_cng.

## 5. P2 Resource Inventory (Snapshot at P2 Start)

### 5.1 P1 Result Subdirs Locked (SHA-256 inventory)
Full SHA-256 of every `.json/.csv/.md/.log/.yaml` file under each of the
following subdirs is recorded in `docs/P2_initial_state_manifest_20260720.json`.

```
results/cross_dataset_transfer_20260719/
results/calibration_error_10seed_20260719/
results/ood_scaffold_template_split_20260719/
results/retrosynthesis_route_ranking_20260719/
results/false_negative_three_layer_20260719/
results/ord_data_quality_audit_20260719/
results/xtb_dft_validation_20260719/
results/external_calibration_heldout_full_beam_5k_20260719/
results/external_calibration_heldout_full_beam_mlp_apply_5k_20260719/
results/external_calibration_heldout_full_beam_paired_significance_5k_20260719/
results/expert_review_20260719/
results/manuscript_tables_p1_baseline_20260719/
results/P1_initial_state_20260719/
results/semi_hard_curriculum_10seed_20260719/
results/semi_hard_curriculum_smoke_20260719/
results/cross_dataset_transfer_smoke_20260719/
results/cross_dataset_transfer_smoke2_20260719/
results/failure_prototype_calibration_smoke_20260719/
```

### 5.2 Manuscript v1 Lock
- `docs/manuscript_v1_20260719.md` (main)
- `docs/manuscript_supplementary_v1_20260719.md`
- `docs/manuscript_v1_20260719/` (figures directory)
- `docs/manuscript_figures_v1_20260719/` (figures v1)

SHA-256 of manuscript main + supplementary locked in
`docs/P2_initial_state_manifest_20260720.json` under `manuscript_v1_lock`.

### 5.3 GPU Availability (at P2 start)
- GPU 0: idle (0% util, ~10 GB used)
- GPU 1: in use (~39 GB used)
- GPU 2: in use (~10.6 GB)
- GPU 3: in use (~39.6 GB)
- GPU 4: **FORBIDDEN** — calibrate PID 2544995 (22h+ elapsed, ~21 GB)
- GPU 5: ~32% util
- GPU 6: N/A (~1.8 GB)
- GPU 7: N/A (~3.7 GB)

Usable GPUs for P2: 0, 5, 6, 7.

### 5.4 Isolated Venvs (Required)
```
/home/cunyuliu/venvs/aizynthfinder/    # for P2-01, P2-06
/home/cunyuliu/venvs/dft/              # for P2-02 (ORCA/xTB)
/home/cunyuliu/venvs/sota/             # for P2-06 (LocalRetro/Graph2SMILES/Molecular Transformer)
```

Status at P2 start: **none exist yet** — must be created in P2.

## 6. P2 Task Plan Summary

| Task   | Title                                            | Priority | Go/No-Go                                |
| ------ | ------------------------------------------------ | -------- | --------------------------------------- |
| P2-00  | P1 integration + baseline lock (this doc)        | must     | n/a                                     |
| P2-01  | AiZynthFinder real route comparison (L2)         | high     | MRR > AiZynthFinder baseline + 1.0 pp   |
| P2-02  | DFT validation chemoselectivity subset (L3)      | high     | DFT support ≥ 60%                       |
| P2-03  | Expert review execution (L4)                    | medium   | κ ≥ 0.6, approval ≥ 70%                |
| P2-04  | MLP calibrator v2 chemformer-aware (L1)          | high     | Top-1 ≥ Chemformer + 1.0 pp             |
| P2-05  | Cross-dataset transfer v2 expanded (L5)          | medium   | ≥ 3/10 pairs CI all positive            |
| P2-06  | SOTA multi-baseline comparison (L6)              | high     | PC-CNG Top-1 ≥ 3/3 SOTA + 1.0 pp        |
| P2-07  | Transformer-based generator ablation (L7)        | medium   | Top-1 ≥ rule-based + 1.0 pp             |
| P2-08  | Condition prediction downstream (L8)             | medium   | Top-1 > baseline + 1.0 pp               |
| P2-09  | Manuscript v2 + submission prep                 | high     | Manuscript v2 + journal decision       |

## 7. Reproducibility Anchors

- `docs/P2_initial_state_manifest_20260720.json` (P1 full SHA-256 inventory)
- `docs/P2_baseline_metrics_20260720.md` (this document)
- 10 seeds fixed across all P2 runs: `20260710`–`20260719`
- All P2 acceptance commands reference the same seed list.
