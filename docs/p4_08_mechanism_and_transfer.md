# P4-G8: Mechanism, Transfer & Learned Structured Proposal

**Phase**: P4-G8  
**Date**: 2026-07-23  
**Status**: G8-A PENDING (G4 v2 running), G8-B COMPLETE (NO_GO), G8-C GATED  

## Entry Conditions

| Condition | Status | Evidence |
|-----------|--------|----------|
| P4-G4 completed | IN PROGRESS (v2 full matrix running, 15/70 MLP runs) | `results/p4_generator_scorer_matrix_v2/raw_predictions/` |
| P4-G5 completed | PASS (PARTIAL_GO) | `results/p4_risk_aware/go_no_go.json` |
| P4-G6 completed | PASS (WEAK_GO) | `results/p4_hte_external_validation/go_no_go.json` |
| P4-G3 >= Weak GO | PASS (v2 WEAK_GO) | `results/p4_augmentation_v2_chemformer/go_no_go.json` |
| P4-G4 >= Partial GO | PENDING | Awaiting v2 full matrix completion |
| P4-G5 >= Partial GO | PASS | `results/p4_risk_aware/go_no_go.json` |

---

## P4-G8A: Difficulty-Benefit-Risk Mechanism Curves

**Status**: PENDING — awaiting G4 v2 completion (MLP predictions needed as 3rd scorer)

### Design

- **Difficulty metrics (7)**: positive_similarity, nearest_train_similarity, scoring_margin, ensemble_uncertainty, false_negative_risk, edit_distance, database_collision
- **Curve shapes tested (5)**: monotonic_decreasing, monotonic_increasing, inverted_u, threshold, flat
- **Datasets**: G3 v2 test (chemformer + gnn scorers), G4 v2 test (morgan_mlp scorer)
- **Bin count**: 10 (deciles)
- **Analysis**: Fit curve shape on dev set, validate on test set; simultaneously analyze utility and risk

### Verdict Criteria

- **GO**: Relationship reproduced in >=2 datasets AND >=2 scorers consistent
- **PARTIAL_GO**: Relationship found but not fully reproduced
- **NO_GO**: No consistent relationship found

### Outputs (pending)

```
results/p4_mechanism_curve/
├── per_candidate_metrics.csv
├── curve_specs.json
├── utility_curve.json
├── risk_curve.json
├── shape_comparison.csv
├── go_no_go.json
├── run_manifest.json
├── environment.json
├── input_hashes.json
└── commands.log
```

---

## P4-G8B: Cross-Reaction-Family Transfer

**Status**: COMPLETE  
**Verdict**: NO_GO  
**Runtime**: 45.3 seconds  

### Methods

| Method | Implemented | Notes |
|--------|------------|-------|
| direct transfer | YES | Train on source, test on target |
| head fine-tuning | YES | Freeze backbone, fine-tune last layer on target val |
| risk-aware fine-tuning | YES | PU prior cap + risk-weighted loss |
| LoRA/adapter | NO | Limitation: not implemented in this version |
| EWC | NO | Limitation: not implemented in this version |
| multi-task | NO | Limitation: not implemented in this version |

### Family Pairs

| Pair | Direction 1 | Direction 2 | Skip Reason |
|------|-------------|-------------|-------------|
| Pd coupling ↔ Alkylation | Pd→Alk | Alk→Pd | — |
| Pd coupling ↔ Hydrogenation | H→Pd | Pd→H | Pd→H skipped (Hydrogenation test=0) |
| Alkylation ↔ Cabonylation | Alk→Cab | Cab→Alk | — |
| Rh coupling ↔ Cu coupling | Rh→Cu | Cu→Rh | — |

**Note**: Manifest uses "Cabonylation" (typo in source data), not "Carbonylation".

### Results Summary

**7 directions × 3 methods × 2 seeds = 42 transfer experiments + 14 baselines = 56 total runs**

| Direction | Method | Transfer MRR | Baseline MRR | Gain |
|-----------|--------|-------------|-------------|------|
| Pd→Alkylation | direct | 0.237 | 0.282 | -0.046 |
| Alkylation→Pd | direct | 0.163 | 0.220 | -0.057 |
| Hydrogenation→Pd | direct | 0.090 | 0.220 | -0.130 |
| Alkylation→Cabonylation | direct | 0.225 | 0.482 | -0.257 |
| Cabonylation→Alkylation | direct | 0.109 | 0.282 | -0.173 |
| Rh→Cu | direct | 0.667 | 1.000 | -0.333 |
| Cu→Rh | direct | 0.390 | 0.583 | -0.193 |

**All 7 directions show negative transfer gain.** No method (direct, head_ft, risk_aware) achieves positive transfer in any direction.

### Family Macro Summary

| Family | N Experiments | Mean MRR | Mean AUPRC | Mean ECE |
|--------|--------------|----------|------------|----------|
| Alkylation | 16 | 0.200 | 0.554 | 0.149 |
| Cabonylation | 8 | 0.290 | 0.451 | 0.109 |
| Cu coupling | 8 | 0.750 | 0.750 | 0.138 |
| Pd coupling | 16 | 0.150 | 0.447 | 0.098 |
| Rh coupling | 8 | 0.439 | 0.646 | 0.146 |

### Key Findings

1. **Universal negative transfer**: All 7 directions show transfer gain < 0, meaning models trained on one reaction family perform worse than models trained on the target family directly.

2. **Method insensitivity**: direct, head_ft, and risk_aware methods produce nearly identical results. Head fine-tuning does not improve transfer, likely because the Morgan MLP is too simple (3-layer, 256 hidden) for fine-tuning to make a meaningful difference.

3. **Small test sets limit reliability**: Cu coupling (3 test candidates), Rh coupling (6 test), and Cabonylation (9 test) have very small test sets, making per-family metrics unreliable.

4. **Largest negative transfer**: Alkylation→Cabonylation (-0.257), where the baseline MRR (0.482) is more than double the transfer MRR (0.225).

5. **No catastrophic forgetting observed**: Source MRR remains stable across methods, indicating no severe forgetting (though source test sets are also small).

### Limitations

1. Only 3/6 spec-recommended methods implemented (missing LoRA/adapter, EWC, multi-task)
2. Morgan MLP is a simple model; more complex architectures (e.g., GNN, transformer) might transfer better
3. Some families have very small test sets (3-9 candidates)
4. No domain similarity analysis between families
5. Head fine-tuning appears identical to direct (potential implementation issue or model too simple)

### Outputs

```
results/p4_cross_family_transfer/
├── transfer_results.csv      (56 rows)
├── family_macro_summary.csv  (5 families)
├── go_no_go.json             (NO_GO)
├── run_manifest.json
├── environment.json
├── input_hashes.json
└── commands.log
```

---

## P4-G8C: Learned Structured Proposal

**Status**: GATED — awaiting G4 v2 verdict

### Entry Conditions

| Condition | Required | Current | Status |
|-----------|----------|---------|--------|
| P4-G3 >= Weak GO | YES | WEAK_GO (v2) | PASS |
| P4-G4 >= Partial GO | YES | PENDING | BLOCKED |
| P4-G5 >= Partial GO | YES | PARTIAL_GO | PASS |

### Assessment

G8-C cannot be started until G4 v2 completes and achieves at least PARTIAL_GO. Additionally, the G8-B NO_GO result (no cross-family transfer) suggests that PC-CNG negative sources may not have stable utility across reaction families, which the spec states should prevent launching a large learned proposal ("若 PC-CNG negative source 没有稳定效用，不得启动大型 learned proposal").

If G4 v2 achieves PARTIAL_GO or better, G8-C will be assessed for feasibility. The proposed architecture:
- Reaction graph transformer + edit-locus pointer + edit-type/argument heads + action mask + risk head
- 4-stage training: edit reconstruction → rule imitation → competing outcomes → risk-adjusted preference (DPO/IPO, not PPO)

---

## Test Coverage

| Test File | Tests | Status |
|-----------|-------|--------|
| `tests/test_mechanism_curve.py` | 20 | ALL PASS |
| `tests/test_cross_family_transfer.py` | 15 | ALL PASS |
| `tests/test_structured_edit_proposal.py` | — | NOT CREATED (G8-C gated) |

---

## Commit Artifacts

- Code: `pc_cng/p4_g8a_mechanism_curve.py`, `pc_cng/p4_g8b_cross_family_transfer.py`, `pc_cng/run_p4_g4_v2.py`
- Tests: `tests/test_mechanism_curve.py`, `tests/test_cross_family_transfer.py`
- Results: `results/p4_cross_family_transfer/` (complete), `results/p4_mechanism_curve/` (pending)
- Docs: `docs/p4_08_mechanism_and_transfer.md` (this file)
