# P4-G8: Mechanism, Transfer & Learned Structured Proposal

**Phase**: P4-G8  
**Date**: 2026-07-24 (updated)  
**Status**: G8-A COMPLETE (GO), G8-B v2 IN PROGRESS, G8-C COMPLETE (NO_GO)  

## Entry Conditions

| Condition | Status | Evidence |
|-----------|--------|----------|
| P4-G5 completed | PASS (PARTIAL_GO) | `results/p4_risk_aware/go_no_go.json` |
| P4-G6 completed | PASS (WEAK_GO) | `results/p4_hte_external_validation/go_no_go.json` |
| P4-G3 >= Weak GO | PASS (v2 WEAK_GO) | `results/p4_augmentation_v2_chemformer/go_no_go.json` |

---

## P4-G8A: Difficulty-Benefit-Risk Mechanism Curves

**Status**: COMPLETE  
**Verdict**: GO  
**Date**: 2026-07-23  

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

### Results

**5 difficulty metrics reproduced across >=2 datasets and >=2 scorers with consistent curve shapes.**

| Metric | Curve Shape | Datasets | Scorers | Consistent |
|--------|------------|----------|---------|------------|
| positive_similarity | monotonic_increasing | 2 | 3 | 3 |
| nearest_train_similarity | threshold | 2 | 3 | 3 |
| scoring_margin | threshold | 2 | 3 | 3 |
| ensemble_uncertainty | inverted_u | 2 | 3 | 2 |
| edit_distance | monotonic_decreasing | 2 | 3 | 3 |

### Outputs

```
results/p4_mechanism_curve/
├── per_candidate_metrics.csv
├── curve_specs.json
├── utility_curve.json
├── risk_curve.json
├── shape_comparison.csv
├── go_no_go.json             (GO)
├── commands.log
├── environment.json
└── input_hashes.json
```

---

## P4-G8B: Cross-Reaction-Family Transfer

### G8-B v1 (Initial)

**Status**: COMPLETE  
**Verdict**: NO_GO  
**Runtime**: 45.3 seconds  

#### Limitations of v1

1. Only 3/6 spec-recommended methods implemented (missing LoRA/adapter, EWC, multi-task)
2. Morgan MLP is a simple model; more complex architectures might transfer better
3. Some families have very small test sets (3-9 candidates)
4. No domain similarity analysis between families
5. Only 2 seeds (spec requires 10)
6. No cluster bootstrap CI, permutation tests, or effect sizes

### G8-B v2 (Full Spec)

**Status**: IN PROGRESS (training phase)  
**Date**: 2026-07-24  
**PIDs**: 263373 (smoke), 320457 (full)  

#### Current Progress

- Classification phase COMPLETE: USPTO EAS=11836, C-N=18964; ORD EAS=35, C-N=20
- Training phase IN PROGRESS: first direction (USPTO:EAS→USPTO:C-N coupling)
- Smoke: 3 directions × 6 methods × 2 seeds = 36 runs
- Full: 7 directions × 6 methods × 10 seeds = 420 runs
  

#### Improvements over v1

| Aspect | v1 | v2 |
|--------|----|----|
| Methods | 3 (direct, head_ft, risk_aware) | 6 (+ LoRA adapter, EWC, multi-task) |
| Seeds | 2 | 10 (predeclared) |
| Statistics | None | Cluster bootstrap CI, exact sign-flip permutation test, Cohen's d |
| Directions | 7 (HTE only) | Expanded (HTE + USPTO + ORD) |
| Reaction parsing | `>>` only | `>>` and `>agents>` formats |

#### Methods (v2)

| Method | Implemented | Description |
|--------|------------|-------------|
| direct | YES | Train on source, test on target |
| head_ft | YES | Freeze backbone, fine-tune last layer on target val |
| lora_adapter | YES | Low-rank adaptation (rank=8) inserted into linear layers |
| ewc | YES | Elastic Weight Consolidation (Fisher information regularization) |
| risk_aware | YES | PU prior cap + risk-weighted loss |
| multi_task | YES | Joint training on source + target with domain labels |

#### Verdict Criteria (v2)

- **GO**: >=2 chemically different directions with CI all positive, p < 0.05, Cohen's d > 0.3
- **PARTIAL_GO**: >=1 direction positive; all failures reported with effect sizes
- **NO_GO**: No positive transfer; or severe catastrophic forgetting

#### Outputs (pending)

```
results/p4_cross_family_transfer_v2/
├── transfer_results.csv
├── family_macro_summary.csv
├── direction_stats.json
├── go_no_go.json
├── run_manifest.json
├── environment.json
├── input_hashes.json
├── commands.log
├── raw_predictions/
└── _cache/
```

---

## P4-G8C: Learned Structured Proposal

**Status**: COMPLETE  
**Verdict**: NO_GO  
**Date**: 2026-07-24  

### Architecture (7 sub-modules)

1. **Reaction graph transformer** - MPNN + multi-head self-attention
2. **Reaction-center encoder** - Encode formed/broken bonds as context
3. **Edit-locus pointer** - Attention pointer to select atom to edit
4. **Edit-type classifier** - atom_transmutation / bond_order_change / formed_bond_migrate / no_edit
5. **Atom/bond argument decoder** - Decode specific arguments
6. **Validity action mask** - Mask chemically invalid edits
7. **Risk / uncertainty head** - False-negative risk + epistemic uncertainty

### Training Stages (4, no PPO)

| Stage | Name | Description |
|-------|------|-------------|
| 1 | Edit reconstruction | Real-reaction edit reconstruction (legal edit grammar) |
| 2 | Rule imitation | Imitate PC-CNG rule proposals |
| 3 | Competing outcomes | Observed competing-outcome learning (real alternative products) |
| 4 | Risk-adjusted preference | DPO/IPO pairwise preference learning |

### Comparison Arms (4)

| Arm | Description |
|-----|-------------|
| rule_pc_cng | Baseline rule generator |
| unconstrained_neural | Neural generator without validity mask |
| learned_structured | Full model |
| learned_structured_risk | Full model + risk reranking |

### GO Criteria

- Pareto-frontier advantage over the rule version
- Downstream utility CI all positive
- Candidate coverage matched (improvement must not come merely from generating more candidates)

### Bug Fixes Applied During G8-C Execution

The initial G8-C run produced 0 candidates for all neural arms. Root cause analysis
identified 6 bugs, all fixed before the final run:

| Bug | Root Cause | Fix |
|-----|-----------|-----|
| 0 candidates generated | `migrate_target` never set for `FORMED_BOND_MIGRATE` edits | Set `migrate_target = atom_arg % n_atoms` in `generate_structured_proposal` |
| False collision (80%) | Candidate SMILES retained atom mapping (`:1`, `:2`) while positives were unmapped | Added `_strip_atom_maps()` helper; strip maps from all edited SMILES and positives |
| `unconstrained_neural` 0 candidates | `locus` indices exceeded product atom count (graph includes all atoms) | Clamp `n_atoms` to `product_n_atoms`; mask `locus_logits` beyond product atoms |
| All edits are `NO_EDIT` | `type_logits.argmax()` always returned `NO_EDIT` (returns original product) | Exclude `NO_EDIT` from type sampling (`type_probs_no_edit`); sample types instead of argmax |
| All edits produce same SMILES | Model argmax produced no-op edits (transmute to same element, change to same bond order) | Added validity pre-check: `_apply_structured_edit` result must differ from original product |
| Stage 4 DPO loss explosion | DPO loss diverged to ~1e18, corrupting model weights | Added magnitude guard (`loss > 1e4` → skip batch); lowered gradient clip 5.0 → 1.0 |

### Full Run Results

**Verdict**: NO_GO  
**Runtime**: 315 seconds (5.25 minutes)  
**Training data**: 399 train / 47 val / 50 test (from 496 pre-mapped USPTO reactions)  
**Architecture**: hidden_dim=128, num_heads=4, num_layers=3, dropout=0.1  
**Training**: 4 stages × 4 rounds × 8 epochs = 128 total epochs  

#### Comparison Results

| Arm | N Candidates | Utility | Validity | Collision Risk | Controllability | Family Coverage | Diversity |
|-----|-------------|---------|----------|---------------|-----------------|-----------------|-----------|
| rule_pc_cng | 42 | 0.960 | 1.000 | 0.000 | 0.000 | 24 | 1.000 |
| unconstrained_neural | 20 | 0.960 | 1.000 | 0.000 | 0.400 | 10 | 0.950 |
| learned_structured | 20 | 0.960 | 1.000 | 0.000 | 0.400 | 13 | 0.950 |
| learned_structured_risk | 20 | 0.960 | 1.000 | 0.000 | 0.400 | 12 | 1.000 |

#### Pareto Frontier

All 4 arms are on the Pareto frontier (none dominates another):
- `rule_pc_cng`: highest family coverage (24) and most candidates (42)
- `learned_structured`: best family coverage among neural arms (13), 40% controllability
- `learned_structured_risk`: highest diversity (1.000), risk reranking effective
- `unconstrained_neural`: matches utility without validity mask

#### Verdict Details

- **Utility delta**: 0.0 (CI [0.0, 0.0]) — downstream AUPRC identical with 20 vs 42 negatives
- **Coverage matched**: False (20 vs 42 candidates)
- **Improvements**: 0/4 metrics improved vs rule
- **Reason**: The learned model generates valid, non-colliding, diverse candidates but
  does not outperform the rule-based generator on downstream utility with 399 training reactions

### Outputs

```
results/p4_learned_proposal_full/
├── go_no_go.json              (NO_GO)
├── comparison_results.csv
├── pareto_frontier.json
├── model_checkpoint.pt
├── train_log.json
└── raw_predictions/
    ├── rule_pc_cng.csv         (42 candidates)
    ├── unconstrained_neural.csv (20 candidates)
    ├── learned_structured.csv   (20 candidates)
    └── learned_structured_risk.csv (20 candidates)
```

### Analysis

The NO_GO verdict is an honest result reflecting the current limitations:

1. **Training data scale**: 399 reactions is insufficient for a graph transformer to
   learn meaningful edit patterns (rule generator uses chemical knowledge directly)
2. **Model capacity**: The 3-layer MPNN + attention model is lightweight; larger
   architectures (e.g., Chemformer-scale) might improve
3. **Edit grammar coverage**: Only 3 edit types (atom_transmutation, bond_order_change,
   formed_bond_migrate); real reaction grammars include more complex transformations
4. **DPO instability**: Stage 4 risk-adjusted preference learning was unstable
   (loss explosion prevented by magnitude guard); the model primarily benefits from
   stages 1-3 (reconstruction + imitation + contrastive)

The learned model DOES demonstrate:
- Valid candidate generation (validity=1.0)
- Zero false collisions (collision_risk=0.0)
- Structural diversity (0.95 diversity, 13 scaffolds)
- Edit controllability (40% ATOM_TRANSMUTATION)
- Risk-aware reranking improves diversity (0.95 → 1.00)

---

## Test Coverage

| Test File | Tests | Status |
|-----------|-------|--------|
| `tests/test_mechanism_curve.py` | 20 | ALL PASS |
| `tests/test_cross_family_transfer.py` | 15 | ALL PASS (v1) |
| `tests/test_cross_family_transfer_v2.py` | 23 | ALL PASS (v2) |
| `tests/test_structured_edit_proposal_v2.py` | 33 | ALL PASS |

---

## Data Preparations

### RXNMapper Pre-mapping

USPTO OpenMolecules reactions are unmapped. RXNMapper was used to batch-map reactions for G8-C:

| Dataset | Reactions | Mapped | File |
|---------|-----------|--------|------|
| G8-C smoke | 80 | 80 | `data/p4/g8c_smoke_mapped_reactions.json` |
| G8-C full | 500 | 496 | `data/p4/g8c_full_mapped_reactions.json` |

---

## Commit Artifacts

- Code: `pc_cng/p4_g8a_mechanism_curve.py`, `pc_cng/p4_g8b_cross_family_transfer.py`, `pc_cng/p4_g8b_transfer_v2.py`, `pc_cng/p4_g8c_learned_structured_proposal.py`
- Tests: `tests/test_mechanism_curve.py`, `tests/test_cross_family_transfer.py`, `tests/test_cross_family_transfer_v2.py`, `tests/test_structured_edit_proposal_v2.py`
- Results: `results/p4_mechanism_curve/` (G8-A GO), `results/p4_cross_family_transfer/` (G8-B v1 NO_GO), `results/p4_cross_family_transfer_v2/` (G8-B v2 IN PROGRESS), `results/p4_learned_proposal_full/` (G8-C NO_GO), `results/p4_learned_proposal_v7_smoke/` (G8-C smoke)
- Docs: `docs/p4_08_mechanism_and_transfer.md` (this file)
