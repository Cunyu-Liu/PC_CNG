# Phase C / G8-C formal source-expert analysis plan v1

Status: `FROZEN_BEFORE_GPU_FORMAL_VALIDATION`  
Frozen date: 2026-07-29  
Scope: source-expert credibility only; this plan does not test superiority over
other negative sources.

## Data and split contract

- Source: `data/processed/p4_hte_normalized.parquet`.
- Stage 1 uses real atom-mapped bond-form, bond-break and bond-order-change
  actions. Unmapped reactions are `NOT_APPLICABLE`, not `NO_EDIT`.
- Stage 2 uses actual rule-generator actions. Rule non-applicability is an
  explicit `NOT_APPLICABLE` target.
- Stage 3 and Stage 4 use only observed distinct products sharing the complete
  parent/context tuple: reactants, catalysts, solvent, temperature and time.
- Because all observed multi-product contexts are assigned to the source
  training split, their model-selection split is frozen by
  `context_hash_80_20_v1`: SHA-256 of the complete context key assigns 80% to
  training and 20% to validation. A complete context never crosses splits.
- Source test rows are sealed. Formal Phase C does not compute any test metric.

## Training contract

- Formal mode must run Stage 1 to Stage 4 in order on a CUDA GPU.
- Missing edit, rule, competing-outcome, preference or risk cache is fatal.
- Formal mode has no batch-halves, shuffled-pair or zero-reference fallback.
- The Stage 4 reference is copied after Stage 3, frozen and hash-checked after
  optimization.
- Stage 4 scores the complete real action: locus, type and observed argument.
- Risk supervision sources are known-positive collisions, observed competing
  products, completed expert labels and held-out HTE outcomes. Absent expert
  labels are reported as unavailable and never synthesized.

## Validation endpoints and frozen thresholds

| Endpoint | Threshold |
|---|---:|
| Edit-locus accuracy | >= 0.20 |
| Edit-type accuracy | >= 0.50 |
| Valid edit rate | >= 0.95 |
| Candidate coverage | >= 0.80 |
| Candidate-level FNR ECE, 10 bins | <= 0.15 |
| Maximum absolute Stage 4 log-ratio | <= 5.0 |
| Mean action-type entropy | >= 0.50 |
| Frozen reference hash | unchanged |

Accuracy, validity and coverage receive Wilson 95% intervals. Argument accuracy
and joint locus/type accuracy are reported as secondary diagnostics and cannot
replace a failed primary endpoint.

## Run tiers

1. GPU engineering pilot: capped data, used only to test the formal execution
   path and artifact contract.
2. Formal validation: full frozen training/validation inputs with the model and
   thresholds fixed before metrics are read.

The pilot cannot support a scientific claim. Formal validation can establish
only that the learned generator is a credible source expert. It cannot establish
SOTA or superiority over another source.

## Stop rules

- Abort on CPU fallback, NaN/Inf, empty formal batches, missing caches, a
  changing reference hash or access to sealed test labels.
- Preserve every negative result.
- If expert labels remain unavailable, the strongest possible status is
  `FORMAL_SOURCE_EXPERT_PARTIAL_EXPERT_LABELS_PENDING`.
- The legacy tiny self-built Morgan MLP utility evaluation is disabled in
  formal mode.

## Required artifacts

- `formal_validation.json`
- `go_no_go.json`
- `model_checkpoint.pt`
- `train_log.json`
- `run_manifest.json`
- `environment.json`
- `input_hashes.json`
- `commands.log`

