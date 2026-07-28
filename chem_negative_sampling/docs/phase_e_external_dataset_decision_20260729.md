# Phase E external blind-test dataset decision

Status: metadata-only planning; no candidate labels were downloaded or inspected.

## Decision rule

A Phase E dataset is eligible only if it:

1. was not used in Phase 3, Phase 4, G6 or Phase D method design;
2. contains observed experimental outcomes rather than model-generated negatives;
3. exposes enough reaction and condition context to build the frozen primary task;
4. has auditable provenance and a usable research license;
5. can remain label-sealed until model, source policy and analysis are frozen;
6. passes exact-hash and source-overlap checks against the development registry.

## Candidate matrix

| Candidate | Independence | Outcome quality | Context/OOD value | License/access | Contamination risk | Decision |
|---|---|---|---|---|---|---|
| Ha et al. JACS 2025 Pd C–N HTE, Figshare 28215923 | High relative to current Phase 4 pools; exact hash audit still required | Real HTE yields; reaction CSV before/after deduplication | Pharmaceutical Pd C–N conditions; useful external source shift | MIT; CSV; 1.24 GB full deposit | Medium because later BH collections may include it | Primary candidate; register metadata only |
| Das et al. JACS 2026 50,688 C–N uHTE, ORD | New campaign, but ORD has been used elsewhere in this project | Dense calibrated UPLC-MS assay yields and controls | Strong condition/metal OOD; only two substrate pairs | Article is open; machine-readable data in ORD; exact data license must be recorded | Medium-high: reject any overlapping ORD record/hash | Secondary candidate after exact ORD record isolation |
| Neves et al. BH-HTE-OOD JnJ subset | JnJ campaign is new; unified table also merges public datasets already used by HiTEA | Real industrial HTE with source labels | Explicit source and substrate OOD design | Apache-2.0 repository/CSV | High unless only source-provenance JnJ rows are isolated before labels are exposed | Validation candidate, JnJ-only custodian extraction |
| Existing HiTEA / RegioSQM20 / NiCOlit / Phase 4 pools / USPTO / current ORD cache | None | Mixed | Already used for method design | Existing local assets | Certain contamination | Forbidden for confirmatory Phase E |

## Selected path

1. Register Figshare 28215923 as `jacs2025_pdcn_external` without downloading labels.
2. Obtain provider file metadata and checksums.
3. Ask an independent custodian to create:
   - a label-free reaction/context pool;
   - a sealed label artifact;
   - a receipt containing only label digest, schema digest and row count.
4. Freeze the Phase D checkpoint/commit and Statistical Analysis Plan.
5. Build a forbidden-artifact index from all development datasets and fixed pools.
6. Create `sealed_test_manifest.json` with `pc_cng.phase_e_sealed_contract`.
7. Run verification before a single formal evaluation.

The current developer cannot truthfully self-certify `labels_unseen_before_model_freeze`
after opening a labelled CSV. If independent custody is unavailable, the result must
be labelled external evaluation, not blind confirmatory evidence.

## Frozen primary endpoint proposal

`real-HTE condition-feasibility source-macro AUPRC`

Primary comparisons:

1. learned source gate versus validation-selected best single source;
2. learned source gate versus uniform union.

Both use paired experimental-cluster inference, the same one-negative-per-parent
budget and the pre-frozen two-backbone panel. Calibration is a safety co-endpoint,
not a substitute for external utility.
