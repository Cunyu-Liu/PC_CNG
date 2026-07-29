# Phase B G6 corrected reanalysis artifacts

This directory contains the small, version-control-safe artifacts from the
authoritative corrected GPU rerun at commit
`080236bd11893e7fccf0a7f3a173250c11a404ff`.

Scientific status:

- `CORRECTED_REANALYSIS_TEST_OUTCOMES_PREVIOUSLY_OBSERVED`
- `confirmatory_blind_test=false`
- benchmark engineering is auditable;
- PC-CNG superiority is not supported;
- only one `source_publication` slice is evaluable.

The 168 MB `predictions_t5_v3.json` stays on `/mnt` and is represented here by
its SHA-256 in `run_manifest_v3.json` and the final artifact manifest. This
avoids placing a large generated prediction file in Git while preserving an
immutable integrity check.

Files:

- `formal_result_v3.json`: complete formal result, secondary diagnostics and
  primary inference.
- `run_manifest_v3.json`: exact command, inputs, code hashes, runtime policy and
  output hashes.
- `excluded_reaction_context_records_v3.json`: 96 fail-closed context
  exclusions.
- `independent_primary_inference_v3_080236b.json`: separate-process
  reconstruction from the frozen prediction artifact.
- `reconstruction_verification_v3_080236b.json`: complete primary-object
  equality check.
- `g6_v3_inference_operating_characteristics_080236b_20260729.json`:
  post-implementation design check, not efficacy evidence.
- `pip_freeze_080236b.txt`, `conda_explicit_080236b.txt` and
  `environment_provenance_080236b.txt`: environment evidence.
- `artifact_manifest_080236b.json`: hashes and remote canonical locations.

The independent reconstruction is process-independent but intentionally
reuses the frozen repository inference library. It verifies artifact
reproducibility, not an independent reimplementation of the statistics.
