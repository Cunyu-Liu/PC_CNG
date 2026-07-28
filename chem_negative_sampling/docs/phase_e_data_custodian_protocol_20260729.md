# Phase E independent data-custodian protocol

## Purpose

Keep external test outcomes inaccessible to the development team until the
model checkpoint, source policy, primary endpoint and analysis code have been
frozen. The custodian must be independent of model/threshold selection.

## Inputs

- Candidate registration JSON from `chem_negative_sampling/docs/`.
- Development forbidden-artifact index.
- Provider dataset downloaded directly from the registered source.
- Frozen feature schema supplied by the development team.

## Custodian procedure

1. Verify the provider URL, dataset version, license and provider checksum.
2. Record a SHA256 digest of the unmodified provider artifact.
3. Apply only the pre-specified row inclusion and schema normalization rules.
4. Isolate the registered source. For the JnJ candidate, retain only dedicated
   JnJ production rows and exclude every merged public source.
5. Compute exact reaction/context hashes and reject any hash present in the
   development forbidden-artifact index.
6. Split the eligible table into:
   - a label-free context pool containing stable row IDs, reactants, reagents,
     catalysts, solvents, temperature/time and candidate/target structures;
   - a sealed label artifact keyed only by the same stable row IDs.
7. Do not send, display or summarize label values, prevalence, yield
   distribution, class balance or model metrics to the development team.
8. Send only the label-free pool and a signed receipt with the fields below.
9. Retain the sealed label artifact until the model and analysis freeze receipt
   is returned.
10. Unseal once for the frozen evaluator and preserve all outputs.

## Required label receipt

```json
{
  "dataset_id": "registered dataset id",
  "sealed_label_sha256": "64 lowercase hex characters",
  "label_schema_sha256": "64 lowercase hex characters",
  "n_rows": 1,
  "custodian": "independent person or service identifier",
  "labels_never_exposed_to_development": true,
  "created_at_utc": "ISO-8601 timestamp"
}
```

The receipt must not contain a label path, individual label, class prevalence,
yield summary or threshold-dependent statistic.

## Formal handoff

The development team returns:

- frozen model/checkpoint SHA256;
- frozen source-policy checkpoint SHA256;
- Git commit;
- frozen Statistical Analysis Plan SHA256;
- label-free evaluation-pool SHA256;
- `sealed_test_manifest.json` verified by
  `pc_cng.phase_e_sealed_contract verify`.

The custodian then runs, or permits a designated evaluator to run, exactly one
formal command. Any second run, changed endpoint or changed threshold is a new
study and must use a new sealed dataset.

## Fail-closed rules

- No independent custodian: report external evaluation, not blind
  confirmatory evidence.
- Any overlap/hash collision: reject the affected rows before unsealing.
- Any missing context required by the frozen task: exclude by the frozen rule,
  never by observed outcome.
- Any mismatch between receipt and artifact hash: abort.
- Any request for prevalence or preliminary metric before freeze: refuse and
  record the request.
