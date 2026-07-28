# Phase C / G8-C analysis-plan amendment 1

Status: `FROZEN_AFTER_ENGINEERING_PILOT_BEFORE_FULL_FORMAL_RUN`  
Date: 2026-07-29

## Why this amendment exists

The capped GPU engineering pilot completed the fail-closed Stage 1 to Stage 4
path and artifact contract. As declared in analysis plan v1, pilot metrics are
development-only. The pilot also showed that the risk head requires explicit
post-hoc calibration before it can be evaluated as a calibrated false-negative
risk estimator.

This amendment does not change the FNR ECE endpoint or its threshold.

## Frozen calibration protocol

- The HTE validation risk examples are partitioned by complete experimental
  group using SHA-256 key `phase_c_risk_calibration_v1`.
- One group-disjoint half is used only to select a scalar temperature.
- The other group-disjoint half is used only once to report ECE, Brier score
  and AUPRC.
- Temperature is selected by minimum binary NLL over 91 fixed log-spaced
  values from 0.5 to 5.0.
- Both partitions must contain both labels; otherwise formal evaluation fails.
- The primary ECE remains 10-bin ECE on the independent evaluation half, with
  the unchanged threshold `ECE <= 0.15`.
- Raw, uncalibrated ECE is reported alongside calibrated ECE.
- Source test rows remain sealed.

## Full-run configuration frozen here

- GPU: an available NVIDIA A100 selected at launch.
- Hidden dimension: 128.
- Attention heads: 4.
- Transformer layers: 3.
- Epochs per stage: 3.
- Stage rounds: 1.
- Batch size: 16.
- Candidate budget: 8.
- Formal edit validation: up to 512 seeded validation reactions.
- Formal candidate validation: up to 128 seeded validation reactions.
- Formal risk validation: up to 2,048 seeded validation examples.
- Formal reward validation: up to 256 seeded validation pairs.
- Seed: 20260724.

No full-run hyperparameter, endpoint or threshold will be changed after full
formal metrics are read. A failed endpoint remains a failed endpoint.

