# Phase C expert-label provenance hardening amendment

Date: 2026-07-29
Status: `VERIFIED_GPU_REGRESSION`

## Scope

This amendment hardens the G8-C Phase C expert-risk ingestion boundary. It
changes neither the learned-generator architecture, frozen v2 data partition,
thresholds, seed, candidate budget, nor the one-shot v2 holdout result.

## Provenance rule

A future row may enter `risk_source="expert_label"` only if it contains all
of the following non-empty fields:

- `sample_id`;
- `reviewer_id`;
- `review_timestamp`;
- `candidate_reaction`;
- a valid numeric feasibility score or a recognized human verdict.

Rows without complete provenance are excluded. They are never converted to
synthetic labels, assigned a default reviewer, or treated as a negative.

## Current-data audit

The two current reviewer forms contain 1,000 offered review rows. At audit
time, zero rows had a review timestamp and zero rows had a feasibility score
or verdict. Therefore the current accepted expert-risk count is exactly zero.
The formal v2 status remains
`FORMAL_SOURCE_EXPERT_PARTIAL_EXPERT_LABELS_PENDING`.

## Verification

- CUDA probe ran in the `pc_cng_gpu` environment on physical GPU 6.
- The focused provenance tests passed: 2/2.
- The complete G8-C Phase C regression suite passed: 27/27.
- The live accepted expert-risk count remained zero after the hardening check.

## Evidence boundary

This is an engineering provenance safeguard. It does not add real expert
evidence, does not rerun the consumed v2 holdout, and does not support any
source-superiority claim.
