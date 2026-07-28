# Phase F / G7 expert pilot v2 protocol

Status: technical protocol ready; real expert responses and independently
verified obvious-negative controls remain external dependencies.

## Why the historical pilot cannot be used

The historical builder is retained for audit, but its materials are deprecated
for scientific evidence because it:

- prepared only two pilot reviewer forms;
- could replace observed real negatives with random-corruption proxies;
- could emit empty reaction contexts when prediction rows did not map back to
  candidates;
- showed candidate products without guaranteed complete parent reaction and
  condition context.

No old expert result is deleted or rewritten. Its status remains `DEFERRED`.

## v2 design

- 80 items, 10 in each of eight strata:
  - independently verified positive control;
  - independently verified obvious negative control;
  - random mismatch;
  - rule PC-CNG;
  - learned structured;
  - shuffled real;
  - uniform union;
  - learned source gate.
- At least three independent chemistry experts.
- Every source stratum uses a different complete-case parent reaction.
- All generated candidates come from the same Phase D six-source cache.
- Gate-selected items use the frozen policy map.
- Review forms expose reactants, conditions, candidate product and reaction
  family, but hide source, FNR, model scores and observed labels.
- Each reviewer receives all 80 items in an independently randomized order.
- Proxy negative controls, blank reaction contexts and fewer than three
  reviewers fail closed.

## Reviewer task

Each item uses six 1–5 ratings:

1. structural validity;
2. mechanistic plausibility;
3. plausible competing outcome;
4. likely low-yield or failure;
5. false-negative risk;
6. reviewer confidence.

One reason code and optional notes are allowed. The form is intentionally short
enough for a target completion time of at most three minutes per item.

## Pilot exit

The main review is not launched unless:

- weighted kappa or ordinal Krippendorff alpha is at least 0.5, target 0.6;
- positive and obvious-negative controls are significantly discriminated;
- no severe reviewer drift is present;
- source comparisons use the same frozen analysis and disclose all results.

Prepared forms do not count as expert validation. G7 remains `DEFERRED` until
real completed forms from at least three independent experts are returned and
analyzed.
