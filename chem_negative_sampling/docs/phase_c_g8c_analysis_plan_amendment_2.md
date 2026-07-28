# Phase C / G8-C analysis-plan amendment 2

Status: `FROZEN_BEFORE_V2_UNSEEN_HOLDOUT_RUN`  
Date: 2026-07-29

## Preserved v1 result

The first full formal run at commit `91dfdf7` is permanently retained as
`FORMAL_SOURCE_EXPERT_NO_GO`:

- edit-locus accuracy: 0.069;
- edit-type accuracy: 0.000;
- valid edit rate: 1.000;
- candidate coverage: 1.000;
- calibrated FNR ECE: 0.113;
- expert labels available: 0.

No threshold or v1 result is changed.

## Root cause established on development data

Real reconstruction actions and counterfactual proposal actions were forced
through one locus/type head even though their label semantics conflict. Real
targets are dominated by `BOND_FORM`; rule targets are dominated by
`NOT_APPLICABLE` and generative edits. Rule imitation therefore overwrote
reconstruction.

A development-only run on the already consumed source validation split showed
that separating the heads and rehearsing real edits removes this semantic
overwrite. Those development metrics are not confirmatory evidence.

The risk path also discarded most valid labels because it reused an action
featurizer that requires a formed bond. Risk supervision now uses a dedicated
product-graph path, covers all parseable candidate outcomes and uses
class-balanced BCE over the full risk pool.

## Frozen v2 method changes

- Separate reconstruction and proposal locus/type heads.
- Stage 1 trains the reconstruction head on real bond-form, bond-break and
  bond-order-change actions.
- Stages 2 to 4 retain a fixed-weight (0.25) real-edit rehearsal loss.
- Rehearsal examples are prevalidated once before each stage; a missing pool is
  fatal.
- Stage 3 iterates over the larger of the competing-pair and risk pools so all
  risk examples are covered.
- Risk examples use a product-graph featurizer that does not require a formed
  bond.
- Risk BCE uses inverse class-frequency weights fixed from the training pool.

## New holdout contract

The v1 source validation split is now development-only and excluded entirely
from v2 training and evaluation.

The v2 holdout is created only from source-training groups that had not
previously been evaluated as a separate endpoint:

- reactions: `phase_c_v2_reaction_holdout_v1`;
- same-context pairs: `phase_c_v2_pair_holdout_v1`;
- risk experimental groups: `phase_c_v2_risk_holdout_v1`.

Each uses SHA-256 group hashing with a fixed 80/20 assignment. Preflight before
metrics found:

- 582 train / 137 holdout reactions;
- 81 train / 15 holdout same-context preference pairs;
- 758 train / 165 holdout risk examples after featurization;
- holdout risk labels: 93 positive / 72 negative.

Source test rows remain sealed.

## Unchanged endpoints and thresholds

All v1 validation thresholds remain unchanged. The full-run model
hyperparameters, seed, candidate budget and independent temperature-calibration
protocol from amendment 1 also remain unchanged.

The v2 holdout is evaluated once. Failure is retained and cannot trigger
further tuning on this holdout.

