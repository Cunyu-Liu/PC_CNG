# G6 v3 formal benchmark runbook

## Scope and scientific status

`p4_g6_benchmark_v3.py` replaces the historical G6 v2 evaluation.  The v2
artifacts remain auditable but are not eligible for headline claims: they used
product-only proxies, approximate ordinal/ranking tasks, a non-paired delta
interval and confounded source arms.

The v3 benchmark is eligible for formal analysis only with the frozen plan in
`docs/phase_b_g6_analysis_plan_v1.json`, CUDA, the frozen Chemformer
checkpoint, five seeds, 2,000 paired-cluster bootstrap draws and 10,000
paired-cluster permutations.  A `--smoke` output is integration evidence only
and carries `SMOKE_NOT_SCIENTIFIC` status.

## Frozen primary comparison

Primary endpoint:

```text
T5 condition-feasibility source-publication-macro AUPRC
```

Primary comparisons, in their fixed Holm-corrected order:

1. `pc_cng` (the current **rule** PC-CNG source) versus `random`;
2. `pc_cng` versus `template_rule`;
3. deterministic `union` versus `pc_cng`.

Each source arm uses the same training parents, one positive and one negative
per parent, the same frozen reaction encoder and the same optimizer schedule.
The runner refuses test-driven baseline selection.  `pc_cng` must never be
described as a learned generator in a v3 result because the frozen manifest
maps it to `rule_pc_cng`.

The test split must retain at least 20 pre-existing `experimental_group`
(plate) clusters and at least 10 clusters containing both endpoint classes.
The current HTE table satisfies this with 48 held-out plate clusters.  This
check prevents a one-cluster bootstrap from being presented as a valid
uncertainty interval.

The current HTE table has one evaluable `source_publication` slice.  The
reported endpoint is therefore a single-source external HTE AUPRC, not a
cross-publication replication.  The runner records this scope explicitly.

## Input and task contract

All tasks use the same frozen reaction-conditioned Chemformer encoder with
separate, explicit fields for reactants, catalysts, solvents, reagents,
candidate product, temperature and reaction time.  Records that lack
reactants or products are explicitly listed in
`excluded_reaction_context_records_v3.json`; they are never silently converted
to product-only input.

- T1: `<10%` yield binary endpoint.
- T2: cumulative-link ordinal model over the frozen yield bins.
- T3: reaction-conditioned yield regression.
- T4: within-plate pairwise logistic ranking loss and plate NDCG.
- T5: condition-feasibility probability, with temperature scaling fit only on
  the validation split and then applied unchanged to test predictions.

The product-only fingerprint code remains a legacy weak baseline and is not
part of the v3 headline model.

## Server runtime contract

On the current server, Conda's `libstdc++` must take precedence while creating
the PyTorch optimizer.  This is injected per command, never exported globally:

```bash
cd /home/cunyuliu/pc_cng_research/chem_negative_sampling
env CUDA_VISIBLE_DEVICES=<GPU> \
    LD_LIBRARY_PATH=/home/cunyuliu/miniconda3/envs/pc_cng/lib \
    /home/cunyuliu/miniconda3/envs/pc_cng/bin/python -m pc_cng.run_p4_g6_v3 ...
```

`pc_cng` and the shared `models` package are both included in the project
wheel. The documented module entrypoint is regression-tested from the
`chem_negative_sampling/` working directory without adding the repository
parent to `PYTHONPATH`. Install the `chem` extra for the formal runner; it
includes the direct Parquet dependency `pyarrow`.

All new result writers use strict JSON (`allow_nan=False`). When between-seed
variance cannot be estimated, such as a one-seed integration smoke, the
standardized seed effect is `null`; it must never be serialized as
`Infinity`, `NaN` or another non-standard JSON token.

Holm adjusted p-values use the standard step-down cumulative maximum of the
ordered Bonferroni products. Rejection stops after the first non-rejection.
The regression suite includes a case where the unmodified ordered products
would decrease.

Primary AUPRC is computed by `sklearn.metrics.average_precision_score`.
This threshold-based implementation is invariant to record ordering when
scores are tied; the regression suite checks that invariant explicitly.

The formal run fails closed if any tracked file differs from `HEAD` or if its
output directory is non-empty. User-owned untracked files do not invalidate
the tracked-code identity. The run manifest records the exact process argv,
input, code and output hashes, git commit, runtime versions, CUDA device and
the clean-tracked-worktree check.

The formal run must use a fresh, immutable output directory.  The result
directory must contain `formal_result_v3.json`, `predictions_t5_v3.json`,
`run_manifest_v3.json` and `excluded_reaction_context_records_v3.json`.
Use `pc_cng.recompute_p4_g6_v3` to reconstruct the primary inference from the
predictions and frozen analysis plan in a separate command.

Retain the simulation artifact from `pc_cng.simulate_p4_g6_v3`. It reports
family-wise type-I error for the three Holm-corrected comparisons using an
exchangeable but non-identical paired null, and power for the first frozen
comparison at 0.02, 0.04 and 0.08 synthetic deltas. Binomial rates are
accompanied by 95% Wilson intervals. It is an implementation design check,
never a PC-CNG efficacy result. A simulation rerun performed during completion
re-audit does not retroactively preregister or strengthen the already executed
efficacy test.

## Stop rules

Stop and preserve the output if any of the following occurs:

- non-CUDA training path or unavailable frozen checkpoint/vocabulary;
- a missing reactant/product record is silently retained;
- source parent/budget matching fails;
- validation calibration uses test labels;
- prediction alignment differs between paired arms/seeds;
- the independent inference reconstruction differs from the formal result.

No outcome from this benchmark can establish cross-publication or second-scorer
replication.  Those require a separate sealed dataset and scorer protocol.
