#!/usr/bin/env python3
"""Compute ALL pairwise CIs for the primary endpoint (T5_condition_feasibility_macro_auprc).

Computes paired cluster bootstrap CIs for all C(6,2)=15 pairs of arms
(B0, B1, B2, B3, B4, B5) using family_macro_auprc_metric (macro-averaged AUPRC
across reaction families), matching the G6 primary-endpoint definition.

For each pair, the T5 label is set explicitly as:
    label = 1 if measured_yield >= 50.0 else 0
matching the original T5 label definition.
"""
from __future__ import annotations

import json
import sys
import time
from itertools import combinations
from pathlib import Path

REPO_ROOT = Path("/home/cunyuliu/pc_cng_research")
sys.path.insert(0, str(REPO_ROOT / "chem_negative_sampling"))

from pc_cng.paired_cluster_inference import (  # noqa: E402
    family_macro_auprc_metric,
    paired_cluster_bootstrap,
)

PRIMARY_ENDPOINT = "T5_condition_feasibility_macro_auprc"
ARMS = ["B0", "B1", "B2", "B3", "B4", "B5"]
CLUSTER_KEY = "experimental_group"
N_BOOTSTRAP = 2000
SEED = 20260723
YIELD_THRESHOLD = 50.0


def set_t5_labels(records: list[dict]) -> list[dict]:
    """Set label = 1 if measured_yield >= 50.0 else 0 (matching T5 label definition)."""
    for r in records:
        r["label"] = 1 if float(r.get("measured_yield", 0)) >= YIELD_THRESHOLD else 0
    return records


def main() -> None:
    scored_file = (
        REPO_ROOT
        / "results"
        / "p4_hte_external_validation_v2"
        / "scored_records_v2.json"
    )
    if not scored_file.exists():
        print(f"ERROR: {scored_file} not found.", file=sys.stderr)
        sys.exit(1)

    with open(scored_file) as f:
        scored_data = json.load(f)

    arm_scored = scored_data["arm_scored_by_task"]

    # Load T5 records for each arm; shallow-copy so we don't mutate saved data,
    # then set labels explicitly.
    arm_t5: dict[str, list[dict]] = {}
    print(f"Loading T5 scored records (label = 1 if measured_yield >= {YIELD_THRESHOLD} else 0)")
    for arm in ARMS:
        recs = arm_scored.get(arm, {}).get("T5", [])
        recs = [dict(r) for r in recs]
        set_t5_labels(recs)
        arm_t5[arm] = recs
        n_pos = sum(1 for r in recs if r["label"] == 1)
        print(f"  {arm}: {len(recs)} records ({n_pos} positive)")

    # Sanity check: all arms must have equal record counts (paired design)
    counts = {arm: len(recs) for arm, recs in arm_t5.items()}
    assert len(set(counts.values())) == 1, f"Unequal record counts: {counts}"

    # Compute all pairwise CIs (challenger - baseline)
    results: dict[str, dict] = {}
    pairs = list(combinations(ARMS, 2))
    print(f"\n=== Pairwise CIs for {PRIMARY_ENDPOINT} ===")
    print(f"metric_fn=family_macro_auprc_metric  cluster_key={CLUSTER_KEY}  "
          f"n_bootstrap={N_BOOTSTRAP}  seed={SEED}")
    header = (
        f"{'Challenger':>11} vs {'Baseline':<9}  "
        f"{'ch_point':>10}  {'bl_point':>10}  {'delta':>10}  "
        f"{'CI_low':>10}  {'CI_high':>10}  {'all_pos':>7}  {'p_value':>8}"
    )
    print(header)
    print("-" * len(header))

    t0 = time.time()
    for i, (challenger, baseline) in enumerate(pairs, 1):
        ci = paired_cluster_bootstrap(
            challenger_records=arm_t5[challenger],
            baseline_records=arm_t5[baseline],
            metric_fn=family_macro_auprc_metric,
            cluster_key=CLUSTER_KEY,
            n_bootstrap=N_BOOTSTRAP,
            seed=SEED,
        )
        key = f"{challenger}_vs_{baseline}_{PRIMARY_ENDPOINT}"
        results[key] = ci
        print(
            f"{challenger:>11} vs {baseline:<9}  "
            f"{ci['challenger_point']:>10.6f}  {ci['baseline_point']:>10.6f}  "
            f"{ci['delta_mean']:>10.6f}  {ci['delta_ci_low']:>10.6f}  "
            f"{ci['delta_ci_high']:>10.6f}  {str(ci['ci_all_positive']):>7}  "
            f"{ci['p_value']:>8.4f}"
        )
        elapsed = time.time() - t0
        print(f"    [{i}/{len(pairs)}] elapsed={elapsed:.1f}s", flush=True)

    # Save results to JSON
    output_file = (
        REPO_ROOT
        / "results"
        / "p4_hte_external_validation_v2"
        / "all_pairwise_ci_primary.json"
    )
    payload = {
        "primary_endpoint": PRIMARY_ENDPOINT,
        "metric_fn": "family_macro_auprc_metric",
        "cluster_key": CLUSTER_KEY,
        "n_bootstrap": N_BOOTSTRAP,
        "seed": SEED,
        "label_definition": f"label = 1 if measured_yield >= {YIELD_THRESHOLD} else 0",
        "arms": ARMS,
        "n_pairs": len(results),
        "pairwise_cis": results,
    }
    with open(output_file, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved {len(results)} pairwise CIs to {output_file}")

    # Summary of key comparisons
    print("\n=== Key comparisons ===")
    b1_b0 = results.get(f"B1_vs_B0_{PRIMARY_ENDPOINT}", {})
    b1_b2 = results.get(f"B1_vs_B2_{PRIMARY_ENDPOINT}", {})
    b4_b0 = results.get(f"B4_vs_B0_{PRIMARY_ENDPOINT}", {})
    print(
        f"B1 vs B0 (primary, PC-CNG vs no-neg): "
        f"delta={b1_b0.get('delta_mean')}, CI=[{b1_b0.get('delta_ci_low')}, "
        f"{b1_b0.get('delta_ci_high')}], all_positive={b1_b0.get('ci_all_positive')}, "
        f"p={b1_b0.get('p_value')}"
    )
    print(
        f"B1 vs B2 (PC-CNG vs random):          "
        f"delta={b1_b2.get('delta_mean')}, CI=[{b1_b2.get('delta_ci_low')}, "
        f"{b1_b2.get('delta_ci_high')}], all_positive={b1_b2.get('ci_all_positive')}, "
        f"p={b1_b2.get('p_value')}"
    )
    print(
        f"B4 vs B0 (secondary, obs+PC-CNG):     "
        f"delta={b4_b0.get('delta_mean')}, CI=[{b4_b0.get('delta_ci_low')}, "
        f"{b4_b0.get('delta_ci_high')}], all_positive={b4_b0.get('ci_all_positive')}, "
        f"p={b4_b0.get('p_value')}"
    )


if __name__ == "__main__":
    main()
