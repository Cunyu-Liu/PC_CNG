#!/usr/bin/env python3
"""Independent script to reconstruct paired cluster CI from saved scored records.

Exit Criterion: "primary endpoint 的 paired cluster CI 可由独立脚本重建"

This script:
1. Loads scored_records_v2.json (saved by p4_g6_hte_eval_v2.py)
2. Loads go_no_go_v2.json (the original evaluation result)
3. Recomputes paired cluster bootstrap CI for the primary endpoint
4. Compares the reconstructed CI with the original CI

If the CIs match (within bootstrap Monte Carlo error), the Exit Criterion is satisfied.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Add the package to path
REPO_ROOT = Path("/home/cunyuliu/pc_cng_research")
sys.path.insert(0, str(REPO_ROOT / "chem_negative_sampling"))

from pc_cng.paired_cluster_inference import (
    auprc_metric,
    macro_auprc_metric,
    paired_cluster_bootstrap,
)


def reconstruct_primary_ci(scored_file: Path, result_file: Path) -> dict:
    """Reconstruct the primary endpoint paired cluster CI from saved scored records.

    Primary endpoint: T5_condition_feasibility_macro_auprc
    Comparison: B4 (observed+synthetic) vs B0 (positive_only)
    """
    # Load saved data
    with open(scored_file) as f:
        scored_data = json.load(f)

    with open(result_file) as f:
        result = json.load(f)

    test_records = scored_data["test_records"]
    arm_scored = scored_data["arm_scored_by_task"]

    primary_metric = result["primary_endpoint"]
    print(f"Primary endpoint: {primary_metric}")
    print(f"Test records: {len(test_records)}")
    print(f"Arms with scored data: {list(arm_scored.keys())}")

    # Reconstruct T5 scored records for B4 (challenger) and B0 (baseline)
    # The scored records have 'score' field set by the task head
    b0_t5 = arm_scored.get("B0", {}).get("T5", [])
    b4_t5 = arm_scored.get("B4", {}).get("T5", [])

    # Ensure labels are set correctly for T5 (high yield = positive)
    for r in b0_t5:
        r["label"] = 1 if float(r.get("measured_yield", 0)) >= 50.0 else 0
    for r in b4_t5:
        r["label"] = 1 if float(r.get("measured_yield", 0)) >= 50.0 else 0

    print(f"B0 T5 records: {len(b0_t5)}")
    print(f"B4 T5 records: {len(b4_t5)}")

    # Choose metric function based on primary endpoint
    if "macro_auprc" in primary_metric:
        metric_fn = macro_auprc_metric
    elif "auprc" in primary_metric:
        metric_fn = auprc_metric
    else:
        raise ValueError(f"Unknown metric: {primary_metric}")

    # Reconstruct paired cluster bootstrap CI
    n_bootstrap = result.get("n_bootstrap", 2000)
    ci_reconstructed = paired_cluster_bootstrap(
        challenger_records=b4_t5,
        baseline_records=b0_t5,
        metric_fn=metric_fn,
        cluster_key="experimental_group",
        n_bootstrap=n_bootstrap,
        seed=20260723,  # Same seed as original
    )

    # Get original CI from result
    ci_key = f"B4_vs_B0_{primary_metric}"
    ci_original = result.get("paired_cis", {}).get(ci_key, {})

    # Compare
    print(f"\n=== CI Comparison for {ci_key} ===")
    print(f"  Original:      delta_mean={ci_original.get('delta_mean')}, CI=[{ci_original.get('delta_ci_low')}, {ci_original.get('delta_ci_high')}]")
    print(f"  Reconstructed: delta_mean={ci_reconstructed['delta_mean']}, CI=[{ci_reconstructed['delta_ci_low']}, {ci_reconstructed['delta_ci_high']}]")

    # Check if they match (within Monte Carlo error)
    delta_diff = abs(ci_reconstructed["delta_mean"] - ci_original.get("delta_mean", 0))
    ci_low_diff = abs(ci_reconstructed["delta_ci_low"] - ci_original.get("delta_ci_low", 0))
    ci_high_diff = abs(ci_reconstructed["delta_ci_high"] - ci_original.get("delta_ci_high", 0))

    # With same seed, they should match exactly
    exact_match = (delta_diff < 1e-10 and ci_low_diff < 1e-10 and ci_high_diff < 1e-10)
    close_match = (delta_diff < 0.01 and ci_low_diff < 0.01 and ci_high_diff < 0.01)

    print(f"\n  delta_mean diff: {delta_diff}")
    print(f"  ci_low diff: {ci_low_diff}")
    print(f"  ci_high diff: {ci_high_diff}")
    print(f"  Exact match (same seed): {exact_match}")
    print(f"  Close match (<0.01): {close_match}")

    # Also reconstruct B4 vs B3
    b3_t5 = arm_scored.get("B3", {}).get("T5", [])
    for r in b3_t5:
        r["label"] = 1 if float(r.get("measured_yield", 0)) >= 50.0 else 0

    if b3_t5:
        ci_b4_b3 = paired_cluster_bootstrap(
            challenger_records=b4_t5,
            baseline_records=b3_t5,
            metric_fn=metric_fn,
            cluster_key="experimental_group",
            n_bootstrap=n_bootstrap,
            seed=20260723,
        )
        ci_b4_b3_key = "B4_vs_B3_T5_condition_feasibility_macro_auprc"
        ci_b4_b3_orig = result.get("paired_cis", {}).get(ci_b4_b3_key, {})
        print(f"\n=== CI Comparison for {ci_b4_b3_key} ===")
        print(f"  Original:      delta_mean={ci_b4_b3_orig.get('delta_mean')}, CI=[{ci_b4_b3_orig.get('delta_ci_low')}, {ci_b4_b3_orig.get('delta_ci_high')}]")
        print(f"  Reconstructed: delta_mean={ci_b4_b3['delta_mean']}, CI=[{ci_b4_b3['delta_ci_low']}, {ci_b4_b3['delta_ci_high']}]")

    # Exit Criterion verdict
    print(f"\n=== Exit Criterion ===")
    if exact_match:
        print("PASS: Paired cluster CI reconstructed EXACTLY by independent script (same seed)")
    elif close_match:
        print("PASS: Paired cluster CI reconstructed within tolerance by independent script")
    else:
        print("FAIL: Reconstructed CI does not match original")

    return {
        "exact_match": exact_match,
        "close_match": close_match,
        "ci_reconstructed": ci_reconstructed,
        "ci_original": ci_original,
    }


if __name__ == "__main__":
    output_dir = Path("/home/cunyuliu/pc_cng_research/results/p4_hte_external_validation_v2")
    scored_file = output_dir / "scored_records_v2.json"
    result_file = output_dir / "go_no_go_v2.json"

    if not scored_file.exists():
        print(f"ERROR: {scored_file} not found. Run G6 v2 evaluation first.")
        sys.exit(1)
    if not result_file.exists():
        print(f"ERROR: {result_file} not found. Run G6 v2 evaluation first.")
        sys.exit(1)

    reconstruct_primary_ci(scored_file, result_file)
