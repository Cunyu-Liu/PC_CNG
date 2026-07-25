#!/usr/bin/env python3
"""Independent script to reconstruct paired cluster CI from saved scored records.

Exit Criterion: "primary endpoint 的 paired cluster CI 可由独立脚本重建"

This script:
1. Loads scored_records_v2.json (saved by p4_g6_hte_eval_v2.py)
2. Loads go_no_go_v2.json (the evaluation result)
3. Reads the `primary_comparison` field (e.g. "B1_vs_B0") to determine which
   comparison to reconstruct. Falls back to "B1_vs_B0" if absent.
4. Recomputes paired cluster bootstrap CI for the primary endpoint.
5. Compares the reconstructed CI with the stored CI in `paired_cis`.

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
    family_macro_auprc_metric,
    macro_auprc_metric,
    paired_cluster_bootstrap,
)

PRIMARY_ENDPOINT = "T5_condition_feasibility_macro_auprc"
YIELD_THRESHOLD = 50.0


def set_t5_labels(records: list[dict]) -> list[dict]:
    """Set label = 1 if measured_yield >= 50.0 else 0 (matching T5 label definition)."""
    for r in records:
        r["label"] = 1 if float(r.get("measured_yield", 0)) >= YIELD_THRESHOLD else 0
    return records


def parse_comparison(comparison: str) -> tuple[str, str]:
    """Parse 'B1_vs_B0' -> ('B1', 'B0')."""
    parts = comparison.split("_vs_")
    if len(parts) != 2:
        raise ValueError(f"Cannot parse comparison: {comparison}")
    return parts[0], parts[1]


def reconstruct_one_ci(
    arm_scored: dict,
    challenger: str,
    baseline: str,
    primary_metric: str,
    n_bootstrap: int,
    seed: int = 20260723,
) -> dict:
    """Reconstruct paired cluster bootstrap CI for one (challenger, baseline) pair."""
    ch_t5 = arm_scored.get(challenger, {}).get("T5", [])
    bl_t5 = arm_scored.get(baseline, {}).get("T5", [])

    # Shallow-copy so we don't mutate saved data, then set labels explicitly.
    ch_t5 = [dict(r) for r in ch_t5]
    bl_t5 = [dict(r) for r in bl_t5]
    set_t5_labels(ch_t5)
    set_t5_labels(bl_t5)

    print(f"{challenger} T5 records: {len(ch_t5)}")
    print(f"{baseline} T5 records: {len(bl_t5)}")

    # Choose metric function based on primary endpoint
    if "macro_auprc" in primary_metric:
        metric_fn = family_macro_auprc_metric
    elif "auprc" in primary_metric:
        metric_fn = auprc_metric
    else:
        raise ValueError(f"Unknown metric: {primary_metric}")

    ci_reconstructed = paired_cluster_bootstrap(
        challenger_records=ch_t5,
        baseline_records=bl_t5,
        metric_fn=metric_fn,
        cluster_key="experimental_group",
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    return ci_reconstructed


def compare_ci(
    label: str,
    ci_reconstructed: dict,
    ci_original: dict,
) -> dict:
    """Compare reconstructed CI with original CI."""
    print(f"\n=== CI Comparison for {label} ===")
    print(
        f"  Original:      delta_mean={ci_original.get('delta_mean')}, "
        f"CI=[{ci_original.get('delta_ci_low')}, {ci_original.get('delta_ci_high')}], "
        f"all_positive={ci_original.get('ci_all_positive')}"
    )
    print(
        f"  Reconstructed: delta_mean={ci_reconstructed['delta_mean']}, "
        f"CI=[{ci_reconstructed['delta_ci_low']}, {ci_reconstructed['delta_ci_high']}], "
        f"all_positive={ci_reconstructed['ci_all_positive']}"
    )

    delta_diff = abs(ci_reconstructed["delta_mean"] - ci_original.get("delta_mean", 0))
    ci_low_diff = abs(ci_reconstructed["delta_ci_low"] - ci_original.get("delta_ci_low", 0))
    ci_high_diff = abs(ci_reconstructed["delta_ci_high"] - ci_original.get("delta_ci_high", 0))

    # With same seed, they should match exactly
    exact_match = delta_diff < 1e-10 and ci_low_diff < 1e-10 and ci_high_diff < 1e-10
    close_match = delta_diff < 0.01 and ci_low_diff < 0.01 and ci_high_diff < 0.01

    print(f"\n  delta_mean diff: {delta_diff}")
    print(f"  ci_low diff: {ci_low_diff}")
    print(f"  ci_high diff: {ci_high_diff}")
    print(f"  Exact match (same seed): {exact_match}")
    print(f"  Close match (<0.01): {close_match}")

    return {
        "label": label,
        "exact_match": exact_match,
        "close_match": close_match,
        "ci_reconstructed": ci_reconstructed,
        "ci_original": ci_original,
    }


def reconstruct_primary_ci(scored_file: Path, result_file: Path) -> dict:
    """Reconstruct the primary endpoint paired cluster CI from saved scored records.

    Primary endpoint: T5_condition_feasibility_macro_auprc
    Primary comparison: read from result["primary_comparison"], default "B1_vs_B0".
    """
    with open(scored_file) as f:
        scored_data = json.load(f)
    with open(result_file) as f:
        result = json.load(f)

    test_records = scored_data["test_records"]
    arm_scored = scored_data["arm_scored_by_task"]

    primary_metric = result.get("primary_endpoint", PRIMARY_ENDPOINT)
    primary_comparison = result.get("primary_comparison", "B1_vs_B0")
    n_bootstrap = result.get("n_bootstrap", 2000)

    print(f"Primary endpoint: {primary_metric}")
    print(f"Primary comparison: {primary_comparison}")
    print(f"Test records: {len(test_records)}")
    print(f"Arms with scored data: {list(arm_scored.keys())}")

    challenger, baseline = parse_comparison(primary_comparison)

    ci_reconstructed = reconstruct_one_ci(
        arm_scored, challenger, baseline, primary_metric, n_bootstrap
    )

    ci_key = f"{challenger}_vs_{baseline}_{primary_metric}"
    ci_original = result.get("paired_cis", {}).get(ci_key, {})
    if not ci_original:
        # Fallback: check secondary_comparison field
        secondary = result.get("secondary_comparison", {})
        if isinstance(secondary, dict) and secondary.get("comparison") == primary_comparison:
            ci_original = secondary
        else:
            print(f"WARNING: {ci_key} not found in paired_cis; using empty dict for comparison")
            ci_original = {}

    primary_result = compare_ci(ci_key, ci_reconstructed, ci_original)

    # Also reconstruct the secondary comparison (B4_vs_B0) if present and different
    secondary_comparison = result.get("secondary_comparison", {})
    if isinstance(secondary_comparison, dict) and "comparison" in secondary_comparison:
        sec_comp_name = secondary_comparison["comparison"]
        if sec_comp_name != primary_comparison:
            print(f"\n--- Secondary comparison: {sec_comp_name} ---")
            sec_ch, sec_bl = parse_comparison(sec_comp_name)
            sec_ci_reconstructed = reconstruct_one_ci(
                arm_scored, sec_ch, sec_bl, primary_metric, n_bootstrap
            )
            sec_ci_key = f"{sec_ch}_vs_{sec_bl}_{primary_metric}"
            sec_ci_original = result.get("paired_cis", {}).get(sec_ci_key, secondary_comparison)
            secondary_result = compare_ci(sec_ci_key, sec_ci_reconstructed, sec_ci_original)
        else:
            secondary_result = None
    else:
        secondary_result = None

    # Exit Criterion verdict
    print(f"\n=== Exit Criterion ===")
    if primary_result["exact_match"]:
        print("PASS: Primary paired cluster CI reconstructed EXACTLY by independent script (same seed)")
    elif primary_result["close_match"]:
        print("PASS: Primary paired cluster CI reconstructed within tolerance by independent script")
    else:
        print("FAIL: Reconstructed primary CI does not match original")

    return {
        "primary_comparison": primary_comparison,
        "primary_result": primary_result,
        "secondary_result": secondary_result,
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
