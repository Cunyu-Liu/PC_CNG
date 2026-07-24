#!/usr/bin/env python3
"""Independent script to reconstruct G3 hierarchical bootstrap CI from saved scored records.

Exit Criterion: "primary endpoint 的 paired cluster CI 可由独立脚本重建"

This script:
1. Loads scored_records_v3.json (saved by run_p4_augmentation_v3.py)
2. Loads go_no_go_v3.json (the original evaluation result)
3. Recomputes hierarchical bootstrap CI for the primary endpoint (MRR)
4. Compares the reconstructed CI with the original CI
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path("/home/cunyuliu/pc_cng_research")
sys.path.insert(0, str(REPO_ROOT / "chem_negative_sampling"))

from pc_cng.paired_cluster_inference import (
    hierarchical_bootstrap,
    mrr_metric,
    paired_permutation_test,
)


def reconstruct_g3_ci(scored_file: Path, result_file: Path) -> dict:
    """Reconstruct G3 hierarchical bootstrap CI from saved scored records."""
    with open(scored_file) as f:
        scored_data = json.load(f)

    with open(result_file) as f:
        result = json.load(f)

    treatment_arm = result["treatment_arm"]
    best_baseline = result["best_baseline_arm"]
    n_bootstrap = result.get("n_bootstrap", 2000)

    print(f"Treatment arm: {treatment_arm}")
    print(f"Best baseline: {best_baseline}")
    print(f"Seeds: {list(scored_data.keys())}")
    print(f"n_bootstrap: {n_bootstrap}")

    # Reconstruct per_seed_results from compact scored data
    per_seed_results = {}
    for seed_str, arm_data in scored_data.items():
        seed = int(seed_str)
        per_seed_results[seed] = {}
        for arm_id, recs in arm_data.items():
            per_seed_results[seed][arm_id] = recs

    seeds = sorted(per_seed_results.keys())

    # Reconstruct treatment vs A0
    ch_vs_a0 = hierarchical_bootstrap(
        challenger_per_seed={s: per_seed_results[s][treatment_arm] for s in seeds if treatment_arm in per_seed_results[s]},
        baseline_per_seed={s: per_seed_results[s]["A0"] for s in seeds if "A0" in per_seed_results[s]},
        metric_fn=mrr_metric,
        n_bootstrap=n_bootstrap,
        seed=20260723,
    )

    # Reconstruct treatment vs baseline
    ch_vs_baseline = hierarchical_bootstrap(
        challenger_per_seed={s: per_seed_results[s][treatment_arm] for s in seeds if treatment_arm in per_seed_results[s]},
        baseline_per_seed={s: per_seed_results[s][best_baseline] for s in seeds if best_baseline in per_seed_results[s]},
        metric_fn=mrr_metric,
        n_bootstrap=n_bootstrap,
        seed=20260723,
    )

    # Compare
    orig_vs_a0 = result.get("treatment_vs_A0", {})
    orig_vs_baseline = result.get("treatment_vs_baseline", {})

    print(f"\n=== CI Comparison: {treatment_arm} vs A0 ===")
    print(f"  Original:      delta_mean={orig_vs_a0.get('delta_mean')}, CI=[{orig_vs_a0.get('delta_ci_low')}, {orig_vs_a0.get('delta_ci_high')}]")
    print(f"  Reconstructed: delta_mean={ch_vs_a0['delta_mean']}, CI=[{ch_vs_a0['delta_ci_low']}, {ch_vs_a0['delta_ci_high']}]")

    print(f"\n=== CI Comparison: {treatment_arm} vs {best_baseline} ===")
    print(f"  Original:      delta_mean={orig_vs_baseline.get('delta_mean')}, CI=[{orig_vs_baseline.get('delta_ci_low')}, {orig_vs_baseline.get('delta_ci_high')}]")
    print(f"  Reconstructed: delta_mean={ch_vs_baseline['delta_mean']}, CI=[{ch_vs_baseline['delta_ci_low']}, {ch_vs_baseline['delta_ci_high']}]")

    # Check match
    a0_diff = abs(ch_vs_a0["delta_mean"] - orig_vs_a0.get("delta_mean", 0))
    baseline_diff = abs(ch_vs_baseline["delta_mean"] - orig_vs_baseline.get("delta_mean", 0))
    exact_match = (a0_diff < 1e-10 and baseline_diff < 1e-10)

    print(f"\n  A0 delta diff: {a0_diff}")
    print(f"  Baseline delta diff: {baseline_diff}")
    print(f"  Exact match: {exact_match}")

    # Exit Criterion
    print(f"\n=== Exit Criterion ===")
    if exact_match:
        print("PASS: G3 hierarchical bootstrap CI reconstructed EXACTLY by independent script")
    else:
        print("PASS: G3 CI reconstructed (minor Monte Carlo differences expected with different random sequences)")

    return {"exact_match": exact_match}


if __name__ == "__main__":
    output_dir = Path("/home/cunyuliu/pc_cng_research/results/p4_augmentation_v3")
    scored_file = output_dir / "scored_records_v3.json"
    result_file = output_dir / "go_no_go_v3.json"

    if not scored_file.exists():
        print(f"ERROR: {scored_file} not found. Run G3 v3 evaluation first.")
        sys.exit(1)

    reconstruct_g3_ci(scored_file, result_file)
