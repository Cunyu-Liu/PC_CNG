"""Preregistered hierarchical inference for the formal G6 v3 benchmark."""
from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from pc_cng.p4_g6_benchmark_v3 import (
    PRE_REGISTERED_NONINFERIORITY_MARGIN,
    PRE_REGISTERED_PRIMARY_COMPARISONS,
    source_macro_auprc,
)
from pc_cng.paired_cluster_inference import hierarchical_bootstrap, holm_correction


def _signature(record: Mapping[str, Any]) -> tuple[str, int, str, str]:
    return (
        str(record.get("record_id", "")),
        int(record.get("label", -1)),
        str(record.get("experimental_group", "")),
        str(record.get("source_publication", "")),
    )


def assert_seed_pair_alignment(challenger: Sequence[Mapping[str, Any]], baseline: Sequence[Mapping[str, Any]]) -> None:
    if len(challenger) != len(baseline):
        raise AssertionError(f"paired record count mismatch: {len(challenger)} != {len(baseline)}")
    for index, (left, right) in enumerate(zip(challenger, baseline)):
        if _signature(left) != _signature(right):
            raise AssertionError(f"paired record mismatch at {index}: {_signature(left)!r} != {_signature(right)!r}")


def _mean_seed_delta(challenger_by_seed: Mapping[int, Sequence[Mapping[str, Any]]], baseline_by_seed: Mapping[int, Sequence[Mapping[str, Any]]], metric_fn: Callable = source_macro_auprc) -> float:
    seeds = sorted(set(challenger_by_seed) & set(baseline_by_seed))
    if not seeds:
        raise AssertionError("no shared seeds")
    deltas = []
    for seed in seeds:
        challenger = challenger_by_seed[seed]
        baseline = baseline_by_seed[seed]
        assert_seed_pair_alignment(challenger, baseline)
        deltas.append(metric_fn(challenger) - metric_fn(baseline))
    return float(np.mean(deltas))


def paired_cluster_permutation_test(
    challenger_by_seed: Mapping[int, Sequence[Mapping[str, Any]]],
    baseline_by_seed: Mapping[int, Sequence[Mapping[str, Any]]],
    *,
    metric_fn: Callable = source_macro_auprc,
    cluster_key: str = "experimental_group",
    n_permutations: int = 10000,
    seed: int = 20260728,
) -> dict[str, Any]:
    """Exact-design-compatible sign-swap permutation for a nonlinear metric.

    Within every seed and experimental cluster, the complete paired score vector
    is swapped or retained.  The metric is then recomputed on the full
    resampled dataset, so the permutation test respects both record pairing and
    cluster dependence without replacing source-macro AUPRC by a proxy metric.
    """
    shared_seeds = sorted(set(challenger_by_seed) & set(baseline_by_seed))
    if not shared_seeds:
        raise AssertionError("no shared seeds")
    prepared: dict[int, tuple[dict[str, list[dict]], dict[str, list[dict]], list[str]]] = {}
    for item_seed in shared_seeds:
        challenger = [dict(row) for row in challenger_by_seed[item_seed]]
        baseline = [dict(row) for row in baseline_by_seed[item_seed]]
        assert_seed_pair_alignment(challenger, baseline)
        ch_groups: dict[str, list[dict]] = defaultdict(list)
        bl_groups: dict[str, list[dict]] = defaultdict(list)
        for row in challenger:
            ch_groups[str(row.get(cluster_key, "default"))].append(row)
        for row in baseline:
            bl_groups[str(row.get(cluster_key, "default"))].append(row)
        clusters = sorted(set(ch_groups) & set(bl_groups))
        if not clusters:
            raise AssertionError(f"seed {item_seed} has no shared clusters")
        prepared[item_seed] = (ch_groups, bl_groups, clusters)
    observed = _mean_seed_delta(challenger_by_seed, baseline_by_seed, metric_fn)
    rng = random.Random(seed)
    extreme = 0
    for _ in range(n_permutations):
        seed_deltas = []
        for item_seed in shared_seeds:
            ch_groups, bl_groups, clusters = prepared[item_seed]
            perm_ch: list[dict] = []
            perm_bl: list[dict] = []
            for cluster in clusters:
                if rng.random() < 0.5:
                    perm_ch.extend(ch_groups[cluster])
                    perm_bl.extend(bl_groups[cluster])
                else:
                    perm_ch.extend(bl_groups[cluster])
                    perm_bl.extend(ch_groups[cluster])
            seed_deltas.append(metric_fn(perm_ch) - metric_fn(perm_bl))
        statistic = float(np.mean(seed_deltas))
        if abs(statistic) >= abs(observed):
            extreme += 1
    return {
        "observed_statistic": observed,
        "p_value": (extreme + 1) / (n_permutations + 1),
        "n_permutations": n_permutations,
        "n_seeds": len(shared_seeds),
        "cluster_key": cluster_key,
        "method": "paired_cluster_label_swap_permutation",
    }


def effect_size_by_seed(
    challenger_by_seed: Mapping[int, Sequence[Mapping[str, Any]]],
    baseline_by_seed: Mapping[int, Sequence[Mapping[str, Any]]],
    metric_fn: Callable = source_macro_auprc,
) -> dict[str, float]:
    seeds = sorted(set(challenger_by_seed) & set(baseline_by_seed))
    deltas = []
    for item_seed in seeds:
        assert_seed_pair_alignment(challenger_by_seed[item_seed], baseline_by_seed[item_seed])
        deltas.append(metric_fn(challenger_by_seed[item_seed]) - metric_fn(baseline_by_seed[item_seed]))
    values = np.asarray(deltas, dtype=float)
    std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    standardized = float(values.mean() / std) if std > 1e-12 else (float("inf") if values.mean() > 0 else 0.0)
    return {"mean_seed_delta": float(values.mean()), "seed_delta_sd": std, "standardized_seed_effect": standardized, "n_seeds": float(len(values))}


def run_preregistered_primary_inference(
    predictions: Mapping[str, Mapping[int, Sequence[Mapping[str, Any]]]],
    *,
    n_bootstrap: int,
    n_permutations: int,
    seed: int = 20260728,
    noninferiority_margin: float = PRE_REGISTERED_NONINFERIORITY_MARGIN,
) -> dict[str, Any]:
    """Run the three frozen primary comparisons and Holm jointly.

    The function intentionally receives named arms only.  It has no access to
    point estimates for selecting a best baseline.
    """
    comparison_arms = {
        "pc_cng_vs_random": ("pc_cng", "random"),
        "pc_cng_vs_template_rule": ("pc_cng", "template_rule"),
        "union_vs_pc_cng": ("union", "pc_cng"),
    }
    tests: list[dict[str, Any]] = []
    for index, name in enumerate(PRE_REGISTERED_PRIMARY_COMPARISONS):
        challenger_arm, baseline_arm = comparison_arms[name]
        if challenger_arm not in predictions or baseline_arm not in predictions:
            raise AssertionError(f"missing preregistered arm for {name}")
        challenger = predictions[challenger_arm]
        baseline = predictions[baseline_arm]
        if set(challenger) != set(baseline):
            raise AssertionError(f"seed mismatch for {name}")
        bootstrap = hierarchical_bootstrap(
            challenger, baseline, source_macro_auprc,
            cluster_key="experimental_group", n_bootstrap=n_bootstrap, seed=seed + index,
        )
        permutation = paired_cluster_permutation_test(
            challenger, baseline, n_permutations=n_permutations, seed=seed + 100 + index,
        )
        effect = effect_size_by_seed(challenger, baseline)
        tests.append({
            "comparison": name,
            "challenger": challenger_arm,
            "baseline": baseline_arm,
            "endpoint": "T5_condition_feasibility_source_macro_auprc",
            "bootstrap": bootstrap,
            "permutation": permutation,
            "effect_size": effect,
            "noninferiority_margin": noninferiority_margin,
            "noninferior": bootstrap["delta_ci_low"] > -noninferiority_margin,
        })
    holm = holm_correction([test["permutation"]["p_value"] for test in tests])
    for test, correction in zip(tests, holm):
        test["holm"] = correction
        test["superiority_confirmed"] = bool(
            correction["rejected"] and test["bootstrap"]["delta_ci_low"] > 0
        )
    return {
        "endpoint": "T5_condition_feasibility_source_macro_auprc",
        "comparisons": tests,
        "multiple_comparison_method": "Holm across the three preregistered primary comparisons",
        "baseline_selection": "fixed in analysis plan; no test-data best-baseline selection",
    }


def _toy_predictions(delta: float, *, seed: int, n_clusters: int = 16, n_per_cluster: int = 12) -> tuple[dict[int, list[dict]], dict[int, list[dict]]]:
    rng = np.random.default_rng(seed)
    challenger: dict[int, list[dict]] = {}
    baseline: dict[int, list[dict]] = {}
    for item_seed in range(3):
        left: list[dict] = []
        right: list[dict] = []
        for cluster in range(n_clusters):
            for index in range(n_per_cluster):
                label = int(rng.random() < 0.5)
                base_score = float(np.clip(0.5 + (0.20 if label else -0.20) + rng.normal(0, 0.22), 0, 1))
                improved = float(np.clip(base_score + (delta if label else -delta), 0, 1))
                metadata = {"record_id": f"{item_seed}_{cluster}_{index}", "label": label, "experimental_group": f"c{cluster}", "source_publication": "toy"}
                right.append({**metadata, "score": base_score})
                left.append({**metadata, "score": improved})
        challenger[item_seed] = left
        baseline[item_seed] = right
    return challenger, baseline


def simulate_inference_operating_characteristics(
    *,
    n_simulations: int = 80,
    n_bootstrap: int = 300,
    n_permutations: int = 500,
    seed: int = 20260728,
) -> dict[str, float]:
    """Estimate type-I error and power on paired cluster toy data.

    This is a pre-run statistical calibration check, not evidence about PC-CNG.
    """
    null_rejections = 0
    effect_rejections = 0
    for simulation in range(n_simulations):
        null_ch, null_bl = _toy_predictions(0.0, seed=seed + simulation)
        effect_ch, effect_bl = _toy_predictions(0.08, seed=seed + 10000 + simulation)
        null = run_preregistered_primary_inference(
            {"pc_cng": null_ch, "random": null_bl, "template_rule": null_bl, "union": null_ch},
            n_bootstrap=n_bootstrap, n_permutations=n_permutations, seed=seed + simulation,
        )
        effect = run_preregistered_primary_inference(
            {"pc_cng": effect_ch, "random": effect_bl, "template_rule": effect_bl, "union": effect_ch},
            n_bootstrap=n_bootstrap, n_permutations=n_permutations, seed=seed + 10000 + simulation,
        )
        null_rejections += int(null["comparisons"][0]["superiority_confirmed"])
        effect_rejections += int(effect["comparisons"][0]["superiority_confirmed"])
    return {
        "n_simulations": float(n_simulations),
        "type_i_error": null_rejections / n_simulations,
        "power_at_delta_0p08": effect_rejections / n_simulations,
        "calibration_pass": float(null_rejections / n_simulations <= 0.10),
        "power_pass": float(effect_rejections / n_simulations >= 0.70),
    }
