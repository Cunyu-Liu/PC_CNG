#!/usr/bin/env python3
"""Paired cluster inference for Phase 1 G3/G6 rebuild.

Implements:
- paired_cluster_bootstrap: within-replicate challenger-baseline difference
- hierarchical_bootstrap: two-level (cluster + seed)
- paired_permutation_test: permutation test on paired differences
- holm_correction: Holm step-down multiple comparison correction
- max_t_correction: Westfall-Young max-T correction

Key principle: challenger and baseline metrics MUST be computed on the SAME
resampled clusters within each bootstrap replicate, preserving their correlation.
"""
from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Callable, Sequence

import numpy as np


def _group_by_cluster(records: Sequence[dict], cluster_key: str) -> dict[str, list[dict]]:
    """Group records by cluster key."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        groups[str(r.get(cluster_key, r.get("experimental_group", "default")))].append(r)
    return dict(groups)


def paired_cluster_bootstrap(
    challenger_records: list[dict],
    baseline_records: list[dict],
    metric_fn: Callable[[list[dict]], float],
    cluster_key: str = "experimental_group",
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    seed: int = 20260723,
) -> dict:
    """Paired cluster bootstrap: compute challenger - baseline within each replicate.

    Within each bootstrap replicate:
    1. Resample cluster IDs (with replacement)
    2. Compute challenger metric on resampled clusters
    3. Compute baseline metric on SAME resampled clusters
    4. Record difference = challenger - baseline

    This preserves the correlation between challenger and baseline.
    """
    assert len(challenger_records) == len(baseline_records), (
        f"Record count mismatch: {len(challenger_records)} vs {len(baseline_records)}"
    )

    # Group both by cluster (assuming same cluster IDs)
    ch_clusters = _group_by_cluster(challenger_records, cluster_key)
    bl_clusters = _group_by_cluster(baseline_records, cluster_key)
    cluster_ids = sorted(set(ch_clusters.keys()) & set(bl_clusters.keys()))
    assert len(cluster_ids) > 0, "No common cluster IDs"

    rng = random.Random(seed)
    n = len(cluster_ids)
    alpha = 1.0 - confidence

    # Point estimates
    ch_point = metric_fn(challenger_records)
    bl_point = metric_fn(baseline_records)
    delta_point = ch_point - bl_point

    # Bootstrap
    deltas = []
    for _ in range(n_bootstrap):
        sampled_ids = [cluster_ids[rng.randrange(n)] for _ in range(n)]
        ch_sample = []
        bl_sample = []
        for cid in sampled_ids:
            ch_sample.extend(ch_clusters[cid])
            bl_sample.extend(bl_clusters[cid])
        ch_metric = metric_fn(ch_sample)
        bl_metric = metric_fn(bl_sample)
        deltas.append(ch_metric - bl_metric)

    deltas.sort()
    ci_low = deltas[int(alpha / 2 * n_bootstrap)]
    ci_high = deltas[int((1 - alpha / 2) * n_bootstrap)]
    # p-value: proportion of bootstrap deltas <= 0 (for positive effect test)
    p_value = sum(1 for d in deltas if d <= 0) / n_bootstrap

    return {
        "challenger_point": ch_point,
        "baseline_point": bl_point,
        "delta_mean": delta_point,
        "delta_ci_low": ci_low,
        "delta_ci_high": ci_high,
        "ci_all_positive": ci_low > 0,
        "p_value": p_value,
        "n_bootstrap": n_bootstrap,
        "n_clusters": n,
        "method": "paired_cluster_bootstrap",
    }


def hierarchical_bootstrap(
    challenger_per_seed: dict[int, list[dict]],
    baseline_per_seed: dict[int, list[dict]],
    metric_fn: Callable[[list[dict]], float],
    cluster_key: str = "experimental_group",
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    seed: int = 20260723,
) -> dict:
    """Hierarchical bootstrap: cluster level + seed level.

    Level 1: resample clusters within each seed
    Level 2: resample seeds
    """
    seeds = sorted(set(challenger_per_seed.keys()) & set(baseline_per_seed.keys()))
    assert len(seeds) > 0, "No common seeds"
    n_seeds = len(seeds)

    # Pre-group by seed -> cluster
    ch_grouped: dict[int, dict[str, list[dict]]] = {}
    bl_grouped: dict[int, dict[str, list[dict]]] = {}
    for s in seeds:
        ch_grouped[s] = _group_by_cluster(challenger_per_seed[s], cluster_key)
        bl_grouped[s] = _group_by_cluster(baseline_per_seed[s], cluster_key)

    rng = random.Random(seed)
    alpha = 1.0 - confidence

    # Point estimate: average across seeds
    ch_points = [metric_fn(challenger_per_seed[s]) for s in seeds]
    bl_points = [metric_fn(baseline_per_seed[s]) for s in seeds]
    delta_point = np.mean(ch_points) - np.mean(bl_points)

    deltas = []
    for _ in range(n_bootstrap):
        # Level 2: resample seeds
        sampled_seeds = [seeds[rng.randrange(n_seeds)] for _ in range(n_seeds)]
        seed_deltas = []
        for s in sampled_seeds:
            ch_clusters = ch_grouped[s]
            bl_clusters = bl_grouped[s]
            cluster_ids = sorted(set(ch_clusters.keys()) & set(bl_clusters.keys()))
            if not cluster_ids:
                continue
            n_clusters = len(cluster_ids)
            # Level 1: resample clusters within this seed
            sampled_cids = [cluster_ids[rng.randrange(n_clusters)] for _ in range(n_clusters)]
            ch_sample = []
            bl_sample = []
            for cid in sampled_cids:
                ch_sample.extend(ch_clusters[cid])
                bl_sample.extend(bl_clusters[cid])
            ch_m = metric_fn(ch_sample)
            bl_m = metric_fn(bl_sample)
            seed_deltas.append(ch_m - bl_m)
        if seed_deltas:
            deltas.append(np.mean(seed_deltas))

    deltas.sort()
    ci_low = deltas[int(alpha / 2 * len(deltas))]
    ci_high = deltas[int((1 - alpha / 2) * len(deltas))]
    p_value = sum(1 for d in deltas if d <= 0) / len(deltas)

    return {
        "challenger_point": float(np.mean(ch_points)),
        "baseline_point": float(np.mean(bl_points)),
        "delta_mean": float(delta_point),
        "delta_ci_low": float(ci_low),
        "delta_ci_high": float(ci_high),
        "ci_all_positive": ci_low > 0,
        "p_value": p_value,
        "n_bootstrap": len(deltas),
        "n_seeds": n_seeds,
        "method": "hierarchical_bootstrap",
    }


def paired_permutation_test(
    challenger_values: list[float],
    baseline_values: list[float],
    n_permutations: int = 10000,
    seed: int = 20260723,
) -> dict:
    """Paired permutation test: flip signs of differences under null.

    Null hypothesis: P(challenger > baseline) = 0.5
    Test statistic: mean of paired differences
    """
    assert len(challenger_values) == len(baseline_values)
    diffs = [c - b for c, b in zip(challenger_values, baseline_values)]
    n = len(diffs)
    observed = sum(diffs) / n

    rng = random.Random(seed)
    count = 0
    for _ in range(n_permutations):
        permuted = [d if rng.random() < 0.5 else -d for d in diffs]
        if abs(sum(permuted) / n) >= abs(observed):
            count += 1

    p_value = (count + 1) / (n_permutations + 1)
    return {
        "observed_statistic": observed,
        "p_value": p_value,
        "n_permutations": n_permutations,
        "n_pairs": n,
        "method": "paired_permutation_test",
    }


def holm_correction(p_values: list[float], alpha: float = 0.05) -> list[dict]:
    """Holm step-down multiple comparison correction.

    Returns list of {index, p_value, adjusted_p, rejected} for each test.
    """
    n = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    results = []
    rejected_any = True
    for rank, (idx, p) in enumerate(indexed):
        adjusted_p = min(1.0, p * (n - rank))
        # Step-down: once we fail to reject, all subsequent also fail
        if rejected_any and adjusted_p <= alpha:
            rejected = True
        else:
            rejected = False
            rejected_any = False
        results.append({
            "index": idx,
            "p_value": p,
            "adjusted_p": adjusted_p,
            "rejected": rejected,
        })
    results.sort(key=lambda x: x["index"])
    return results


def max_t_correction(
    observed_deltas: list[float],
    bootstrap_deltas: list[list[float]],
    alpha: float = 0.05,
) -> list[dict]:
    """Westfall-Young max-T correction using bootstrap distribution.

    observed_deltas: list of observed test statistics (one per comparison)
    bootstrap_deltas: list of lists, each inner list is bootstrap deltas for one comparison
    """
    n_tests = len(observed_deltas)
    n_boot = len(bootstrap_deltas[0]) if bootstrap_deltas else 0

    # For each bootstrap replicate, find max |T| across all tests
    max_ts = []
    for b in range(n_boot):
        max_t = max(abs(bootstrap_deltas[t][b]) for t in range(n_tests))
        max_ts.append(max_t)
    max_ts.sort()

    results = []
    for t in range(n_tests):
        obs = abs(observed_deltas[t])
        # p-value: proportion of max-T >= observed
        p_value = sum(1 for mt in max_ts if mt >= obs) / len(max_ts)
        results.append({
            "index": t,
            "observed_delta": observed_deltas[t],
            "adjusted_p": p_value,
            "rejected": p_value <= alpha,
        })
    return results


# ---------------------------------------------------------------------------
# Metric functions for common tasks
# ---------------------------------------------------------------------------

def mrr_metric(records: list[dict]) -> float:
    """Mean Reciprocal Rank for reranking tasks."""
    # Group by reaction group and compute MRR
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        groups[str(r.get("group_id", r.get("reaction_id", "default")))].append(r)
    rrs = []
    for gid, group in groups.items():
        group.sort(key=lambda x: x.get("score", 0), reverse=True)
        for rank, item in enumerate(group, 1):
            if item.get("label", 0) == 1:
                rrs.append(1.0 / rank)
                break
        else:
            rrs.append(0.0)
    return float(np.mean(rrs)) if rrs else 0.0


def auprc_metric(records: list[dict], positive_label: int = 1) -> float:
    """Area Under Precision-Recall Curve."""
    from sklearn.metrics import average_precision_score
    y_true = [r.get("label", 0) for r in records]
    y_score = [r.get("score", 0.0) for r in records]
    if len(set(y_true)) < 2:
        return 0.0
    return float(average_precision_score(y_true, y_score))


def macro_auprc_metric(records: list[dict], bin_key: str = "yield_bin") -> float:
    """Macro-averaged AUPRC across yield bins."""
    bins = defaultdict(list)
    for r in records:
        b = r.get(bin_key, 0)
        bins[b].append(r)
    auprcs = []
    for b, bin_records in bins.items():
        if len(set(r.get("label", 0) for r in bin_records)) < 2:
            continue
        auprcs.append(auprc_metric(bin_records))
    return float(np.mean(auprcs)) if auprcs else 0.0


def family_macro_auprc_metric(records: list[dict], family_key: str = "reaction_family") -> float:
    """Macro-averaged AUPRC across reaction families (matches G6 primary endpoint definition).

    Groups records by reaction_family and computes binary AUPRC within each family,
    then macro-averages. Families with only one label class are skipped.
    """
    families = defaultdict(list)
    for r in records:
        fam = r.get(family_key, "unknown")
        families[fam].append(r)
    auprcs = []
    for fam, fam_records in families.items():
        if len(set(r.get("label", 0) for r in fam_records)) < 2:
            continue
        auprcs.append(auprc_metric(fam_records))
    return float(np.mean(auprcs)) if auprcs else 0.0


def mae_metric(records: list[dict]) -> float:
    """Mean Absolute Error for regression."""
    errors = [abs(r.get("score", 0.0) - r.get("measured_yield", 0.0)) for r in records]
    return float(np.mean(errors)) if errors else 0.0


def spearman_metric(records: list[dict]) -> float:
    """Spearman rank correlation."""
    from scipy.stats import spearmanr
    scores = [r.get("score", 0.0) for r in records]
    yields = [r.get("measured_yield", 0.0) for r in records]
    if len(set(scores)) < 2 or len(set(yields)) < 2:
        return 0.0
    rho, _ = spearmanr(scores, yields)
    return float(rho) if not math.isnan(rho) else 0.0


def ndcg_metric(records: list[dict], k: int = 10) -> float:
    """Normalized Discounted Cumulative Gain at k for plate ranking."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        groups[str(r.get("plate_id", r.get("experimental_group", "default")))].append(r)
    ndcgs = []
    for gid, group in groups.items():
        group.sort(key=lambda x: x.get("score", 0), reverse=True)
        # DCG
        dcg = sum(
            (2 ** group[i].get("measured_yield", 0) - 1) / math.log2(i + 2)
            for i in range(min(k, len(group)))
        )
        # IDCG
        ideal = sorted(group, key=lambda x: x.get("measured_yield", 0), reverse=True)
        idcg = sum(
            (2 ** ideal[i].get("measured_yield", 0) - 1) / math.log2(i + 2)
            for i in range(min(k, len(ideal)))
        )
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
    return float(np.mean(ndcgs)) if ndcgs else 0.0


def ece_metric(records: list[dict], n_bins: int = 10) -> float:
    """Expected Calibration Error."""
    scores = np.array([r.get("score", 0.0) for r in records])
    labels = np.array([r.get("label", 0) for r in records])
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(records)
    for i in range(n_bins):
        mask = (scores >= bin_edges[i]) & (scores < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        avg_conf = scores[mask].mean()
        avg_acc = labels[mask].mean()
        ece += abs(avg_conf - avg_acc) * mask.sum() / n
    return float(ece)
