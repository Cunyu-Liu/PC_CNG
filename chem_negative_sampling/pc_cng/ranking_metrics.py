"""Dependency-free ranking metrics for same-context candidate evaluation."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Sequence


def dcg(labels: Sequence[int]) -> float:
    """Discounted cumulative gain for binary relevance labels."""
    return sum((1.0 if label else 0.0) / math.log2(rank + 1) for rank, label in enumerate(labels, start=1))


def ranking_metrics(rows: Sequence[Dict[str, object]]) -> Dict[str, float | int]:
    """Compute Top-k, MRR, NDCG, and candidate-count diagnostics by group_id.

    Each row must contain ``group_id``, ``label``, and ``score``. Groups without
    both a positive and a negative candidate are ignored because they do not
    define a ranking decision.
    """
    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["group_id"])].append(row)

    groups = 0
    top1 = 0.0
    top3 = 0.0
    mrr = 0.0
    ndcg = 0.0
    rank_sum = 0.0
    candidate_count = 0
    positive_count = 0
    random_top1 = 0.0
    for group_rows in grouped.values():
        labels = [int(row["label"]) for row in group_rows]
        if not any(labels) or all(labels):
            continue
        ranked = sorted(group_rows, key=lambda row: float(row["score"]), reverse=True)
        ranked_labels = [int(row["label"]) for row in ranked]
        first_positive_rank = next(rank for rank, label in enumerate(ranked_labels, start=1) if label == 1)
        ideal = sorted(ranked_labels, reverse=True)
        groups += 1
        top1 += 1.0 if ranked_labels[0] == 1 else 0.0
        top3 += 1.0 if any(ranked_labels[:3]) else 0.0
        mrr += 1.0 / first_positive_rank
        ndcg += dcg(ranked_labels) / max(dcg(ideal), 1e-12)
        rank_sum += first_positive_rank
        candidate_count += len(group_rows)
        positives = sum(labels)
        positive_count += positives
        random_top1 += positives / len(group_rows)

    if groups == 0:
        return {
            "groups": 0,
            "candidate_rows": 0,
            "top1": 0.0,
            "top3": 0.0,
            "mrr": 0.0,
            "ndcg": 0.0,
            "mean_first_positive_rank": 0.0,
            "mean_candidates_per_group": 0.0,
            "mean_positives_per_group": 0.0,
            "random_top1_expected": 0.0,
        }
    return {
        "groups": groups,
        "candidate_rows": candidate_count,
        "top1": top1 / groups,
        "top3": top3 / groups,
        "mrr": mrr / groups,
        "ndcg": ndcg / groups,
        "mean_first_positive_rank": rank_sum / groups,
        "mean_candidates_per_group": candidate_count / groups,
        "mean_positives_per_group": positive_count / groups,
        "random_top1_expected": random_top1 / groups,
    }


def grouped_metrics(rows: Sequence[Dict[str, object]], field: str) -> Dict[str, Dict[str, float | int]]:
    """Compute ranking metrics for each value of ``field``."""
    values = sorted({str(row.get(field, "") or "unknown") for row in rows})
    out: Dict[str, Dict[str, float | int]] = {}
    for value in values:
        subset = [row for row in rows if str(row.get(field, "") or "unknown") == value]
        metrics = ranking_metrics(subset)
        if int(metrics["groups"]) > 0:
            out[value] = metrics
    return out
