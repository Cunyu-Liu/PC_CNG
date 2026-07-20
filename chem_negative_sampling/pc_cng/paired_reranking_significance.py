"""Paired significance tests for same-context reranking outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from typing import Dict, Iterable, List, Sequence, Tuple


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def safe_int(value: object, default: int = 0) -> int:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return default


def dcg(labels: Sequence[int]) -> float:
    return sum((1.0 if label else 0.0) / math.log2(rank + 1) for rank, label in enumerate(labels, start=1))


def read_rows(path: str, score_column: str) -> Dict[str, List[Dict[str, object]]]:
    groups: Dict[str, List[Dict[str, object]]] = {}
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if score_column not in (reader.fieldnames or []):
            raise ValueError(f"Missing score column {score_column!r} in {path}")
        for row in reader:
            group_id = row.get("group_id", "")
            if not group_id:
                continue
            item: Dict[str, object] = dict(row)
            item["label"] = safe_int(row.get("label"))
            item["score"] = safe_float(row.get(score_column))
            groups.setdefault(group_id, []).append(item)
    return groups


def group_metrics(rows: Sequence[Dict[str, object]]) -> Dict[str, float] | None:
    labels = [safe_int(row.get("label")) for row in rows]
    if not any(labels) or all(labels):
        return None
    ranked = sorted(rows, key=lambda row: safe_float(row.get("score")), reverse=True)
    ranked_labels = [safe_int(row.get("label")) for row in ranked]
    first_positive_rank = next(rank for rank, label in enumerate(ranked_labels, start=1) if label == 1)
    ideal = sorted(ranked_labels, reverse=True)
    return {
        "top1": 1.0 if ranked_labels[0] == 1 else 0.0,
        "mrr": 1.0 / first_positive_rank,
        "ndcg": dcg(ranked_labels) / max(dcg(ideal), 1e-12),
    }


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lo = math.floor(index)
    hi = math.ceil(index)
    if lo == hi:
        return ordered[int(index)]
    return ordered[lo] * (hi - index) + ordered[hi] * (index - lo)


def bootstrap_ci(values: Sequence[float], iterations: int, seed: int) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    rng = random.Random(seed)
    n = len(values)
    estimates = []
    for _ in range(iterations):
        estimates.append(mean([values[rng.randrange(n)] for _ in range(n)]))
    return percentile(estimates, 0.025), percentile(estimates, 0.975)


def paired_permutation_p_value(values: Sequence[float], iterations: int, seed: int) -> float:
    """Two-sided paired sign-flip permutation p-value for mean delta."""
    if not values:
        return 1.0
    observed = abs(mean(values))
    rng = random.Random(seed)
    extreme = 1
    for _ in range(iterations):
        permuted = [value if rng.random() < 0.5 else -value for value in values]
        if abs(mean(permuted)) >= observed:
            extreme += 1
    return extreme / (iterations + 1)


def sign_test_p_value(values: Sequence[float]) -> float:
    positive = sum(1 for value in values if value > 0.0)
    negative = sum(1 for value in values if value < 0.0)
    n = positive + negative
    if n == 0:
        return 1.0
    k = min(positive, negative)
    cdf = sum(math.comb(n, i) * (0.5**n) for i in range(k + 1))
    return min(1.0, 2.0 * cdf)


def compare_groups(
    baseline_groups: Dict[str, List[Dict[str, object]]],
    candidate_groups: Dict[str, List[Dict[str, object]]],
) -> Tuple[List[Dict[str, object]], Dict[str, Dict[str, float | int]]]:
    rows: List[Dict[str, object]] = []
    metric_values: Dict[str, Dict[str, float | int]] = {
        metric: {"baseline_sum": 0.0, "candidate_sum": 0.0}
        for metric in ["top1", "mrr", "ndcg"]
    }
    for group_id in sorted(set(baseline_groups) & set(candidate_groups)):
        base = group_metrics(baseline_groups[group_id])
        cand = group_metrics(candidate_groups[group_id])
        if base is None or cand is None:
            continue
        row: Dict[str, object] = {"group_id": group_id}
        for metric in ["top1", "mrr", "ndcg"]:
            baseline_value = float(base[metric])
            candidate_value = float(cand[metric])
            row[f"baseline_{metric}"] = baseline_value
            row[f"candidate_{metric}"] = candidate_value
            row[f"delta_{metric}"] = candidate_value - baseline_value
            metric_values[metric]["baseline_sum"] = float(metric_values[metric]["baseline_sum"]) + baseline_value
            metric_values[metric]["candidate_sum"] = float(metric_values[metric]["candidate_sum"]) + candidate_value
        rows.append(row)
    return rows, metric_values


def summarize(rows: Sequence[Dict[str, object]], iterations: int, seed: int) -> Dict[str, Dict[str, float | int]]:
    out: Dict[str, Dict[str, float | int]] = {}
    n = len(rows)
    for offset, metric in enumerate(["top1", "mrr", "ndcg"]):
        deltas = [safe_float(row.get(f"delta_{metric}")) for row in rows]
        baseline_values = [safe_float(row.get(f"baseline_{metric}")) for row in rows]
        candidate_values = [safe_float(row.get(f"candidate_{metric}")) for row in rows]
        ci_low, ci_high = bootstrap_ci(deltas, iterations, seed + offset)
        out[metric] = {
            "groups": n,
            "baseline_mean": mean(baseline_values),
            "candidate_mean": mean(candidate_values),
            "delta_mean": mean(deltas),
            "delta_ci95_low": ci_low,
            "delta_ci95_high": ci_high,
            "paired_permutation_p": paired_permutation_p_value(deltas, iterations, seed + 100 + offset),
            "sign_test_p": sign_test_p_value(deltas),
            "candidate_better_groups": sum(1 for value in deltas if value > 0.0),
            "baseline_better_groups": sum(1 for value in deltas if value < 0.0),
            "tie_groups": sum(1 for value in deltas if value == 0.0),
        }
    return out


def write_csv(path: str, rows: Iterable[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--baseline-name", default="baseline")
    parser.add_argument("--candidate-name", default="candidate")
    parser.add_argument("--score-column", default="score")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260711)
    args = parser.parse_args()

    baseline_groups = read_rows(args.baseline, args.score_column)
    candidate_groups = read_rows(args.candidate, args.score_column)
    paired_rows, _ = compare_groups(baseline_groups, candidate_groups)
    summary = summarize(paired_rows, args.bootstrap_iterations, args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    detail_fields = [
        "group_id",
        "baseline_top1",
        "candidate_top1",
        "delta_top1",
        "baseline_mrr",
        "candidate_mrr",
        "delta_mrr",
        "baseline_ndcg",
        "candidate_ndcg",
        "delta_ndcg",
    ]
    write_csv(os.path.join(args.output_dir, "paired_group_deltas.csv"), paired_rows, detail_fields)
    summary_rows = []
    for metric, stats in summary.items():
        row = {"metric": metric}
        row.update(stats)
        summary_rows.append(row)
    summary_fields = [
        "metric",
        "groups",
        "baseline_mean",
        "candidate_mean",
        "delta_mean",
        "delta_ci95_low",
        "delta_ci95_high",
        "paired_permutation_p",
        "sign_test_p",
        "candidate_better_groups",
        "baseline_better_groups",
        "tie_groups",
    ]
    write_csv(os.path.join(args.output_dir, "summary.csv"), summary_rows, summary_fields)
    payload = {
        "config": vars(args),
        "baseline_name": args.baseline_name,
        "candidate_name": args.candidate_name,
        "common_evaluable_groups": len(paired_rows),
        "summary": summary,
        "outputs": {
            "summary_csv": os.path.join(args.output_dir, "summary.csv"),
            "paired_group_deltas_csv": os.path.join(args.output_dir, "paired_group_deltas.csv"),
        },
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
