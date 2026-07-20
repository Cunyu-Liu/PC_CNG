"""Multi-seed ensemble paired significance for reranking.

Builds ensemble (mean) scores across multiple seeds for both baseline and
candidate models, then runs group-level paired bootstrap and permutation
tests. Also computes seed-level bootstrap confidence intervals by resampling
seeds with replacement.

Top-journal framing:
- Group-level ensemble test: are the deltas consistent across groups when
  we use the full 10-seed ensemble scores?
- Seed-level bootstrap: how stable is the mean delta when we resample the
  10 seeds with replacement?
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from .paired_reranking_significance import (
    bootstrap_ci,
    compare_groups,
    group_metrics,
    mean,
    paired_permutation_p_value,
    percentile,
    sign_test_p_value,
    summarize,
)


def read_score_rows(path: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(dict(row))
    return rows


def row_key(row: Dict[str, object]) -> str:
    group_id = str(row.get("group_id", ""))
    reaction = str(row.get("reaction_smiles", ""))
    source_id = str(row.get("source_id", ""))
    return f"{group_id}||{source_id}||{reaction}"


def build_ensemble_scores(score_csvs: Sequence[str]) -> List[Dict[str, object]]:
    """Average scores across seeds for each (group_id, source_id, reaction_smiles)."""
    accumulated: Dict[str, Dict[str, object]] = {}
    count: Dict[str, int] = {}
    for csv_path in score_csvs:
        for row in read_score_rows(csv_path):
            key = row_key(row)
            score = float(row.get("score", 0.0) or 0.0)
            if key not in accumulated:
                item: Dict[str, object] = dict(row)
                item["score"] = 0.0
                item["score_min"] = float("inf")
                item["score_max"] = float("-inf")
                accumulated[key] = item
                count[key] = 0
            accumulated[key]["score"] = float(accumulated[key]["score"]) + score
            accumulated[key]["score_min"] = min(
                float(accumulated[key]["score_min"]), score
            )
            accumulated[key]["score_max"] = max(
                float(accumulated[key]["score_max"]), score
            )
            count[key] += 1

    ensemble: List[Dict[str, object]] = []
    for key, item in accumulated.items():
        n = count[key]
        item["score"] = float(item["score"]) / n
        if float(item["score_min"]) == float("inf"):
            item["score_min"] = 0.0
        if float(item["score_max"]) == float("-inf"):
            item["score_max"] = 0.0
        item["models_scored"] = n
        ensemble.append(item)
    return ensemble


def write_scores_csv(path: str, rows: Sequence[Dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def per_seed_group_metrics(
    score_csvs: Sequence[str],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Compute per-seed per-group metrics for seed-level bootstrap.

    Returns: seed_idx -> group_id -> {top1, mrr, ndcg}
    """
    result: Dict[str, Dict[str, Dict[str, float]]] = {}
    for idx, csv_path in enumerate(score_csvs):
        groups: Dict[str, List[Dict[str, object]]] = {}
        for row in read_score_rows(csv_path):
            group_id = str(row.get("group_id", ""))
            if not group_id:
                continue
            row["label"] = int(float(row.get("label", 0) or 0))
            row["score"] = float(row.get("score", 0.0) or 0.0)
            groups.setdefault(group_id, []).append(row)
        seed_metrics: Dict[str, Dict[str, float]] = {}
        for group_id, rows in groups.items():
            m = group_metrics(rows)
            if m is not None:
                seed_metrics[group_id] = m
        result[str(idx)] = seed_metrics
    return result


def seed_level_bootstrap(
    baseline_seed_metrics: Dict[str, Dict[str, Dict[str, float]]],
    candidate_seed_metrics: Dict[str, Dict[str, float]],
    iterations: int,
    seed: int,
) -> Dict[str, Dict[str, float]]:
    """Bootstrap over seeds (resample seed indices with replacement).

    For each bootstrap sample of seeds, compute mean Top-1/MRR/NDCG across
    groups, then delta (candidate - baseline). Returns CI on delta.
    """
    rng = random.Random(seed)
    n_seeds = len(baseline_seed_metrics)
    seed_ids = list(baseline_seed_metrics.keys())
    common_groups = sorted(
        set.intersection(
            *[set(baseline_seed_metrics[s].keys()) for s in seed_ids],
            set(candidate_seed_metrics.keys()) if candidate_seed_metrics else set(),
        )
        if candidate_seed_metrics
        else set.intersection(*[set(baseline_seed_metrics[s].keys()) for s in seed_ids])
    )
    if not common_groups:
        common_groups = sorted(
            set.intersection(*[set(baseline_seed_metrics[s].keys()) for s in seed_ids])
        )

    deltas_by_metric: Dict[str, List[float]] = {m: [] for m in ["top1", "mrr", "ndcg"]}
    for _ in range(iterations):
        sample = [seed_ids[rng.randrange(n_seeds)] for _ in range(n_seeds)]
        for metric in ["top1", "mrr", "ndcg"]:
            base_values = []
            cand_values = []
            for group_id in common_groups:
                base_avg = mean(
                    [baseline_seed_metrics[s][group_id][metric] for s in sample]
                )
                if candidate_seed_metrics:
                    cand_avg = mean(
                        [candidate_seed_metrics[s][group_id][metric] for s in sample]
                    )
                    cand_values.append(cand_avg)
                base_values.append(base_avg)
            if candidate_seed_metrics:
                delta = mean(
                    [c - b for c, b in zip(cand_values, base_values)]
                )
            else:
                delta = mean(base_values)
            deltas_by_metric[metric].append(delta)

    out: Dict[str, Dict[str, float]] = {}
    for metric in ["top1", "mrr", "ndcg"]:
        values = deltas_by_metric[metric]
        out[metric] = {
            "ci95_low": percentile(values, 0.025),
            "ci95_high": percentile(values, 0.975),
            "mean": mean(values),
            "std": math.sqrt(sum((v - mean(values)) ** 2 for v in values) / max(len(values), 1)),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", action="append", required=True,
                        help="Baseline candidate_scores.csv (repeat for multiple seeds)")
    parser.add_argument("--candidate", action="append", required=True,
                        help="Candidate candidate_scores.csv (repeat for multiple seeds)")
    parser.add_argument("--baseline-name", default="baseline_ensemble")
    parser.add_argument("--candidate-name", default="candidate_ensemble")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260712)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    baseline_ensemble = build_ensemble_scores(args.baseline)
    candidate_ensemble = build_ensemble_scores(args.candidate)

    baseline_ensemble_path = os.path.join(args.output_dir, "baseline_ensemble_scores.csv")
    candidate_ensemble_path = os.path.join(args.output_dir, "candidate_ensemble_scores.csv")
    write_scores_csv(baseline_ensemble_path, baseline_ensemble)
    write_scores_csv(candidate_ensemble_path, candidate_ensemble)

    baseline_groups: Dict[str, List[Dict[str, object]]] = {}
    for row in baseline_ensemble:
        gid = str(row.get("group_id", ""))
        if not gid:
            continue
        row["label"] = int(float(row.get("label", 0) or 0))
        row["score"] = float(row.get("score", 0.0) or 0.0)
        baseline_groups.setdefault(gid, []).append(row)

    candidate_groups: Dict[str, List[Dict[str, object]]] = {}
    for row in candidate_ensemble:
        gid = str(row.get("group_id", ""))
        if not gid:
            continue
        row["label"] = int(float(row.get("label", 0) or 0))
        row["score"] = float(row.get("score", 0.0) or 0.0)
        candidate_groups.setdefault(gid, []).append(row)

    paired_rows, _ = compare_groups(baseline_groups, candidate_groups)
    ensemble_summary = summarize(paired_rows, args.bootstrap_iterations, args.seed)

    baseline_seed_metrics = per_seed_group_metrics(args.baseline)
    candidate_seed_metrics = per_seed_group_metrics(args.candidate)

    seed_bootstrap = seed_level_bootstrap(
        baseline_seed_metrics,
        candidate_seed_metrics,
        args.bootstrap_iterations,
        args.seed + 500,
    )

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
    with open(os.path.join(args.output_dir, "paired_group_deltas.csv"), "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=detail_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(paired_rows)

    summary_rows = []
    for metric, stats in ensemble_summary.items():
        row = {"metric": metric, "level": "group_ensemble"}
        row.update(stats)
        summary_rows.append(row)
    for metric, stats in seed_bootstrap.items():
        summary_rows.append({
            "metric": metric,
            "level": "seed_bootstrap",
            "delta_mean": stats["mean"],
            "delta_ci95_low": stats["ci95_low"],
            "delta_ci95_high": stats["ci95_high"],
            "delta_std": stats["std"],
        })

    summary_fields = [
        "metric",
        "level",
        "groups",
        "baseline_mean",
        "candidate_mean",
        "delta_mean",
        "delta_ci95_low",
        "delta_ci95_high",
        "delta_std",
        "paired_permutation_p",
        "sign_test_p",
        "candidate_better_groups",
        "baseline_better_groups",
        "tie_groups",
    ]
    with open(os.path.join(args.output_dir, "summary.csv"), "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary_rows)

    payload = {
        "config": vars(args),
        "baseline_name": args.baseline_name,
        "candidate_name": args.candidate_name,
        "n_baseline_seeds": len(args.baseline),
        "n_candidate_seeds": len(args.candidate),
        "common_evaluable_groups": len(paired_rows),
        "ensemble_summary": ensemble_summary,
        "seed_bootstrap": seed_bootstrap,
        "outputs": {
            "baseline_ensemble_scores": baseline_ensemble_path,
            "candidate_ensemble_scores": candidate_ensemble_path,
            "summary_csv": os.path.join(args.output_dir, "summary.csv"),
            "paired_group_deltas_csv": os.path.join(args.output_dir, "paired_group_deltas.csv"),
        },
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    print(json.dumps({
        "common_evaluable_groups": len(paired_rows),
        "ensemble_top1_delta": ensemble_summary["top1"]["delta_mean"],
        "ensemble_top1_ci95": [
            ensemble_summary["top1"]["delta_ci95_low"],
            ensemble_summary["top1"]["delta_ci95_high"],
        ],
        "seed_bootstrap_top1_ci95": [
            seed_bootstrap["top1"]["ci95_low"],
            seed_bootstrap["top1"]["ci95_high"],
        ],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
