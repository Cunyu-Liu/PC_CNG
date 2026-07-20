"""Analyze action-family contributions for PC-CNG hard negatives.

This module separates three different questions that are easy to conflate:

1. Generation/review: which action families survived false-negative review?
2. Reranking challenge: when a family appears as a competing negative, does the
   scorer still rank the observed product above it?
3. Score separation: how large is the positive-vs-family-negative margin?

The output is intended for paper evidence and generator debugging, not for
training. It reads existing scored candidate CSVs produced by
evaluate_candidate_reranking.py or train_dpo_reward_mlp.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import Counter, defaultdict
from statistics import mean, median, pstdev
from typing import Dict, Iterable, List, Sequence, Tuple

from .evaluate_candidate_reranking import ranking_metrics


DEFAULT_FAMILIES = ["heteroatom", "regio", "tautomer", "low_yield_seed"]


def family_from_row(row: Dict[str, str]) -> str:
    family = (row.get("candidate_family") or row.get("action_family") or "").strip()
    if family:
        return family
    failure_type = (row.get("failure_type") or "").strip()
    if failure_type.endswith("_hard_negative"):
        return failure_type[: -len("_hard_negative")]
    return failure_type or "unknown"


def read_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def safe_float(value: object, default: float = float("nan")) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def safe_int(value: object, default: int = 0) -> int:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return default


def finite(values: Iterable[float]) -> List[float]:
    return [value for value in values if value == value and math.isfinite(value)]


def summarize_numeric(values: Sequence[float]) -> Dict[str, float | int]:
    values = finite(values)
    if not values:
        return {"n": 0, "mean": 0.0, "std": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    return {
        "n": len(values),
        "mean": mean(values),
        "std": pstdev(values) if len(values) > 1 else 0.0,
        "median": median(values),
        "min": min(values),
        "max": max(values),
    }


def read_source_metadata(paths: Sequence[str]) -> Dict[str, Dict[str, str]]:
    metadata: Dict[str, Dict[str, str]] = {}
    for path in paths:
        for row in read_csv_rows(path):
            source_id = row.get("source_id", "")
            if not source_id:
                continue
            metadata[source_id] = {
                "split": row.get("split", "") or "unknown",
                "dataset": row.get("source", "") or row.get("dataset", "") or os.path.basename(path),
                "label_type": row.get("label_type", ""),
            }
    return metadata


def generation_summary(
    rows: Sequence[Dict[str, str]],
    families: Sequence[str],
    source_metadata: Dict[str, Dict[str, str]] | None = None,
) -> Dict[str, Dict[str, object]]:
    source_metadata = source_metadata or {}
    by_family: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_family[family_from_row(row)].append(row)

    out: Dict[str, Dict[str, object]] = {}
    all_families = sorted(set(families) | set(by_family))
    for family in all_families:
        fam_rows = by_family.get(family, [])
        status_counts = Counter(row.get("review_status", "unreviewed") or "unreviewed" for row in fam_rows)
        split_counts = Counter(source_metadata.get(row.get("source_id", ""), {}).get("split", "unknown") for row in fam_rows)
        kept_split_counts = Counter(
            source_metadata.get(row.get("source_id", ""), {}).get("split", "unknown")
            for row in fam_rows
            if (row.get("review_status", "keep_synthetic_negative") or "keep_synthetic_negative")
            == "keep_synthetic_negative"
        )
        dataset_counts = Counter(
            source_metadata.get(row.get("source_id", ""), {}).get("dataset", "unknown") for row in fam_rows
        )
        keep = int(status_counts.get("keep_synthetic_negative", 0))
        needs_review = int(status_counts.get("needs_review_or_downweight", 0))
        discard = int(status_counts.get("discard_known_positive", 0))
        hard_scores = [safe_float(row.get("hard_score")) for row in fam_rows]
        risks = [safe_float(row.get("false_negative_risk")) for row in fam_rows]
        out[family] = {
            "family": family,
            "total": len(fam_rows),
            "keep_synthetic_negative": keep,
            "needs_review_or_downweight": needs_review,
            "discard_known_positive": discard,
            "keep_rate": keep / len(fam_rows) if fam_rows else 0.0,
            "status_counts": dict(status_counts),
            "split_counts": dict(split_counts),
            "kept_split_counts": dict(kept_split_counts),
            "dataset_counts": dict(dataset_counts),
            "hard_score": summarize_numeric(hard_scores),
            "false_negative_risk": summarize_numeric(risks),
        }
    return out


def normalize_scored_rows(path: str, score_column: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for row in read_csv_rows(path):
        if score_column not in row:
            raise ValueError(f"Missing score column {score_column!r} in {path}")
        reaction = row.get("reaction_smiles") or row.get("candidate_reaction") or ""
        if not reaction:
            continue
        item: Dict[str, object] = dict(row)
        item["reaction_smiles"] = reaction
        item["label"] = safe_int(row.get("label"))
        item["score"] = safe_float(row.get(score_column))
        item["candidate_family"] = family_from_row(row)
        item["candidate_source"] = row.get("candidate_source", "") or "unknown"
        item["split"] = row.get("split", "") or "unknown"
        item["dataset"] = row.get("dataset", "") or "unknown"
        rows.append(item)
    return rows


def group_rows(rows: Sequence[Dict[str, object]]) -> Dict[str, List[Dict[str, object]]]:
    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("group_id", ""))].append(row)
    return grouped


def family_groups(rows: Sequence[Dict[str, object]], family: str) -> Dict[str, List[Dict[str, object]]]:
    grouped = group_rows(rows)
    selected: Dict[str, List[Dict[str, object]]] = {}
    for group_id, group in grouped.items():
        has_family_negative = any(
            int(row.get("label", 0)) == 0 and str(row.get("candidate_family", "")) == family for row in group
        )
        if has_family_negative:
            selected[group_id] = group
    return selected


def row_score_stats(rows: Sequence[Dict[str, object]], family: str) -> Dict[str, object]:
    subset = [row for row in rows if str(row.get("candidate_family", "")) == family]
    scores = [float(row.get("score", 0.0)) for row in subset]
    labels = [int(row.get("label", 0)) for row in subset]
    return {
        "rows": len(subset),
        "positive_rows": sum(1 for label in labels if label == 1),
        "negative_rows": sum(1 for label in labels if label == 0),
        "score": summarize_numeric(scores),
    }


def score_margin_stats(selected_groups: Dict[str, List[Dict[str, object]]], family: str) -> Dict[str, object]:
    margins: List[float] = []
    pos_scores: List[float] = []
    neg_scores: List[float] = []
    hard_neg_beats_positive = 0
    for group in selected_groups.values():
        positives = [float(row.get("score", 0.0)) for row in group if int(row.get("label", 0)) == 1]
        negatives = [
            float(row.get("score", 0.0))
            for row in group
            if int(row.get("label", 0)) == 0 and str(row.get("candidate_family", "")) == family
        ]
        if not positives or not negatives:
            continue
        pos_max = max(positives)
        neg_max = max(negatives)
        pos_scores.append(pos_max)
        neg_scores.append(neg_max)
        margin = pos_max - neg_max
        margins.append(margin)
        if neg_max >= pos_max:
            hard_neg_beats_positive += 1
    return {
        "paired_groups": len(margins),
        "positive_wins_rate": sum(1 for margin in margins if margin > 0) / len(margins) if margins else 0.0,
        "hard_negative_beats_positive": hard_neg_beats_positive,
        "margin": summarize_numeric(margins),
        "positive_max_score": summarize_numeric(pos_scores),
        "family_negative_max_score": summarize_numeric(neg_scores),
    }


def challenge_rows_for_family(selected_groups: Dict[str, List[Dict[str, object]]], family: str) -> List[Dict[str, object]]:
    challenge_rows: List[Dict[str, object]] = []
    for group in selected_groups.values():
        for row in group:
            label = int(row.get("label", 0))
            if label == 1 or str(row.get("candidate_family", "")) == family:
                challenge_rows.append(row)
    return challenge_rows


def removal_rows(rows: Sequence[Dict[str, object]], family: str) -> List[Dict[str, object]]:
    out = []
    for row in rows:
        label = int(row.get("label", 0))
        row_family = str(row.get("candidate_family", ""))
        if label == 0 and row_family == family:
            continue
        out.append(row)
    return out


def delta_metrics(a: Dict[str, float | int], b: Dict[str, float | int]) -> Dict[str, float]:
    return {
        key: float(a.get(key, 0.0)) - float(b.get(key, 0.0))
        for key in ["top1", "top3", "mrr", "ndcg"]
    }


def analyze_scored_file(name: str, path: str, score_column: str, families: Sequence[str]) -> Dict[str, object]:
    rows = normalize_scored_rows(path, score_column)
    overall = ranking_metrics(rows)
    out: Dict[str, object] = {
        "name": name,
        "path": path,
        "score_column": score_column,
        "rows": len(rows),
        "overall": overall,
        "families": {},
    }
    for family in families:
        selected = family_groups(rows, family)
        full_context_rows = [row for group in selected.values() for row in group]
        challenge_rows = challenge_rows_for_family(selected, family)
        full_context_metrics = ranking_metrics(full_context_rows)
        family_challenge_metrics = ranking_metrics(challenge_rows)
        removed_metrics = ranking_metrics(removal_rows(rows, family))
        out["families"][family] = {
            "row_stats": row_score_stats(rows, family),
            "groups_with_family_negative": len(selected),
            "full_context_metrics": full_context_metrics,
            "family_only_challenge_metrics": family_challenge_metrics,
            "removal_metrics": removed_metrics,
            "removal_delta_vs_overall": delta_metrics(removed_metrics, overall),
            "score_margins": score_margin_stats(selected, family),
        }
    return out


def parse_named_path(value: str) -> Tuple[str, str]:
    if "=" not in value:
        path = value
        name = os.path.splitext(os.path.basename(path))[0]
        return name, path
    name, path = value.split("=", 1)
    return name, path


def flatten_generation_rows(summary: Dict[str, Dict[str, object]]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for family, raw in sorted(summary.items()):
        item = dict(raw)
        rows.append(
            {
                "family": family,
                "total": str(item.get("total", 0)),
                "keep": str(item.get("keep_synthetic_negative", 0)),
                "keep_train": str(dict(item.get("kept_split_counts", {})).get("train", 0)),
                "keep_val": str(dict(item.get("kept_split_counts", {})).get("val", 0)),
                "keep_test": str(dict(item.get("kept_split_counts", {})).get("test", 0)),
                "needs_review": str(item.get("needs_review_or_downweight", 0)),
                "discard_known_positive": str(item.get("discard_known_positive", 0)),
                "keep_rate": f"{float(item.get('keep_rate', 0.0)):.4f}",
                "hard_score_mean": f"{float(dict(item.get('hard_score', {})).get('mean', 0.0)):.4f}",
                "false_negative_risk_mean": f"{float(dict(item.get('false_negative_risk', {})).get('mean', 0.0)):.4f}",
            }
        )
    return rows


def flatten_score_rows(score_summaries: Sequence[Dict[str, object]], families: Sequence[str]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for score_summary in score_summaries:
        model = str(score_summary["name"])
        overall = dict(score_summary.get("overall", {}))
        family_payload = dict(score_summary.get("families", {}))
        for family in families:
            raw = dict(family_payload.get(family, {}))
            row_stats = dict(raw.get("row_stats", {}))
            challenge = dict(raw.get("family_only_challenge_metrics", {}))
            full_context = dict(raw.get("full_context_metrics", {}))
            margins = dict(raw.get("score_margins", {}))
            margin_stats = dict(margins.get("margin", {}))
            rows.append(
                {
                    "model": model,
                    "family": family,
                    "family_rows": str(row_stats.get("rows", 0)),
                    "family_negative_rows": str(row_stats.get("negative_rows", 0)),
                    "groups": str(raw.get("groups_with_family_negative", 0)),
                    "family_challenge_top1": f"{float(challenge.get('top1', 0.0)):.4f}",
                    "family_challenge_mrr": f"{float(challenge.get('mrr', 0.0)):.4f}",
                    "full_context_top1": f"{float(full_context.get('top1', 0.0)):.4f}",
                    "overall_top1": f"{float(overall.get('top1', 0.0)):.4f}",
                    "positive_wins_rate": f"{float(margins.get('positive_wins_rate', 0.0)):.4f}",
                    "margin_mean": f"{float(margin_stats.get('mean', 0.0)):.4f}",
                    "margin_median": f"{float(margin_stats.get('median', 0.0)):.4f}",
                    "hard_neg_beats_positive": str(margins.get("hard_negative_beats_positive", 0)),
                }
            )
    return rows


def write_csv(path: str, rows: Sequence[Dict[str, str]], fieldnames: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: str, rows: Sequence[Dict[str, str]], fieldnames: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lines = [
        "| " + " | ".join(fieldnames) + " |",
        "| " + " | ".join(["---"] * len(fieldnames)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row.get(field, "") for field in fieldnames) + " |")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic-csv", required=True, help="Reviewed hard-negative CSV")
    parser.add_argument("--real-csv", action="append", default=[], help="Real labeled CSV used to map source_id to split")
    parser.add_argument("--score-csv", action="append", default=[], help="NAME=PATH scored candidate CSV")
    parser.add_argument("--score-column", default="score")
    parser.add_argument("--family", action="append", default=DEFAULT_FAMILIES)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    families = list(dict.fromkeys(args.family))
    synthetic_rows = read_csv_rows(args.synthetic_csv)
    source_metadata = read_source_metadata(args.real_csv)
    generation = generation_summary(synthetic_rows, families, source_metadata)

    score_summaries = []
    for item in args.score_csv:
        name, path = parse_named_path(item)
        score_summaries.append(analyze_scored_file(name, path, args.score_column, families))

    os.makedirs(args.output_dir, exist_ok=True)
    payload = {
        "synthetic_csv": args.synthetic_csv,
        "real_csv": args.real_csv,
        "families": families,
        "generation": generation,
        "score_summaries": score_summaries,
    }
    with open(os.path.join(args.output_dir, "action_family_contribution.json"), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    generation_rows = flatten_generation_rows(generation)
    generation_fields = [
        "family",
        "total",
        "keep",
        "keep_train",
        "keep_val",
        "keep_test",
        "needs_review",
        "discard_known_positive",
        "keep_rate",
        "hard_score_mean",
        "false_negative_risk_mean",
    ]
    write_csv(os.path.join(args.output_dir, "generation_family_table.csv"), generation_rows, generation_fields)
    write_markdown(os.path.join(args.output_dir, "generation_family_table.md"), generation_rows, generation_fields)

    score_rows = flatten_score_rows(score_summaries, families)
    score_fields = [
        "model",
        "family",
        "family_rows",
        "family_negative_rows",
        "groups",
        "family_challenge_top1",
        "family_challenge_mrr",
        "full_context_top1",
        "overall_top1",
        "positive_wins_rate",
        "margin_mean",
        "margin_median",
        "hard_neg_beats_positive",
    ]
    write_csv(os.path.join(args.output_dir, "score_family_table.csv"), score_rows, score_fields)
    write_markdown(os.path.join(args.output_dir, "score_family_table.md"), score_rows, score_fields)

    print(json.dumps({"output_dir": args.output_dir, "families": families, "score_files": len(score_summaries)}, indent=2))


if __name__ == "__main__":
    main()
