"""Reaction-class diagnostics for PC-CNG reranking benchmarks."""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Dict, Iterable, List, Sequence, Tuple

from .ranking_metrics import ranking_metrics


def parse_score_spec(value: str) -> Tuple[str, str, str]:
    if "=" not in value:
        path = value
        name = os.path.splitext(os.path.basename(path))[0]
    else:
        name, path = value.split("=", 1)
    if ":" in path:
        path, score_column = path.rsplit(":", 1)
    else:
        score_column = "score"
    if not name or not path:
        raise ValueError(f"Expected NAME=PATH[:SCORE_COLUMN], got {value!r}")
    return name, path, score_column


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


def normalize_reaction_class(row: Dict[str, object]) -> str:
    reaction_class = str(row.get("reaction_class", "") or "").strip()
    if reaction_class:
        return reaction_class
    if str(row.get("dataset", "") or "").strip() == "regiosqm20":
        return "RegioSQM20"
    return "unknown"


def read_scored_rows(path: str, score_column: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if score_column not in (reader.fieldnames or []):
            raise ValueError(f"Missing score column {score_column!r} in {path}")
        for row in reader:
            reaction = row.get("reaction_smiles") or row.get("candidate_reaction") or ""
            group_id = row.get("group_id", "")
            if not reaction or not group_id:
                continue
            item: Dict[str, object] = dict(row)
            item["reaction_smiles"] = reaction
            item["label"] = safe_int(row.get("label"))
            item["score"] = safe_float(row.get(score_column))
            item["reaction_class"] = normalize_reaction_class(item)
            item["split"] = row.get("split", "") or "unknown"
            item["dataset"] = row.get("dataset", "") or "unknown"
            rows.append(item)
    return rows


def unique_group_count(rows: Sequence[Dict[str, object]]) -> int:
    return len({str(row.get("group_id", "")) for row in rows})


def source_counts(rows: Sequence[Dict[str, object]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        source = str(row.get("candidate_source", "") or "unknown")
        counts[source] = counts.get(source, 0) + 1
    return counts


def tie_aware_top1_audit(rows: Sequence[Dict[str, object]], eps: float = 1e-12) -> Dict[str, float | int]:
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("group_id", "")), []).append(row)

    groups = 0
    strict_errors = 0
    tie_aware_top1 = 0
    top_score_tie_groups = 0
    tie_only_errors = 0
    for group_rows in grouped.values():
        labels = [safe_int(row.get("label")) for row in group_rows]
        if not any(labels) or all(labels):
            continue
        groups += 1
        ranked = sorted(group_rows, key=lambda row: safe_float(row.get("score")), reverse=True)
        strict_ok = safe_int(ranked[0].get("label")) == 1
        if not strict_ok:
            strict_errors += 1
        max_score = safe_float(ranked[0].get("score"))
        top_tied = [row for row in ranked if abs(safe_float(row.get("score")) - max_score) <= eps]
        tie_ok = any(safe_int(row.get("label")) == 1 for row in top_tied)
        if len(top_tied) > 1:
            top_score_tie_groups += 1
        if tie_ok:
            tie_aware_top1 += 1
        if not strict_ok and tie_ok:
            tie_only_errors += 1
    return {
        "groups": groups,
        "top1_tie_aware": tie_aware_top1 / groups if groups else 0.0,
        "strict_error_groups": strict_errors,
        "top_score_tie_groups": top_score_tie_groups,
        "tie_only_error_groups": tie_only_errors,
    }


def diagnose_class(
    metrics: Dict[str, float | int],
    min_groups: int,
    weak_top1: float,
    weak_mrr: float,
) -> Tuple[str, str]:
    groups = int(metrics.get("groups", 0))
    top1 = float(metrics.get("top1", 0.0))
    mrr = float(metrics.get("mrr", 0.0))
    if groups == 0:
        return "missing", "no_evaluable_groups"
    if groups < min_groups:
        return "low_support", f"add_candidate_quota_to_reach_{min_groups}_groups"
    if top1 < weak_top1 or mrr < weak_mrr:
        return "weak_performance", "class_targeted_generator_or_error_analysis"
    return "ok", "keep_monitoring"


def class_rows(
    model: str,
    rows: Sequence[Dict[str, object]],
    min_groups: int,
    weak_top1: float,
    weak_mrr: float,
) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    classes = sorted({str(row.get("reaction_class", "") or "unknown") for row in rows})
    for reaction_class in classes:
        subset = [row for row in rows if str(row.get("reaction_class", "") or "unknown") == reaction_class]
        metrics = ranking_metrics(subset)
        tie_audit = tie_aware_top1_audit(subset)
        status, recommendation = diagnose_class(metrics, min_groups, weak_top1, weak_mrr)
        tie_aware_metrics = dict(metrics)
        tie_aware_metrics["top1"] = float(tie_audit.get("top1_tie_aware", 0.0))
        tie_aware_status, tie_aware_recommendation = diagnose_class(
            tie_aware_metrics,
            min_groups,
            weak_top1,
            weak_mrr,
        )
        groups = int(metrics.get("groups", 0))
        candidate_rows = int(metrics.get("candidate_rows", 0))
        out.append(
            {
                "model": model,
                "reaction_class": reaction_class,
                "status": status,
                "recommendation": recommendation,
                "tie_aware_status": tie_aware_status,
                "tie_aware_recommendation": tie_aware_recommendation,
                "groups": groups,
                "candidate_rows": candidate_rows,
                "quota_group_deficit": max(0, min_groups - groups),
                "mean_candidates_per_group": float(metrics.get("mean_candidates_per_group", 0.0)),
                "random_top1_expected": float(metrics.get("random_top1_expected", 0.0)),
                "top1": float(metrics.get("top1", 0.0)),
                "top1_tie_aware": float(tie_audit.get("top1_tie_aware", 0.0)),
                "strict_error_groups": int(tie_audit.get("strict_error_groups", 0)),
                "top_score_tie_groups": int(tie_audit.get("top_score_tie_groups", 0)),
                "tie_only_error_groups": int(tie_audit.get("tie_only_error_groups", 0)),
                "top3": float(metrics.get("top3", 0.0)),
                "mrr": float(metrics.get("mrr", 0.0)),
                "ndcg": float(metrics.get("ndcg", 0.0)),
                "source_counts": source_counts(subset),
            }
        )
    return out


def split_class_rows(
    model: str,
    rows: Sequence[Dict[str, object]],
    min_groups: int,
    weak_top1: float,
    weak_mrr: float,
) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    splits = sorted({str(row.get("split", "") or "unknown") for row in rows})
    for split in splits:
        split_rows = [row for row in rows if str(row.get("split", "") or "unknown") == split]
        for item in class_rows(model, split_rows, min_groups, weak_top1, weak_mrr):
            item = dict(item)
            item["split"] = split
            out.append(item)
    return out


def fmt_pct(value: object) -> str:
    return f"{float(value) * 100.0:.2f}"


def flatten_rows(rows: Iterable[Dict[str, object]], include_split: bool = False) -> List[Dict[str, str]]:
    flat: List[Dict[str, str]] = []
    for row in rows:
        item = {
            "model": str(row.get("model", "")),
            "reaction_class": str(row.get("reaction_class", "")),
            "status": str(row.get("status", "")),
            "recommendation": str(row.get("recommendation", "")),
            "tie_aware_status": str(row.get("tie_aware_status", "")),
            "tie_aware_recommendation": str(row.get("tie_aware_recommendation", "")),
            "groups": str(row.get("groups", 0)),
            "candidate_rows": str(row.get("candidate_rows", 0)),
            "quota_group_deficit": str(row.get("quota_group_deficit", 0)),
            "mean_candidates_per_group": f"{float(row.get('mean_candidates_per_group', 0.0)):.2f}",
            "random_top1_expected": fmt_pct(row.get("random_top1_expected", 0.0)),
            "top1": fmt_pct(row.get("top1", 0.0)),
            "top1_tie_aware": fmt_pct(row.get("top1_tie_aware", 0.0)),
            "strict_error_groups": str(row.get("strict_error_groups", 0)),
            "top_score_tie_groups": str(row.get("top_score_tie_groups", 0)),
            "tie_only_error_groups": str(row.get("tie_only_error_groups", 0)),
            "top3": fmt_pct(row.get("top3", 0.0)),
            "mrr": fmt_pct(row.get("mrr", 0.0)),
            "ndcg": fmt_pct(row.get("ndcg", 0.0)),
        }
        if include_split:
            item = {"split": str(row.get("split", "")), **item}
        flat.append(item)
    return flat


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
    parser.add_argument("--score-csv", action="append", required=True, help="NAME=PATH[:SCORE_COLUMN]")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-groups", type=int, default=20)
    parser.add_argument("--weak-top1", type=float, default=0.80)
    parser.add_argument("--weak-mrr", type=float, default=0.85)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    summaries: Dict[str, object] = {}
    all_class_rows: List[Dict[str, object]] = []
    all_split_rows: List[Dict[str, object]] = []
    for spec in args.score_csv:
        model, path, score_column = parse_score_spec(spec)
        rows = read_scored_rows(path, score_column)
        overall = ranking_metrics(rows)
        class_summary = class_rows(model, rows, args.min_groups, args.weak_top1, args.weak_mrr)
        split_summary = split_class_rows(model, rows, args.min_groups, args.weak_top1, args.weak_mrr)
        summaries[model] = {
            "path": path,
            "score_column": score_column,
            "rows": len(rows),
            "groups": unique_group_count(rows),
            "overall": overall,
            "class_summary": class_summary,
            "split_class_summary": split_summary,
        }
        all_class_rows.extend(class_summary)
        all_split_rows.extend(split_summary)

    fieldnames = [
        "model",
        "reaction_class",
        "status",
        "recommendation",
        "tie_aware_status",
        "tie_aware_recommendation",
        "groups",
        "candidate_rows",
        "quota_group_deficit",
        "mean_candidates_per_group",
        "random_top1_expected",
        "top1",
        "top1_tie_aware",
        "strict_error_groups",
        "top_score_tie_groups",
        "tie_only_error_groups",
        "top3",
        "mrr",
        "ndcg",
    ]
    class_table = flatten_rows(all_class_rows)
    split_fieldnames = ["split"] + fieldnames
    split_table = flatten_rows(all_split_rows, include_split=True)
    write_csv(os.path.join(args.output_dir, "reaction_class_summary.csv"), class_table, fieldnames)
    write_markdown(os.path.join(args.output_dir, "reaction_class_summary.md"), class_table, fieldnames)
    write_csv(os.path.join(args.output_dir, "reaction_class_by_split.csv"), split_table, split_fieldnames)
    write_markdown(os.path.join(args.output_dir, "reaction_class_by_split.md"), split_table, split_fieldnames)
    with open(os.path.join(args.output_dir, "reaction_class_benchmark.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "config": vars(args),
                "summaries": summaries,
                "outputs": {
                    "reaction_class_summary": os.path.join(args.output_dir, "reaction_class_summary.md"),
                    "reaction_class_by_split": os.path.join(args.output_dir, "reaction_class_by_split.md"),
                },
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )
    print(json.dumps({"output_dir": args.output_dir, "models": list(summaries)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
