#!/usr/bin/env python3
"""Audit external-product score calibration without training a new model.

The script evaluates existing score columns in a candidate_scores.csv file.
It is intentionally CPU-only and dependency-free so it can be run as a
reproducibility diagnostic on large bridge outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


Row = Dict[str, str]
Group = List[Row]


EXCLUDED_NUMERIC_COLUMNS = {
    "label",
    "lm_rank",
}


def parse_float(value: object) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = float(text)
    except ValueError:
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def parse_label(row: Row) -> int:
    value = parse_float(row.get("label"))
    return int(value or 0)


def read_rows(path: str) -> List[Row]:
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def group_rows(rows: Iterable[Row]) -> Dict[str, Group]:
    groups: Dict[str, Group] = defaultdict(list)
    for row in rows:
        groups[row.get("group_id", "")].append(row)
    return dict(groups)


def group_meta(group: Group, column: str) -> str:
    for row in group:
        if parse_label(row) > 0 and row.get(column):
            return row[column]
    for row in group:
        if row.get(column):
            return row[column]
    return "unknown"


def discover_score_columns(rows: Sequence[Row]) -> List[str]:
    if not rows:
        return []
    columns = []
    for column in rows[0].keys():
        if column in EXCLUDED_NUMERIC_COLUMNS or column.endswith("_rank"):
            continue
        values = [parse_float(row.get(column)) for row in rows[:1000]]
        usable = sum(value is not None for value in values)
        if usable >= max(3, len(values) // 10):
            columns.append(column)
    preferred = [
        "chemformer_likelihood",
        "pc_cng",
        "hybrid_pc_cng_w0p00",
        "hybrid_pc_cng_w0p25",
        "hybrid_pc_cng_w0p50",
        "hybrid_pc_cng_w0p75",
        "hybrid_pc_cng_w1p00",
    ]
    ordered = [column for column in preferred if column in columns]
    ordered.extend(column for column in columns if column not in ordered)
    return ordered


def score_group(group: Group, score_column: str) -> Optional[Dict[str, float]]:
    scored: List[Tuple[float, int]] = []
    for row in group:
        score = parse_float(row.get(score_column))
        if score is None:
            continue
        scored.append((score, parse_label(row)))
    if not scored:
        return None
    positives = sum(label > 0 for _, label in scored)
    negatives = sum(label <= 0 for _, label in scored)
    if positives == 0 or negatives == 0:
        return None
    ranked = sorted(scored, key=lambda item: item[0], reverse=True)
    labels = [label for _, label in ranked]
    first_positive_rank = next((idx + 1 for idx, label in enumerate(labels) if label > 0), None)
    if first_positive_rank is None:
        return None
    dcg = sum((1.0 if label > 0 else 0.0) / math.log2(idx + 2) for idx, label in enumerate(labels))
    ideal_labels = sorted(labels, reverse=True)
    idcg = sum((1.0 if label > 0 else 0.0) / math.log2(idx + 2) for idx, label in enumerate(ideal_labels))
    return {
        "candidate_rows": float(len(scored)),
        "positives": float(positives),
        "top1": 1.0 if labels[0] > 0 else 0.0,
        "top3": 1.0 if any(label > 0 for label in labels[:3]) else 0.0,
        "mrr": 1.0 / float(first_positive_rank),
        "ndcg": dcg / idcg if idcg > 0 else 0.0,
        "first_positive_rank": float(first_positive_rank),
        "random_top1_expected": float(positives) / float(len(scored)),
    }


def score_items(scored: Sequence[Tuple[float, int]]) -> Optional[Dict[str, float]]:
    if not scored:
        return None
    positives = sum(label > 0 for _, label in scored)
    negatives = sum(label <= 0 for _, label in scored)
    if positives == 0 or negatives == 0:
        return None
    ranked = sorted(scored, key=lambda item: item[0], reverse=True)
    labels = [label for _, label in ranked]
    first_positive_rank = next((idx + 1 for idx, label in enumerate(labels) if label > 0), None)
    if first_positive_rank is None:
        return None
    dcg = sum((1.0 if label > 0 else 0.0) / math.log2(idx + 2) for idx, label in enumerate(labels))
    ideal_labels = sorted(labels, reverse=True)
    idcg = sum((1.0 if label > 0 else 0.0) / math.log2(idx + 2) for idx, label in enumerate(ideal_labels))
    return {
        "candidate_rows": float(len(scored)),
        "positives": float(positives),
        "top1": 1.0 if labels[0] > 0 else 0.0,
        "top3": 1.0 if any(label > 0 for label in labels[:3]) else 0.0,
        "mrr": 1.0 / float(first_positive_rank),
        "ndcg": dcg / idcg if idcg > 0 else 0.0,
        "first_positive_rank": float(first_positive_rank),
        "random_top1_expected": float(positives) / float(len(scored)),
    }


def aggregate(group_metrics: Sequence[Dict[str, float]]) -> Dict[str, object]:
    if not group_metrics:
        return {
            "groups": 0,
            "candidate_rows": 0,
            "top1": None,
            "top3": None,
            "mrr": None,
            "ndcg": None,
        }
    metrics = ["top1", "top3", "mrr", "ndcg", "first_positive_rank", "random_top1_expected"]
    out: Dict[str, object] = {
        "groups": len(group_metrics),
        "candidate_rows": int(sum(item["candidate_rows"] for item in group_metrics)),
        "mean_candidates_per_group": sum(item["candidate_rows"] for item in group_metrics) / len(group_metrics),
        "mean_positives_per_group": sum(item["positives"] for item in group_metrics) / len(group_metrics),
    }
    for metric in metrics:
        out[metric] = sum(item[metric] for item in group_metrics) / len(group_metrics)
    return out


def filter_groups(groups: Dict[str, Group], split: Optional[str] = None, subgroup: Optional[Tuple[str, str]] = None) -> Dict[str, Group]:
    selected = {}
    for group_id, group in groups.items():
        if split is not None and group_meta(group, "split") != split:
            continue
        if subgroup is not None and group_meta(group, subgroup[0]) != subgroup[1]:
            continue
        selected[group_id] = group
    return selected


def evaluate(groups: Dict[str, Group], score_column: str) -> Dict[str, object]:
    return aggregate([result for group in groups.values() if (result := score_group(group, score_column)) is not None])


def paired_compare(groups: Dict[str, Group], primary: str, candidate: str) -> Dict[str, object]:
    deltas = []
    primary_better = 0
    candidate_better = 0
    ties = 0
    shared_rows = 0
    for group in groups.values():
        primary_items = []
        candidate_items = []
        for row in group:
            primary_score = parse_float(row.get(primary))
            candidate_score = parse_float(row.get(candidate))
            if primary_score is None or candidate_score is None:
                continue
            label = parse_label(row)
            primary_items.append((primary_score, label))
            candidate_items.append((candidate_score, label))
        primary_metrics = score_items(primary_items)
        candidate_metrics = score_items(candidate_items)
        if primary_metrics is None or candidate_metrics is None:
            continue
        shared_rows += int(primary_metrics["candidate_rows"])
        delta = candidate_metrics["top1"] - primary_metrics["top1"]
        deltas.append(delta)
        if delta > 0:
            candidate_better += 1
        elif delta < 0:
            primary_better += 1
        else:
            ties += 1
    if not deltas:
        return {"shared_groups": 0}
    return {
        "shared_groups": len(deltas),
        "shared_rows": shared_rows,
        "candidate_minus_primary_top1": sum(deltas) / len(deltas),
        "candidate_better_groups": candidate_better,
        "primary_better_groups": primary_better,
        "tie_groups": ties,
    }


def sorted_subgroups(groups: Dict[str, Group], column: str) -> List[str]:
    return sorted({group_meta(group, column) for group in groups.values()})


def validation_selected_by_subgroup(
    groups: Dict[str, Group],
    score_columns: Sequence[str],
    subgroup_column: str,
    selection_split: str,
    report_split: str,
    selection_metric: str,
) -> Dict[str, object]:
    selections: Dict[str, object] = {}
    for value in sorted_subgroups(groups, subgroup_column):
        val_groups = filter_groups(groups, selection_split, (subgroup_column, value))
        test_groups = filter_groups(groups, report_split, (subgroup_column, value))
        scored = []
        for column in score_columns:
            val_metrics = evaluate(val_groups, column)
            metric_value = val_metrics.get(selection_metric)
            if metric_value is None:
                continue
            scored.append((float(metric_value), int(val_metrics["groups"]), column, val_metrics))
        if not scored:
            continue
        scored.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
        _, _, selected_column, selected_val = scored[0]
        selections[value] = {
            "selected_score": selected_column,
            "selection_split": selection_split,
            "selection_metric": selection_metric,
            "validation_metrics": selected_val,
            "report_split": report_split,
            "report_metrics": evaluate(test_groups, selected_column),
        }
    return selections


def make_markdown(summary: Dict[str, object]) -> str:
    lines = [
        "# External Score Calibration Audit",
        "",
        f"Input: `{summary['candidate_scores']}`",
        "",
        "## Test Split Metrics",
        "",
        "| Score | Groups | Top-1 | Top-3 | MRR | NDCG | Mean candidates/group |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    metrics_by_score = summary["metrics_by_score"]
    for score in summary["score_columns"]:
        split_metrics = metrics_by_score[score]["by_split"].get("test", {})
        if not split_metrics or split_metrics.get("groups", 0) == 0:
            continue
        lines.append(
            "| {score} | {groups} | {top1:.4f} | {top3:.4f} | {mrr:.4f} | {ndcg:.4f} | {cand:.2f} |".format(
                score=score,
                groups=split_metrics["groups"],
                top1=split_metrics["top1"],
                top3=split_metrics["top3"],
                mrr=split_metrics["mrr"],
                ndcg=split_metrics["ndcg"],
                cand=split_metrics.get("mean_candidates_per_group", 0.0),
            )
        )
    lines.extend(["", "## Paired Test Delta vs Primary", ""])
    lines.append("Paired deltas are computed on rows where both scores are present, so validity-aware full-beam rows without PC-CNG scores do not inflate hybrid comparisons.")
    lines.extend(["", "| Score | Shared groups | Shared rows | ΔTop-1 | Candidate better | Primary better | Ties |", "|---|---:|---:|---:|---:|---:|---:|"])
    primary = summary["primary_score"]
    for score, payload in summary["paired_vs_primary"].items():
        test_payload = payload.get("test", {})
        if score == primary or test_payload.get("shared_groups", 0) == 0:
            continue
        lines.append(
            "| {score} | {groups} | {rows} | {delta:.4f} | {cb} | {pb} | {ties} |".format(
                score=score,
                groups=test_payload["shared_groups"],
                rows=test_payload.get("shared_rows", 0),
                delta=test_payload["candidate_minus_primary_top1"],
                cb=test_payload["candidate_better_groups"],
                pb=test_payload["primary_better_groups"],
                ties=test_payload["tie_groups"],
            )
        )
    lines.extend(["", "## Validation-Selected Scores by Subgroup", ""])
    for subgroup_column, selections in summary["validation_selected_by_subgroup"].items():
        lines.extend([f"### {subgroup_column}", "", "| Value | Selected score | Val Top-1 | Test groups | Test Top-1 |", "|---|---|---:|---:|---:|"])
        for value, payload in selections.items():
            val_top1 = payload["validation_metrics"].get("top1")
            test_metrics = payload["report_metrics"]
            test_top1 = test_metrics.get("top1")
            lines.append(
                "| {value} | {score} | {val_top1:.4f} | {groups} | {test_top1:.4f} |".format(
                    value=value,
                    score=payload["selected_score"],
                    val_top1=val_top1 if val_top1 is not None else float("nan"),
                    groups=test_metrics.get("groups", 0),
                    test_top1=test_top1 if test_top1 is not None else float("nan"),
                )
            )
        lines.append("")
    lines.append(
        "Interpretation note: this audit is diagnostic. Any adaptive selection must be treated as a calibration signal unless it is predeclared and rerun on a held-out benchmark."
    )
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> Dict[str, object]:
    rows = read_rows(args.candidate_scores)
    groups = group_rows(rows)
    score_columns = args.score_column or discover_score_columns(rows)
    if args.primary_score not in score_columns:
        raise SystemExit(f"primary score {args.primary_score!r} is not in score columns: {score_columns}")

    split_values = ["overall"] + sorted({group_meta(group, "split") for group in groups.values() if group_meta(group, "split")})
    metrics_by_score: Dict[str, object] = {}
    paired: Dict[str, object] = {}
    for score in score_columns:
        by_split = {}
        for split in split_values:
            split_groups = groups if split == "overall" else filter_groups(groups, split)
            by_split[split] = evaluate(split_groups, score)
        metrics_by_score[score] = {"by_split": by_split}
        paired[score] = {}
        for split in split_values:
            split_groups = groups if split == "overall" else filter_groups(groups, split)
            paired[score][split] = paired_compare(split_groups, args.primary_score, score)

    validation_selected = {}
    for subgroup_column in args.subgroup_column:
        validation_selected[subgroup_column] = validation_selected_by_subgroup(
            groups,
            score_columns,
            subgroup_column,
            args.selection_split,
            args.report_split,
            args.selection_metric,
        )

    summary = {
        "candidate_scores": os.path.abspath(args.candidate_scores),
        "rows": len(rows),
        "groups": len(groups),
        "score_columns": score_columns,
        "primary_score": args.primary_score,
        "metrics_by_score": metrics_by_score,
        "paired_vs_primary": paired,
        "validation_selected_by_subgroup": validation_selected,
        "notes": [
            "Scores are ranked descending within each group.",
            "Groups without at least one scored positive and one scored negative for a method are excluded for that method.",
            "Validation-selected subgroup scores are diagnostics, not SOTA claims, unless predeclared before held-out evaluation.",
        ],
    }
    os.makedirs(args.output_dir, exist_ok=True)
    summary_path = os.path.join(args.output_dir, "external_score_calibration_summary.json")
    md_path = os.path.join(args.output_dir, "external_score_calibration_summary.md")
    with open(summary_path, "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    with open(md_path, "w") as handle:
        handle.write(make_markdown(summary))
    print(json.dumps({"summary": summary_path, "markdown": md_path, "rows": len(rows), "groups": len(groups)}, indent=2))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-scores", required=True, help="candidate_scores.csv from external-product benchmark")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--primary-score", default="chemformer_likelihood")
    parser.add_argument("--score-column", action="append", default=[])
    parser.add_argument("--subgroup-column", action="append", default=["dataset"])
    parser.add_argument("--selection-split", default="val")
    parser.add_argument("--report-split", default="test")
    parser.add_argument("--selection-metric", default="top1", choices=["top1", "top3", "mrr", "ndcg"])
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
