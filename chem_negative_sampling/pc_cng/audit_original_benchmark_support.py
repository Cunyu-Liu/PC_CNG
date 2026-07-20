"""Audit original same-context benchmark support and test-group deficits."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Sequence

from .chem_utils import canonicalize_smiles


def read_csv(path: str) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def count_by(rows: Iterable[Dict[str, str]], field: str) -> Dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts[row.get(field, "") or "missing"] += 1
    return dict(counts)


def row_dataset(row: Dict[str, str], fallback: str) -> str:
    return row.get("source") or row.get("dataset") or fallback


def group_value(row: Dict[str, str], group_by: str) -> str:
    if group_by == "canonical_reactants":
        reactants = row.get("reactants", "")
        return canonicalize_smiles(reactants) or reactants
    return row.get(group_by) or row.get("reactants") or row.get("split_key") or row.get("source_id") or ""


def summarize_grouped_real(
    rows: Sequence[Dict[str, str]],
    dataset_id: str,
    group_by: str,
) -> tuple[List[Dict[str, object]], Dict[str, object]]:
    grouped: Dict[tuple[str, str, str], List[Dict[str, str]]] = defaultdict(list)
    group_splits: Dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if row.get("label_type") not in {"positive", "real_negative"}:
            continue
        split = row.get("split", "unknown")
        dataset = row_dataset(row, dataset_id)
        value = group_value(row, group_by)
        if not value:
            continue
        grouped[(dataset, split, value)].append(row)
        group_splits[f"{dataset}|{value}"].add(split)

    groups: List[Dict[str, object]] = []
    for (dataset, split, value), group_rows in grouped.items():
        labels = Counter(row.get("label_type", "") for row in group_rows)
        positives = labels.get("positive", 0)
        negatives = labels.get("real_negative", 0)
        groups.append(
            {
                "group_type": "real",
                "dataset": dataset,
                "split": split,
                "group_by": group_by,
                "group_value": value,
                "candidate_rows": len(group_rows),
                "positives": positives,
                "negatives": negatives,
                "evaluable": positives > 0 and negatives > 0,
            }
        )
    leakage = [
        {"group_key": key, "splits": sorted(splits), "n_splits": len(splits)}
        for key, splits in sorted(group_splits.items())
        if len(splits) > 1
    ]
    summary = {
        "group_type": "real",
        "dataset_id": dataset_id,
        "group_by": group_by,
        "rows": len(rows),
        "groups": len(groups),
        "evaluable_groups": sum(1 for group in groups if group["evaluable"]),
        "split_counts": count_by(groups, "split"),
        "evaluable_split_counts": count_by([g for g in groups if g["evaluable"]], "split"),
        "dataset_counts": count_by(groups, "dataset"),
        "leakage_groups_across_splits": len(leakage),
        "leakage_examples": leakage[:20],
    }
    return groups, summary


def positive_lookup(real_rows: Sequence[Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    for row in real_rows:
        if row.get("label_type") == "positive" and row.get("source_id"):
            out[row["source_id"]] = row
    return out


def summarize_synthetic_groups(
    synthetic_paths: Sequence[str],
    positives: Dict[str, Dict[str, str]],
    review_statuses: set[str],
) -> tuple[List[Dict[str, object]], Dict[str, object]]:
    by_source: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    status_counts: Counter[str] = Counter()
    parent_split_counts: Counter[str] = Counter()
    for path in synthetic_paths:
        for row in read_csv(path):
            source_id = row.get("source_id", "")
            if source_id not in positives:
                continue
            status = row.get("review_status", "keep_synthetic_negative")
            status_counts[status] += 1
            if review_statuses and status not in review_statuses:
                continue
            reaction = row.get("candidate_reaction") or row.get("reaction_smiles") or ""
            if not reaction:
                continue
            parent = positives[source_id]
            parent_split_counts[parent.get("split", "unknown")] += 1
            by_source[source_id].append(row)

    groups: List[Dict[str, object]] = []
    for source_id, rows in by_source.items():
        parent = positives[source_id]
        split = parent.get("split", "unknown")
        dataset = row_dataset(parent, "real")
        groups.append(
            {
                "group_type": "synthetic",
                "dataset": dataset,
                "split": split,
                "group_by": "source_id",
                "group_value": source_id,
                "candidate_rows": len(rows) + 1,
                "positives": 1,
                "negatives": len(rows),
                "evaluable": len(rows) > 0,
            }
        )
    summary = {
        "group_type": "synthetic",
        "rows_after_review_filter": sum(len(rows) for rows in by_source.values()),
        "source_groups": len(groups),
        "evaluable_groups": sum(1 for group in groups if group["evaluable"]),
        "split_counts": count_by(groups, "split"),
        "evaluable_split_counts": count_by([g for g in groups if g["evaluable"]], "split"),
        "dataset_counts": count_by(groups, "dataset"),
        "review_status_counts_seen": dict(status_counts),
        "parent_split_counts_seen_after_filter": dict(parent_split_counts),
    }
    return groups, summary


def table_summary(groups: Sequence[Dict[str, object]], target_test_groups: int) -> Dict[str, object]:
    evaluable = [group for group in groups if group["evaluable"]]
    split_counts = count_by(evaluable, "split")
    dataset_counts = count_by(evaluable, "dataset")
    test_groups = int(split_counts.get("test", 0))
    return {
        "groups": len(groups),
        "evaluable_groups": len(evaluable),
        "evaluable_split_counts": split_counts,
        "evaluable_dataset_counts": dataset_counts,
        "test_groups": test_groups,
        "target_test_groups": target_test_groups,
        "test_group_deficit": max(0, target_test_groups - test_groups),
    }


def write_csv(path: str, rows: Sequence[Dict[str, object]], fields: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: str, rows: Sequence[Dict[str, object]], fields: Sequence[str]) -> None:
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join(["---"] * len(fields)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-csv", action="append", required=True)
    parser.add_argument("--synthetic-csv", action="append", default=[])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-test-groups", type=int, default=200)
    parser.add_argument("--real-group-by", action="append", default=["reactants", "source_id", "split_key", "canonical_reactants"])
    parser.add_argument("--review-status", action="append", default=["keep_synthetic_negative"])
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    real_rows: List[Dict[str, str]] = []
    for path in args.real_csv:
        real_rows.extend(read_csv(path))

    real_summaries = []
    real_group_tables: Dict[str, List[Dict[str, object]]] = {}
    for group_by in args.real_group_by:
        groups, summary = summarize_grouped_real(real_rows, "original", group_by)
        real_group_tables[group_by] = groups
        real_summaries.append(summary)

    synthetic_groups, synthetic_summary = summarize_synthetic_groups(
        synthetic_paths=args.synthetic_csv,
        positives=positive_lookup(real_rows),
        review_statuses=set(args.review_status),
    )

    combined_reactants = list(real_group_tables.get("reactants", [])) + synthetic_groups
    compact_rows = []
    for summary in real_summaries:
        compact = table_summary(real_group_tables[summary["group_by"]], args.target_test_groups)
        compact_rows.append(
            {
                "scope": f"real_{summary['group_by']}",
                "evaluable_groups": compact["evaluable_groups"],
                "test_groups": compact["test_groups"],
                "target_test_groups": compact["target_test_groups"],
                "test_group_deficit": compact["test_group_deficit"],
                "split_counts": json.dumps(compact["evaluable_split_counts"], sort_keys=True),
                "dataset_counts": json.dumps(compact["evaluable_dataset_counts"], sort_keys=True),
                "leakage_groups_across_splits": summary.get("leakage_groups_across_splits", 0),
            }
        )
    synthetic_compact = table_summary(synthetic_groups, args.target_test_groups)
    compact_rows.append(
        {
            "scope": "synthetic_source_id",
            "evaluable_groups": synthetic_compact["evaluable_groups"],
            "test_groups": synthetic_compact["test_groups"],
            "target_test_groups": synthetic_compact["target_test_groups"],
            "test_group_deficit": synthetic_compact["test_group_deficit"],
            "split_counts": json.dumps(synthetic_compact["evaluable_split_counts"], sort_keys=True),
            "dataset_counts": json.dumps(synthetic_compact["evaluable_dataset_counts"], sort_keys=True),
            "leakage_groups_across_splits": "",
        }
    )
    combined_compact = table_summary(combined_reactants, args.target_test_groups)
    compact_rows.append(
        {
            "scope": "combined_real_reactants_plus_synthetic_source_id",
            "evaluable_groups": combined_compact["evaluable_groups"],
            "test_groups": combined_compact["test_groups"],
            "target_test_groups": combined_compact["target_test_groups"],
            "test_group_deficit": combined_compact["test_group_deficit"],
            "split_counts": json.dumps(combined_compact["evaluable_split_counts"], sort_keys=True),
            "dataset_counts": json.dumps(combined_compact["evaluable_dataset_counts"], sort_keys=True),
            "leakage_groups_across_splits": "",
        }
    )

    fields = [
        "scope",
        "evaluable_groups",
        "test_groups",
        "target_test_groups",
        "test_group_deficit",
        "split_counts",
        "dataset_counts",
        "leakage_groups_across_splits",
    ]
    write_csv(os.path.join(args.output_dir, "original_benchmark_support_summary.csv"), compact_rows, fields)
    write_markdown(os.path.join(args.output_dir, "original_benchmark_support_summary.md"), compact_rows, fields)

    group_fields = [
        "group_type",
        "dataset",
        "split",
        "group_by",
        "group_value",
        "candidate_rows",
        "positives",
        "negatives",
        "evaluable",
    ]
    write_csv(os.path.join(args.output_dir, "combined_reactants_plus_synthetic_groups.csv"), combined_reactants, group_fields)

    payload = {
        "config": vars(args),
        "real_summaries": real_summaries,
        "synthetic_summary": synthetic_summary,
        "combined_reactants_plus_synthetic_summary": combined_compact,
        "summary_table": compact_rows,
        "outputs": {
            "summary_csv": os.path.join(args.output_dir, "original_benchmark_support_summary.csv"),
            "summary_md": os.path.join(args.output_dir, "original_benchmark_support_summary.md"),
            "combined_group_csv": os.path.join(args.output_dir, "combined_reactants_plus_synthetic_groups.csv"),
            "summary_json": os.path.join(args.output_dir, "original_benchmark_support_audit.json"),
        },
    }
    with open(os.path.join(args.output_dir, "original_benchmark_support_audit.json"), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    print(json.dumps({"combined": combined_compact, "summary_csv": payload["outputs"]["summary_csv"]}, indent=2))


if __name__ == "__main__":
    main()
