"""Audit source-context support versus molecular duplicate support.

Reaction-class reranking uses source_id-level synthetic groups, while candidate
generation currently performs global canonical reaction de-duplication. This
audit makes that policy mismatch visible without changing the benchmark.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Set, Tuple

from .chem_utils import canonicalize_reaction, canonicalize_smiles, split_reaction


def normalize_reaction_class(row: Dict[str, str]) -> str:
    reaction_class = (row.get("reaction_class") or "").strip()
    if reaction_class:
        return reaction_class
    source = (row.get("source") or row.get("dataset") or "").strip().lower()
    if source == "regiosqm20":
        return "RegioSQM20"
    return "unknown"


def canonical_parent(reaction: str) -> str:
    return canonicalize_reaction(reaction) or reaction


def canonical_reactants(reaction: str) -> str:
    try:
        reactants, _, _ = split_reaction(reaction)
    except ValueError:
        return ""
    return canonicalize_smiles(reactants) or reactants


def read_positive_rows(paths: Sequence[str]) -> Dict[str, Dict[str, str]]:
    positives: Dict[str, Dict[str, str]] = {}
    for path in paths:
        with open(path, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                label_type = (row.get("label_type") or "positive").strip().lower()
                if label_type not in {"positive", "success"}:
                    continue
                source_id = (row.get("source_id") or "").strip()
                reaction = (row.get("reaction_smiles") or "").strip()
                if not source_id or not reaction:
                    continue
                item = dict(row)
                item["source_id"] = source_id
                item["reaction_smiles"] = reaction
                item["reaction_class"] = normalize_reaction_class(item)
                item["split"] = item.get("split") or "unknown"
                item["dataset"] = item.get("source") or os.path.basename(path)
                item["canonical_parent"] = canonical_parent(reaction)
                item["canonical_reactants"] = canonical_reactants(reaction)
                positives[source_id] = item
    return positives


def read_synthetic_rows(
    paths: Sequence[str],
    positives: Dict[str, Dict[str, str]],
    review_statuses: Set[str],
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for path in paths:
        with open(path, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                source_id = (row.get("source_id") or "").strip()
                if source_id not in positives:
                    continue
                status = row.get("review_status", "keep_synthetic_negative")
                if review_statuses and status not in review_statuses:
                    continue
                reaction = (row.get("candidate_reaction") or row.get("reaction_smiles") or "").strip()
                if not reaction:
                    continue
                pos = positives[source_id]
                item = dict(row)
                item["source_id"] = source_id
                item["candidate_reaction"] = reaction
                item["reaction_class"] = pos["reaction_class"]
                item["split"] = pos["split"]
                item["canonical_parent"] = pos["canonical_parent"]
                item["canonical_reactants"] = pos["canonical_reactants"]
                item["canonical_candidate"] = canonical_parent(reaction)
                rows.append(item)
    return rows


def split_counts(rows: Iterable[Dict[str, str]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        split = row.get("split") or "unknown"
        counts[split] = counts.get(split, 0) + 1
    return counts


def ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def diagnose(
    positive_parent_count: int,
    candidate_source_count: int,
    candidate_parent_count: int,
    min_groups: int,
) -> Tuple[str, str]:
    if candidate_source_count >= min_groups and candidate_parent_count >= min_groups:
        return "ok", "keep_monitoring"
    if candidate_source_count >= min_groups and candidate_parent_count < min_groups:
        return "source_duplicate_risk", "do_not_claim_molecular_support_without_duplicate_sensitivity"
    if positive_parent_count < min_groups:
        return "data_source_gap", "add_external_or_curated_weak_class_contexts"
    return "generator_coverage_gap", f"generate_distinct_candidates_for_{min_groups - candidate_source_count}_more_sources"


def summarize_class(
    reaction_class: str,
    positives: Sequence[Dict[str, str]],
    synthetic_rows: Sequence[Dict[str, str]],
    min_groups: int,
) -> Dict[str, object]:
    positive_sources = {row["source_id"] for row in positives}
    positive_parents = {row["canonical_parent"] for row in positives if row.get("canonical_parent")}
    positive_reactants = {row["canonical_reactants"] for row in positives if row.get("canonical_reactants")}
    candidate_sources = {row["source_id"] for row in synthetic_rows}
    candidate_parents = {row["canonical_parent"] for row in synthetic_rows if row.get("canonical_parent")}
    candidate_reactants = {row["canonical_reactants"] for row in synthetic_rows if row.get("canonical_reactants")}
    candidate_reactions = {row["canonical_candidate"] for row in synthetic_rows if row.get("canonical_candidate")}
    status, recommendation = diagnose(
        positive_parent_count=len(positive_parents),
        candidate_source_count=len(candidate_sources),
        candidate_parent_count=len(candidate_parents),
        min_groups=min_groups,
    )
    return {
        "reaction_class": reaction_class,
        "status": status,
        "recommendation": recommendation,
        "positive_sources": len(positive_sources),
        "positive_parent_reactions": len(positive_parents),
        "positive_reactant_contexts": len(positive_reactants),
        "candidate_rows": len(synthetic_rows),
        "candidate_sources": len(candidate_sources),
        "candidate_parent_reactions": len(candidate_parents),
        "candidate_reactant_contexts": len(candidate_reactants),
        "candidate_reactions": len(candidate_reactions),
        "source_group_deficit": max(0, min_groups - len(candidate_sources)),
        "molecular_parent_deficit": max(0, min_groups - len(candidate_parents)),
        "coverage_of_positive_sources": ratio(len(candidate_sources), len(positive_sources)),
        "coverage_of_positive_parent_reactions": ratio(len(candidate_parents), len(positive_parents)),
        "duplicate_candidate_pressure": 1.0 - ratio(len(candidate_reactions), len(synthetic_rows)),
        "positive_split_counts": split_counts(positives),
        "candidate_split_counts": split_counts(synthetic_rows),
    }


def flatten(rows: Sequence[Dict[str, object]]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for row in rows:
        out.append(
            {
                "reaction_class": str(row.get("reaction_class", "")),
                "status": str(row.get("status", "")),
                "recommendation": str(row.get("recommendation", "")),
                "positive_sources": str(row.get("positive_sources", 0)),
                "positive_parent_reactions": str(row.get("positive_parent_reactions", 0)),
                "positive_reactant_contexts": str(row.get("positive_reactant_contexts", 0)),
                "candidate_rows": str(row.get("candidate_rows", 0)),
                "candidate_sources": str(row.get("candidate_sources", 0)),
                "candidate_parent_reactions": str(row.get("candidate_parent_reactions", 0)),
                "candidate_reactant_contexts": str(row.get("candidate_reactant_contexts", 0)),
                "candidate_reactions": str(row.get("candidate_reactions", 0)),
                "source_group_deficit": str(row.get("source_group_deficit", 0)),
                "molecular_parent_deficit": str(row.get("molecular_parent_deficit", 0)),
                "coverage_of_positive_sources": f"{float(row.get('coverage_of_positive_sources', 0.0)) * 100.0:.2f}",
                "coverage_of_positive_parent_reactions": f"{float(row.get('coverage_of_positive_parent_reactions', 0.0)) * 100.0:.2f}",
                "duplicate_candidate_pressure": f"{float(row.get('duplicate_candidate_pressure', 0.0)) * 100.0:.2f}",
            }
        )
    return out


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


def missing_context_rows(
    positives_by_class: Dict[str, List[Dict[str, str]]],
    synthetic_by_class: Dict[str, List[Dict[str, str]]],
    max_per_class: int,
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for reaction_class in sorted(positives_by_class):
        candidate_parents = {
            row["canonical_parent"]
            for row in synthetic_by_class.get(reaction_class, [])
            if row.get("canonical_parent")
        }
        seen_missing: Set[str] = set()
        emitted = 0
        for row in positives_by_class[reaction_class]:
            parent = row.get("canonical_parent", "")
            if not parent or parent in candidate_parents or parent in seen_missing:
                continue
            seen_missing.add(parent)
            rows.append(
                {
                    "reaction_class": reaction_class,
                    "source_id": row.get("source_id", ""),
                    "split": row.get("split", ""),
                    "dataset": row.get("dataset", ""),
                    "canonical_parent": parent,
                    "canonical_reactants": row.get("canonical_reactants", ""),
                    "reaction_smiles": row.get("reaction_smiles", ""),
                }
            )
            emitted += 1
            if max_per_class > 0 and emitted >= max_per_class:
                break
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--positive-csv", action="append", required=True)
    parser.add_argument("--synthetic-csv", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-groups", type=int, default=20)
    parser.add_argument("--review-status", action="append", default=["keep_synthetic_negative"])
    parser.add_argument("--include-reaction-class", action="append", default=[])
    parser.add_argument("--max-missing-contexts-per-class", type=int, default=200)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    positives_by_source = read_positive_rows(args.positive_csv)
    synthetic_rows = read_synthetic_rows(args.synthetic_csv, positives_by_source, set(args.review_status))
    include_classes = set(args.include_reaction_class)
    positives_by_class: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    synthetic_by_class: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in positives_by_source.values():
        reaction_class = row["reaction_class"]
        if include_classes and reaction_class not in include_classes:
            continue
        positives_by_class[reaction_class].append(row)
    for row in synthetic_rows:
        reaction_class = row["reaction_class"]
        if include_classes and reaction_class not in include_classes:
            continue
        synthetic_by_class[reaction_class].append(row)

    classes = sorted(set(positives_by_class) | set(synthetic_by_class))
    summaries = [
        summarize_class(
            reaction_class=reaction_class,
            positives=positives_by_class.get(reaction_class, []),
            synthetic_rows=synthetic_by_class.get(reaction_class, []),
            min_groups=args.min_groups,
        )
        for reaction_class in classes
    ]
    fields = [
        "reaction_class",
        "status",
        "recommendation",
        "positive_sources",
        "positive_parent_reactions",
        "positive_reactant_contexts",
        "candidate_rows",
        "candidate_sources",
        "candidate_parent_reactions",
        "candidate_reactant_contexts",
        "candidate_reactions",
        "source_group_deficit",
        "molecular_parent_deficit",
        "coverage_of_positive_sources",
        "coverage_of_positive_parent_reactions",
        "duplicate_candidate_pressure",
    ]
    table = flatten(summaries)
    write_csv(os.path.join(args.output_dir, "source_support_audit.csv"), table, fields)
    write_markdown(os.path.join(args.output_dir, "source_support_audit.md"), table, fields)
    missing_fields = [
        "reaction_class",
        "source_id",
        "split",
        "dataset",
        "canonical_parent",
        "canonical_reactants",
        "reaction_smiles",
    ]
    missing_rows = missing_context_rows(
        positives_by_class=positives_by_class,
        synthetic_by_class=synthetic_by_class,
        max_per_class=args.max_missing_contexts_per_class,
    )
    write_csv(os.path.join(args.output_dir, "missing_parent_contexts.csv"), missing_rows, missing_fields)
    write_markdown(os.path.join(args.output_dir, "missing_parent_contexts.md"), missing_rows, missing_fields)
    payload = {
        "config": vars(args),
        "positive_sources_total": len(positives_by_source),
        "synthetic_rows_total": len(synthetic_rows),
        "class_summary": summaries,
        "missing_parent_contexts": missing_rows,
        "outputs": {
            "csv": os.path.join(args.output_dir, "source_support_audit.csv"),
            "markdown": os.path.join(args.output_dir, "source_support_audit.md"),
            "missing_contexts_csv": os.path.join(args.output_dir, "missing_parent_contexts.csv"),
            "missing_contexts_markdown": os.path.join(args.output_dir, "missing_parent_contexts.md"),
        },
    }
    with open(os.path.join(args.output_dir, "source_support_audit.json"), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
