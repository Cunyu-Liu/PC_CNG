"""Scan held-out positive parents for original benchmark test expansion.

The scanner is deliberately conservative: it does not claim new evaluable
same-context groups until negatives are generated/reviewed. Its job is to find
positive parent reactions whose reactant contexts can safely enter the next
negative-generation stage for closing the original held-out test-group deficit.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .chem_utils import canonicalize_smiles, split_reaction

try:  # pragma: no cover - depends on optional RDKit install
    from rdkit import RDLogger  # type: ignore

    RDLogger.DisableLog("rdApp.*")
except Exception:  # pragma: no cover
    pass


POSITIVE_LABEL = "positive"
USABLE_CURRENT_LABELS = {"positive", "real_negative"}
SMILES_CACHE: Dict[str, Optional[str]] = {}
REACTION_CACHE: Dict[str, Optional[str]] = {}


def read_csv_rows(path: str) -> Tuple[List[Dict[str, str]], List[str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader], list(reader.fieldnames or [])


def iter_csv_rows(path: str) -> Iterable[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            yield dict(row)


def count_by(rows: Iterable[Dict[str, str]], field: str) -> Dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts[row.get(field, "") or "missing"] += 1
    return dict(counts)


def reaction_parts(row: Dict[str, str]) -> Tuple[str, str, str]:
    reaction = row.get("reaction_smiles", "")
    if reaction:
        try:
            return split_reaction(reaction)
        except ValueError:
            return "", "", ""
    return row.get("reactants", ""), row.get("agents", ""), row.get("products", "")


def cached_canonicalize_smiles(smiles: str) -> Optional[str]:
    key = smiles or ""
    if key not in SMILES_CACHE:
        SMILES_CACHE[key] = canonicalize_smiles(key)
    return SMILES_CACHE[key]


def canonical_reactants(row: Dict[str, str]) -> Optional[str]:
    reactants = row.get("reactants", "")
    if not reactants:
        reactants, _, _ = reaction_parts(row)
    return cached_canonicalize_smiles(reactants)


def canonical_product(row: Dict[str, str]) -> Optional[str]:
    products = row.get("products", "")
    if not products:
        _, _, products = reaction_parts(row)
    return cached_canonicalize_smiles(products)


def canonical_parent(row: Dict[str, str]) -> Optional[str]:
    reaction = row.get("reaction_smiles", "")
    cache_key = reaction or f"{row.get('reactants', '')}>{row.get('agents', '')}>{row.get('products', '')}"
    if cache_key in REACTION_CACHE:
        return REACTION_CACHE[cache_key]
    reactants, agents, products = reaction_parts(row)
    if not reactants or not products:
        REACTION_CACHE[cache_key] = None
        return None
    can_reactants = cached_canonicalize_smiles(reactants)
    can_products = cached_canonicalize_smiles(products)
    if can_reactants is None or can_products is None:
        REACTION_CACHE[cache_key] = None
        return None
    parent = f"{can_reactants}>{agents}>{can_products}" if agents else f"{can_reactants}>>{can_products}"
    REACTION_CACHE[cache_key] = parent
    return parent


def parse_yield(value: str) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip().replace("%", "")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def is_better_representative(new_row: Dict[str, object], old_row: Dict[str, object]) -> bool:
    new_yield = new_row.get("yield_numeric")
    old_yield = old_row.get("yield_numeric")
    new_score = float(new_yield) if isinstance(new_yield, (int, float)) else -1.0
    old_score = float(old_yield) if isinstance(old_yield, (int, float)) else -1.0
    if new_score != old_score:
        return new_score > old_score
    return str(new_row.get("source_id", "")) < str(old_row.get("source_id", ""))


def write_csv(path: str, rows: Sequence[Dict[str, object]], fields: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_markdown_table(path: str, rows: Sequence[Dict[str, object]], fields: Sequence[str]) -> None:
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join(["---"] * len(fields)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def build_current_exclusion_sets(current_csvs: Sequence[str]) -> Dict[str, object]:
    context_splits: Dict[str, Set[str]] = defaultdict(set)
    parent_reactions: Set[str] = set()
    positive_products: Set[str] = set()
    split_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    rows_seen = 0

    for path in current_csvs:
        for row in iter_csv_rows(path):
            if row.get("label_type") not in USABLE_CURRENT_LABELS:
                continue
            rows_seen += 1
            split = row.get("split", "") or "missing"
            split_counts[split] += 1
            label_counts[row.get("label_type", "") or "missing"] += 1
            context = canonical_reactants(row)
            if context:
                context_splits[context].add(split)
            parent = canonical_parent(row)
            if parent:
                parent_reactions.add(parent)
            if row.get("label_type") == POSITIVE_LABEL:
                product = canonical_product(row)
                if product:
                    positive_products.add(product)

    return {
        "rows_seen": rows_seen,
        "split_counts": dict(split_counts),
        "label_counts": dict(label_counts),
        "context_splits": context_splits,
        "parent_reactions": parent_reactions,
        "positive_products": positive_products,
    }


def first_pass_source(source_csv: str) -> Dict[str, object]:
    context_splits: Dict[str, Set[str]] = defaultdict(set)
    reaction_splits: Dict[str, Set[str]] = defaultdict(set)
    context_positive_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    invalid_rows = 0
    rows_seen = 0

    for row in iter_csv_rows(source_csv):
        rows_seen += 1
        split = row.get("split", "") or "missing"
        label = row.get("label_type", "") or "missing"
        split_counts[split] += 1
        label_counts[label] += 1
        context = canonical_reactants(row)
        parent = canonical_parent(row)
        if not context or not parent:
            invalid_rows += 1
            continue
        context_splits[context].add(split)
        reaction_splits[parent].add(split)
        if label == POSITIVE_LABEL:
            context_positive_counts[context] += 1

    return {
        "rows_seen": rows_seen,
        "split_counts": dict(split_counts),
        "label_counts": dict(label_counts),
        "invalid_or_uncanonicalizable_rows": invalid_rows,
        "context_splits": context_splits,
        "reaction_splits": reaction_splits,
        "context_positive_counts": context_positive_counts,
    }


def format_splits(splits: Iterable[str]) -> str:
    return ",".join(sorted(split for split in splits if split))


def scan_candidates(
    source_csv: str,
    source_split: str,
    current_sets: Dict[str, object],
    source_stats: Dict[str, object],
    reject_example_limit: int,
) -> Tuple[List[Dict[str, object]], Dict[str, int], List[Dict[str, object]], Dict[str, int]]:
    current_context_splits = current_sets["context_splits"]
    current_parent_reactions = current_sets["parent_reactions"]
    current_positive_products = current_sets["positive_products"]
    source_context_splits = source_stats["context_splits"]
    source_reaction_splits = source_stats["reaction_splits"]
    source_context_positive_counts = source_stats["context_positive_counts"]

    assert isinstance(current_context_splits, dict)
    assert isinstance(current_parent_reactions, set)
    assert isinstance(current_positive_products, set)
    assert isinstance(source_context_splits, dict)
    assert isinstance(source_reaction_splits, dict)
    assert isinstance(source_context_positive_counts, Counter)

    best_by_context: Dict[str, Dict[str, object]] = {}
    reject_reason_counts: Counter[str] = Counter()
    rejected_examples: List[Dict[str, object]] = []
    eligible_positive_rows = 0
    product_overlap_rows = 0

    for row in iter_csv_rows(source_csv):
        reasons: List[str] = []
        split = row.get("split", "") or "missing"
        label = row.get("label_type", "") or "missing"
        context = canonical_reactants(row)
        parent = canonical_parent(row)
        product = canonical_product(row)
        if label != POSITIVE_LABEL:
            reasons.append("not_positive")
        if split != source_split:
            reasons.append("not_source_split")
        if not context or not parent or not product:
            reasons.append("invalid_or_uncanonicalizable")

        if context:
            other_source_context_splits = set(source_context_splits.get(context, set())) - {source_split}
            if other_source_context_splits:
                reasons.append("source_context_cross_split")
            if context in current_context_splits:
                reasons.append("context_seen_in_current_original")
        if parent:
            other_source_reaction_splits = set(source_reaction_splits.get(parent, set())) - {source_split}
            if other_source_reaction_splits:
                reasons.append("source_reaction_cross_split")
            if parent in current_parent_reactions:
                reasons.append("reaction_seen_in_current_original")
        if product and product in current_positive_products:
            product_overlap_rows += 1

        if reasons:
            for reason in reasons:
                reject_reason_counts[reason] += 1
            if len(rejected_examples) < reject_example_limit:
                rejected_examples.append(
                    {
                        "source_id": row.get("source_id", ""),
                        "split": split,
                        "label_type": label,
                        "reasons": ";".join(reasons),
                        "reaction_smiles": row.get("reaction_smiles", ""),
                    }
                )
            continue

        eligible_positive_rows += 1
        assert context is not None
        assert parent is not None
        assert product is not None
        candidate = {
            "source_id": row.get("source_id", ""),
            "source": row.get("source", ""),
            "split": split,
            "split_key": row.get("split_key", ""),
            "yield": row.get("yield", ""),
            "yield_numeric": parse_yield(row.get("yield", "")),
            "reaction_smiles": row.get("reaction_smiles", ""),
            "reactants": row.get("reactants", ""),
            "products": row.get("products", ""),
            "canonical_reactants": context,
            "canonical_product": product,
            "canonical_reaction": parent,
            "source_context_positive_rows": source_context_positive_counts.get(context, 0),
            "source_context_splits": format_splits(source_context_splits.get(context, set())),
            "product_seen_in_current_positive": product in current_positive_products,
            "expansion_role": "heldout_positive_parent_for_negative_generation",
            "recommended_next_step": "generate_and_review_boundary_negatives",
        }
        old = best_by_context.get(context)
        if old is None or is_better_representative(candidate, old):
            best_by_context[context] = candidate

    candidates = sorted(
        best_by_context.values(),
        key=lambda row: (
            -(float(row["yield_numeric"]) if isinstance(row.get("yield_numeric"), (int, float)) else -1.0),
            str(row.get("source_id", "")),
        ),
    )
    for index, row in enumerate(candidates, start=1):
        row["pool_rank"] = index
    scan_counts = {
        "eligible_positive_rows": eligible_positive_rows,
        "eligible_unique_contexts": len(candidates),
        "eligible_duplicate_positive_rows_within_contexts": max(0, eligible_positive_rows - len(candidates)),
        "positive_product_overlap_rows_with_current_original": product_overlap_rows,
    }
    return candidates, dict(reject_reason_counts), rejected_examples, scan_counts


def summary_status(eligible_contexts: int, needed_groups: int) -> str:
    if needed_groups <= 0:
        return "no_additional_groups_needed"
    if eligible_contexts >= needed_groups:
        return "positive_parent_pool_sufficient_for_next_generation_stage"
    if eligible_contexts > 0:
        return "positive_parent_pool_insufficient"
    return "no_eligible_positive_parent_contexts"


def write_summary_markdown(path: str, payload: Dict[str, object]) -> None:
    rows = [
        {"metric": "existing_combined_test_groups", "value": payload["existing_test_groups"]},
        {"metric": "target_test_groups", "value": payload["target_test_groups"]},
        {"metric": "needed_additional_test_groups", "value": payload["needed_additional_test_groups"]},
        {"metric": "eligible_unique_contexts", "value": payload["scan_counts"]["eligible_unique_contexts"]},
        {"metric": "selected_candidate_contexts", "value": payload["selected_candidate_contexts"]},
        {"metric": "projected_test_groups_if_selected_get_negatives", "value": payload["projected_test_groups_if_selected_get_negatives"]},
        {"metric": "status", "value": payload["status"]},
    ]
    lines = [
        "# USPTO/OpenMolecules Original Test Expansion Scan",
        "",
        "| metric | value |",
        "| --- | --- |",
    ]
    for row in rows:
        lines.append(f"| {row['metric']} | {row['value']} |")
    lines.extend(
        [
            "",
            "## Filter Reason Counts",
            "",
            "| reason | rows |",
            "| --- | ---: |",
        ]
    )
    reject_counts = payload.get("reject_reason_counts", {})
    if isinstance(reject_counts, dict):
        for reason, count in sorted(reject_counts.items(), key=lambda item: (-int(item[1]), item[0])):
            lines.append(f"| {reason} | {count} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Selected rows are held-out positive parent reactions only. They still need boundary-negative generation, known-positive filtering, review/status assignment, and a support re-audit before they can count as evaluable same-context test groups.",
        ]
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current-real-csv", action="append", required=True)
    parser.add_argument("--source-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-split", default="test")
    parser.add_argument("--existing-test-groups", type=int, required=True)
    parser.add_argument("--target-test-groups", type=int, default=200)
    parser.add_argument("--max-selected", type=int, default=256)
    parser.add_argument("--eligible-output-limit", type=int, default=1000)
    parser.add_argument("--reject-example-limit", type=int, default=200)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    needed_groups = max(0, args.target_test_groups - args.existing_test_groups)

    current_sets = build_current_exclusion_sets(args.current_real_csv)
    source_stats = first_pass_source(args.source_csv)
    candidates, reject_counts, rejected_examples, scan_counts = scan_candidates(
        source_csv=args.source_csv,
        source_split=args.source_split,
        current_sets=current_sets,
        source_stats=source_stats,
        reject_example_limit=args.reject_example_limit,
    )

    selected_count = len(candidates) if args.max_selected <= 0 else min(len(candidates), args.max_selected)
    selected = [dict(row) for row in candidates[:selected_count]]
    for index, row in enumerate(selected, start=1):
        row["selection_rank"] = index
        row.pop("yield_numeric", None)
    eligible_output = [dict(row) for row in candidates[: max(0, args.eligible_output_limit)]]
    for row in eligible_output:
        row.pop("yield_numeric", None)

    fields = [
        "selection_rank",
        "pool_rank",
        "source_id",
        "source",
        "split",
        "split_key",
        "yield",
        "reaction_smiles",
        "reactants",
        "products",
        "canonical_reactants",
        "canonical_product",
        "canonical_reaction",
        "source_context_positive_rows",
        "source_context_splits",
        "product_seen_in_current_positive",
        "expansion_role",
        "recommended_next_step",
    ]
    selected_csv = os.path.join(args.output_dir, "uspto_original_test_expansion_candidates.csv")
    eligible_csv = os.path.join(args.output_dir, "uspto_original_test_expansion_eligible_pool_top.csv")
    rejected_csv = os.path.join(args.output_dir, "uspto_original_test_expansion_rejected_examples.csv")
    summary_json = os.path.join(args.output_dir, "uspto_original_test_expansion_scan.json")
    summary_md = os.path.join(args.output_dir, "uspto_original_test_expansion_scan.md")

    write_csv(selected_csv, selected, fields)
    write_csv(eligible_csv, eligible_output, [field for field in fields if field != "selection_rank"])
    write_csv(rejected_csv, rejected_examples, ["source_id", "split", "label_type", "reasons", "reaction_smiles"])

    projected = args.existing_test_groups + selected_count
    payload: Dict[str, object] = {
        "config": vars(args),
        "current_original": {
            "rows_seen": current_sets["rows_seen"],
            "split_counts": current_sets["split_counts"],
            "label_counts": current_sets["label_counts"],
            "unique_contexts": len(current_sets["context_splits"]),
            "unique_parent_reactions": len(current_sets["parent_reactions"]),
            "unique_positive_products": len(current_sets["positive_products"]),
        },
        "source": {
            "rows_seen": source_stats["rows_seen"],
            "split_counts": source_stats["split_counts"],
            "label_counts": source_stats["label_counts"],
            "invalid_or_uncanonicalizable_rows": source_stats["invalid_or_uncanonicalizable_rows"],
            "unique_contexts": len(source_stats["context_splits"]),
            "unique_parent_reactions": len(source_stats["reaction_splits"]),
        },
        "existing_test_groups": args.existing_test_groups,
        "target_test_groups": args.target_test_groups,
        "needed_additional_test_groups": needed_groups,
        "scan_counts": scan_counts,
        "reject_reason_counts": reject_counts,
        "selected_candidate_contexts": selected_count,
        "selected_can_cover_deficit": selected_count >= needed_groups,
        "projected_test_groups_if_selected_get_negatives": projected,
        "status": summary_status(int(scan_counts["eligible_unique_contexts"]), needed_groups),
        "outputs": {
            "selected_candidates_csv": selected_csv,
            "eligible_pool_top_csv": eligible_csv,
            "rejected_examples_csv": rejected_csv,
            "summary_json": summary_json,
            "summary_md": summary_md,
        },
    }
    with open(summary_json, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    write_summary_markdown(summary_md, payload)

    table_rows = [
        {
            "scope": "uspto_openmolecules_heldout_positive_parents",
            "existing_test_groups": args.existing_test_groups,
            "target_test_groups": args.target_test_groups,
            "needed_additional_test_groups": needed_groups,
            "eligible_unique_contexts": scan_counts["eligible_unique_contexts"],
            "selected_candidate_contexts": selected_count,
            "projected_if_selected_get_negatives": projected,
            "status": payload["status"],
        }
    ]
    write_markdown_table(
        os.path.join(args.output_dir, "uspto_original_test_expansion_summary.md"),
        table_rows,
        [
            "scope",
            "existing_test_groups",
            "target_test_groups",
            "needed_additional_test_groups",
            "eligible_unique_contexts",
            "selected_candidate_contexts",
            "projected_if_selected_get_negatives",
            "status",
        ],
    )
    print(json.dumps({"status": payload["status"], "selected": selected_count, "needed": needed_groups}, indent=2))


if __name__ == "__main__":
    main()
