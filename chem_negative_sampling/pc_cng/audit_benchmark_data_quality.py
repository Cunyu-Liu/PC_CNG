"""Audit benchmark manifest data quality for PC-CNG v3.

The audit is intentionally file-based and reproducible: it reads a machine
manifest, checks required CSV schemas, split/context leakage, duplicate parent
contexts, RDKit-valid reaction rates, synthetic review status, and known-positive
product overlap for synthetic negatives.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

from .chem_utils import canonicalize_reaction, canonicalize_smiles, is_valid_reaction, split_reaction


REAL_REQUIRED_FIELDS = ["source_id", "reaction_smiles", "label_type", "split"]
SYN_REQUIRED_FIELDS = ["source_id"]
KEEP_STATUS = "keep_synthetic_negative"


def read_csv(path: str) -> Tuple[List[Dict[str, str]], List[str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
        return rows, list(reader.fieldnames or [])


def rel_or_abs(root: str, path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(root, path)


def count_by(rows: Iterable[Dict[str, str]], field: str) -> Dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts[row.get(field, "") or "missing"] += 1
    return dict(counts)


def ratio(numerator: int, denominator: int) -> float:
    return 0.0 if denominator <= 0 else float(numerator) / float(denominator)


def reaction_parts(reaction: str) -> Tuple[str, str, str]:
    try:
        return split_reaction(reaction)
    except ValueError:
        return "", "", ""


def canonical_parent(reaction: str) -> str:
    return canonicalize_reaction(reaction) or ""


def canonical_reactants(reaction: str) -> str:
    reactants, _, _ = reaction_parts(reaction)
    return canonicalize_smiles(reactants) or ""


def canonical_product(reaction: str) -> str:
    _, _, products = reaction_parts(reaction)
    return canonicalize_smiles(products) or ""


def write_csv(path: str, rows: Sequence[Dict[str, object]], fields: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_markdown(path: str, rows: Sequence[Dict[str, object]], fields: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join(["---"] * len(fields)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def split_leakage(groups: Dict[str, set[str]]) -> List[Dict[str, object]]:
    out = []
    for key, splits in sorted(groups.items()):
        clean = {split for split in splits if split}
        if len(clean) > 1:
            out.append({"key": key, "splits": ",".join(sorted(clean)), "n_splits": len(clean)})
    return out


def audit_real_dataset(path: str, dataset_id: str) -> Tuple[Dict[str, object], List[Dict[str, str]]]:
    rows, fields = read_csv(path)
    missing_fields = [field for field in REAL_REQUIRED_FIELDS if field not in fields]
    positive_rows = [row for row in rows if row.get("label_type") == "positive"]
    usable_rows = [row for row in rows if row.get("label_type") in {"positive", "real_negative"}]
    valid_count = 0
    invalid_examples = []
    source_splits: Dict[str, set[str]] = defaultdict(set)
    reactant_splits: Dict[str, set[str]] = defaultdict(set)
    parent_splits: Dict[str, set[str]] = defaultdict(set)
    parent_counts: Counter[str] = Counter()
    reactant_counts: Counter[str] = Counter()
    positive_products = []

    for row in usable_rows:
        reaction = row.get("reaction_smiles", "")
        split = row.get("split", "") or "missing"
        source_id = row.get("source_id", "")
        if is_valid_reaction(reaction):
            valid_count += 1
        elif len(invalid_examples) < 20:
            invalid_examples.append(
                {
                    "dataset": dataset_id,
                    "source_id": source_id,
                    "split": split,
                    "reaction_smiles": reaction,
                }
            )
        parent = canonical_parent(reaction)
        reactants = canonical_reactants(reaction)
        if source_id:
            source_splits[source_id].add(split)
        if parent:
            parent_splits[parent].add(split)
            parent_counts[parent] += 1
        if reactants:
            reactant_splits[reactants].add(split)
            reactant_counts[reactants] += 1
        if row.get("label_type") == "positive":
            product = canonical_product(reaction)
            if product:
                positive_products.append(
                    {
                        "dataset": dataset_id,
                        "source_id": source_id,
                        "split": split,
                        "canonical_product": product,
                        "reaction_smiles": reaction,
                    }
                )

    source_leaks = split_leakage(source_splits)
    reactant_leaks = split_leakage(reactant_splits)
    parent_leaks = split_leakage(parent_splits)
    duplicate_parent_contexts = sum(1 for count in parent_counts.values() if count > 1)
    duplicate_reactant_contexts = sum(1 for count in reactant_counts.values() if count > 1)
    summary = {
        "dataset_id": dataset_id,
        "path": path,
        "exists": os.path.exists(path),
        "fields_present": fields,
        "missing_required_fields": missing_fields,
        "rows": len(rows),
        "usable_rows": len(usable_rows),
        "positive_rows": len(positive_rows),
        "split_counts": count_by(usable_rows, "split"),
        "label_counts": count_by(usable_rows, "label_type"),
        "rdkit_valid_reaction_rows": valid_count,
        "rdkit_valid_reaction_rate": ratio(valid_count, len(usable_rows)),
        "unique_source_ids": len(source_splits),
        "unique_parent_reactions": len(parent_counts),
        "unique_reactant_contexts": len(reactant_counts),
        "source_id_split_leaks": len(source_leaks),
        "reactant_context_split_leaks": len(reactant_leaks),
        "parent_reaction_split_leaks": len(parent_leaks),
        "duplicate_parent_reactions": duplicate_parent_contexts,
        "duplicate_reactant_contexts": duplicate_reactant_contexts,
        "invalid_examples": invalid_examples,
        "leak_examples": {
            "source_id": source_leaks[:20],
            "reactant_context": reactant_leaks[:20],
            "parent_reaction": parent_leaks[:20],
        },
    }
    return summary, positive_products


def audit_synthetic_dataset(
    path: str,
    dataset_id: str,
    source_split: Dict[str, str],
    known_positive_products: set[str],
) -> Dict[str, object]:
    rows, fields = read_csv(path)
    missing_fields = [field for field in SYN_REQUIRED_FIELDS if field not in fields]
    has_reaction_field = "candidate_reaction" in fields or "reaction_smiles" in fields
    status_counts: Counter[str] = Counter()
    source_split_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    valid_count = 0
    unknown_source_ids = 0
    non_train_parent_rows = 0
    known_positive_product_overlaps = []
    keep_known_positive_product_overlaps = []
    candidate_counts: Counter[str] = Counter()
    invalid_examples = []

    for row in rows:
        source_id = row.get("source_id", "")
        status = row.get("review_status", "") or "missing"
        status_counts[status] += 1
        source_parent_split = source_split.get(source_id, "unknown_source")
        source_split_counts[source_parent_split] += 1
        if source_parent_split == "unknown_source":
            unknown_source_ids += 1
        elif source_parent_split != "train":
            non_train_parent_rows += 1
        family_counts[row.get("action_family", "") or row.get("failure_type", "") or "missing"] += 1
        reaction = row.get("candidate_reaction") or row.get("reaction_smiles") or ""
        if is_valid_reaction(reaction):
            valid_count += 1
        elif len(invalid_examples) < 20:
            invalid_examples.append({"dataset": dataset_id, "source_id": source_id, "candidate_reaction": reaction})
        candidate = canonical_parent(reaction)
        if candidate:
            candidate_counts[candidate] += 1
        product = canonical_product(reaction)
        if product and product in known_positive_products:
            overlap = {
                "dataset": dataset_id,
                "source_id": source_id,
                "parent_split": source_parent_split,
                "review_status": status,
                "canonical_product": product,
                "candidate_reaction": reaction,
            }
            if len(known_positive_product_overlaps) < 200:
                known_positive_product_overlaps.append(overlap)
            if status == KEEP_STATUS and len(keep_known_positive_product_overlaps) < 200:
                keep_known_positive_product_overlaps.append(overlap)

    duplicate_candidates = sum(1 for count in candidate_counts.values() if count > 1)
    summary = {
        "dataset_id": dataset_id,
        "path": path,
        "exists": os.path.exists(path),
        "fields_present": fields,
        "missing_required_fields": missing_fields,
        "has_candidate_reaction_field": has_reaction_field,
        "rows": len(rows),
        "review_status_counts": dict(status_counts),
        "keep_synthetic_negative_rows": status_counts.get(KEEP_STATUS, 0),
        "parent_split_counts": dict(source_split_counts),
        "unknown_source_id_rows": unknown_source_ids,
        "non_train_parent_rows": non_train_parent_rows,
        "rdkit_valid_reaction_rows": valid_count,
        "rdkit_valid_reaction_rate": ratio(valid_count, len(rows)),
        "unique_candidate_reactions": len(candidate_counts),
        "duplicate_candidate_reactions": duplicate_candidates,
        "action_family_counts": dict(family_counts),
        "known_positive_product_overlap_rows": len(known_positive_product_overlaps),
        "keep_known_positive_product_overlap_rows": len(keep_known_positive_product_overlaps),
        "known_positive_product_overlap_examples": known_positive_product_overlaps[:20],
        "keep_known_positive_product_overlap_examples": keep_known_positive_product_overlaps[:20],
        "invalid_examples": invalid_examples,
    }
    return summary


def flatten_real(summary: Dict[str, object]) -> Dict[str, object]:
    return {
        "dataset_id": summary["dataset_id"],
        "rows": summary["rows"],
        "usable_rows": summary["usable_rows"],
        "positive_rows": summary["positive_rows"],
        "split_counts": json.dumps(summary["split_counts"], sort_keys=True),
        "label_counts": json.dumps(summary["label_counts"], sort_keys=True),
        "rdkit_valid_rate": f"{float(summary['rdkit_valid_reaction_rate']) * 100.0:.2f}",
        "source_id_split_leaks": summary["source_id_split_leaks"],
        "reactant_context_split_leaks": summary["reactant_context_split_leaks"],
        "parent_reaction_split_leaks": summary["parent_reaction_split_leaks"],
        "duplicate_parent_reactions": summary["duplicate_parent_reactions"],
        "missing_required_fields": ",".join(summary["missing_required_fields"]),
    }


def flatten_synthetic(summary: Dict[str, object]) -> Dict[str, object]:
    return {
        "dataset_id": summary["dataset_id"],
        "rows": summary["rows"],
        "keep_rows": summary["keep_synthetic_negative_rows"],
        "parent_split_counts": json.dumps(summary["parent_split_counts"], sort_keys=True),
        "rdkit_valid_rate": f"{float(summary['rdkit_valid_reaction_rate']) * 100.0:.2f}",
        "unknown_source_id_rows": summary["unknown_source_id_rows"],
        "non_train_parent_rows": summary["non_train_parent_rows"],
        "unique_candidate_reactions": summary["unique_candidate_reactions"],
        "duplicate_candidate_reactions": summary["duplicate_candidate_reactions"],
        "known_positive_product_overlap_rows": summary["known_positive_product_overlap_rows"],
        "keep_known_positive_product_overlap_rows": summary["keep_known_positive_product_overlap_rows"],
        "missing_required_fields": ",".join(summary["missing_required_fields"]),
        "has_candidate_reaction_field": summary["has_candidate_reaction_field"],
    }


def load_manifest(path: str) -> Dict[str, object]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


# --------------------------------------------------------------------------- #
# Single-CSV audit mode (added in P1-09 for ORD data quality)
# --------------------------------------------------------------------------- #

ATOM_MAPPING_MARKER = ":]"


def _overlap_with_csvs(target_reactions: set[str], overlap_csv_paths: List[str]) -> List[Dict[str, object]]:
    """Compute canonical-reaction overlap between the target set and each
    reference CSV (e.g. USPTO / HiTEA / RegioSQM20 normalized).
    """
    results: List[Dict[str, object]] = []
    target_count = len(target_reactions)
    for ref_path in overlap_csv_paths:
        entry: Dict[str, object] = {
            "reference_csv": ref_path,
            "exists": os.path.isfile(ref_path),
            "reference_rows": 0,
            "reference_unique_reactions": 0,
            "overlap_rows": 0,
            "overlap_rate": 0.0,
        }
        if not os.path.isfile(ref_path):
            results.append(entry)
            continue
        try:
            ref_rows, _ = read_csv(ref_path)
        except Exception as exc:  # noqa: BLE001
            entry["error"] = str(exc)
            results.append(entry)
            continue
        ref_reactions = set()
        for row in ref_rows:
            rxn = row.get("reaction_smiles", "") or ""
            canon = canonical_parent(rxn)
            if canon:
                ref_reactions.add(canon)
        entry["reference_rows"] = len(ref_rows)
        entry["reference_unique_reactions"] = len(ref_reactions)
        overlap = target_reactions & ref_reactions
        entry["overlap_rows"] = len(overlap)
        entry["overlap_rate"] = ratio(len(overlap), target_count) if target_count else 0.0
        results.append(entry)
    return results


def audit_single_normalized_csv(
    input_csv: str,
    output_dir: str,
    overlap_csvs: List[str] | None = None,
) -> Dict[str, object]:
    """Audit a single normalized CSV (e.g. ord_normalized.csv).

    Computes: SMILES validity rate, atom-mapping coverage, reaction-type
    (``source`` field) distribution, split distribution, and overlap with
    reference normalized datasets.  Writes ``single_csv_audit.json``,
    ``single_csv_audit.md``, and ``overlap_analysis.csv`` to ``output_dir``.
    """
    os.makedirs(output_dir, exist_ok=True)
    rows, fields = read_csv(input_csv)
    n = len(rows)
    valid_count = 0
    mapped_count = 0
    invalid_examples: List[Dict[str, str]] = []
    source_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    yield_non_empty = 0
    unique_reactions: set[str] = set()
    duplicate_count = 0

    for row in rows:
        rxn = row.get("reaction_smiles", "") or ""
        source = row.get("source", "") or "<empty>"
        split = row.get("split", "") or "<empty>"
        label = row.get("label_type", "") or "<empty>"
        source_counts[source] += 1
        split_counts[split] += 1
        label_counts[label] += 1
        if row.get("yield", ""):
            yield_non_empty += 1
        if is_valid_reaction(rxn):
            valid_count += 1
        elif len(invalid_examples) < 20:
            invalid_examples.append({
                "source_id": row.get("source_id", ""),
                "reaction_smiles": rxn,
            })
        if ATOM_MAPPING_MARKER in rxn:
            mapped_count += 1
        canon = canonical_parent(rxn)
        if canon:
            if canon in unique_reactions:
                duplicate_count += 1
            else:
                unique_reactions.add(canon)

    overlap_results = _overlap_with_csvs(unique_reactions, overlap_csvs or [])

    summary: Dict[str, object] = {
        "input_csv": input_csv,
        "exists": os.path.isfile(input_csv),
        "fields_present": fields,
        "rows": n,
        "rdkit_valid_reaction_rows": valid_count,
        "rdkit_valid_reaction_rate": ratio(valid_count, n),
        "atom_mapping_coverage_rows": mapped_count,
        "atom_mapping_coverage_rate": ratio(mapped_count, n),
        "unique_reactions": len(unique_reactions),
        "duplicate_reactions": duplicate_count,
        "yield_non_empty_rows": yield_non_empty,
        "yield_non_empty_rate": ratio(yield_non_empty, n),
        "source_distribution": dict(source_counts),
        "split_distribution": dict(split_counts),
        "label_type_distribution": dict(label_counts),
        "invalid_examples": invalid_examples,
        "overlap_analysis": overlap_results,
    }

    # write outputs
    with open(os.path.join(output_dir, "single_csv_audit.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    md_lines = [
        "# Single-CSV Data Quality Audit",
        "",
        f"- **Input**: `{input_csv}`",
        f"- **Rows**: {n}",
        f"- **RDKit-valid rate**: {ratio(valid_count, n) * 100:.2f}%",
        f"- **Atom-mapping coverage**: {ratio(mapped_count, n) * 100:.2f}%",
        f"- **Unique reactions**: {len(unique_reactions)}",
        f"- **Duplicate reactions**: {duplicate_count}",
        "",
        "## Source distribution",
        "",
        "| source | count |",
        "| --- | --- |",
    ]
    for source, count in source_counts.most_common():
        md_lines.append(f"| {source} | {count} |")
    md_lines += ["", "## Overlap analysis", "", "| reference | ref_rows | overlap | overlap_rate |", "| --- | --- | --- | --- |"]
    for entry in overlap_results:
        md_lines.append(
            f"| {entry['reference_csv']} | {entry['reference_rows']} | "
            f"{entry['overlap_rows']} | {entry['overlap_rate'] * 100:.4f}% |"
        )
    with open(os.path.join(output_dir, "single_csv_audit.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(md_lines) + "\n")

    overlap_fields = ["reference_csv", "exists", "reference_rows",
                      "reference_unique_reactions", "overlap_rows", "overlap_rate"]
    write_csv(os.path.join(output_dir, "overlap_analysis.csv"), overlap_results, overlap_fields)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-json", default=None,
                        help="Manifest JSON (manifest audit mode)")
    parser.add_argument("--input", default=None,
                        help="Single normalized CSV to audit (single-CSV audit mode, "
                             "added in P1-09 for ORD data quality)")
    parser.add_argument("--overlap-csvs", default=None,
                        help="Comma-separated reference CSVs for overlap analysis "
                             "(single-CSV mode only)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--root", default=None, help="Override manifest root path")
    args = parser.parse_args()

    # Single-CSV audit mode (P1-09)
    if args.input:
        overlap_csvs = [p.strip() for p in (args.overlap_csvs or "").split(",") if p.strip()]
        summary = audit_single_normalized_csv(args.input, args.output_dir, overlap_csvs)
        print(json.dumps({
            "mode": "single_csv_audit",
            "input": args.input,
            "rows": summary["rows"],
            "rdkit_valid_rate": summary["rdkit_valid_reaction_rate"],
            "atom_mapping_coverage_rate": summary["atom_mapping_coverage_rate"],
            "outputs": {
                "audit_json": os.path.join(args.output_dir, "single_csv_audit.json"),
                "audit_md": os.path.join(args.output_dir, "single_csv_audit.md"),
                "overlap_csv": os.path.join(args.output_dir, "overlap_analysis.csv"),
            },
        }, indent=2, ensure_ascii=False))
        return

    if not args.manifest_json:
        parser.error("either --manifest-json or --input is required")

    manifest = load_manifest(args.manifest_json)
    root = args.root or str(manifest.get("server_root") or ".")
    real_specs = manifest.get("datasets", {}).get("real", [])
    synthetic_specs = manifest.get("datasets", {}).get("synthetic", [])
    os.makedirs(args.output_dir, exist_ok=True)

    source_split: Dict[str, str] = {}
    real_summaries = []
    all_positive_products: List[Dict[str, str]] = []
    for spec in real_specs:
        dataset_id = str(spec["id"])
        path = rel_or_abs(root, str(spec["path"]))
        summary, positive_products = audit_real_dataset(path, dataset_id)
        real_summaries.append(summary)
        all_positive_products.extend(positive_products)
        rows, _ = read_csv(path)
        for row in rows:
            source_id = row.get("source_id", "")
            split = row.get("split", "")
            if source_id and split:
                source_split[source_id] = split

    known_positive_products = {row["canonical_product"] for row in all_positive_products if row.get("canonical_product")}
    synthetic_summaries = []
    for spec in synthetic_specs:
        dataset_id = str(spec["id"])
        path = rel_or_abs(root, str(spec["path"]))
        synthetic_summaries.append(audit_synthetic_dataset(path, dataset_id, source_split, known_positive_products))

    real_fields = [
        "dataset_id",
        "rows",
        "usable_rows",
        "positive_rows",
        "split_counts",
        "label_counts",
        "rdkit_valid_rate",
        "source_id_split_leaks",
        "reactant_context_split_leaks",
        "parent_reaction_split_leaks",
        "duplicate_parent_reactions",
        "missing_required_fields",
    ]
    synthetic_fields = [
        "dataset_id",
        "rows",
        "keep_rows",
        "parent_split_counts",
        "rdkit_valid_rate",
        "unknown_source_id_rows",
        "non_train_parent_rows",
        "unique_candidate_reactions",
        "duplicate_candidate_reactions",
        "known_positive_product_overlap_rows",
        "keep_known_positive_product_overlap_rows",
        "missing_required_fields",
        "has_candidate_reaction_field",
    ]
    real_table = [flatten_real(summary) for summary in real_summaries]
    synthetic_table = [flatten_synthetic(summary) for summary in synthetic_summaries]
    write_csv(os.path.join(args.output_dir, "real_dataset_audit.csv"), real_table, real_fields)
    write_markdown(os.path.join(args.output_dir, "real_dataset_audit.md"), real_table, real_fields)
    write_csv(os.path.join(args.output_dir, "synthetic_dataset_audit.csv"), synthetic_table, synthetic_fields)
    write_markdown(os.path.join(args.output_dir, "synthetic_dataset_audit.md"), synthetic_table, synthetic_fields)

    gate_failures = []
    warnings = []
    for summary in real_summaries:
        if summary["missing_required_fields"]:
            gate_failures.append(f"{summary['dataset_id']}: missing required fields")
        if summary["source_id_split_leaks"]:
            gate_failures.append(f"{summary['dataset_id']}: source_id split leakage")
        if summary["parent_reaction_split_leaks"]:
            gate_failures.append(f"{summary['dataset_id']}: parent reaction split leakage")
        if summary["reactant_context_split_leaks"]:
            warnings.append(f"{summary['dataset_id']}: reactant-context appears in multiple splits")
        if float(summary["rdkit_valid_reaction_rate"]) < 0.99:
            gate_failures.append(f"{summary['dataset_id']}: RDKit-valid rate below 99%")
    for summary in synthetic_summaries:
        if summary["missing_required_fields"] or not summary["has_candidate_reaction_field"]:
            gate_failures.append(f"{summary['dataset_id']}: missing synthetic schema")
        if summary["unknown_source_id_rows"]:
            gate_failures.append(f"{summary['dataset_id']}: synthetic rows with unknown source_id")
        if summary["non_train_parent_rows"]:
            warnings.append(
                f"{summary['dataset_id']}: CSV contains synthetic rows attached to non-train parent; training reader must keep filtering train parents"
            )
        if summary["keep_known_positive_product_overlap_rows"]:
            gate_failures.append(f"{summary['dataset_id']}: keep_synthetic_negative rows overlap known positive products")
        elif summary["known_positive_product_overlap_rows"]:
            warnings.append(f"{summary['dataset_id']}: non-keep or mixed rows overlap known positive products")
        if float(summary["rdkit_valid_reaction_rate"]) < 0.99:
            gate_failures.append(f"{summary['dataset_id']}: RDKit-valid rate below 99%")

    status = "pass" if not gate_failures and not warnings else ("pass_with_warnings" if not gate_failures else "needs_review")
    payload = {
        "manifest_json": args.manifest_json,
        "root": root,
        "real_datasets": real_summaries,
        "synthetic_datasets": synthetic_summaries,
        "known_positive_products": len(known_positive_products),
        "gate_failures": gate_failures,
        "warnings": warnings,
        "status": status,
        "outputs": {
            "real_dataset_audit_csv": os.path.join(args.output_dir, "real_dataset_audit.csv"),
            "real_dataset_audit_md": os.path.join(args.output_dir, "real_dataset_audit.md"),
            "synthetic_dataset_audit_csv": os.path.join(args.output_dir, "synthetic_dataset_audit.csv"),
            "synthetic_dataset_audit_md": os.path.join(args.output_dir, "synthetic_dataset_audit.md"),
            "summary_json": os.path.join(args.output_dir, "benchmark_data_quality_audit.json"),
        },
    }
    with open(os.path.join(args.output_dir, "benchmark_data_quality_audit.json"), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    with open(os.path.join(args.output_dir, "manifest_snapshot.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "gate_failures": gate_failures,
                "warnings": warnings,
                "outputs": payload["outputs"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
