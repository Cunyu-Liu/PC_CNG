"""Select additional contexts for the external product-prediction bridge.

This prepares CPU-only inputs for scaling the Chemformer/Molecular Transformer
bridge. It does not generate beams or score models; it only selects safe
positive reaction contexts and writes Chemformer input tables for the next step.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .build_external_product_prediction_candidate_set import CONTEXT_FIELDS
from .chem_utils import canonicalize_smiles, split_reaction
from .reaction_lm_scorer import chemformer_table_value

try:  # pragma: no cover - depends on optional RDKit install
    from rdkit import RDLogger  # type: ignore

    RDLogger.DisableLog("rdApp.*")
except Exception:  # pragma: no cover
    pass


POSITIVE_LABEL = "positive"
SMILES_CACHE: Dict[str, Optional[str]] = {}


def cached_canonicalize_smiles(smiles: str) -> Optional[str]:
    key = smiles or ""
    if key not in SMILES_CACHE:
        SMILES_CACHE[key] = canonicalize_smiles(key)
    return SMILES_CACHE[key]


def reaction_parts(row: Dict[str, str]) -> Tuple[str, str, str]:
    reaction = row.get("reaction_smiles", "")
    if reaction:
        try:
            return split_reaction(reaction)
        except ValueError:
            return "", "", ""
    return row.get("reactants", ""), row.get("agents", ""), row.get("products", "")


def canonical_reactants(row: Dict[str, str]) -> Optional[str]:
    reactants = row.get("reactants", "")
    if not reactants:
        reactants, _, _ = reaction_parts(row)
    if not reactants:
        return None
    return cached_canonicalize_smiles(reactants)


def parse_yield(value: str) -> Optional[float]:
    text = str(value or "").strip().replace("%", "")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def iter_csv_rows(path: str) -> Iterable[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            yield dict(row)


def external_context_id(dataset: str, split: str, source_id: str) -> str:
    return f"external_product_prediction|{dataset}|{split}|{source_id}"


def read_existing_contexts(paths: Sequence[str]) -> Dict[str, object]:
    source_ids: Set[str] = set()
    canonical_contexts: Set[str] = set()
    groups: Set[str] = set()
    split_counts: Counter[str] = Counter()
    dataset_counts: Counter[str] = Counter()
    rows = 0
    context_rows: List[Dict[str, str]] = []

    for path in paths:
        for row in iter_csv_rows(path):
            rows += 1
            context_rows.append(dict(row))
            source_id = (row.get("source_id") or "").strip()
            group_id = (row.get("group_id") or "").strip()
            split = row.get("split") or "unknown"
            dataset = row.get("dataset") or "unknown"
            if source_id:
                source_ids.add(source_id)
            if group_id:
                groups.add(group_id)
            split_counts[split] += 1
            dataset_counts[dataset] += 1
            can_context = canonical_reactants(row)
            if can_context:
                canonical_contexts.add(can_context)

    return {
        "rows": rows,
        "groups": groups,
        "source_ids": source_ids,
        "canonical_contexts": canonical_contexts,
        "split_counts": dict(sorted(split_counts.items())),
        "dataset_counts": dict(sorted(dataset_counts.items())),
        "context_rows": context_rows,
    }


def source_context_splits(source_csv: str) -> Tuple[Dict[str, Set[str]], Dict[str, int]]:
    splits_by_context: Dict[str, Set[str]] = defaultdict(set)
    stats = {"rows": 0, "positive_rows": 0, "canonical_context_failures": 0}
    for row in iter_csv_rows(source_csv):
        stats["rows"] += 1
        if row.get("label_type") != POSITIVE_LABEL:
            continue
        stats["positive_rows"] += 1
        context = canonical_reactants(row)
        if not context:
            stats["canonical_context_failures"] += 1
            continue
        splits_by_context[context].add(row.get("split") or "unknown")
    return splits_by_context, stats


def better_representative(new_row: Dict[str, object], old_row: Dict[str, object]) -> bool:
    new_yield = new_row.get("yield_numeric")
    old_yield = old_row.get("yield_numeric")
    new_score = float(new_yield) if isinstance(new_yield, (int, float)) else -1.0
    old_score = float(old_yield) if isinstance(old_yield, (int, float)) else -1.0
    if new_score != old_score:
        return new_score > old_score
    return str(new_row.get("source_id", "")) < str(old_row.get("source_id", ""))


def to_context_row(row: Dict[str, object]) -> Dict[str, str]:
    reactants = str(row.get("reactants") or "")
    agents = str(row.get("agents") or "")
    products = str(row.get("observed_product") or row.get("products") or "")
    split = str(row.get("split") or "unknown")
    dataset = str(row.get("dataset") or "external_context_expansion")
    source_id = str(row.get("source_id") or "")
    return {
        "row_index": "",
        "group_id": external_context_id(dataset, split, source_id),
        "source_id": source_id,
        "reactants": reactants,
        "agents": agents,
        "observed_product": products,
        "split": split,
        "dataset": dataset,
        "reaction_class": str(row.get("reaction_class") or ""),
    }


def scan_source_candidates(
    source_csv: str,
    existing_source_ids: Set[str],
    existing_contexts: Set[str],
    source_splits: Dict[str, Set[str]],
    reject_example_limit: int,
    include_splits: Optional[Set[str]] = None,
    exclude_source_ids: Optional[Set[str]] = None,
) -> Tuple[List[Dict[str, object]], Dict[str, int], List[Dict[str, object]]]:
    representatives: Dict[str, Dict[str, object]] = {}
    reject_counts: Counter[str] = Counter()
    reject_examples: List[Dict[str, object]] = []

    for row in iter_csv_rows(source_csv):
        if row.get("label_type") != POSITIVE_LABEL:
            reject_counts["non_positive"] += 1
            continue
        source_id = (row.get("source_id") or "").strip()
        if not source_id:
            reject_counts["missing_source_id"] += 1
            continue
        if exclude_source_ids is not None and source_id in exclude_source_ids:
            reject_counts["source_id_explicitly_excluded"] += 1
            if len(reject_examples) < reject_example_limit:
                reject_examples.append({"reason": "source_id_explicitly_excluded", "source_id": source_id})
            continue
        split = row.get("split") or "unknown"
        if include_splits is not None and split not in include_splits:
            reject_counts["split_not_selected"] += 1
            continue
        if source_id in existing_source_ids:
            reject_counts["source_id_seen_in_existing_external"] += 1
            continue

        reactants, agents, products = reaction_parts(row)
        if not reactants or not products:
            reject_counts["invalid_reaction_parts"] += 1
            continue
        context = canonical_reactants(row)
        if not context:
            reject_counts["canonical_context_failed"] += 1
            continue
        if context in existing_contexts:
            reject_counts["context_seen_in_existing_external"] += 1
            if len(reject_examples) < reject_example_limit:
                reject_examples.append({"reason": "context_seen_in_existing_external", "source_id": source_id})
            continue
        if len(source_splits.get(context, set())) > 1:
            reject_counts["source_context_cross_split"] += 1
            if len(reject_examples) < reject_example_limit:
                reject_examples.append({"reason": "source_context_cross_split", "source_id": source_id})
            continue

        item: Dict[str, object] = {
            "source_id": source_id,
            "reactants": reactants,
            "agents": agents,
            "products": products,
            "observed_product": products,
            "split": split,
            "dataset": row.get("source") or os.path.splitext(os.path.basename(source_csv))[0],
            "reaction_class": row.get("reaction_class", ""),
            "reaction_smiles": row.get("reaction_smiles", ""),
            "yield": row.get("yield", ""),
            "yield_numeric": parse_yield(row.get("yield", "")),
            "canonical_reactants": context,
        }
        old = representatives.get(context)
        if old is None or better_representative(item, old):
            representatives[context] = item

    candidates = sorted(
        representatives.values(),
        key=lambda row: (
            str(row.get("split") or ""),
            -(float(row["yield_numeric"]) if isinstance(row.get("yield_numeric"), (int, float)) else -1.0),
            str(row.get("source_id") or ""),
        ),
    )
    return candidates, dict(reject_counts), reject_examples


def write_contexts(path: str, rows: Sequence[Dict[str, str]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CONTEXT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for idx, row in enumerate(rows):
            out = dict(row)
            out["row_index"] = str(idx)
            writer.writerow(out)


def write_chemformer_input(path: str, contexts: Sequence[Dict[str, str]], include_agents: bool) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["reactants", "products", "set"], delimiter="\t")
        writer.writeheader()
        for context in contexts:
            source = context["reactants"]
            if include_agents and context.get("agents"):
                source = f"{source}>{context['agents']}"
            writer.writerow(
                {
                    "reactants": chemformer_table_value(source),
                    "products": chemformer_table_value(context["observed_product"]),
                    "set": context.get("split") or "test",
                }
            )


def write_markdown(path: str, summary: Dict[str, object]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lines = [
        "# External Product Prediction Context Expansion",
        "",
        "| Item | Value |",
        "|---|---|",
        f"| Existing contexts | `{summary['existing_context_groups']}` |",
        f"| Target contexts | `{summary['target_total_contexts']}` |",
        f"| Needed additional contexts | `{summary['needed_additional_contexts']}` |",
        f"| Eligible unique source contexts | `{summary['eligible_unique_source_contexts']}` |",
        f"| Selected contexts | `{summary['selected_contexts']}` |",
        f"| Merged contexts | `{summary['merged_contexts']}` |",
        f"| Can cover target | `{summary['selected_can_cover_target']}` |",
        f"| Selected split counts | `{summary['selected_split_counts']}` |",
        f"| Selected dataset counts | `{summary['selected_dataset_counts']}` |",
    ]
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def select_contexts(
    existing_context_csvs: Sequence[str],
    source_csv: str,
    output_dir: str,
    target_total_contexts: int,
    max_selected: Optional[int],
    reject_example_limit: int,
    include_agents: bool,
    include_splits: Optional[Set[str]] = None,
    exclude_source_ids: Optional[Set[str]] = None,
) -> Dict[str, object]:
    os.makedirs(output_dir, exist_ok=True)
    existing = read_existing_contexts(existing_context_csvs)
    existing_rows = [dict(row) for row in existing["context_rows"]]  # type: ignore[index]
    source_splits, source_stats = source_context_splits(source_csv)
    candidates, reject_counts, reject_examples = scan_source_candidates(
        source_csv=source_csv,
        existing_source_ids=existing["source_ids"],  # type: ignore[arg-type]
        existing_contexts=existing["canonical_contexts"],  # type: ignore[arg-type]
        source_splits=source_splits,
        reject_example_limit=reject_example_limit,
        include_splits=include_splits,
        exclude_source_ids=exclude_source_ids,
    )
    existing_count = len(existing["groups"])  # type: ignore[arg-type]
    needed = max(0, target_total_contexts - existing_count)
    select_n = needed if max_selected is None else min(max_selected, needed)
    selected = [to_context_row(row) for row in candidates[:select_n]]
    merged = existing_rows + selected

    expansion_contexts_csv = os.path.join(output_dir, "external_product_prediction_context_expansion.csv")
    merged_contexts_csv = os.path.join(output_dir, "external_product_prediction_contexts_merged.csv")
    expansion_chemformer_input = os.path.join(output_dir, "external_product_prediction_context_expansion_chemformer_input.csv")
    merged_chemformer_input = os.path.join(output_dir, "external_product_prediction_contexts_merged_chemformer_input.csv")
    summary_json = os.path.join(output_dir, "external_product_prediction_context_expansion_summary.json")
    summary_md = os.path.join(output_dir, "external_product_prediction_context_expansion_summary.md")
    reject_examples_csv = os.path.join(output_dir, "external_product_prediction_context_expansion_reject_examples.csv")

    write_contexts(expansion_contexts_csv, selected)
    write_contexts(merged_contexts_csv, merged)
    write_chemformer_input(expansion_chemformer_input, selected, include_agents)
    write_chemformer_input(merged_chemformer_input, merged, include_agents)

    selected_split_counts = Counter(row["split"] for row in selected)
    selected_dataset_counts = Counter(row["dataset"] for row in selected)
    summary: Dict[str, object] = {
        "source_csv": source_csv,
        "existing_context_csv": list(existing_context_csvs),
        "target_total_contexts": target_total_contexts,
        "include_splits": sorted(include_splits) if include_splits is not None else None,
        "exclude_source_ids": sorted(exclude_source_ids) if exclude_source_ids else None,
        "existing_context_groups": existing_count,
        "needed_additional_contexts": needed,
        "eligible_unique_source_contexts": len(candidates),
        "selected_contexts": len(selected),
        "merged_contexts": len(merged),
        "selected_can_cover_target": len(selected) >= needed,
        "selected_split_counts": dict(sorted(selected_split_counts.items())),
        "selected_dataset_counts": dict(sorted(selected_dataset_counts.items())),
        "existing_split_counts": existing["split_counts"],
        "existing_dataset_counts": existing["dataset_counts"],
        "source_first_pass": source_stats,
        "reject_reason_counts": reject_counts,
        "outputs": {
            "expansion_contexts_csv": expansion_contexts_csv,
            "merged_contexts_csv": merged_contexts_csv,
            "expansion_chemformer_input_csv": expansion_chemformer_input,
            "merged_chemformer_input_csv": merged_chemformer_input,
            "summary_json": summary_json,
            "summary_md": summary_md,
            "reject_examples_csv": reject_examples_csv,
        },
    }
    with open(summary_json, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    write_markdown(summary_md, summary)

    if reject_examples:
        with open(reject_examples_csv, "w", newline="", encoding="utf-8") as handle:
            fields = ["reason", "source_id"]
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(reject_examples)
    else:
        with open(reject_examples_csv, "w", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=["reason", "source_id"]).writeheader()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--existing-context-csv", action="append", required=True)
    parser.add_argument("--source-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-total-contexts", type=int, default=25000)
    parser.add_argument("--max-selected", type=int, default=None)
    parser.add_argument("--reject-example-limit", type=int, default=50)
    parser.add_argument("--include-agents", dest="include_agents", action="store_true", default=True)
    parser.add_argument("--exclude-agents", dest="include_agents", action="store_false")
    parser.add_argument("--include-split", action="append", default=None, help="Only select source rows from this split; repeatable")
    parser.add_argument("--exclude-source-id", action="append", default=None, help="Exclude a source_id from selection; repeatable")
    args = parser.parse_args()

    summary = select_contexts(
        existing_context_csvs=args.existing_context_csv,
        source_csv=args.source_csv,
        output_dir=args.output_dir,
        target_total_contexts=args.target_total_contexts,
        max_selected=args.max_selected,
        reject_example_limit=args.reject_example_limit,
        include_agents=args.include_agents,
        include_splits=set(args.include_split) if args.include_split else None,
        exclude_source_ids=set(args.exclude_source_id) if args.exclude_source_id else None,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
