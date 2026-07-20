"""Audit the 25k external bridge base candidate set before beam scoring.

This CPU-only audit verifies that the merged external contexts and the
observed+PC-CNG candidate set are ready for later strict/validity-aware scoring.
It deliberately does not require Chemformer beams or model scores.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from statistics import mean, median
from typing import Dict, Iterable, List, Mapping, Sequence, Set

from .chem_utils import is_valid_reaction
from .reaction_lm_scorer import canonical_smiles


def nonempty(value: object) -> str:
    return str(value or "").strip()


def read_csv(path: str) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def product_key(product: str) -> str:
    return canonical_smiles(product) or nonempty(product)


def split_source_tags(candidate_source: str) -> List[str]:
    tags = [tag.strip() for tag in nonempty(candidate_source).split("+") if tag.strip()]
    return tags or ["unknown"]


def is_pc_cng(row: Mapping[str, str]) -> bool:
    return any("pc_cng" in tag for tag in split_source_tags(row.get("candidate_source", "")))


def is_observed(row: Mapping[str, str]) -> bool:
    return any(tag == "observed_positive" for tag in split_source_tags(row.get("candidate_source", "")))


def is_negative(row: Mapping[str, str]) -> bool:
    return nonempty(row.get("label")) == "0"


def numeric_summary(values: Sequence[int]) -> Dict[str, float | int]:
    if not values:
        return {"min": 0, "median": 0, "mean": 0.0, "p90": 0, "max": 0}
    ordered = sorted(values)
    p90_index = min(len(ordered) - 1, int(0.9 * (len(ordered) - 1)))
    return {
        "min": ordered[0],
        "median": float(median(ordered)),
        "mean": float(mean(ordered)),
        "p90": ordered[p90_index],
        "max": ordered[-1],
    }


def first_examples(rows: Iterable[Mapping[str, object]], limit: int) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for row in rows:
        out.append(dict(row))
        if len(out) >= limit:
            break
    return out


def source_counter(rows: Iterable[Mapping[str, str]]) -> Dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        for tag in split_source_tags(row.get("candidate_source", "")):
            counts[tag] += 1
    return dict(sorted(counts.items()))


def decision_flags(
    missing_positive_groups: int,
    bad_positive_multiplicity_groups: int,
    duplicate_product_groups: int,
    pc_cng_negative_coverage: float,
    invalid_candidate_reaction_rows: int,
    invalid_pc_cng_negative_rows: int,
) -> Dict[str, List[str]]:
    hard_failures: List[str] = []
    warnings: List[str] = []
    if missing_positive_groups:
        hard_failures.append("missing_observed_positive_groups")
    if bad_positive_multiplicity_groups:
        hard_failures.append("observed_positive_multiplicity_not_one")
    if duplicate_product_groups:
        warnings.append("duplicate_candidate_product_within_group")
    if pc_cng_negative_coverage < 0.99:
        warnings.append("pc_cng_negative_group_coverage_below_99pct")
    if invalid_candidate_reaction_rows:
        warnings.append("invalid_candidate_reactions")
    if invalid_pc_cng_negative_rows:
        warnings.append("invalid_pc_cng_negative_reactions")
    return {"hard_failures": hard_failures, "warnings": warnings}


def write_markdown(path: str, payload: Mapping[str, object]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    flags = dict(payload.get("decision_flags", {}))
    lines = [
        "# External 25k Base Candidate Quality Audit",
        "",
        "| Item | Value |",
        "|---|---|",
        f"| Context rows / groups | `{payload['context_rows']}` / `{payload['context_groups']}` |",
        f"| Candidate rows / groups | `{payload['candidate_rows']}` / `{payload['candidate_groups']}` |",
        f"| Observed-positive groups | `{payload['observed_positive_groups']}` |",
        f"| PC-CNG negative rows / groups | `{payload['pc_cng_negative_rows']}` / `{payload['pc_cng_negative_groups']}` |",
        f"| PC-CNG negative coverage | `{payload['pc_cng_negative_group_coverage']}` |",
        f"| Missing observed-positive groups | `{payload['missing_observed_positive_groups']}` |",
        f"| Missing PC-CNG negative groups | `{payload['missing_pc_cng_negative_groups']}` |",
        f"| Duplicate candidate-product groups | `{payload['duplicate_candidate_product_groups']}` |",
        f"| Same-product PC-CNG negatives | `{payload['same_product_pc_cng_negative_rows']}` |",
        f"| Invalid candidate reactions | `{payload['invalid_candidate_reaction_rows']}` |",
        f"| Invalid PC-CNG negative reactions | `{payload['invalid_pc_cng_negative_rows']}` |",
        f"| Decision | `{payload['decision']}` |",
        f"| Hard failures | `{', '.join(flags.get('hard_failures', []))}` |",
        f"| Warnings | `{', '.join(flags.get('warnings', []))}` |",
        "",
        "## Candidate Count Summaries",
        "",
        "| Scope | Summary |",
        "|---|---|",
        f"| All candidates per context | `{payload['candidate_count_summary']}` |",
        f"| PC-CNG negatives per context | `{payload['pc_cng_negative_count_summary']}` |",
        "",
        "## Candidate Source Counts",
        "",
        "| Source | Rows |",
        "|---|---:|",
    ]
    for source, count in dict(payload.get("candidate_source_counts", {})).items():
        lines.append(f"| `{source}` | {count} |")
    lines.append("")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def run_audit(
    contexts_csv: str,
    candidates_csv: str,
    output_dir: str,
    example_limit: int,
) -> Dict[str, object]:
    contexts = read_csv(contexts_csv)
    candidates = read_csv(candidates_csv)
    os.makedirs(output_dir, exist_ok=True)

    context_groups: Set[str] = {nonempty(row.get("group_id")) for row in contexts if nonempty(row.get("group_id"))}
    observed_product_by_group = {
        nonempty(row.get("group_id")): product_key(nonempty(row.get("observed_product")))
        for row in contexts
        if nonempty(row.get("group_id"))
    }

    groups_seen: Set[str] = set()
    group_rows: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    positive_counts: Counter[str] = Counter()
    pc_cng_negative_counts: Counter[str] = Counter()
    duplicate_product_groups: Set[str] = set()
    duplicate_product_rows = 0
    same_product_pc_cng_negatives: List[Dict[str, object]] = []
    invalid_rows: List[Dict[str, object]] = []
    invalid_pc_cng_negative_rows: List[Dict[str, object]] = []
    label_counts: Counter[str] = Counter()

    for row_index, row in enumerate(candidates, start=1):
        group = nonempty(row.get("group_id"))
        if group:
            groups_seen.add(group)
            group_rows[group].append(row)
        label_counts[nonempty(row.get("label")) or "missing"] += 1
        if is_observed(row) and nonempty(row.get("label")) == "1":
            positive_counts[group] += 1
        if is_pc_cng(row) and is_negative(row):
            pc_cng_negative_counts[group] += 1
            if product_key(nonempty(row.get("candidate_product"))) == observed_product_by_group.get(group, ""):
                same_product_pc_cng_negatives.append(
                    {
                        "row_index": row_index,
                        "group_id": group,
                        "source_id": row.get("source_id", ""),
                        "candidate_product": row.get("candidate_product", ""),
                    }
                )
        reaction = nonempty(row.get("candidate_reaction"))
        if reaction and not is_valid_reaction(reaction):
            entry = {
                "row_index": row_index,
                "group_id": group,
                "source_id": row.get("source_id", ""),
                "candidate_source": row.get("candidate_source", ""),
                "candidate_reaction": reaction,
            }
            invalid_rows.append(entry)
            if is_pc_cng(row) and is_negative(row):
                invalid_pc_cng_negative_rows.append(entry)

    for group, rows in group_rows.items():
        product_counts: Counter[str] = Counter(product_key(nonempty(row.get("candidate_product"))) for row in rows)
        duplicate_rows = sum(count - 1 for count in product_counts.values() if count > 1)
        if duplicate_rows:
            duplicate_product_groups.add(group)
            duplicate_product_rows += duplicate_rows

    missing_positive_groups = sorted(group for group in context_groups if positive_counts[group] == 0)
    bad_positive_multiplicity = sorted(group for group in context_groups if positive_counts[group] not in {1})
    missing_pc_cng_groups = sorted(group for group in context_groups if pc_cng_negative_counts[group] == 0)
    candidate_counts = [len(group_rows.get(group, [])) for group in sorted(context_groups)]
    pc_cng_counts = [pc_cng_negative_counts[group] for group in sorted(context_groups)]
    pc_cng_negative_coverage = (
        (len(context_groups) - len(missing_pc_cng_groups)) / len(context_groups) if context_groups else 0.0
    )
    flags = decision_flags(
        missing_positive_groups=len(missing_positive_groups),
        bad_positive_multiplicity_groups=len(bad_positive_multiplicity),
        duplicate_product_groups=len(duplicate_product_groups),
        pc_cng_negative_coverage=pc_cng_negative_coverage,
        invalid_candidate_reaction_rows=len(invalid_rows),
        invalid_pc_cng_negative_rows=len(invalid_pc_cng_negative_rows),
    )
    decision = "fail" if flags["hard_failures"] else ("pass_with_warnings" if flags["warnings"] else "pass")

    payload: Dict[str, object] = {
        "audit": "external_25k_base_candidate_quality",
        "contexts_csv": contexts_csv,
        "candidates_csv": candidates_csv,
        "context_rows": len(contexts),
        "context_groups": len(context_groups),
        "candidate_rows": len(candidates),
        "candidate_groups": len(groups_seen),
        "observed_positive_rows": sum(positive_counts.values()),
        "observed_positive_groups": sum(1 for group in context_groups if positive_counts[group] > 0),
        "pc_cng_negative_rows": sum(pc_cng_negative_counts.values()),
        "pc_cng_negative_groups": sum(1 for group in context_groups if pc_cng_negative_counts[group] > 0),
        "pc_cng_negative_group_coverage": pc_cng_negative_coverage,
        "missing_observed_positive_groups": len(missing_positive_groups),
        "bad_observed_positive_multiplicity_groups": len(bad_positive_multiplicity),
        "missing_pc_cng_negative_groups": len(missing_pc_cng_groups),
        "duplicate_candidate_product_groups": len(duplicate_product_groups),
        "duplicate_candidate_product_extra_rows": duplicate_product_rows,
        "same_product_pc_cng_negative_rows": len(same_product_pc_cng_negatives),
        "invalid_candidate_reaction_rows": len(invalid_rows),
        "invalid_pc_cng_negative_rows": len(invalid_pc_cng_negative_rows),
        "label_counts": dict(sorted(label_counts.items())),
        "candidate_source_counts": source_counter(candidates),
        "candidate_count_summary": numeric_summary(candidate_counts),
        "pc_cng_negative_count_summary": numeric_summary(pc_cng_counts),
        "decision_flags": flags,
        "decision": decision,
        "examples": {
            "missing_observed_positive_groups": missing_positive_groups[:example_limit],
            "bad_observed_positive_multiplicity_groups": bad_positive_multiplicity[:example_limit],
            "missing_pc_cng_negative_groups": missing_pc_cng_groups[:example_limit],
            "same_product_pc_cng_negatives": first_examples(same_product_pc_cng_negatives, example_limit),
            "invalid_candidate_reactions": first_examples(invalid_rows, example_limit),
        },
    }
    json_path = os.path.join(output_dir, "external_25k_base_candidate_quality_audit.json")
    md_path = os.path.join(output_dir, "external_25k_base_candidate_quality_audit.md")
    payload["outputs"] = {"json": json_path, "md": md_path}
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    write_markdown(md_path, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contexts-csv", required=True)
    parser.add_argument("--candidates-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--example-limit", type=int, default=20)
    args = parser.parse_args()
    payload = run_audit(
        contexts_csv=args.contexts_csv,
        candidates_csv=args.candidates_csv,
        output_dir=args.output_dir,
        example_limit=args.example_limit,
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
