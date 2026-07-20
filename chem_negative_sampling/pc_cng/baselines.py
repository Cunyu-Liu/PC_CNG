"""Baseline negative generators for fair PC-CNG comparisons."""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Dict, Iterable, List

from .chem_utils import join_reaction, molecule_parts, replace_first, split_reaction, token_jaccard
from .counterfactual import CounterfactualGenerator, PRODUCT_REPLACEMENTS, REACTANT_REPLACEMENTS
from .validator import CounterfactualValidator


def load_positive_rows(path: str, limit: int | None = None) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader, start=1):
            if limit is not None and len(rows) >= limit:
                break
            reaction = (row.get("reaction_smiles") or "").strip()
            if not reaction:
                continue
            rows.append(
                {
                    "source_id": (row.get("source_id") or row.get("id") or f"row_{index:09d}").strip(),
                    "reaction_smiles": reaction,
                }
            )
    return rows


def random_mismatch(rows: List[Dict[str, str]]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    if len(rows) < 2:
        return out
    validator = CounterfactualValidator()
    products = []
    for row in rows:
        try:
            _, _, product = split_reaction(row["reaction_smiles"])
            products.append(product)
        except Exception:
            products.append("")
    for index, row in enumerate(rows):
        try:
            reactants, agents, product = split_reaction(row["reaction_smiles"])
        except Exception:
            continue
        other_product = products[(index + 1) % len(products)]
        candidate = join_reaction(reactants, other_product, agents)
        scores = validator.score(row["reaction_smiles"], candidate, "random_product_mismatch", "forward_outcome")
        out.append(_record(row, candidate, "random", "random_product_mismatch", "rotate_product", scores.to_dict()))
    return out


def template_perturbation(rows: List[Dict[str, str]]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    validator = CounterfactualValidator()
    for row in rows:
        try:
            reactants, agents, product = split_reaction(row["reaction_smiles"])
        except Exception:
            continue
        edited = replace_first(product, PRODUCT_REPLACEMENTS)
        if not edited:
            continue
        new_product, action = edited
        candidate = join_reaction(reactants, new_product, agents)
        scores = validator.score(row["reaction_smiles"], candidate, "template_product_perturbation", "forward_outcome")
        out.append(_record(row, candidate, "template_perturbation", "template_product_perturbation", action, scores.to_dict()))
    return out


def dora_alternate_center(rows: List[Dict[str, str]]) -> List[Dict[str, object]]:
    """Organic-reaction analogue of DORA alternate reaction center.

    This is only a baseline approximation: it mutates one precursor functional
    group while keeping the target product fixed.
    """
    out: List[Dict[str, object]] = []
    validator = CounterfactualValidator()
    for row in rows:
        try:
            reactants, agents, product = split_reaction(row["reaction_smiles"])
        except Exception:
            continue
        parts = molecule_parts(reactants)
        for part_index, part in enumerate(parts):
            edited = replace_first(part, REACTANT_REPLACEMENTS)
            if not edited:
                continue
            new_part, action = edited
            new_parts = list(parts)
            new_parts[part_index] = new_part
            candidate = join_reaction(".".join(new_parts), product, agents)
            scores = validator.score(row["reaction_smiles"], candidate, "dora_style_alternate_center", "retro_precursor")
            out.append(_record(row, candidate, "dora_alternate_center", "dora_style_alternate_center", action, scores.to_dict()))
            break
    return out


def pu_reliable_negative(rows: List[Dict[str, str]], threshold: float = 0.2) -> List[Dict[str, object]]:
    """Simple PU-style reliable-negative baseline.

    A candidate is considered reliable negative if product token overlap with
    the parent product is low after random mismatch. This approximates a
    high-precision reliable-negative extraction, not a final PU algorithm.
    """
    out: List[Dict[str, object]] = []
    mismatched = random_mismatch(rows)
    for row in mismatched:
        try:
            _, _, parent_product = split_reaction(str(row["positive_reaction"]))
            _, _, candidate_product = split_reaction(str(row["candidate_reaction"]))
        except Exception:
            continue
        if token_jaccard(parent_product, candidate_product) <= threshold:
            row = dict(row)
            row["baseline"] = "pu_reliable_negative"
            row["failure_type"] = "pu_low_overlap_reliable_negative"
            out.append(row)
    return out


def pc_cng_rule(rows: List[Dict[str, str]]) -> List[Dict[str, object]]:
    generator = CounterfactualGenerator()
    out: List[Dict[str, object]] = []
    for row in rows:
        for candidate in generator.generate_for_reaction(row["reaction_smiles"], row["source_id"]):
            data = candidate.to_dict()
            data["baseline"] = "pc_cng_rule_mvp"
            out.append(data)
    return out


def _record(
    parent: Dict[str, str],
    candidate: str,
    baseline: str,
    failure_type: str,
    edit_action: str,
    scores: Dict[str, object],
) -> Dict[str, object]:
    reactants, _, products = split_reaction(parent["reaction_smiles"])
    cand_reactants, _, cand_products = split_reaction(candidate)
    return {
        "source_id": parent["source_id"],
        "positive_reaction": parent["reaction_smiles"],
        "candidate_reaction": candidate,
        "task": "baseline",
        "failure_type": failure_type,
        "edit_action": edit_action,
        "parent_reactants": reactants,
        "parent_product": products,
        "candidate_reactants": cand_reactants,
        "candidate_product": cand_products,
        "label": 0,
        "provenance": "baseline_synthetic_counterfactual",
        "baseline": baseline,
        **scores,
    }


def write_rows(path: str, rows: Iterable[Dict[str, object]]) -> int:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = [
        "baseline",
        "source_id",
        "positive_reaction",
        "candidate_reaction",
        "task",
        "failure_type",
        "edit_action",
        "parent_reactants",
        "parent_product",
        "candidate_reactants",
        "candidate_product",
        "valid",
        "atom_balance",
        "locality",
        "closeness",
        "hard_score",
        "false_negative_risk",
        "passes_filter",
        "label",
        "provenance",
    ]
    count = 0
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    rows = load_positive_rows(args.input, args.limit)
    generators = {
        "random": random_mismatch,
        "template_perturbation": template_perturbation,
        "dora_alternate_center": dora_alternate_center,
        "pu_reliable_negative": pu_reliable_negative,
        "pc_cng_rule_mvp": pc_cng_rule,
    }
    all_rows: List[Dict[str, object]] = []
    counts: Dict[str, int] = {}
    for name, fn in generators.items():
        generated = fn(rows)
        counts[name] = len(generated)
        all_rows.extend(generated)

    total = write_rows(args.output, all_rows)
    summary = {"input": args.input, "output": args.output, "positive_rows": len(rows), "total": total, "counts": counts}
    os.makedirs(os.path.dirname(args.summary), exist_ok=True)
    with open(args.summary, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

