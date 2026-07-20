"""Export rule-selected hard negatives from edit-decoder candidates."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from typing import Dict, List

from .reaction_boundary_generator import BoundaryCandidate


OUTPUT_FIELDS = list(BoundaryCandidate.__dataclass_fields__.keys()) + ["rule_score", "rule_rank"]


def score_candidate(row: Dict[str, str]) -> float:
    similarity = float(row.get("product_similarity", 0.0) or 0.0)
    atom_balance = float(row.get("atom_balance", 0.0) or 0.0)
    distance = float(row.get("candidate_distance_to_true_anchor", 99.0) or 99.0)
    same_atom = float(row.get("candidate_same_atomic_num_as_true", 0.0) or 0.0)
    distance_score = 1.0 / (1.0 + max(0.0, distance - 1.0))
    return 0.40 * similarity + 0.30 * atom_balance + 0.20 * distance_score + 0.10 * same_atom


def to_boundary_row(row: Dict[str, str], score: float, rank: int) -> Dict[str, object]:
    product_similarity = float(row.get("product_similarity", 0.0) or 0.0)
    return {
        "source_id": row["source_id"],
        "positive_reaction": row["positive_reaction"],
        "candidate_reaction": row["candidate_reaction"],
        "task": "forward_outcome",
        "failure_type": "rule_hard_reaction_center_alternative",
        "edit_action": row["edit_action"],
        "parent_reactants": row["reactants"],
        "parent_product": row["parent_product"],
        "candidate_reactants": row["reactants"],
        "candidate_product": row["candidate_product"],
        "valid": 1.0,
        "atom_balance": float(row.get("atom_balance", 0.0) or 0.0),
        "locality": product_similarity,
        "closeness": product_similarity,
        "hard_score": float(score),
        "false_negative_risk": max(0.0, min(1.0, max(0.0, product_similarity - 0.90) / 0.10)),
        "passes_filter": True,
        "mapped": True,
        "center_maps": f"{row.get('fragment_map', '')};{row.get('true_anchor_map', '')};{row.get('candidate_anchor_map', '')}",
        "label": 0,
        "provenance": "pc_cng_v3_rule_hard_negative",
        "rule_score": float(score),
        "rule_rank": int(rank),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Edit-decoder candidate CSV")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--min-score", type=float, default=0.0)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    os.makedirs(os.path.dirname(args.summary), exist_ok=True)
    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    seen = 0
    with open(args.input, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            seen += 1
            if row.get("candidate_role") != "hard_negative":
                continue
            if int(row.get("is_known_positive", 0) or 0):
                continue
            score = score_candidate(row)
            if score < args.min_score:
                continue
            row["_rule_score"] = str(score)
            grouped[row["pair_id"]].append(row)

    written = 0
    with open(args.output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for pair_id, rows in sorted(grouped.items()):
            ranked = sorted(rows, key=lambda item: float(item["_rule_score"]), reverse=True)
            for rank, row in enumerate(ranked[: args.top_k], start=1):
                writer.writerow(to_boundary_row(row, float(row["_rule_score"]), rank))
                written += 1

    summary = {
        "input": args.input,
        "output": args.output,
        "seen_rows": seen,
        "candidate_groups": len(grouped),
        "written": written,
        "top_k": args.top_k,
        "min_score": args.min_score,
        "notes": [
            "Rule hard negatives are selected from candidate_role=hard_negative rows.",
            "Known positives are masked before export.",
        ],
    }
    with open(args.summary, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
