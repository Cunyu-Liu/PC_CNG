"""False-negative risk review for synthetic counterfactual reactions."""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Dict, Iterable, List, Set

from .chem_utils import canonicalize_reaction, split_reaction, token_jaccard
from .known_positive_cache import load_known_positive_reactions

try:  # pragma: no cover - depends on optional RDKit install
    from rdkit import RDLogger  # type: ignore

    RDLogger.DisableLog("rdApp.*")
except Exception:  # pragma: no cover
    pass


def load_known_positive_set(paths: List[str]) -> Set[str]:
    return load_known_positive_reactions(paths)


def review_row(row: Dict[str, str], known_positives: Set[str]) -> Dict[str, str]:
    candidate = row.get("candidate_reaction", "")
    canonical = canonicalize_reaction(candidate)
    exact_known_positive = bool(canonical and canonical in known_positives)
    try:
        _, _, parent_product = split_reaction(row.get("positive_reaction", ""))
        _, _, candidate_product = split_reaction(candidate)
        product_overlap = token_jaccard(parent_product, candidate_product)
    except Exception:
        product_overlap = 0.0

    hard_score = float(row.get("hard_score") or 0.0)
    fn_risk = float(row.get("false_negative_risk") or 0.0)
    reasons: List[str] = []
    if exact_known_positive:
        reasons.append("exact_known_positive")
    if fn_risk >= 0.75:
        reasons.append("high_model_false_negative_risk")
    if hard_score >= 0.85 and product_overlap >= 0.85:
        reasons.append("very_close_to_positive_product")
    if row.get("failure_type") in {"side_product", "chemoselectivity_error"} and product_overlap >= 0.8:
        reasons.append("plausible_alternative_outcome")

    status = "keep_synthetic_negative"
    if exact_known_positive:
        status = "discard_known_positive"
    elif reasons:
        status = "needs_review_or_downweight"

    out = dict(row)
    out["review_status"] = status
    out["review_reasons"] = ";".join(reasons)
    out["product_overlap"] = f"{product_overlap:.6f}"
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Synthetic negatives CSV")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--known-positive", action="append", default=[], help="CSV of known positive reactions")
    args = parser.parse_args()

    known_positives = load_known_positive_set(args.known_positive)
    counts: Dict[str, int] = {}
    reviewed_rows = 0

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.input, newline="", encoding="utf-8") as handle, open(
        args.output, "w", newline="", encoding="utf-8"
    ) as output_handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        output_fields = list(dict.fromkeys(fieldnames + ["review_status", "review_reasons", "product_overlap"]))
        writer = csv.DictWriter(output_handle, fieldnames=output_fields, extrasaction="ignore")
        writer.writeheader()
        for row in reader:
            reviewed_row = review_row(row, known_positives)
            writer.writerow(reviewed_row)
            reviewed_rows += 1
            status = reviewed_row["review_status"]
            counts[status] = counts.get(status, 0) + 1

    summary = {
        "input": args.input,
        "output": args.output,
        "known_positive_count": len(known_positives),
        "reviewed_rows": reviewed_rows,
        "status_counts": counts,
        "notes": [
            "Rows marked needs_review_or_downweight should not be treated as strong negatives.",
            "Rows marked discard_known_positive must be removed from synthetic-negative training.",
        ],
    }
    os.makedirs(os.path.dirname(args.summary), exist_ok=True)
    with open(args.summary, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
