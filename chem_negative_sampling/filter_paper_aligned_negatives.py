"""Filter synthetic negatives to match the Science Advances negative-data lesson.

The Science Advances paper distinguishes informative type-1 negatives
(unexpected but chemically meaningful products) from weaker/noisier negatives
such as random product pairings or ambiguous no-reaction cases. This filter
keeps only synthetic candidates that look like type-1 forward-outcome
alternatives, with moderate product overlap and bounded false-negative risk.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Dict


def as_float(row: Dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default) or default)
    except ValueError:
        return default


def keep_type1(row: Dict[str, str], args: argparse.Namespace) -> tuple[bool, str]:
    if row.get("review_status", "keep_synthetic_negative") != "keep_synthetic_negative":
        return False, "review_rejected"
    if row.get("task") != "forward_outcome":
        return False, "not_forward_outcome"
    if row.get("failure_type") not in set(args.failure_type):
        return False, "wrong_failure_type"
    if row.get("candidate_reaction") == row.get("positive_reaction"):
        return False, "identical_positive"
    product_overlap = as_float(row, "product_overlap")
    false_negative_risk = as_float(row, "false_negative_risk")
    hard_score = as_float(row, "hard_score")
    atom_balance = as_float(row, "atom_balance")
    valid = as_float(row, "valid")
    if valid < args.min_valid:
        return False, "invalid"
    if atom_balance < args.min_atom_balance:
        return False, "low_atom_balance"
    if product_overlap < args.min_product_overlap:
        return False, "too_distant_product"
    if product_overlap > args.max_product_overlap:
        return False, "too_close_product"
    if false_negative_risk > args.max_false_negative_risk:
        return False, "high_false_negative_risk"
    if hard_score < args.min_hard_score:
        return False, "too_easy"
    if hard_score > args.max_hard_score:
        return False, "too_hard_or_suspicious"
    return True, "keep"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--failure-type", action="append", default=["chemoselectivity_error"])
    parser.add_argument("--min-product-overlap", type=float, default=0.30)
    parser.add_argument("--max-product-overlap", type=float, default=0.85)
    parser.add_argument("--max-false-negative-risk", type=float, default=0.70)
    parser.add_argument("--min-hard-score", type=float, default=0.55)
    parser.add_argument("--max-hard-score", type=float, default=0.90)
    parser.add_argument("--min-atom-balance", type=float, default=0.45)
    parser.add_argument("--min-valid", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    os.makedirs(os.path.dirname(args.summary), exist_ok=True)
    counts: Dict[str, int] = {}
    kept = 0
    total = 0
    output_fields = None

    with open(args.output, "w", newline="", encoding="utf-8") as out_handle:
        writer = None
        for path in args.input:
            with open(path, newline="", encoding="utf-8") as in_handle:
                reader = csv.DictReader(in_handle)
                if output_fields is None:
                    output_fields = list(reader.fieldnames or [])
                    writer = csv.DictWriter(out_handle, fieldnames=output_fields, extrasaction="ignore")
                    writer.writeheader()
                assert writer is not None
                for row in reader:
                    total += 1
                    should_keep, reason = keep_type1(row, args)
                    counts[reason] = counts.get(reason, 0) + 1
                    if not should_keep:
                        continue
                    writer.writerow(row)
                    kept += 1
                    if args.limit is not None and kept >= args.limit:
                        break
            if args.limit is not None and kept >= args.limit:
                break

    summary = {
        "input": args.input,
        "output": args.output,
        "total_seen": total,
        "kept": kept,
        "counts": counts,
        "criteria": {
            "failure_type": args.failure_type,
            "min_product_overlap": args.min_product_overlap,
            "max_product_overlap": args.max_product_overlap,
            "max_false_negative_risk": args.max_false_negative_risk,
            "min_hard_score": args.min_hard_score,
            "max_hard_score": args.max_hard_score,
            "min_atom_balance": args.min_atom_balance,
        },
        "notes": [
            "Designed to keep type-1 negatives: unexpected but chemically meaningful alternative products.",
            "Retro/no-disconnection/no-reaction artifacts are intentionally excluded by default.",
        ],
    }
    with open(args.summary, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
