"""Streaming PC-CNG v2 reaction-boundary generation CLI."""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Dict

from .reaction_boundary_generator import BoundaryCandidate, ReactionBoundaryGenerator


BOUNDARY_FIELDS = list(BoundaryCandidate.__dataclass_fields__.keys())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="CSV with reaction_smiles and source_id/id columns")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--include-failed", action="store_true")
    parser.add_argument("--allow-unmapped-fallback", action="store_true")
    parser.add_argument("--max-candidates-per-reaction", type=int, default=4)
    parser.add_argument("--min-product-similarity", type=float, default=0.30)
    parser.add_argument("--max-product-similarity", type=float, default=0.97)
    parser.add_argument("--max-false-negative-risk", type=float, default=0.85)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    os.makedirs(os.path.dirname(args.summary), exist_ok=True)
    generator = ReactionBoundaryGenerator(
        min_product_similarity=args.min_product_similarity,
        max_product_similarity=args.max_product_similarity,
        max_false_negative_risk=args.max_false_negative_risk,
        max_candidates_per_reaction=args.max_candidates_per_reaction,
        allow_unmapped_fallback=args.allow_unmapped_fallback,
    )

    processed = 0
    mapped_generated = 0
    generated = 0
    failed_generation = 0
    by_action: Dict[str, int] = {}

    with open(args.input, newline="", encoding="utf-8") as input_handle, open(
        args.output, "w", newline="", encoding="utf-8"
    ) as output_handle:
        reader = csv.DictReader(input_handle)
        if "reaction_smiles" not in (reader.fieldnames or []):
            raise ValueError("Input CSV must contain reaction_smiles")
        writer = csv.DictWriter(output_handle, fieldnames=BOUNDARY_FIELDS, extrasaction="ignore")
        writer.writeheader()

        for index, row in enumerate(reader, start=1):
            if args.limit is not None and processed >= args.limit:
                break
            reaction = (row.get("reaction_smiles") or "").strip()
            if not reaction:
                continue
            source_id = (row.get("source_id") or row.get("id") or f"row_{index:09d}").strip()
            processed += 1
            try:
                candidates = generator.generate_for_reaction(
                    reaction,
                    source_id=source_id,
                    include_failed=args.include_failed,
                )
            except Exception:
                failed_generation += 1
                continue
            if not candidates:
                failed_generation += 1
                continue
            for candidate in candidates:
                writer.writerow(candidate.to_dict())
                generated += 1
                if candidate.mapped:
                    mapped_generated += 1
                by_action[candidate.edit_action] = by_action.get(candidate.edit_action, 0) + 1

    summary = {
        "input": args.input,
        "output": args.output,
        "processed_reactions": processed,
        "generated": generated,
        "failed_or_empty": failed_generation,
        "mapped_generated": mapped_generated,
        "avg_generated_per_processed": generated / processed if processed else 0.0,
        "by_action": by_action,
        "criteria": {
            "include_failed": args.include_failed,
            "allow_unmapped_fallback": args.allow_unmapped_fallback,
            "max_candidates_per_reaction": args.max_candidates_per_reaction,
            "min_product_similarity": args.min_product_similarity,
            "max_product_similarity": args.max_product_similarity,
            "max_false_negative_risk": args.max_false_negative_risk,
        },
        "notes": [
            "PC-CNG v2 generates type-1 forward-outcome alternatives near atom-mapped reaction centers.",
            "Use false-negative review and pairwise reward training before treating these as strong negatives.",
        ],
    }
    with open(args.summary, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
