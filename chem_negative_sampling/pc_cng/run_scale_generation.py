"""Streaming PC-CNG generation for larger positive-reaction CSV files.

This is the scale-up bridge after the MVP. It avoids loading the full dataset
into memory and writes synthetic counterfactual negatives incrementally.

Example:
    python3 -m pc_cng.run_scale_generation \
      --input data/uspto_positives.csv \
      --output results/uspto_pc_cng_negatives.csv \
      --summary results/uspto_pc_cng_summary.json \
      --limit 100000
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Dict

from .counterfactual import CounterfactualGenerator


NEGATIVE_FIELDS = [
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="CSV with reaction_smiles and optional source_id/id")
    parser.add_argument("--output", required=True, help="Output CSV for generated synthetic negatives")
    parser.add_argument("--summary", required=True, help="Output JSON summary")
    parser.add_argument("--limit", type=int, default=None, help="Optional max positive rows to process")
    parser.add_argument("--include-failed", action="store_true", help="Keep candidates failing MVP filters")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    os.makedirs(os.path.dirname(args.summary), exist_ok=True)

    generator = CounterfactualGenerator()
    processed = 0
    generated = 0
    by_task: Dict[str, int] = {}
    by_failure: Dict[str, int] = {}
    hard_score_sum = 0.0
    fn_risk_sum = 0.0

    with open(args.input, newline="", encoding="utf-8") as input_handle, open(
        args.output, "w", newline="", encoding="utf-8"
    ) as output_handle:
        reader = csv.DictReader(input_handle)
        if "reaction_smiles" not in (reader.fieldnames or []):
            raise ValueError("Input CSV must contain a reaction_smiles column")
        writer = csv.DictWriter(output_handle, fieldnames=NEGATIVE_FIELDS, extrasaction="ignore")
        writer.writeheader()

        for index, row in enumerate(reader, start=1):
            if args.limit is not None and processed >= args.limit:
                break
            reaction = (row.get("reaction_smiles") or "").strip()
            if not reaction:
                continue
            source_id = (row.get("source_id") or row.get("id") or f"row_{index:09d}").strip()
            try:
                candidates = generator.generate_for_reaction(
                    reaction,
                    source_id=source_id,
                    include_failed=args.include_failed,
                )
            except Exception:
                # Full data can contain malformed reactions. Keep streaming and
                # account for valid outputs rather than terminating the run.
                processed += 1
                continue

            processed += 1
            for candidate in candidates:
                data = candidate.to_dict()
                writer.writerow(data)
                generated += 1
                task = str(data["task"])
                failure_type = str(data["failure_type"])
                by_task[task] = by_task.get(task, 0) + 1
                by_failure[failure_type] = by_failure.get(failure_type, 0) + 1
                hard_score_sum += float(data["hard_score"])
                fn_risk_sum += float(data["false_negative_risk"])

    summary = {
        "input": args.input,
        "output": args.output,
        "processed_positive_reactions": processed,
        "generated_negative_reactions": generated,
        "avg_negatives_per_positive": generated / processed if processed else 0.0,
        "by_task": by_task,
        "by_failure_type": by_failure,
        "avg_hard_score": hard_score_sum / generated if generated else 0.0,
        "avg_false_negative_risk": fn_risk_sum / generated if generated else 0.0,
        "include_failed": args.include_failed,
        "limit": args.limit,
        "notes": [
            "Streaming output contains synthetic counterfactual negatives, not real failed experiments.",
            "For publishable scale-up, run RDKit validation, atom mapping, leakage checks, and external baselines.",
        ],
    }
    with open(args.summary, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

