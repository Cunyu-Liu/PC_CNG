"""Run heteroatom/regio/tautomer/low-yield hard-negative actions."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from typing import Dict, Set

from .false_negative_review import load_known_positive_set
from .hard_negative_actions import (
    anchor_candidate_actions,
    low_yield_seed_action,
    output_fieldnames,
    tautomer_actions,
)
from .reaction_boundary_generator import RXNMapperAdapter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", default=[], help="Positive/normalized CSV for candidate actions")
    parser.add_argument("--low-yield-input", action="append", default=[], help="CSV containing real_negative or low-yield rows")
    parser.add_argument("--known-positive", action="append", default=[])
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--action", action="append", choices=["heteroatom", "regio", "tautomer", "low_yield_seed"], required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--map-unmapped", action="store_true")
    parser.add_argument("--max-candidates-per-reaction", type=int, default=4)
    parser.add_argument("--max-candidates-per-pair", type=int, default=12)
    parser.add_argument("--max-anchor-distance", type=int, default=6)
    parser.add_argument("--max-tautomers", type=int, default=4)
    parser.add_argument("--yield-threshold", type=float, default=5.0)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    os.makedirs(os.path.dirname(args.summary), exist_ok=True)
    actions: Set[str] = set(args.action)
    known_positives = load_known_positive_set(args.known_positive)
    mapper = RXNMapperAdapter() if args.map_unmapped else None

    counts = Counter()
    seen = 0
    written = 0
    seen_reactions: Set[str] = set()

    with open(args.output, "w", newline="", encoding="utf-8") as output_handle:
        writer = csv.DictWriter(output_handle, fieldnames=output_fieldnames(), extrasaction="ignore")
        writer.writeheader()

        for path in args.input:
            with open(path, newline="", encoding="utf-8") as input_handle:
                reader = csv.DictReader(input_handle)
                for row in reader:
                    if args.limit is not None and seen >= args.limit:
                        break
                    if row.get("label_type", "positive") != "positive":
                        continue
                    reaction = row.get("reaction_smiles", "")
                    if not reaction:
                        continue
                    seen += 1
                    candidates = []
                    if actions & {"heteroatom", "regio"}:
                        candidates.extend(
                            anchor_candidate_actions(
                                reaction_smiles=reaction,
                                source_id=row.get("source_id", f"row_{seen:09d}"),
                                split=row.get("split", "train"),
                                label_type=row.get("label_type", "positive"),
                                action_families=actions,
                                mapper=mapper,
                                map_unmapped=args.map_unmapped,
                                known_positives=known_positives,
                                max_candidates_per_pair=args.max_candidates_per_pair,
                                max_anchor_distance=args.max_anchor_distance,
                            )
                        )
                    if "tautomer" in actions:
                        candidates.extend(
                            tautomer_actions(
                                reaction_smiles=reaction,
                                source_id=row.get("source_id", f"row_{seen:09d}"),
                                known_positives=known_positives,
                                max_tautomers=args.max_tautomers,
                            )
                        )
                    candidates.sort(key=lambda item: item.hard_score, reverse=True)
                    per_reaction = 0
                    for candidate in candidates:
                        if per_reaction >= args.max_candidates_per_reaction:
                            break
                        key = candidate.candidate_reaction
                        if key in seen_reactions:
                            continue
                        seen_reactions.add(key)
                        writer.writerow(candidate.to_dict())
                        counts[candidate.action_family] += 1
                        written += 1
                        per_reaction += 1
                if args.limit is not None and seen >= args.limit:
                    break

        if "low_yield_seed" in actions:
            for path in args.low_yield_input:
                with open(path, newline="", encoding="utf-8") as input_handle:
                    reader = csv.DictReader(input_handle)
                    for row in reader:
                        candidate = low_yield_seed_action(row, yield_threshold=args.yield_threshold)
                        if candidate is None:
                            continue
                        if candidate.candidate_reaction in seen_reactions:
                            continue
                        seen_reactions.add(candidate.candidate_reaction)
                        writer.writerow(candidate.to_dict())
                        counts[candidate.action_family] += 1
                        written += 1

    summary: Dict[str, object] = {
        "input": args.input,
        "low_yield_input": args.low_yield_input,
        "known_positive_count": len(known_positives),
        "actions": sorted(actions),
        "seen_positive_rows": seen,
        "written": written,
        "counts": dict(counts),
        "output": args.output,
    }
    with open(args.summary, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
