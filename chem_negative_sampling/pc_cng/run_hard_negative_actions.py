"""Run heteroatom/regio/tautomer/low-yield hard-negative actions."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from typing import Dict, Set

from .false_negative_review import load_known_positive_set
from .chem_utils import canonicalize_reaction
from .hard_negative_actions import (
    anchor_candidate_actions,
    class_fallback_actions,
    diversity_anchor_actions,
    low_yield_seed_action,
    output_fieldnames,
    partial_product_actions,
    tautomer_actions,
    unreacted_substrate_actions,
)
from .reaction_boundary_generator import RXNMapperAdapter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", default=[], help="Positive/normalized CSV for candidate actions")
    parser.add_argument("--low-yield-input", action="append", default=[], help="CSV containing real_negative or low-yield rows")
    parser.add_argument("--known-positive", action="append", default=[])
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument(
        "--action",
        action="append",
        choices=[
            "heteroatom",
            "regio",
            "class_fallback",
            "partial_product",
            "unreacted_substrate",
            "tautomer",
            "low_yield_seed",
        ],
        required=True,
    )
    parser.add_argument("--include-reaction-class", action="append", default=[], help="Only generate candidates for these reaction_class values")
    parser.add_argument("--exclude-candidate-csv", action="append", default=[], help="CSV containing candidate_reaction rows to avoid duplicating")
    parser.add_argument(
        "--exclude-review-status",
        action="append",
        default=[],
        help="When excluding reviewed CSVs, only exclude rows with these review_status values. Defaults to all rows.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--map-unmapped", action="store_true")
    parser.add_argument("--max-candidates-per-reaction", type=int, default=4)
    parser.add_argument("--max-candidates-per-pair", type=int, default=12)
    parser.add_argument("--max-anchor-distance", type=int, default=6)
    parser.add_argument("--min-product-similarity", type=float, default=0.65)
    parser.add_argument("--max-product-similarity", type=float, default=0.98)
    parser.add_argument("--min-atom-balance", type=float, default=0.55)
    parser.add_argument("--diverse-anchor", action="store_true", help="Use product-graph terminal-substituent shifts")
    parser.add_argument("--max-tautomers", type=int, default=4)
    parser.add_argument("--yield-threshold", type=float, default=5.0)
    parser.add_argument("--progress-every", type=int, default=500)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    os.makedirs(os.path.dirname(args.summary), exist_ok=True)
    actions: Set[str] = set(args.action)
    include_classes: Set[str] = set(args.include_reaction_class)
    known_positives = load_known_positive_set(args.known_positive)
    mapper = RXNMapperAdapter() if args.map_unmapped else None
    exclude_statuses: Set[str] = set(args.exclude_review_status)

    counts = Counter()
    anchor_diagnostics = Counter()
    diverse_anchor_diagnostics = Counter()
    selection_diagnostics = Counter()
    seen = 0
    written = 0
    seen_reactions: Set[str] = set()
    for path in args.exclude_candidate_csv:
        with open(path, newline="", encoding="utf-8") as exclude_handle:
            for row in csv.DictReader(exclude_handle):
                status = row.get("review_status", "keep_synthetic_negative")
                if exclude_statuses and status not in exclude_statuses:
                    continue
                reaction = row.get("candidate_reaction") or row.get("reaction_smiles") or ""
                canonical = canonicalize_reaction(reaction) if reaction else None
                if canonical:
                    seen_reactions.add(canonical)
    initial_excluded_candidate_count = len(seen_reactions)

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
                    reaction_class = row.get("reaction_class", "") or "unknown"
                    if include_classes and reaction_class not in include_classes:
                        counts[f"skip_class:{reaction_class}"] += 1
                        continue
                    reaction = row.get("reaction_smiles", "")
                    if not reaction:
                        continue
                    seen += 1
                    if args.progress_every > 0 and seen % args.progress_every == 0:
                        print(
                            f"[run_hard_negative_actions] seen_positive_rows={seen} written={written} counts={dict(counts)}",
                            file=sys.stderr,
                            flush=True,
                        )
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
                                min_product_similarity=args.min_product_similarity,
                                max_product_similarity=args.max_product_similarity,
                                min_atom_balance=args.min_atom_balance,
                                diagnostics=anchor_diagnostics,
                            )
                        )
                        if args.diverse_anchor:
                            candidates.extend(
                                diversity_anchor_actions(
                                    reaction_smiles=reaction,
                                    source_id=row.get("source_id", f"row_{seen:09d}"),
                                    action_families=actions,
                                    known_positives=known_positives,
                                    max_candidates_per_reaction=args.max_candidates_per_reaction,
                                    max_anchor_distance=args.max_anchor_distance,
                                    min_product_similarity=args.min_product_similarity,
                                    max_product_similarity=args.max_product_similarity,
                                    min_atom_balance=args.min_atom_balance,
                                    diagnostics=diverse_anchor_diagnostics,
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
                    if "class_fallback" in actions:
                        candidates.extend(
                            class_fallback_actions(
                                reaction_smiles=reaction,
                                source_id=row.get("source_id", f"row_{seen:09d}"),
                                known_positives=known_positives,
                                max_candidates_per_reaction=args.max_candidates_per_reaction,
                                min_product_similarity=args.min_product_similarity,
                                max_product_similarity=args.max_product_similarity,
                                diagnostics=diverse_anchor_diagnostics,
                            )
                        )
                    if "partial_product" in actions:
                        candidates.extend(
                            partial_product_actions(
                                reaction_smiles=reaction,
                                source_id=row.get("source_id", f"row_{seen:09d}"),
                                known_positives=known_positives,
                                max_candidates_per_reaction=args.max_candidates_per_reaction,
                                min_product_similarity=args.min_product_similarity,
                                max_product_similarity=args.max_product_similarity,
                                diagnostics=diverse_anchor_diagnostics,
                            )
                        )
                    if "unreacted_substrate" in actions:
                        candidates.extend(
                            unreacted_substrate_actions(
                                reaction_smiles=reaction,
                                source_id=row.get("source_id", f"row_{seen:09d}"),
                                known_positives=known_positives,
                                max_candidates_per_reaction=args.max_candidates_per_reaction,
                                min_product_similarity=args.min_product_similarity,
                                max_product_similarity=args.max_product_similarity,
                                diagnostics=diverse_anchor_diagnostics,
                            )
                        )
                    candidates.sort(key=lambda item: item.hard_score, reverse=True)
                    per_reaction = 0
                    for candidate in candidates:
                        selection_diagnostics[f"candidate_seen:{candidate.action_family}"] += 1
                        if per_reaction >= args.max_candidates_per_reaction:
                            selection_diagnostics[f"skip_per_reaction_cap:{candidate.action_family}"] += 1
                            continue
                        key = canonicalize_reaction(candidate.candidate_reaction) or candidate.candidate_reaction
                        if key in seen_reactions:
                            selection_diagnostics[f"skip_global_duplicate:{candidate.action_family}"] += 1
                            continue
                        seen_reactions.add(key)
                        candidate.reaction_class = reaction_class
                        writer.writerow(candidate.to_dict())
                        counts[candidate.action_family] += 1
                        counts[f"class:{reaction_class}"] += 1
                        selection_diagnostics[f"written:{candidate.action_family}"] += 1
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
                        key = canonicalize_reaction(candidate.candidate_reaction) or candidate.candidate_reaction
                        if key in seen_reactions:
                            continue
                        seen_reactions.add(key)
                        candidate.reaction_class = row.get("reaction_class", "") or "unknown"
                        writer.writerow(candidate.to_dict())
                        counts[candidate.action_family] += 1
                        counts[f"class:{candidate.reaction_class}"] += 1
                        written += 1
                        if args.progress_every > 0 and written % args.progress_every == 0:
                            print(
                                f"[run_hard_negative_actions] low_yield_written_progress written={written} counts={dict(counts)}",
                                file=sys.stderr,
                                flush=True,
                            )

    summary: Dict[str, object] = {
        "input": args.input,
        "low_yield_input": args.low_yield_input,
        "known_positive_count": len(known_positives),
        "actions": sorted(actions),
        "include_reaction_class": sorted(include_classes),
        "excluded_candidate_csv": args.exclude_candidate_csv,
        "exclude_review_status": sorted(exclude_statuses),
        "initial_excluded_candidate_count": initial_excluded_candidate_count,
        "seen_positive_rows": seen,
        "written": written,
        "counts": dict(counts),
        "anchor_thresholds": {
            "max_candidates_per_pair": args.max_candidates_per_pair,
            "max_anchor_distance": args.max_anchor_distance,
            "min_product_similarity": args.min_product_similarity,
            "max_product_similarity": args.max_product_similarity,
            "min_atom_balance": args.min_atom_balance,
        },
        "anchor_diagnostics": dict(anchor_diagnostics),
        "diverse_anchor_enabled": args.diverse_anchor,
        "diverse_anchor_diagnostics": dict(diverse_anchor_diagnostics),
        "selection_diagnostics": dict(selection_diagnostics),
        "output": args.output,
    }
    with open(args.summary, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
