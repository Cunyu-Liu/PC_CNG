"""Build candidate-anchor datasets for the trainable reaction-center decoder."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from typing import Dict, List

from .reaction_boundary_generator import RXNMapperAdapter
from .reaction_center_edit_decoder import build_edit_candidate_groups, candidate_fieldnames
from .false_negative_review import load_known_positive_set
from .chem_utils import canonicalize_reaction


REQUIRED_COLUMNS = {"source_id", "reaction_smiles", "label_type", "split"}


def validate_fieldnames(path: str, fieldnames: List[str] | None) -> None:
    missing = REQUIRED_COLUMNS - set(fieldnames or [])
    if missing:
        raise ValueError(f"{path} is missing required columns for edit decoder: {sorted(missing)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True, help="Normalized CSV files")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--map-unmapped", action="store_true", help="Use RXNMapper for unmapped reactions")
    parser.add_argument("--positive-only", action="store_true", default=True)
    parser.add_argument("--max-candidates-per-pair", type=int, default=8)
    parser.add_argument("--max-anchor-distance", type=int, default=6)
    parser.add_argument("--known-positive", action="append", default=[])
    parser.add_argument("--hard-negative-min-similarity", type=float, default=0.70)
    parser.add_argument("--hard-negative-max-similarity", type=float, default=0.95)
    parser.add_argument("--hard-negative-min-atom-balance", type=float, default=0.65)
    parser.add_argument("--hard-negative-max-distance", type=float, default=4.0)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    os.makedirs(os.path.dirname(args.summary), exist_ok=True)
    mapper = RXNMapperAdapter() if args.map_unmapped else None
    known_positives = load_known_positive_set(args.known_positive)
    fieldnames = candidate_fieldnames()

    stats: Dict[str, object] = {
        "inputs": args.input,
        "output": args.output,
        "map_unmapped": args.map_unmapped,
        "positive_only": args.positive_only,
        "seen_rows": 0,
        "candidate_rows": 0,
        "candidate_groups": 0,
        "labels": Counter(),
        "splits": Counter(),
        "skip_reasons": Counter(),
        "groups_by_source": Counter(),
        "candidate_roles": Counter(),
        "known_positive_count": len(known_positives),
    }

    with open(args.output, "w", newline="", encoding="utf-8") as output_handle:
        writer = csv.DictWriter(output_handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()

        for path in args.input:
            with open(path, newline="", encoding="utf-8") as input_handle:
                reader = csv.DictReader(input_handle)
                validate_fieldnames(path, reader.fieldnames)
                for row in reader:
                    if args.limit is not None and int(stats["seen_rows"]) >= args.limit:
                        break
                    stats["seen_rows"] = int(stats["seen_rows"]) + 1
                    label_type = row.get("label_type", "")
                    split = row.get("split", "")
                    source = row.get("source", os.path.basename(path))
                    stats["labels"][label_type] += 1  # type: ignore[index]
                    stats["splits"][split] += 1  # type: ignore[index]
                    if args.positive_only and label_type != "positive":
                        stats["skip_reasons"]["not_positive"] += 1  # type: ignore[index]
                        continue
                    groups, reason = build_edit_candidate_groups(
                        reaction_smiles=row["reaction_smiles"],
                        source_id=row["source_id"],
                        split=split,
                        label_type=label_type,
                        mapper=mapper,
                        map_unmapped=args.map_unmapped,
                        max_candidates_per_pair=args.max_candidates_per_pair,
                        max_anchor_distance=args.max_anchor_distance,
                    )
                    if reason != "ok":
                        stats["skip_reasons"][reason] += 1  # type: ignore[index]
                        continue
                    for group in groups:
                        stats["candidate_groups"] = int(stats["candidate_groups"]) + 1
                        stats["groups_by_source"][source] += 1  # type: ignore[index]
                        for candidate in group.rows:
                            role = classify_candidate(candidate, known_positives, args)
                            candidate["candidate_role"] = role
                            candidate["is_known_positive"] = 1 if role == "known_positive_alt" else 0
                            candidate["is_hard_negative"] = 1 if role == "hard_negative" else 0
                            candidate["hard_negative_weight"] = hard_negative_weight(candidate) if role == "hard_negative" else 0.0
                            stats["candidate_roles"][role] += 1  # type: ignore[index]
                            writer.writerow(candidate)
                            stats["candidate_rows"] = int(stats["candidate_rows"]) + 1
                if args.limit is not None and int(stats["seen_rows"]) >= args.limit:
                    break

    serializable = {
        key: dict(value) if isinstance(value, Counter) else value
        for key, value in stats.items()
    }
    with open(args.summary, "w", encoding="utf-8") as handle:
        json.dump(serializable, handle, indent=2, ensure_ascii=False)
    print(json.dumps(serializable, indent=2, ensure_ascii=False))


def classify_candidate(row: Dict[str, object], known_positives: set[str], args: argparse.Namespace) -> str:
    if int(row.get("is_true_anchor", 0) or 0) == 1:
        return "observed_positive"
    canonical = canonicalize_reaction(str(row.get("candidate_reaction", "")))
    if canonical and canonical in known_positives:
        return "known_positive_alt"
    similarity = float(row.get("product_similarity", 0.0) or 0.0)
    atom_balance = float(row.get("atom_balance", 0.0) or 0.0)
    distance = float(row.get("candidate_distance_to_true_anchor", 99.0) or 99.0)
    if (
        args.hard_negative_min_similarity <= similarity <= args.hard_negative_max_similarity
        and atom_balance >= args.hard_negative_min_atom_balance
        and distance <= args.hard_negative_max_distance
    ):
        return "hard_negative"
    return "artifact"


def hard_negative_weight(row: Dict[str, object]) -> float:
    similarity = float(row.get("product_similarity", 0.0) or 0.0)
    atom_balance = float(row.get("atom_balance", 0.0) or 0.0)
    distance = float(row.get("candidate_distance_to_true_anchor", 99.0) or 99.0)
    distance_score = 1.0 / (1.0 + max(0.0, distance - 1.0))
    return max(0.1, min(1.0, 0.45 * similarity + 0.35 * atom_balance + 0.20 * distance_score))


if __name__ == "__main__":
    main()
