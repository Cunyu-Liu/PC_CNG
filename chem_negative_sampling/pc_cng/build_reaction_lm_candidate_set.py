"""Build a standardized candidate CSV for reaction LM scoring."""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Dict, Iterable, List

from .chem_utils import split_reaction
from .evaluate_candidate_reranking import (
    build_real_candidate_rows,
    build_synthetic_candidate_rows,
    positive_lookup,
    read_real_rows,
)
from .reaction_lm_scorer import INPUT_FIELDS


def to_lm_row(row: Dict[str, object]) -> Dict[str, str]:
    reaction = str(row.get("reaction_smiles", ""))
    try:
        reactants, agents, product = split_reaction(reaction)
    except ValueError:
        reactants, agents, product = "", "", ""
    return {
        "group_id": str(row.get("group_id", "")),
        "source_id": str(row.get("source_id", "")),
        "reactants": reactants,
        "agents": agents,
        "candidate_product": product,
        "candidate_reaction": reaction,
        "label": str(int(row.get("label", 0) or 0)),
        "split": str(row.get("split", "")),
        "dataset": str(row.get("dataset", "")),
        "candidate_source": str(row.get("candidate_source", "")),
        "candidate_family": str(row.get("candidate_family", "")),
        "reaction_class": str(row.get("reaction_class", "")),
    }


def write_rows(path: str, rows: Iterable[Dict[str, str]]) -> int:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    count = 0
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=INPUT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-csv", action="append", default=[])
    parser.add_argument("--synthetic-csv", action="append", default=[])
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--group-by", default="reactants")
    parser.add_argument("--candidate-scope", choices=["same_split", "all_group"], default="same_split")
    parser.add_argument("--review-status", action="append", default=["keep_synthetic_negative"])
    args = parser.parse_args()

    real_rows = read_real_rows(args.real_csv)
    candidates: List[Dict[str, object]] = []
    if args.real_csv:
        candidates.extend(
            build_real_candidate_rows(
                real_rows=real_rows,
                group_by=args.group_by,
                candidate_scope=args.candidate_scope,
            )
        )
    if args.synthetic_csv:
        candidates.extend(
            build_synthetic_candidate_rows(
                synthetic_paths=args.synthetic_csv,
                positives=positive_lookup(real_rows),
                review_statuses=args.review_status,
            )
        )

    lm_rows = [to_lm_row(row) for row in candidates]
    written = write_rows(args.output, lm_rows)
    summary = {
        "real_csv": args.real_csv,
        "synthetic_csv": args.synthetic_csv,
        "candidate_scope": args.candidate_scope,
        "group_by": args.group_by,
        "rows": written,
        "output": args.output,
    }
    os.makedirs(os.path.dirname(args.summary), exist_ok=True)
    with open(args.summary, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
