"""Filter reviewed synthetic negatives whose product is a known real positive."""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Dict, List, Sequence, Set

from .chem_utils import canonicalize_smiles, split_reaction
from .known_positive_cache import load_known_positive_products

try:  # pragma: no cover - depends on optional RDKit install
    from rdkit import RDLogger  # type: ignore

    RDLogger.DisableLog("rdApp.*")
except Exception:  # pragma: no cover
    pass


KEEP_STATUS = "keep_synthetic_negative"


def canonical_product(reaction: str) -> str:
    try:
        _, _, products = split_reaction(reaction)
    except ValueError:
        return ""
    return canonicalize_smiles(products) or ""


def read_known_positive_products(paths: Sequence[str]) -> Set[str]:
    return load_known_positive_products(paths)


def filter_rows(
    input_path: str,
    known_positive_products: Set[str],
    review_status: str,
) -> tuple[List[Dict[str, str]], List[Dict[str, str]], List[str]]:
    kept: List[Dict[str, str]] = []
    removed: List[Dict[str, str]] = []
    with open(input_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        for row in reader:
            reaction = row.get("candidate_reaction") or row.get("reaction_smiles") or ""
            product = canonical_product(reaction)
            should_remove = (
                row.get("review_status", "") == review_status
                and product
                and product in known_positive_products
            )
            if should_remove:
                out = dict(row)
                out["known_positive_canonical_product"] = product
                removed.append(out)
            else:
                kept.append(dict(row))
    return kept, removed, fields


def write_rows(path: str, rows: Sequence[Dict[str, str]], fields: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--positive-csv", action="append", required=True)
    parser.add_argument("--synthetic-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--removed-csv", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--review-status", default=KEEP_STATUS)
    args = parser.parse_args()

    known_products = read_known_positive_products(args.positive_csv)
    kept, removed, fields = filter_rows(args.synthetic_csv, known_products, args.review_status)
    removed_fields = list(fields)
    if "known_positive_canonical_product" not in removed_fields:
        removed_fields.append("known_positive_canonical_product")
    write_rows(args.output_csv, kept, fields)
    write_rows(args.removed_csv, removed, removed_fields)
    summary = {
        "positive_csv": args.positive_csv,
        "synthetic_csv": args.synthetic_csv,
        "output_csv": args.output_csv,
        "removed_csv": args.removed_csv,
        "review_status": args.review_status,
        "known_positive_products": len(known_products),
        "input_rows": len(kept) + len(removed),
        "kept_rows": len(kept),
        "removed_rows": len(removed),
    }
    os.makedirs(os.path.dirname(args.summary_json), exist_ok=True)
    with open(args.summary_json, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
