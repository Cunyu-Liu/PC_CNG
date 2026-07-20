"""Normalize the public HiTEA HTE dataset into PC-CNG schema."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from typing import Dict, Iterable, List

import pandas as pd

from .chem_utils import canonicalize_reaction, split_reaction
from .data_ingestion import assign_splits, scaffoldless_split_key


def stable_id(reaction: str, source: str, index: int) -> str:
    digest = hashlib.sha1(f"{source}:{reaction}:{index}".encode("utf-8")).hexdigest()[:12]
    return f"{source}_{index:09d}_{digest}"


def normalize_hitea(input_csv: str, output_csv: str, summary_json: str, source_name: str, limit: int | None) -> None:
    df = pd.read_csv(input_csv)
    if limit is not None:
        df = df.head(limit)
    rows: List[Dict[str, str]] = []
    skipped = 0
    for index, row in df.iterrows():
        reaction = str(row.get("RXN_SMILES", "")).strip()
        if not reaction or reaction.lower() == "nan":
            skipped += 1
            continue
        canonical = canonicalize_reaction(reaction) or reaction
        try:
            reactants, agents, products = split_reaction(canonical)
        except ValueError:
            skipped += 1
            continue
        yield_raw = row.get("Product_Yield_PCT_Area_UV", "")
        try:
            yield_value = float(yield_raw)
        except Exception:
            skipped += 1
            continue

        label_type = "positive" if yield_value > 1.0 else "real_negative"
        source_id = str(row.get("REACTION_ID", "")).strip()
        if not source_id or source_id.lower() == "nan":
            source_id = stable_id(canonical, source_name, index + 1)
        rows.append(
            {
                "source_id": source_id,
                "reaction_smiles": canonical,
                "reactants": reactants,
                "agents": agents,
                "products": products,
                "label_type": label_type,
                "yield": f"{yield_value:.6g}",
                "source": source_name,
                "split_key": scaffoldless_split_key(reactants, products),
                "reaction_class": str(row.get("ReactionClass", "")),
            }
        )
    assign_splits(rows)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    fieldnames = [
        "source_id",
        "reaction_smiles",
        "reactants",
        "agents",
        "products",
        "label_type",
        "yield",
        "source",
        "split_key",
        "split",
        "reaction_class",
    ]
    with open(output_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "input": input_csv,
        "output": output_csv,
        "source": source_name,
        "rows": len(rows),
        "skipped": skipped,
        "labels": count_by(rows, "label_type"),
        "splits": count_by(rows, "split"),
        "reaction_classes": count_by(rows, "reaction_class"),
        "label_rule": "positive if Product_Yield_PCT_Area_UV > 1.0 else real_negative",
    }
    os.makedirs(os.path.dirname(summary_json), exist_ok=True)
    with open(summary_json, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def count_by(rows: Iterable[Dict[str, str]], key: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        value = row.get(key, "")
        counts[value] = counts.get(value, 0) + 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--source-name", default="hitea")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    normalize_hitea(args.input, args.output, args.summary, args.source_name, args.limit)


if __name__ == "__main__":
    main()

