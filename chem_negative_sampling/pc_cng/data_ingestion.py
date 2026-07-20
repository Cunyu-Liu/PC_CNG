"""Dataset ingestion and split utilities for PC-CNG experiments.

The functions are intentionally schema-tolerant: public reaction datasets use
different column names, so we normalize them into one minimal schema before
generation/training.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .chem_utils import canonicalize_reaction, split_reaction


REACTION_COLUMNS = [
    "reaction_smiles",
    "rxn",
    "rxn_smiles",
    "reaction",
    "ReactionSmiles",
    "canonical_rxn",
    "mapped_rxn",
    "mapped_reaction_smiles",
]

LABEL_COLUMNS = ["label", "label_type", "score", "outcome", "success", "is_positive", "class"]
YIELD_COLUMNS = ["yield", "Yield", "yield_percent", "percent_yield", "isolated_yield"]
SOURCE_COLUMNS = ["source", "dataset", "origin"]
ID_COLUMNS = ["source_id", "id", "reaction_id", "uuid"]


def _first_present(row: Dict[str, str], names: Sequence[str]) -> Optional[str]:
    for name in names:
        if name in row and str(row[name]).strip():
            return str(row[name]).strip()
    return None


def infer_label(row: Dict[str, str]) -> str:
    raw = _first_present(row, LABEL_COLUMNS)
    if raw is not None:
        value = raw.lower()
        if value in {"1", "true", "positive", "success", "successful", "pos"}:
            return "positive"
        if value in {"0", "false", "negative", "failed", "failure", "neg"}:
            return "real_negative"

    raw_yield = _first_present(row, YIELD_COLUMNS)
    if raw_yield is not None:
        try:
            value = float(raw_yield.replace("%", ""))
            if value <= 5.0:
                return "real_negative"
            if value >= 20.0:
                return "positive"
        except ValueError:
            pass
    return "positive"


def stable_source_id(reaction_smiles: str, prefix: str, index: int) -> str:
    digest = hashlib.sha1(reaction_smiles.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{index:09d}_{digest}"


def normalize_rows(
    input_path: str,
    source_name: str,
    limit: Optional[int] = None,
) -> Tuple[List[Dict[str, str]], Dict[str, object]]:
    rows: List[Dict[str, str]] = []
    skipped = 0
    with open(input_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        if not any(name in fieldnames for name in REACTION_COLUMNS):
            raise ValueError(f"No recognized reaction column found in {input_path}: {fieldnames}")

        for index, raw in enumerate(reader, start=1):
            if limit is not None and len(rows) >= limit:
                break
            reaction = _first_present(raw, REACTION_COLUMNS)
            if not reaction:
                skipped += 1
                continue
            canonical = canonicalize_reaction(reaction) or reaction
            try:
                reactants, agents, products = split_reaction(canonical)
            except ValueError:
                skipped += 1
                continue

            source_id = _first_present(raw, ID_COLUMNS) or stable_source_id(canonical, source_name, index)
            label = infer_label(raw)
            yield_value = _first_present(raw, YIELD_COLUMNS) or ""
            rows.append(
                {
                    "source_id": source_id,
                    "reaction_smiles": canonical,
                    "reactants": reactants,
                    "agents": agents,
                    "products": products,
                    "label_type": label,
                    "yield": yield_value,
                    "source": _first_present(raw, SOURCE_COLUMNS) or source_name,
                    "split_key": scaffoldless_split_key(reactants, products),
                }
            )

    stats = {
        "input_path": input_path,
        "source_name": source_name,
        "rows": len(rows),
        "skipped": skipped,
        "labels": count_by(rows, "label_type"),
    }
    return rows, stats


def scaffoldless_split_key(reactants: str, products: str) -> str:
    """Approximate split key when Bemis-Murcko/RDKit is unavailable."""
    token = f"{reactants}>>{products}"
    heavy = "".join(ch for ch in token if ch.isalpha() or ch in "=#")
    return hashlib.sha1(heavy.encode("utf-8")).hexdigest()[:10]


def count_by(rows: Iterable[Dict[str, str]], key: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        value = row.get(key, "")
        counts[value] = counts.get(value, 0) + 1
    return counts


def write_csv(path: str, rows: Sequence[Dict[str, str]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
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
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def assign_splits(rows: List[Dict[str, str]], train: float = 0.8, val: float = 0.1) -> None:
    keys = sorted({row["split_key"] for row in rows})
    train_cut = int(len(keys) * train)
    val_cut = int(len(keys) * (train + val))
    train_keys = set(keys[:train_cut])
    val_keys = set(keys[train_cut:val_cut])
    for row in rows:
        if row["split_key"] in train_keys:
            row["split"] = "train"
        elif row["split_key"] in val_keys:
            row["split"] = "val"
        else:
            row["split"] = "test"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--source-name", default="dataset")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    rows, stats = normalize_rows(args.input, args.source_name, args.limit)
    assign_splits(rows)
    stats["splits"] = count_by(rows, "split")
    write_csv(args.output, rows)
    os.makedirs(os.path.dirname(args.summary), exist_ok=True)
    with open(args.summary, "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2, ensure_ascii=False)
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
