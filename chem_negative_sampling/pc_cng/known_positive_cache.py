"""Known-positive reaction/product cache helpers.

Large positive CSVs such as USPTO/OpenMolecules are expensive to canonicalize
repeatedly. These helpers build a reusable JSON cache with canonical reactions
and products, while keeping the existing CSV input path behavior unchanged.
"""

from __future__ import annotations

import csv
import json
import os
from typing import Dict, List, Sequence, Set

from .chem_utils import canonicalize_reaction, canonicalize_smiles, split_reaction

try:  # pragma: no cover - depends on optional RDKit install
    from rdkit import RDLogger  # type: ignore

    RDLogger.DisableLog("rdApp.*")
except Exception:  # pragma: no cover
    pass


def canonical_product_from_reaction(reaction: str) -> str:
    try:
        _, _, products = split_reaction(reaction)
    except ValueError:
        return ""
    return canonicalize_smiles(products) or ""


def is_cache_path(path: str) -> bool:
    return path.endswith(".json")


def load_cache(path: str) -> Dict[str, object]:
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if "canonical_reactions" not in payload and "canonical_products" not in payload:
        raise ValueError(f"Not a known-positive cache JSON: {path}")
    return payload


def build_known_positive_cache(paths: Sequence[str]) -> Dict[str, object]:
    canonical_reactions: Set[str] = set()
    canonical_products: Set[str] = set()
    per_file: List[Dict[str, object]] = []

    for path in paths:
        rows_seen = 0
        positive_rows = 0
        reaction_hits = 0
        product_hits = 0
        skipped_missing_reaction = 0
        skipped_uncanonicalizable = 0
        with open(path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                rows_seen += 1
                label_type = (row.get("label_type") or "").strip().lower()
                if label_type and label_type not in {"positive", "success"}:
                    continue
                positive_rows += 1
                reaction = (row.get("reaction_smiles") or row.get("positive_reaction") or "").strip()
                if not reaction:
                    skipped_missing_reaction += 1
                    continue
                canonical = canonicalize_reaction(reaction)
                if canonical:
                    canonical_reactions.add(canonical)
                    reaction_hits += 1
                else:
                    skipped_uncanonicalizable += 1
                product = canonical_product_from_reaction(reaction)
                if product:
                    canonical_products.add(product)
                    product_hits += 1
        per_file.append(
            {
                "path": path,
                "rows_seen": rows_seen,
                "positive_rows": positive_rows,
                "canonical_reaction_rows": reaction_hits,
                "canonical_product_rows": product_hits,
                "skipped_missing_reaction": skipped_missing_reaction,
                "skipped_uncanonicalizable_reaction": skipped_uncanonicalizable,
            }
        )

    return {
        "schema": "pc_cng_known_positive_cache_v1",
        "source_csv": list(paths),
        "source_summary": per_file,
        "canonical_reactions": sorted(canonical_reactions),
        "canonical_products": sorted(canonical_products),
        "canonical_reaction_count": len(canonical_reactions),
        "canonical_product_count": len(canonical_products),
    }


def write_cache(path: str, payload: Dict[str, object]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def load_known_positive_reactions(paths: Sequence[str]) -> Set[str]:
    out: Set[str] = set()
    csv_paths: List[str] = []
    for path in paths:
        if not path:
            continue
        if is_cache_path(path):
            payload = load_cache(path)
            out.update(str(item) for item in payload.get("canonical_reactions", []))
        else:
            csv_paths.append(path)
    if csv_paths:
        payload = build_known_positive_cache(csv_paths)
        out.update(str(item) for item in payload.get("canonical_reactions", []))
    return out


def load_known_positive_products(paths: Sequence[str]) -> Set[str]:
    out: Set[str] = set()
    csv_paths: List[str] = []
    for path in paths:
        if not path:
            continue
        if is_cache_path(path):
            payload = load_cache(path)
            out.update(str(item) for item in payload.get("canonical_products", []))
        else:
            csv_paths.append(path)
    if csv_paths:
        payload = build_known_positive_cache(csv_paths)
        out.update(str(item) for item in payload.get("canonical_products", []))
    return out
