"""Build curated weak-class positive contexts from external sources.

The current HiTEA weak-class slice does not contain enough distinct molecular
parent reactions for Amide/Cu/Ni. This builder creates an auditable positive
context CSV from explicit HiTEA cleaned datasets and rule-classified USPTO
reactions. It does not generate negatives by itself.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .chem_utils import canonicalize_reaction, canonicalize_smiles, split_reaction
from .data_ingestion import assign_splits, scaffoldless_split_key

try:  # pragma: no cover - environment dependent
    from rdkit import Chem, RDLogger  # type: ignore

    RDLogger.DisableLog("rdApp.*")
except Exception:  # pragma: no cover
    Chem = None  # type: ignore


AMIDE_PRODUCT_RE = re.compile(r"C\(=O\)N|NC\(=O\)|C\(=O\)\[N|N\]C\(=O\)")
ACYL_RE = re.compile(r"C\(=O\)(Cl|Br|I|O|OC|\[O)")
AMINE_RE = re.compile(r"N|\[NH|\[NH2|\[NH3")


def rough_amide_text_prefilter(reactants: str, product: str) -> bool:
    """Avoid expensive RDKit matching for reactions that cannot be amides."""
    product_has_nitrogen = "N" in product or "n" in product or "[N" in product or "[n" in product
    reactants_have_nitrogen = "N" in reactants or "n" in reactants or "[N" in reactants or "[n" in reactants
    return product_has_nitrogen and reactants_have_nitrogen and "=O" in product and "=O" in reactants


def stable_id(prefix: str, reaction: str, index: int) -> str:
    digest = hashlib.sha1(f"{prefix}:{reaction}:{index}".encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{index:09d}_{digest}"


def parse_named_path(value: str) -> Tuple[str, str]:
    if "=" not in value:
        raise ValueError(f"Expected NAME=PATH, got {value!r}")
    name, path = value.split("=", 1)
    if not name or not path:
        raise ValueError(f"Expected NAME=PATH, got {value!r}")
    return name, path


def rdkit_has_substructure(smiles: str, smarts: str) -> bool:
    if Chem is None:
        return False
    mol = Chem.MolFromSmiles(smiles)
    pattern = Chem.MolFromSmarts(smarts)
    if mol is None or pattern is None:
        return False
    return bool(mol.HasSubstructMatch(pattern))


def is_amide_coupling(reaction: str) -> bool:
    try:
        reactants, _, product = split_reaction(reaction)
    except ValueError:
        return False
    if not rough_amide_text_prefilter(reactants, product):
        return False
    if Chem is not None:
        product_has_amide = rdkit_has_substructure(product, "[CX3](=[OX1])[NX3]")
        reactants_have_amine = rdkit_has_substructure(reactants, "[NX3;H2,H1,H0;!$(N=*)]")
        reactants_have_acyl = any(
            rdkit_has_substructure(reactants, smarts)
            for smarts in [
                "[CX3](=[OX1])[Cl,Br,I]",
                "[CX3](=[OX1])[OX2H]",
                "[CX3](=[OX1])[OX2][#6]",
            ]
        )
        return product_has_amide and reactants_have_amine and reactants_have_acyl
    return bool(AMIDE_PRODUCT_RE.search(product) and AMINE_RE.search(reactants) and ACYL_RE.search(reactants))


def is_cu_coupling(reaction: str) -> bool:
    text = reaction.lower()
    return "[cu" in text or ".cu" in text or "cui" in text


def is_ni_coupling(reaction: str) -> bool:
    text = reaction.lower()
    return "[ni" in text or ".ni" in text or "ni(" in text


def classify_reaction(reaction: str, enabled: Sequence[str]) -> Optional[str]:
    enabled_set = set(enabled)
    if "Amide coupling" in enabled_set and is_amide_coupling(reaction):
        return "Amide coupling"
    if "Cu coupling" in enabled_set and is_cu_coupling(reaction):
        return "Cu coupling"
    if "Ni coupling" in enabled_set and is_ni_coupling(reaction):
        return "Ni coupling"
    return None


def normalize_context(
    reaction: str,
    reaction_class: str,
    source: str,
    source_id: str,
    yield_value: str = "",
) -> Optional[Dict[str, str]]:
    canonical = canonicalize_reaction(reaction) or reaction
    try:
        reactants, agents, products = split_reaction(canonical)
    except ValueError:
        return None
    return {
        "source_id": source_id,
        "reaction_smiles": canonical,
        "reactants": reactants,
        "agents": agents,
        "products": products,
        "label_type": "positive",
        "yield": yield_value,
        "source": source,
        "split_key": scaffoldless_split_key(reactants, products),
        "split": "train",
        "reaction_class": reaction_class,
    }


def read_hitea_cleaned(named_paths: Sequence[str], yield_threshold: float) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for spec in named_paths:
        reaction_class, path = parse_named_path(spec)
        with open(path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for index, row in enumerate(reader, start=1):
                reaction = (row.get("RXN_SMILES") or "").strip()
                if not reaction:
                    continue
                try:
                    yield_value = float(row.get("Product_Yield_PCT_Area_UV") or 0.0)
                except ValueError:
                    yield_value = 0.0
                if yield_value <= yield_threshold:
                    continue
                source_id = (row.get("REACTION_ID") or "").strip() or stable_id("curated_hitea", reaction, index)
                item = normalize_context(
                    reaction=reaction,
                    reaction_class=reaction_class,
                    source=f"curated_hitea_{reaction_class.replace(' ', '_').lower()}",
                    source_id=source_id,
                    yield_value=f"{yield_value:.6g}",
                )
                if item:
                    rows.append(item)
    return rows


def read_uspto_rule_classified(
    paths: Sequence[str],
    classes: Sequence[str],
    max_rows_per_class: int,
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    counts: Dict[str, int] = {}
    for path in paths:
        with open(path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for index, row in enumerate(reader, start=1):
                remaining_classes = [
                    reaction_class
                    for reaction_class in classes
                    if max_rows_per_class <= 0 or counts.get(reaction_class, 0) < max_rows_per_class
                ]
                if not remaining_classes:
                    break
                reaction = (row.get("reaction_smiles") or "").strip()
                if not reaction:
                    continue
                reaction_class = classify_reaction(reaction, remaining_classes)
                if not reaction_class:
                    continue
                source_id = (row.get("source_id") or "").strip() or stable_id("curated_uspto", reaction, index)
                item = normalize_context(
                    reaction=reaction,
                    reaction_class=reaction_class,
                    source="curated_uspto_openmolecules_rule",
                    source_id=f"curated_{source_id}",
                    yield_value=row.get("yield", ""),
                )
                if item:
                    rows.append(item)
                    counts[reaction_class] = counts.get(reaction_class, 0) + 1
    return rows


def dedupe_rows(rows: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    seen = set()
    for row in rows:
        key = canonicalize_reaction(row["reaction_smiles"]) or row["reaction_smiles"]
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def count_by(rows: Iterable[Dict[str, str]], field: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        value = row.get(field, "")
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
        "reaction_class",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hitea-cleaned-csv", action="append", default=[], help="REACTION_CLASS=PATH")
    parser.add_argument("--uspto-csv", action="append", default=[])
    parser.add_argument("--class-name", action="append", default=["Amide coupling", "Cu coupling", "Ni coupling"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--yield-threshold", type=float, default=1.0)
    parser.add_argument("--max-uspto-rows-per-class", type=int, default=200)
    args = parser.parse_args()

    hitea_rows = read_hitea_cleaned(args.hitea_cleaned_csv, args.yield_threshold)
    uspto_rows = read_uspto_rule_classified(args.uspto_csv, args.class_name, args.max_uspto_rows_per_class)
    rows = dedupe_rows([*hitea_rows, *uspto_rows])
    assign_splits(rows)
    write_csv(args.output, rows)
    summary = {
        "hitea_cleaned_csv": args.hitea_cleaned_csv,
        "uspto_csv": args.uspto_csv,
        "class_name": args.class_name,
        "yield_threshold": args.yield_threshold,
        "max_uspto_rows_per_class": args.max_uspto_rows_per_class,
        "input_rows": {
            "hitea": len(hitea_rows),
            "uspto_rule": len(uspto_rows),
        },
        "rows": len(rows),
        "reaction_classes": count_by(rows, "reaction_class"),
        "sources": count_by(rows, "source"),
        "splits": count_by(rows, "split"),
        "output": args.output,
    }
    os.makedirs(os.path.dirname(args.summary), exist_ok=True)
    with open(args.summary, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
