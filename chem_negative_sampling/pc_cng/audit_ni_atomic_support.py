"""Audit nickel atom support in reaction datasets.

The weak-class Ni claim should be backed by molecular evidence, not only by
class labels. This script uses a fast text prefilter followed by RDKit atom
number detection (atomic number 28) and writes reproducible JSON/CSV/MD
artifacts for the data-source gap.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .chem_utils import canonicalize_smiles, molecule_parts, split_reaction

try:  # pragma: no cover - depends on environment
    from rdkit import Chem, RDLogger  # type: ignore

    RDLogger.DisableLog("rdApp.*")
except Exception:  # pragma: no cover
    Chem = None  # type: ignore


NI_ATOMIC_NUMBER = 28


def dataset_name(path: str) -> str:
    name = os.path.basename(path)
    if name.endswith(".csv"):
        name = name[:-4]
    return name


def reaction_components(reaction: str) -> Optional[Tuple[str, str, str]]:
    try:
        return split_reaction(reaction)
    except ValueError:
        return None


def smiles_has_ni_atom(smiles: str) -> Tuple[bool, int]:
    """Return (has_ni, invalid_component_count) for a SMILES field."""
    if "Ni" not in smiles:
        return False, 0

    invalid = 0
    if Chem is None:  # Lightweight fallback for environments without RDKit.
        return "[Ni" in smiles or "Ni]" in smiles or "Ni+" in smiles, 0

    found = False
    for part in molecule_parts(smiles):
        if "Ni" not in part:
            continue
        mol = Chem.MolFromSmiles(part, sanitize=False)
        if mol is None:
            invalid += 1
            continue
        if any(atom.GetAtomicNum() == NI_ATOMIC_NUMBER for atom in mol.GetAtoms()):
            found = True
    return found, invalid


def reaction_has_ni_atom(reaction: str) -> Tuple[bool, Dict[str, bool], int, bool]:
    """Return Ni status, component locations, invalid components, invalid rxn."""
    components = reaction_components(reaction)
    if components is None:
        has_ni, invalid = smiles_has_ni_atom(reaction)
        return has_ni, {"reaction": has_ni}, invalid, True

    locations: Dict[str, bool] = {}
    invalid_components = 0
    for name, smiles in zip(("reactants", "agents", "products"), components):
        has_ni, invalid = smiles_has_ni_atom(smiles)
        locations[name] = has_ni
        invalid_components += invalid
    return any(locations.values()), locations, invalid_components, False


def split_counts(rows: Iterable[Dict[str, str]]) -> Dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts[(row.get("split") or "unknown").strip() or "unknown"] += 1
    return dict(sorted(counts.items()))


def read_reaction(row: Dict[str, str], reaction_column: str) -> str:
    return (row.get(reaction_column) or row.get("reaction_smiles") or "").strip()


def summarize_dataset(
    path: str,
    reaction_column: str,
    max_examples: int,
) -> Tuple[Dict[str, object], List[Dict[str, str]]]:
    total_rows = 0
    reaction_rows = 0
    ni_rows: List[Dict[str, str]] = []
    prefilter_rows = 0
    invalid_reaction_rows = 0
    invalid_component_rows = 0
    distinct_reactants = set()
    distinct_reactions = set()

    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            total_rows += 1
            reaction = read_reaction(row, reaction_column)
            if not reaction:
                continue
            reaction_rows += 1
            if "Ni" not in reaction:
                continue
            prefilter_rows += 1

            has_ni, locations, invalid_components, invalid_reaction = reaction_has_ni_atom(reaction)
            invalid_component_rows += invalid_components
            if invalid_reaction:
                invalid_reaction_rows += 1
            if not has_ni:
                continue

            components = reaction_components(reaction)
            reactants = components[0] if components is not None else ""
            canonical_reactants = canonicalize_smiles(reactants) or reactants
            distinct_reactants.add(canonical_reactants)
            distinct_reactions.add(reaction)
            if len(ni_rows) < max_examples:
                ni_rows.append(
                    {
                        "dataset": dataset_name(path),
                        "source_id": row.get("source_id", ""),
                        "split": row.get("split", ""),
                        "reaction_class": row.get("reaction_class", ""),
                        "ni_locations": ",".join(k for k, v in locations.items() if v),
                        "canonical_reactants": canonical_reactants,
                        "reaction_smiles": reaction,
                    }
                )

    summary = {
        "dataset": dataset_name(path),
        "path": path,
        "total_rows": total_rows,
        "reaction_rows": reaction_rows,
        "ni_prefilter_rows": prefilter_rows,
        "ni_reactions": len(distinct_reactions),
        "distinct_ni_parent_reactants": len({item for item in distinct_reactants if item}),
        "invalid_reaction_rows_with_ni_prefilter": invalid_reaction_rows,
        "invalid_component_rows_with_ni_prefilter": invalid_component_rows,
        "split_counts_in_examples": split_counts(ni_rows),
    }
    return summary, ni_rows


def write_csv(path: str, rows: Sequence[Dict[str, str]], fieldnames: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: str, dataset_summaries: Sequence[Dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = [
        "dataset",
        "reaction_rows",
        "ni_prefilter_rows",
        "ni_reactions",
        "distinct_ni_parent_reactants",
        "invalid_reaction_rows_with_ni_prefilter",
        "invalid_component_rows_with_ni_prefilter",
    ]
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join(["---"] * len(fields)) + " |",
    ]
    for summary in dataset_summaries:
        lines.append("| " + " | ".join(str(summary.get(field, "")) for field in fields) + " |")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def run_audit(
    input_csvs: Sequence[str],
    output_dir: str,
    reaction_column: str,
    max_examples_per_dataset: int,
) -> Dict[str, object]:
    os.makedirs(output_dir, exist_ok=True)
    summaries: List[Dict[str, object]] = []
    examples: List[Dict[str, str]] = []
    for path in input_csvs:
        summary, rows = summarize_dataset(path, reaction_column, max_examples_per_dataset)
        summaries.append(summary)
        examples.extend(rows)

    payload = {
        "config": {
            "input_csv": list(input_csvs),
            "reaction_column": reaction_column,
            "max_examples_per_dataset": max_examples_per_dataset,
            "detection": "text prefilter containing 'Ni' followed by RDKit atomic number 28 when RDKit is available",
            "rdkit_available": Chem is not None,
        },
        "datasets": summaries,
        "total_ni_reactions": sum(int(row["ni_reactions"]) for row in summaries),
        "total_distinct_ni_parent_reactants": sum(
            int(row["distinct_ni_parent_reactants"]) for row in summaries
        ),
    }

    json_path = os.path.join(output_dir, "ni_atomic_support_audit.json")
    csv_path = os.path.join(output_dir, "ni_atomic_examples.csv")
    md_path = os.path.join(output_dir, "ni_atomic_support_audit.md")
    payload["outputs"] = {"json": json_path, "examples_csv": csv_path, "summary_md": md_path}
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    write_csv(
        csv_path,
        examples,
        [
            "dataset",
            "source_id",
            "split",
            "reaction_class",
            "ni_locations",
            "canonical_reactants",
            "reaction_smiles",
        ],
    )
    write_markdown(md_path, summaries)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reaction-column", default="reaction_smiles")
    parser.add_argument("--max-examples-per-dataset", type=int, default=50)
    args = parser.parse_args()

    payload = run_audit(
        input_csvs=args.input_csv,
        output_dir=args.output_dir,
        reaction_column=args.reaction_column,
        max_examples_per_dataset=args.max_examples_per_dataset,
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
