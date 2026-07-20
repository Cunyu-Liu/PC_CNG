"""
P3-04: USPTO/ORD Real Condition Extraction Module
=================================================

翻盘 P2-08 NO-GO by using REAL ORD conditions from the `agents` field
instead of synthetic conditions.

P2-08 was NO-GO at -2.50pp because it used synthetic conditions. P3-04
extracts REAL conditions from the ORD/HITEa `agents` field and classifies
them into:
  - catalyst (contains Pd/Pt/Ru/Cu/Ni/Fe)
  - solvent (matches common solvent SMILES list)
  - reagent (everything else)

Output: data/processed/ord_conditions.json consumed by train_condition.py.

Hard constraints respected:
  - HC #4: unit tests in tests/test_extract_conditions.py (>=80% coverage)
  - Pure Python 3.10 stdlib + RDKit (no new dependencies)

Usage:
    python extract_conditions.py \
        --input data/processed/ord_normalized.csv \
        --output data/processed/ord_conditions.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Optional


# Common solvent SMILES (canonical forms + alternates). Keys are SMILES
# strings (canonicalized when RDKit is available), values are human-readable
# solvent names.
COMMON_SOLVENTS: Dict[str, str] = {
    # water
    "O": "water",
    # DMSO
    "CS(=O)C": "DMSO",
    "CS(C)=O": "DMSO",
    # DMF
    "CN(C)C=O": "DMF",
    # THF
    "C1CCOC1": "THF",
    "O1CCCC1": "THF",
    # toluene
    "Cc1ccccc1": "toluene",
    "C1=CC=C(C)C=C1": "toluene",
    # MeCN (acetonitrile)
    "CC#N": "MeCN",
    "CC(=N)": "MeCN",
    # EtOH
    "CCO": "EtOH",
    # MeOH
    "CO": "MeOH",
    # DCM (dichloromethane)
    "ClCCl": "DCM",
    "C(Cl)Cl": "DCM",
    # EtOAc (ethyl acetate)
    "CC(=O)OCC": "EtOAc",
    # 1,4-dioxane
    "C1COCCO1": "dioxane",
    "O1CCOCC1": "dioxane",
    # acetone
    "CC(=O)C": "acetone",
    "CC(C)=O": "acetone",
    # chloroform
    "ClC(Cl)Cl": "chloroform",
    "C(Cl)(Cl)Cl": "chloroform",
    # hexane
    "CCCCCC": "hexane",
    # heptane
    "CCCCCCC": "heptane",
    # benzene
    "c1ccccc1": "benzene",
    "C1=CC=CC=C1": "benzene",
    # xylene (mixed isomers)
    "Cc1cccc(C)c1": "xylene",
    "Cc1ccccc1C": "xylene",
    # 1,2-dichloroethane (DCE)
    "ClCCCl": "DCE",
    "ClCH2CClH2": "DCE",
    # nitromethane
    "C[N+](=O)[O-]": "nitromethane",
    "[N+](=O)([O-])C": "nitromethane",
    # DMAc (dimethylacetamide)
    "CC(=O)N(C)C": "DMAc",
    # NMP (N-methyl-2-pyrrolidone)
    "CCCC1CC(=O)N1": "NMP",
    "O=C1CCC-CN1C": "NMP",
    # ethyl ether (Et2O)
    "CCOCC": "Et2O",
    "CCOCC": "diethyl_ether",
    # methoxybenzene (anisole)
    "COc1ccccc1": "anisole",
    # carbon tetrachloride
    "ClC(Cl)(Cl)Cl": "CCl4",
    # formamide
    "C(=O)N": "formamide",
    "NC=O": "formamide",
    # acetic acid (used as solvent sometimes)
    "CC(=O)O": "acetic_acid",
    # diethylamine
    "CCNCC": "Et2NH",
}


# Catalyst element symbols (transition metals commonly used in catalysis).
CATALYST_ELEMENTS = {"Pd", "Pt", "Ru", "Cu", "Ni", "Fe"}


def parse_agents_field(agents_str: Optional[str]) -> List[str]:
    """Parse comma-separated agents SMILES into a list of individual SMILES.

    Handles missing/empty/malformed agents fields gracefully.

    Args:
        agents_str: Raw `agents` field from ORD/HITEa CSV. Comma-separated
            SMILES, may contain solvents/catalysts/reagents. May be empty
            or None.

    Returns:
        List of stripped, non-empty SMILES strings.

    Examples:
        >>> parse_agents_field("CCO, c1ccccc1, [Pd]")
        ['CCO', 'c1ccccc1', '[Pd]']
        >>> parse_agents_field("")
        []
        >>> parse_agents_field(None)
        []
    """
    if not agents_str or not isinstance(agents_str, str):
        return []
    parts = [s.strip() for s in agents_str.split(",")]
    return [p for p in parts if p]


def classify_agent(smiles: str) -> str:
    """Classify a single agent SMILES as 'catalyst', 'solvent', or 'reagent'.

    Classification rules (in priority order):
      1. Catalyst: SMILES contains Pd/Pt/Ru/Cu/Ni/Fe (any atom).
      2. Solvent: matches common solvent SMILES (canonicalized if RDKit
         is available).
      3. Reagent: everything else.

    Args:
        smiles: Single SMILES string (already parsed from comma-separated
            `agents` field).

    Returns:
        One of 'catalyst', 'solvent', or 'reagent'.

    Examples:
        >>> classify_agent("[Pd]")
        'catalyst'
        >>> classify_agent("CCO")
        'solvent'
        >>> classify_agent("c1ccccc1")  # benzene is a solvent in our list
        'solvent'
        >>> classify_agent("CC(=O)NC")  # acetamide-like, not a solvent
        'reagent'
    """
    if not smiles:
        return "reagent"

    # Try canonicalize via RDKit if available (preferred path).
    canonical = smiles
    rdkit_available = False
    try:
        from rdkit import Chem  # type: ignore

        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            rdkit_available = True
            # Check atoms for catalyst elements.
            for atom in mol.GetAtoms():
                if atom.GetSymbol() in CATALYST_ELEMENTS:
                    return "catalyst"
            canonical = Chem.MolToSmiles(mol, canonical=True)
        else:
            # SMILES parse failed; fall back to substring check for metals.
            for elem in CATALYST_ELEMENTS:
                if elem in smiles:
                    return "catalyst"
    except ImportError:
        # RDKit not available; fall back to substring check.
        for elem in CATALYST_ELEMENTS:
            if elem in smiles:
                return "catalyst"

    # Check solvent by canonical or raw SMILES.
    if canonical in COMMON_SOLVENTS or smiles in COMMON_SOLVENTS:
        return "solvent"

    # Additional RDKit-based solvent canonicalization fallback (sometimes
    # canonicalization produces different forms).
    if rdkit_available:
        for sol_smiles in COMMON_SOLVENTS:
            try:
                from rdkit import Chem  # type: ignore

                sol_mol = Chem.MolFromSmiles(sol_smiles)
                if sol_mol is not None:
                    sol_canon = Chem.MolToSmiles(sol_mol, canonical=True)
                    if sol_canon == canonical:
                        return "solvent"
            except Exception:
                continue

    return "reagent"


def extract_conditions_from_row(row: dict) -> dict:
    """Extract and classify agents from a CSV row.

    Args:
        row: Dict mapping CSV column names to values. Expected keys:
            `source_id`, `reaction_smiles`, `agents`, `split`.

    Returns:
        Dict with keys: source_id, reaction_smiles, catalyst, solvent,
        reagent, temperature, split. Multiple catalysts/solvents/reagents
        are joined with '.'.
    """
    agents = parse_agents_field(row.get("agents", ""))
    catalysts, solvents, reagents = [], [], []
    for smi in agents:
        cls = classify_agent(smi)
        if cls == "catalyst":
            catalysts.append(smi)
        elif cls == "solvent":
            solvents.append(smi)
        else:
            reagents.append(smi)
    return {
        "source_id": row.get("source_id", ""),
        "reaction_smiles": row.get("reaction_smiles", ""),
        "catalyst": ".".join(catalysts) if catalysts else "",
        "solvent": ".".join(solvents) if solvents else "",
        "reagent": ".".join(reagents) if reagents else "",
        "temperature": None,
        "split": row.get("split", "train"),
    }


def extract_conditions(input_csv: str, output_json: str) -> int:
    """Extract conditions from ORD/HITEa CSV to a JSON file.

    Args:
        input_csv: Path to input CSV (e.g., ord_normalized.csv or
            hitea_full_normalized.csv). Must have columns: source_id,
            reaction_smiles, agents, split.
        output_json: Path to write JSON output. Parent directory is
            created if it does not exist.

    Returns:
        Number of records written.

    Raises:
        FileNotFoundError: if input_csv does not exist.
        KeyError: if required columns are missing (handled gracefully,
            values default to empty strings).
    """
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    records: List[dict] = []
    with open(input_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(extract_conditions_from_row(row))

    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(records, f, indent=2)
    return len(records)


def main() -> None:
    """CLI entry point for P3-04 condition extraction."""
    parser = argparse.ArgumentParser(
        description="P3-04: Extract REAL ORD/HITEa conditions from `agents` field "
                    "(翻盘 P2-08 NO-GO)"
    )
    parser.add_argument(
        "--input", required=True,
        help="Input CSV path (ord_normalized.csv or hitea_full_normalized.csv)"
    )
    parser.add_argument(
        "--output", required=True,
        help="Output JSON path (e.g., data/processed/ord_conditions.json)"
    )
    args = parser.parse_args()
    n = extract_conditions(args.input, args.output)
    print(f"[P3-04] Wrote {n} records to {args.output}")


if __name__ == "__main__":
    main()
