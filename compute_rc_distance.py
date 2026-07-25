#!/usr/bin/env python3
"""Compute reaction-center distance for G8-A supplementary metric.

Spec L1812 lists "reaction-center distance" as a difficulty metric.
G8-A's original run did not include it (manifest lacked the field).
This script computes it post-hoc from atom-mapped HTE reaction SMILES.

Reaction-center distance = shortest molecular graph path from the
PC-CNG edit locus to the nearest reaction-center atom.

Reaction center = atoms whose bond environment changes between
reactants and products (identified via atom mapping).

Outputs:
  results/p4_mechanism_curve/supplementary_rc_distance.csv
  results/p4_mechanism_curve/rc_distance_curve.json
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

os.environ["RDKitRDLogger"] = "0"
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Chem.rdFMCS import FindMCS
except ImportError:
    print("ERROR: RDKit not available", file=sys.stderr)
    sys.exit(1)

RESEARCH_DIR = Path("/home/cunyuliu/pc_cng_research")
MANIFEST_PATH = RESEARCH_DIR / "data/p4/manifests/hte_feasibility_v2.json"
HTE_CSV_PATH = RESEARCH_DIR / "data/processed/hitea_full_normalized.csv"
METRICS_CSV_PATH = RESEARCH_DIR / "results/p4_mechanism_curve/per_candidate_metrics.csv"
OUTPUT_CSV = RESEARCH_DIR / "results/p4_mechanism_curve/supplementary_rc_distance.csv"
OUTPUT_CURVE = RESEARCH_DIR / "results/p4_mechanism_curve/rc_distance_curve.json"


def load_hte_reactions() -> Dict[str, str]:
    """Map source_id -> reaction_smiles from HTE CSV."""
    out: Dict[str, str] = {}
    with open(HTE_CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = row.get("source_id", "").strip()
            rxn = row.get("reaction_smiles", "").strip()
            if sid and rxn:
                out[sid] = rxn
    return out


def load_manifest() -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
    """Return (group_to_source_rxn, candidate_to_smiles, group_to_gold_smiles)."""
    with open(MANIFEST_PATH) as f:
        m = json.load(f)
    group_to_source: Dict[str, str] = {}
    cand_to_smiles: Dict[str, str] = {}
    group_to_gold: Dict[str, str] = {}
    for g in m.get("groups", []):
        gid = g.get("group_id", "")
        srid = g.get("source_reaction_id", "")
        if gid and srid:
            group_to_source[gid] = srid
        for c in g.get("candidates", []):
            cid = c.get("candidate_id", "")
            smi = c.get("candidate_smiles", "")
            if cid and smi:
                cand_to_smiles[cid] = smi
            if c.get("gold_candidate") and smi:
                group_to_gold[gid] = smi
    return group_to_source, cand_to_smiles, group_to_gold


def identify_reaction_center(reaction_smiles: str) -> Set[int]:
    """Identify reaction-center atom map numbers from an atom-mapped reaction.

    Returns the set of atom map numbers whose bond environment changes
    between reactants and products.
    """
    if ">>" in reaction_smiles:
        left, right = reaction_smiles.split(">>", 1)
    else:
        parts = reaction_smiles.split(">")
        if len(parts) >= 3:
            left, right = parts[0], parts[2]
        elif len(parts) == 2:
            left, right = parts[0], parts[1]
        else:
            return set()

    reactants = Chem.MolFromSmiles(left)
    products = Chem.MolFromSmiles(right)
    if reactants is None or products is None:
        return set()

    # Build bond environment per atom map number: frozenset of (neighbor_mapnum, bond_order)
    def bond_env(mol) -> Dict[int, frozenset]:
        env: Dict[int, frozenset] = {}
        for atom in mol.GetAtoms():
            mn = atom.GetAtomMapNum()
            if mn == 0:
                continue
            bonds = []
            for bond in atom.GetBonds():
                other = bond.GetOtherAtom(atom)
                other_mn = other.GetAtomMapNum()
                if other_mn == 0:
                    continue
                bonds.append((other_mn, bond.GetBondTypeAsDouble()))
            env[mn] = frozenset(bonds)
        return env

    r_env = bond_env(reactants)
    p_env = bond_env(products)

    # Reaction center = atoms with changed bond environment
    rc_atoms: Set[int] = set()
    for mn in set(r_env.keys()) | set(p_env.keys()):
        if r_env.get(mn) != p_env.get(mn):
            rc_atoms.add(mn)
    return rc_atoms


def find_edit_locus_atoms(
    candidate_smiles: str, gold_smiles: str
) -> Set[int]:
    """Find atom indices in the candidate that differ from the gold product.

    Uses Maximum Common Substructure (MCS) to align candidate and gold,
    then returns the atom indices in the candidate that are NOT in the MCS.
    """
    cand_mol = Chem.MolFromSmiles(candidate_smiles)
    gold_mol = Chem.MolFromSmiles(gold_smiles)
    if cand_mol is None or gold_mol is None:
        return set()

    # If identical, no edit locus
    if Chem.MolToSmiles(cand_mol) == Chem.MolToSmiles(gold_mol):
        return set()

    try:
        mcs = FindMCS([cand_mol, gold_mol], timeout=5, matchValences=False)
        if mcs.smartsString == "" or mcs.smartsString is None:
            return set(range(cand_mol.GetNumAtoms()))
        mcs_mol = Chem.MolFromSmarts(mcs.smartsString)
        if mcs_mol is None:
            return set(range(cand_mol.GetNumAtoms()))
        match = cand_mol.GetSubstructMatch(mcs_mol)
        if not match:
            return set(range(cand_mol.GetNumAtoms()))
        matched = set(match)
        return set(range(cand_mol.GetNumAtoms())) - matched
    except Exception:
        return set(range(cand_mol.GetNumAtoms()))


def map_locus_to_reaction_center_distance(
    candidate_smiles: str,
    gold_smiles: str,
    reaction_smiles: str,
) -> float:
    """Compute the shortest graph distance from edit locus to reaction center.

    Steps:
    1. Identify reaction-center atom map numbers from the reaction.
    2. Find edit locus atoms in the candidate (atoms differing from gold).
    3. Map reaction-center atom map numbers to atom indices in the candidate
       (via the gold product, which shares atom mapping with the reaction).
    4. Compute shortest path distance from each edit locus atom to the nearest
       reaction-center atom in the candidate's molecular graph.
    5. Return the minimum distance (0 if edit is at the reaction center).
    """
    rc_mapnums = identify_reaction_center(reaction_smiles)
    if not rc_mapnums:
        return -1.0  # Cannot compute

    edit_atoms = find_edit_locus_atoms(candidate_smiles, gold_smiles)
    if not edit_atoms:
        return 0.0  # Gold candidate: edit is at reaction center by definition

    cand_mol = Chem.MolFromSmiles(candidate_smiles)
    if cand_mol is None:
        return -1.0

    # Map reaction-center atom map numbers to candidate atom indices.
    # The candidate is derived from the gold product, which has the same
    # atom mapping as the reaction. However, the candidate SMILES might
    # not have atom mapping (PC-CNG strips it). So we need to align
    # the candidate with the gold product to find corresponding atoms.
    gold_mol = Chem.MolFromSmiles(gold_smiles)
    if gold_mol is None:
        return -1.0

    # Check if gold product has atom mapping
    gold_has_mapping = any(a.GetAtomMapNum() != 0 for a in gold_mol.GetAtoms())

    rc_indices_in_cand: Set[int] = set()

    if gold_has_mapping:
        # Map RC map numbers to gold product atom indices
        gold_mapnum_to_idx: Dict[int, int] = {}
        for atom in gold_mol.GetAtoms():
            mn = atom.GetAtomMapNum()
            if mn != 0:
                gold_mapnum_to_idx[mn] = atom.GetIdx()

        gold_rc_indices = set()
        for mn in rc_mapnums:
            if mn in gold_mapnum_to_idx:
                gold_rc_indices.add(gold_mapnum_to_idx[mn])

        if not gold_rc_indices:
            return -1.0

        # Map gold RC indices to candidate indices via MCS alignment
        try:
            mcs = FindMCS([cand_mol, gold_mol], timeout=5, matchValences=False)
            if mcs.smartsString and mcs.smartsString != "":
                mcs_mol = Chem.MolFromSmarts(mcs.smartsString)
                if mcs_mol is not None:
                    cand_match = cand_mol.GetSubstructMatch(mcs_mol)
                    gold_match = gold_mol.GetSubstructMatch(mcs_mol)
                    if cand_match and gold_match and len(cand_match) == len(gold_match):
                        for ci, gi in zip(cand_match, gold_match):
                            if gi in gold_rc_indices:
                                rc_indices_in_cand.add(ci)
        except Exception:
            pass

    if not rc_indices_in_cand:
        # Fallback: use edit distance as proxy (higher edit distance = farther from RC)
        return -1.0

    # Compute distance matrix for candidate
    from rdkit import Chem
    dist_matrix = Chem.GetDistanceMatrix(cand_mol)

    # For each edit locus atom, find min distance to any RC atom
    min_distances: List[float] = []
    for ea in edit_atoms:
        if ea >= cand_mol.GetNumAtoms():
            continue
        for rc_idx in rc_indices_in_cand:
            if rc_idx >= cand_mol.GetNumAtoms():
                continue
            d = dist_matrix[ea][rc_idx]
            min_distances.append(d)

    if not min_distances:
        return -1.0

    return float(min(min_distances))


def compute_curve(
    rc_distances: List[float],
    downstream_losses: List[float],
    labels: List[int],
) -> Dict:
    """Compute binned curve for reaction-center distance vs downstream loss."""
    # Filter valid entries (rc_distance >= 0)
    valid = [(d, l, lb) for d, l, lb in zip(rc_distances, downstream_losses, labels) if d >= 0]
    if len(valid) < 10:
        return {"status": "insufficient_data", "n_valid": len(valid)}

    distances = [v[0] for v in valid]
    losses = [v[1] for v in valid]
    labs = [v[2] for v in valid]

    # Bin by distance (0, 1, 2, 3, 4+)
    bins = {0: [], 1: [], 2: [], 3: [], 4: []}
    for d, l, lb in valid:
        b = min(int(d), 4)
        bins[b].append((l, lb))

    bin_stats = {}
    for b, items in sorted(bins.items()):
        if not items:
            continue
        bl = [i[0] for i in items]
        bn = [i[1] for i in items]
        pos_rate = sum(bn) / len(bn) if bn else 0
        bin_stats[f"bin_{b}"] = {
            "n": len(items),
            "mean_downstream_loss": sum(bl) / len(bl),
            "positive_rate": pos_rate,
        }

    # Correlation
    import math
    n = len(distances)
    mean_d = sum(distances) / n
    mean_l = sum(losses) / n
    cov = sum((d - mean_d) * (l - mean_l) for d, l in zip(distances, losses)) / n
    std_d = math.sqrt(sum((d - mean_d) ** 2 for d in distances) / n)
    std_l = math.sqrt(sum((l - mean_l) ** 2 for l in losses) / n)
    pearson_r = cov / (std_d * std_l) if std_d > 0 and std_l > 0 else 0

    # Curve shape: expect monotonic_decreasing (farther from RC = easier = lower loss)
    # or monotonic_increasing (farther from RC = harder to detect = higher loss)
    bin_means = [bin_stats[f"bin_{b}"]["mean_downstream_loss"] for b in range(5) if f"bin_{b}" in bin_stats]
    is_monotonic_incr = all(bin_means[i] <= bin_means[i + 1] for i in range(len(bin_means) - 1)) if len(bin_means) > 1 else False
    is_monotonic_decr = all(bin_means[i] >= bin_means[i + 1] for i in range(len(bin_means) - 1)) if len(bin_means) > 1 else False

    return {
        "n_valid": n,
        "n_total": len(rc_distances),
        "coverage": n / len(rc_distances) if rc_distances else 0,
        "pearson_r": pearson_r,
        "curve_shape": "monotonic_increasing" if is_monotonic_incr else ("monotonic_decreasing" if is_monotonic_decr else "non_monotonic"),
        "bins": bin_stats,
    }


def main():
    print("[rc_distance] Loading HTE reactions...")
    hte_rxns = load_hte_reactions()
    print(f"  Loaded {len(hte_rxns)} reactions")

    print("[rc_distance] Loading manifest...")
    group_to_source, cand_to_smiles, group_to_gold = load_manifest()
    print(f"  {len(group_to_source)} groups, {len(cand_to_smiles)} candidates")

    print("[rc_distance] Loading per_candidate_metrics.csv...")
    with open(METRICS_CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    print(f"  {len(rows)} rows")

    # Cache reaction center per group
    rc_cache: Dict[str, Set[int]] = {}
    rxn_cache: Dict[str, str] = {}

    results: List[Dict] = []
    rc_distances: List[float] = []
    downstream_losses: List[float] = []
    labels: List[int] = []

    for i, row in enumerate(rows):
        cid = row.get("candidate_id", "")
        if i % 200 == 0:
            print(f"  Processing {i}/{len(rows)}...")

        # Extract group_id from candidate_id
        # Format: hte_{hash}_{source}_{index}
        parts = cid.rsplit("_", 2)
        if len(parts) < 3:
            results.append({**row, "reaction_center_distance": "-1"})
            rc_distances.append(-1.0)
            downstream_losses.append(float(row.get("downstream_loss", 0)))
            labels.append(int(row.get("label", 0)))
            continue
        group_id = parts[0]

        # Get reaction SMILES
        if group_id not in rxn_cache:
            srid = group_to_source.get(group_id, "")
            rxn = hte_rxns.get(srid, "")
            rxn_cache[group_id] = rxn
        rxn = rxn_cache.get(group_id, "")

        if not rxn:
            results.append({**row, "reaction_center_distance": "-1"})
            rc_distances.append(-1.0)
            downstream_losses.append(float(row.get("downstream_loss", 0)))
            labels.append(int(row.get("label", 0)))
            continue

        # Get candidate and gold SMILES
        cand_smi = cand_to_smiles.get(cid, "")
        gold_smi = group_to_gold.get(group_id, "")

        if not cand_smi or not gold_smi:
            results.append({**row, "reaction_center_distance": "-1"})
            rc_distances.append(-1.0)
            downstream_losses.append(float(row.get("downstream_loss", 0)))
            labels.append(int(row.get("label", 0)))
            continue

        # Compute reaction-center distance
        try:
            d = map_locus_to_reaction_center_distance(cand_smi, gold_smi, rxn)
        except Exception as e:
            d = -1.0

        results.append({**row, "reaction_center_distance": str(d)})
        rc_distances.append(d)
        downstream_losses.append(float(row.get("downstream_loss", 0)))
        labels.append(int(row.get("label", 0)))

    # Write supplementary CSV
    print(f"[rc_distance] Writing {OUTPUT_CSV}...")
    if results:
        fieldnames = list(results[0].keys())
        with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

    # Compute curve
    print("[rc_distance] Computing curve...")
    curve = compute_curve(rc_distances, downstream_losses, labels)
    curve["metric"] = "reaction_center_distance"
    curve["description"] = "Shortest molecular graph path from PC-CNG edit locus to nearest reaction-center atom"

    with open(OUTPUT_CURVE, "w") as f:
        json.dump(curve, f, indent=2)

    print(f"[rc_distance] Done. Coverage: {curve.get('coverage', 0):.2%}")
    print(f"  Pearson r: {curve.get('pearson_r', 0):.4f}")
    print(f"  Curve shape: {curve.get('curve_shape', '?')}")


if __name__ == "__main__":
    main()
