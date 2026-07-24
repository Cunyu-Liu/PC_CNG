#!/usr/bin/env python3
"""Compute reaction-center distance for G8-A supplementary metric (v2).

Approach:
  1. Gold candidate SMILES has atom mapping matching the reaction product.
  2. Identify RC atoms via reaction SMILES (bond environment change).
  3. Map RC map numbers to gold candidate atom indices directly.
  4. Find edit locus: gold atoms NOT in MCS(gold, candidate).
  5. Compute shortest path in GOLD's molecular graph from edit locus to RC.

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
from typing import Dict, List, Set, Tuple

os.environ["RDKitRDLogger"] = "0"
from rdkit import Chem
from rdkit.Chem.rdFMCS import FindMCS

RESEARCH_DIR = Path("/home/cunyuliu/pc_cng_research")
MANIFEST_PATH = RESEARCH_DIR / "data/p4/manifests/hte_feasibility_v2.json"
HTE_CSV_PATH = RESEARCH_DIR / "data/processed/hitea_full_normalized.csv"
METRICS_CSV_PATH = RESEARCH_DIR / "results/p4_mechanism_curve/per_candidate_metrics.csv"
OUTPUT_CSV = RESEARCH_DIR / "results/p4_mechanism_curve/supplementary_rc_distance.csv"
OUTPUT_CURVE = RESEARCH_DIR / "results/p4_mechanism_curve/rc_distance_curve.json"


def load_hte_reactions() -> Dict[str, str]:
    out: Dict[str, str] = {}
    with open(HTE_CSV_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sid = row.get("source_id", "").strip()
            rxn = row.get("reaction_smiles", "").strip()
            if sid and rxn:
                out[sid] = rxn
    return out


def load_manifest():
    with open(MANIFEST_PATH) as f:
        m = json.load(f)
    group_to_source: Dict[str, str] = {}
    cand_to_smiles: Dict[str, str] = {}
    cand_to_group: Dict[str, str] = {}
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
                cand_to_group[cid] = gid
            if c.get("gold_candidate") and smi:
                group_to_gold[gid] = smi
    return group_to_source, cand_to_smiles, cand_to_group, group_to_gold


def identify_reaction_center(reaction_smiles: str) -> Set[int]:
    """RC atom map numbers = atoms whose bond environment changes."""
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
    rc_atoms: Set[int] = set()
    for mn in set(r_env.keys()) | set(p_env.keys()):
        if r_env.get(mn) != p_env.get(mn):
            rc_atoms.add(mn)
    return rc_atoms


def compute_rc_distance(
    candidate_smiles: str,
    gold_smiles: str,
    rc_mapnums: Set[int],
) -> float:
    """Compute shortest graph distance from edit locus to RC in gold product."""
    gold_mol = Chem.MolFromSmiles(gold_smiles)
    if gold_mol is None:
        return -1.0

    # Map RC map numbers to gold atom indices
    gold_rc_indices: Set[int] = set()
    for atom in gold_mol.GetAtoms():
        mn = atom.GetAtomMapNum()
        if mn in rc_mapnums:
            gold_rc_indices.add(atom.GetIdx())

    if not gold_rc_indices:
        return -1.0

    # Gold candidate: distance = 0
    cand_mol = Chem.MolFromSmiles(candidate_smiles)
    if cand_mol is None:
        return -1.0

    # Quick check: if canonical SMILES match, it's gold
    gold_nomap = Chem.RWMol(gold_mol)
    for a in gold_nomap.GetAtoms():
        a.SetAtomMapNum(0)
    cand_nomap = Chem.RWMol(cand_mol)
    for a in cand_nomap.GetAtoms():
        a.SetAtomMapNum(0)
    if Chem.MolToSmiles(gold_nomap) == Chem.MolToSmiles(cand_nomap):
        return 0.0

    # Find MCS between gold and candidate (strip atom mapping first)
    try:
        mcs = FindMCS([gold_nomap, cand_nomap], timeout=5, matchValences=False)
        if not mcs.smartsString or mcs.smartsString == "":
            return -1.0
        mcs_mol = Chem.MolFromSmarts(mcs.smartsString)
        if mcs_mol is None:
            return -1.0
        gold_match = gold_nomap.GetSubstructMatch(mcs_mol)
        if not gold_match:
            return -1.0
        matched_gold = set(gold_match)
        edit_locus = set(range(gold_nomap.GetNumAtoms())) - matched_gold
    except Exception:
        return -1.0

    if not edit_locus:
        # All gold atoms in MCS — edit is bond change or addition
        # Use candidate's extra atoms' nearest gold neighbor
        cand_match = cand_nomap.GetSubstructMatch(mcs_mol)
        if not cand_match:
            return 0.0
        edit_in_cand = set(range(cand_nomap.GetNumAtoms())) - set(cand_match)
        if not edit_in_cand:
            return 0.0
        # Map edit in candidate to nearest gold atom via bonds
        cand_to_gold = dict(zip(cand_match, gold_match))
        nearest_gold: Set[int] = set()
        for ea in edit_in_cand:
            for bond in cand_nomap.GetAtomWithIdx(ea).GetBonds():
                other = bond.GetOtherAtomIdx(ea)
                if other in cand_to_gold:
                    nearest_gold.add(cand_to_gold[other])
        if not nearest_gold:
            return 0.0
        edit_locus = nearest_gold

    # Compute distance matrix in gold product
    dist_matrix = Chem.GetDistanceMatrix(gold_mol)

    min_distances: List[float] = []
    for ea in edit_locus:
        if ea >= gold_mol.GetNumAtoms():
            continue
        for rc_idx in gold_rc_indices:
            if rc_idx >= gold_mol.GetNumAtoms():
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
    import math
    valid = [(d, l, lb) for d, l, lb in zip(rc_distances, downstream_losses, labels) if d >= 0]
    if len(valid) < 10:
        return {"status": "insufficient_data", "n_valid": len(valid)}

    distances = [v[0] for v in valid]
    losses = [v[1] for v in valid]

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
        bin_stats[f"bin_{b}"] = {
            "n": len(items),
            "mean_downstream_loss": sum(bl) / len(bl),
            "positive_rate": sum(bn) / len(bn) if bn else 0,
        }

    n = len(distances)
    mean_d = sum(distances) / n
    mean_l = sum(losses) / n
    cov = sum((d - mean_d) * (l - mean_l) for d, l in zip(distances, losses)) / n
    std_d = math.sqrt(sum((d - mean_d) ** 2 for d in distances) / n)
    std_l = math.sqrt(sum((l - mean_l) ** 2 for l in losses) / n)
    pearson_r = cov / (std_d * std_l) if std_d > 0 and std_l > 0 else 0

    bin_means = [bin_stats[f"bin_{b}"]["mean_downstream_loss"] for b in range(5) if f"bin_{b}" in bin_stats]
    is_incr = all(bin_means[i] <= bin_means[i + 1] for i in range(len(bin_means) - 1)) if len(bin_means) > 1 else False
    is_decr = all(bin_means[i] >= bin_means[i + 1] for i in range(len(bin_means) - 1)) if len(bin_means) > 1 else False

    return {
        "n_valid": n,
        "n_total": len(rc_distances),
        "coverage": n / len(rc_distances) if rc_distances else 0,
        "pearson_r": pearson_r,
        "curve_shape": "monotonic_increasing" if is_incr else ("monotonic_decreasing" if is_decr else "non_monotonic"),
        "bins": bin_stats,
    }


def main():
    print("[rc_distance] Loading HTE reactions...")
    hte_rxns = load_hte_reactions()
    print(f"  {len(hte_rxns)} reactions")

    print("[rc_distance] Loading manifest...")
    group_to_source, cand_to_smiles, cand_to_group, group_to_gold = load_manifest()
    print(f"  {len(group_to_source)} groups, {len(cand_to_smiles)} candidates")

    print("[rc_distance] Loading per_candidate_metrics.csv...")
    with open(METRICS_CSV_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"  {len(rows)} rows")

    # Pre-compute RC atoms per group
    rc_cache: Dict[str, Set[int]] = {}
    for gid, srid in group_to_source.items():
        rxn = hte_rxns.get(srid, "")
        if rxn:
            rc = identify_reaction_center(rxn)
            if rc:
                rc_cache[gid] = rc

    print(f"  RC identified for {len(rc_cache)}/{len(group_to_source)} groups")

    results: List[Dict] = []
    rc_distances: List[float] = []
    downstream_losses: List[float] = []
    labels: List[int] = []

    for i, row in enumerate(rows):
        cid = row.get("candidate_id", "")
        if i % 200 == 0:
            print(f"  Processing {i}/{len(rows)}...")

        group_id = cand_to_group.get(cid, "")
        if not group_id:
            results.append({**row, "reaction_center_distance": "-1"})
            rc_distances.append(-1.0)
            downstream_losses.append(float(row.get("downstream_loss", 0)))
            labels.append(int(row.get("label", 0)))
            continue

        rc_mapnums = rc_cache.get(group_id, set())
        if not rc_mapnums:
            results.append({**row, "reaction_center_distance": "-1"})
            rc_distances.append(-1.0)
            downstream_losses.append(float(row.get("downstream_loss", 0)))
            labels.append(int(row.get("label", 0)))
            continue

        cand_smi = cand_to_smiles.get(cid, "")
        gold_smi = group_to_gold.get(group_id, "")

        if not cand_smi or not gold_smi:
            results.append({**row, "reaction_center_distance": "-1"})
            rc_distances.append(-1.0)
            downstream_losses.append(float(row.get("downstream_loss", 0)))
            labels.append(int(row.get("label", 0)))
            continue

        try:
            d = compute_rc_distance(cand_smi, gold_smi, rc_mapnums)
        except Exception:
            d = -1.0

        results.append({**row, "reaction_center_distance": str(d)})
        rc_distances.append(d)
        downstream_losses.append(float(row.get("downstream_loss", 0)))
        labels.append(int(row.get("label", 0)))

    print(f"[rc_distance] Writing {OUTPUT_CSV}...")
    if results:
        fieldnames = list(results[0].keys())
        with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

    print("[rc_distance] Computing curve...")
    curve = compute_curve(rc_distances, downstream_losses, labels)
    curve["metric"] = "reaction_center_distance"
    curve["description"] = "Shortest molecular graph path from PC-CNG edit locus to nearest reaction-center atom"

    with open(OUTPUT_CURVE, "w") as f:
        json.dump(curve, f, indent=2)

    print(f"[rc_distance] Done. Coverage: {curve.get('coverage', 0):.2%}")
    print(f"  Pearson r: {curve.get('pearson_r', 0):.4f}")
    print(f"  Curve shape: {curve.get('curve_shape', '?')}")
    print(f"  Bins: {list(curve.get('bins', {}).keys())}")


if __name__ == "__main__":
    main()
