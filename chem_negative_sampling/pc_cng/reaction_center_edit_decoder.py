"""Trainable reaction-center edit decoder utilities.

This module builds candidate reaction-center edits from atom-mapped reactions.
The decoder learns to rank the observed reaction-center anchor above plausible
alternative anchors. At generation time, high-scoring non-observed anchors are
used as type-1 boundary negatives.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .atom_mapped_graph_edit import extract_reaction_center, has_atom_mapping, reaction_center_signature
from .chem_utils import atom_balance_score, join_reaction, molecule_parts, split_reaction, string_similarity, token_jaccard
from .reaction_boundary_generator import RXNMapperAdapter

try:  # pragma: no cover - environment dependent
    from rdkit import Chem, RDLogger  # type: ignore

    RDLogger.DisableLog("rdApp.*")
except Exception:  # pragma: no cover
    Chem = None  # type: ignore


ATOM_VOCAB: Tuple[Tuple[str, int], ...] = (
    ("H", 1),
    ("C", 6),
    ("N", 7),
    ("O", 8),
    ("F", 9),
    ("P", 15),
    ("S", 16),
    ("Cl", 17),
    ("Br", 35),
    ("I", 53),
)

ANCHOR_ATOMIC_NUMS: Set[int] = {6, 7, 8, 15, 16, 17, 35, 53}


def _feature_names() -> List[str]:
    names: List[str] = []
    for prefix in ["fragment", "candidate_anchor", "true_anchor"]:
        for label, _ in ATOM_VOCAB:
            names.append(f"{prefix}_atom_{label}")
        names.extend(
            [
                f"{prefix}_atom_other",
                f"{prefix}_degree",
                f"{prefix}_formal_charge",
                f"{prefix}_total_h",
                f"{prefix}_is_aromatic",
                f"{prefix}_is_ring",
                f"{prefix}_mass_log",
            ]
        )
    names.extend(
        [
            "candidate_same_atomic_num_as_true",
            "candidate_same_aromatic_as_true",
            "candidate_same_ring_as_true",
            "candidate_distance_to_true_anchor",
            "candidate_distance_to_fragment",
            "product_similarity",
            "atom_balance",
            "num_formed_bonds",
            "num_broken_bonds",
            "num_changed_bonds",
            "product_num_atoms_log",
            "reactant_num_atoms_log",
        ]
    )
    return names


FEATURE_NAMES = _feature_names()


@dataclass
class EditCandidateGroup:
    source_id: str
    pair_id: str
    split: str
    label_type: str
    mapped_reaction: str
    reactants: str
    product: str
    fragment_map: int
    true_anchor_map: int
    center_signature: str
    rows: List[Dict[str, object]]


def _safe_log(value: float) -> float:
    import math

    return math.log1p(max(0.0, float(value)))


def _atom_feature_values(prefix: str, atom) -> Dict[str, float]:
    values: Dict[str, float] = {}
    atomic_num = atom.GetAtomicNum()
    matched = False
    for label, number in ATOM_VOCAB:
        key = f"{prefix}_atom_{label}"
        values[key] = 1.0 if atomic_num == number else 0.0
        matched = matched or atomic_num == number
    values[f"{prefix}_atom_other"] = 0.0 if matched else 1.0
    values[f"{prefix}_degree"] = float(atom.GetDegree())
    values[f"{prefix}_formal_charge"] = float(atom.GetFormalCharge())
    values[f"{prefix}_total_h"] = float(atom.GetTotalNumHs())
    values[f"{prefix}_is_aromatic"] = 1.0 if atom.GetIsAromatic() else 0.0
    values[f"{prefix}_is_ring"] = 1.0 if atom.IsInRing() else 0.0
    values[f"{prefix}_mass_log"] = _safe_log(atom.GetMass())
    return values


def _mol_num_atoms(smiles: str) -> int:
    if Chem is None:
        return 0
    total = 0
    for part in molecule_parts(smiles):
        mol = Chem.MolFromSmiles(part)
        if mol is not None:
            total += mol.GetNumAtoms()
    return total


def _map_to_idx(mol) -> Dict[int, int]:
    return {atom.GetAtomMapNum(): atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomMapNum()}


def _looks_like_transfer_fragment(atom, anchor_atom) -> bool:
    if atom.GetDegree() > 1:
        return False
    if atom.GetAtomicNum() not in {6, 7, 8, 16}:
        return False
    if atom.IsInRing() or atom.GetIsAromatic():
        return False
    return anchor_atom.GetIdx() in [neighbor.GetIdx() for neighbor in atom.GetNeighbors()]


def _candidate_anchor_atoms(mol, fragment_idx: int, true_anchor_idx: int, max_distance: int) -> List:
    candidates = []
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        if idx in {fragment_idx, true_anchor_idx}:
            continue
        if atom.GetAtomicNum() not in ANCHOR_ATOMIC_NUMS:
            continue
        if atom.GetFormalCharge() != 0:
            continue
        if mol.GetBondBetweenAtoms(fragment_idx, idx) is not None:
            continue
        try:
            distance = len(Chem.GetShortestPath(mol, true_anchor_idx, idx)) - 1
        except Exception:
            distance = 999
        if distance <= max_distance:
            candidates.append((distance, atom))
    candidates.sort(key=lambda item: (item[0], item[1].GetIdx()))
    return [atom for _, atom in candidates]


def move_formed_bond_in_product(product: str, fragment_map: int, true_anchor_map: int, candidate_anchor_map: int) -> Optional[str]:
    """Move a formed substituent bond from true anchor to candidate anchor."""
    if Chem is None:
        return None
    parts = molecule_parts(product)
    for part_index, part in enumerate(parts):
        mol = Chem.MolFromSmiles(part)
        if mol is None:
            continue
        mapping = _map_to_idx(mol)
        if not {fragment_map, true_anchor_map, candidate_anchor_map}.issubset(mapping):
            continue
        fragment_idx = mapping[fragment_map]
        true_anchor_idx = mapping[true_anchor_map]
        candidate_anchor_idx = mapping[candidate_anchor_map]
        if mol.GetBondBetweenAtoms(fragment_idx, true_anchor_idx) is None:
            continue
        if mol.GetBondBetweenAtoms(fragment_idx, candidate_anchor_idx) is not None:
            continue
        rw = Chem.RWMol(mol)
        rw.RemoveBond(fragment_idx, true_anchor_idx)
        rw.AddBond(fragment_idx, candidate_anchor_idx, Chem.BondType.SINGLE)
        try:
            new_mol = rw.GetMol()
            Chem.SanitizeMol(new_mol)
        except Exception:
            return None
        new_parts = list(parts)
        new_parts[part_index] = Chem.MolToSmiles(new_mol, isomericSmiles=True)
        return ".".join(new_parts)
    return None


def _build_feature_row(
    *,
    mol,
    reactants: str,
    parent_product: str,
    candidate_product: str,
    fragment_idx: int,
    candidate_anchor_idx: int,
    true_anchor_idx: int,
    num_formed: int,
    num_broken: int,
    num_changed: int,
) -> Dict[str, float]:
    fragment = mol.GetAtomWithIdx(fragment_idx)
    candidate_anchor = mol.GetAtomWithIdx(candidate_anchor_idx)
    true_anchor = mol.GetAtomWithIdx(true_anchor_idx)
    values: Dict[str, float] = {}
    values.update(_atom_feature_values("fragment", fragment))
    values.update(_atom_feature_values("candidate_anchor", candidate_anchor))
    values.update(_atom_feature_values("true_anchor", true_anchor))
    values["candidate_same_atomic_num_as_true"] = 1.0 if candidate_anchor.GetAtomicNum() == true_anchor.GetAtomicNum() else 0.0
    values["candidate_same_aromatic_as_true"] = 1.0 if candidate_anchor.GetIsAromatic() == true_anchor.GetIsAromatic() else 0.0
    values["candidate_same_ring_as_true"] = 1.0 if candidate_anchor.IsInRing() == true_anchor.IsInRing() else 0.0
    try:
        values["candidate_distance_to_true_anchor"] = float(len(Chem.GetShortestPath(mol, candidate_anchor_idx, true_anchor_idx)) - 1)
    except Exception:
        values["candidate_distance_to_true_anchor"] = 99.0
    try:
        values["candidate_distance_to_fragment"] = float(len(Chem.GetShortestPath(mol, candidate_anchor_idx, fragment_idx)) - 1)
    except Exception:
        values["candidate_distance_to_fragment"] = 99.0
    values["product_similarity"] = 0.5 * token_jaccard(parent_product, candidate_product) + 0.5 * string_similarity(
        parent_product, candidate_product
    )
    values["atom_balance"] = atom_balance_score(reactants, candidate_product)
    values["num_formed_bonds"] = float(num_formed)
    values["num_broken_bonds"] = float(num_broken)
    values["num_changed_bonds"] = float(num_changed)
    values["product_num_atoms_log"] = _safe_log(_mol_num_atoms(parent_product))
    values["reactant_num_atoms_log"] = _safe_log(_mol_num_atoms(reactants))
    return {name: float(values.get(name, 0.0)) for name in FEATURE_NAMES}


def _find_product_mol_with_maps(product: str, required_maps: Set[int]):
    for part in molecule_parts(product):
        mol = Chem.MolFromSmiles(part)
        if mol is None:
            continue
        mapping = _map_to_idx(mol)
        if required_maps.issubset(mapping):
            return mol
    return None


def build_edit_candidate_groups(
    reaction_smiles: str,
    source_id: str,
    split: str,
    label_type: str,
    mapper: Optional[RXNMapperAdapter] = None,
    map_unmapped: bool = False,
    max_candidates_per_pair: int = 8,
    max_anchor_distance: int = 6,
) -> Tuple[List[EditCandidateGroup], str]:
    """Build candidate anchor groups for trainable edit decoding."""
    if Chem is None:
        return [], "rdkit_unavailable"
    mapped_reaction = reaction_smiles
    if not has_atom_mapping(mapped_reaction):
        if not map_unmapped:
            return [], "missing_atom_mapping"
        mapper = mapper or RXNMapperAdapter()
        mapped = mapper.map_reaction(reaction_smiles)
        if not mapped or not has_atom_mapping(mapped):
            return [], "mapping_failed"
        mapped_reaction = mapped

    try:
        reactants, agents, product = split_reaction(mapped_reaction)
    except ValueError:
        return [], "invalid_reaction"
    edit = extract_reaction_center(mapped_reaction)
    if not edit.mapped or not edit.formed_bonds:
        return [], "no_formed_bond"
    signature = reaction_center_signature(edit)

    groups: List[EditCandidateGroup] = []
    for formed_index, (left_map, right_map, _order) in enumerate(edit.formed_bonds):
        mol = _find_product_mol_with_maps(product, {left_map, right_map})
        if mol is None:
            continue
        mapping = _map_to_idx(mol)
        left_idx = mapping[left_map]
        right_idx = mapping[right_map]
        left_atom = mol.GetAtomWithIdx(left_idx)
        right_atom = mol.GetAtomWithIdx(right_idx)
        if _looks_like_transfer_fragment(left_atom, right_atom):
            fragment_map, true_anchor_map = left_map, right_map
            fragment_idx, true_anchor_idx = left_idx, right_idx
        elif _looks_like_transfer_fragment(right_atom, left_atom):
            fragment_map, true_anchor_map = right_map, left_map
            fragment_idx, true_anchor_idx = right_idx, left_idx
        else:
            continue

        candidate_atoms = [mol.GetAtomWithIdx(true_anchor_idx)]
        candidate_atoms.extend(_candidate_anchor_atoms(mol, fragment_idx, true_anchor_idx, max_anchor_distance))
        rows: List[Dict[str, object]] = []
        seen_candidate_maps: Set[int] = set()
        for candidate_atom in candidate_atoms:
            candidate_anchor_map = int(candidate_atom.GetAtomMapNum())
            if not candidate_anchor_map or candidate_anchor_map in seen_candidate_maps:
                continue
            seen_candidate_maps.add(candidate_anchor_map)
            is_true = candidate_anchor_map == true_anchor_map
            if is_true:
                candidate_product = product
                edit_action = "observed_anchor"
            else:
                candidate_product = move_formed_bond_in_product(product, fragment_map, true_anchor_map, candidate_anchor_map)
                edit_action = f"move_anchor:{fragment_map}:{true_anchor_map}->{candidate_anchor_map}"
            if not candidate_product:
                continue
            features = _build_feature_row(
                mol=mol,
                reactants=reactants,
                parent_product=product,
                candidate_product=candidate_product,
                fragment_idx=fragment_idx,
                candidate_anchor_idx=candidate_atom.GetIdx(),
                true_anchor_idx=true_anchor_idx,
                num_formed=len(edit.formed_bonds),
                num_broken=len(edit.broken_bonds),
                num_changed=len(edit.changed_bonds),
            )
            pair_id = f"{source_id}|formed{formed_index}|{fragment_map}->{true_anchor_map}"
            candidate_reaction = join_reaction(reactants, candidate_product, agents)
            row: Dict[str, object] = {
                "source_id": source_id,
                "pair_id": pair_id,
                "split": split,
                "label_type": label_type,
                "mapped_reaction": mapped_reaction,
                "positive_reaction": join_reaction(reactants, product, agents),
                "candidate_reaction": candidate_reaction,
                "reactants": reactants,
                "parent_product": product,
                "candidate_product": candidate_product,
                "fragment_map": fragment_map,
                "true_anchor_map": true_anchor_map,
                "candidate_anchor_map": candidate_anchor_map,
                "is_true_anchor": 1 if is_true else 0,
                "candidate_role": "observed_positive" if is_true else "unannotated_candidate",
                "is_known_positive": 0,
                "is_hard_negative": 0,
                "hard_negative_weight": 0.0,
                "edit_action": edit_action,
                "center_signature": signature,
            }
            row.update(features)
            rows.append(row)
            if len(rows) >= max_candidates_per_pair + 1:
                break
        if any(int(row["is_true_anchor"]) == 1 for row in rows) and any(int(row["is_true_anchor"]) == 0 for row in rows):
            groups.append(
                EditCandidateGroup(
                    source_id=source_id,
                    pair_id=str(rows[0]["pair_id"]),
                    split=split,
                    label_type=label_type,
                    mapped_reaction=mapped_reaction,
                    reactants=reactants,
                    product=product,
                    fragment_map=fragment_map,
                    true_anchor_map=true_anchor_map,
                    center_signature=signature,
                    rows=rows,
                )
            )

    if not groups:
        return [], "no_candidate_anchor"
    return groups, "ok"


def candidate_fieldnames() -> List[str]:
    base = [
        "source_id",
        "pair_id",
        "split",
        "label_type",
        "mapped_reaction",
        "positive_reaction",
        "candidate_reaction",
        "reactants",
        "parent_product",
        "candidate_product",
        "fragment_map",
        "true_anchor_map",
        "candidate_anchor_map",
        "is_true_anchor",
        "candidate_role",
        "is_known_positive",
        "is_hard_negative",
        "hard_negative_weight",
        "edit_action",
        "center_signature",
    ]
    return base + FEATURE_NAMES
