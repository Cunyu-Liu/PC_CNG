"""Atom-mapped reaction-center graph edit utilities.

This module is the bridge from the rule MVP to the publishable model. It can
extract reaction-center bond edits from atom-mapped reactions when map numbers
are present. If mapping is absent, it returns explicit reasons instead of
silently pretending that a graph edit was inferred.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Dict, Iterable, List, Optional, Set, Tuple

from .chem_utils import split_reaction

try:  # pragma: no cover - environment dependent
    from rdkit import Chem  # type: ignore
except Exception:  # pragma: no cover
    Chem = None  # type: ignore


MAP_RE = re.compile(r":(\d+)\]")


Bond = Tuple[int, int, str]


@dataclass
class ReactionCenterEdit:
    formed_bonds: List[Bond]
    broken_bonds: List[Bond]
    changed_bonds: List[Tuple[int, int, str, str]]
    reacting_atoms: List[int]
    mapped: bool
    reason: str = ""

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def has_atom_mapping(smiles: str) -> bool:
    return bool(MAP_RE.search(smiles))


def _fallback_mapped_bonds(smiles: str) -> Set[Bond]:
    """Very conservative fallback for simple atom-mapped linear strings."""
    bonds: Set[Bond] = set()
    maps = [int(value) for value in MAP_RE.findall(smiles)]
    for left, right in zip(maps, maps[1:]):
        a, b = sorted((left, right))
        bonds.add((a, b, "unknown"))
    return bonds


def _rdkit_mapped_bonds(smiles: str) -> Optional[Set[Bond]]:
    if Chem is None:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    bonds: Set[Bond] = set()
    for bond in mol.GetBonds():
        left = bond.GetBeginAtom().GetAtomMapNum()
        right = bond.GetEndAtom().GetAtomMapNum()
        if not left or not right:
            continue
        a, b = sorted((int(left), int(right)))
        bonds.add((a, b, str(bond.GetBondType())))
    return bonds


def mapped_bonds(smiles: str) -> Set[Bond]:
    rdkit_bonds = _rdkit_mapped_bonds(smiles)
    if rdkit_bonds is not None:
        return rdkit_bonds
    return _fallback_mapped_bonds(smiles)


def strip_bond_order(bonds: Iterable[Bond]) -> Dict[Tuple[int, int], str]:
    return {(left, right): order for left, right, order in bonds}


def extract_reaction_center(reaction_smiles: str) -> ReactionCenterEdit:
    reactants, _, products = split_reaction(reaction_smiles)
    if not has_atom_mapping(reactants) or not has_atom_mapping(products):
        return ReactionCenterEdit([], [], [], [], mapped=False, reason="missing_atom_mapping")

    reactant_bonds = strip_bond_order(mapped_bonds(reactants))
    product_bonds = strip_bond_order(mapped_bonds(products))
    reactant_pairs = set(reactant_bonds)
    product_pairs = set(product_bonds)

    formed_pairs = product_pairs - reactant_pairs
    broken_pairs = reactant_pairs - product_pairs
    common_pairs = product_pairs & reactant_pairs

    formed = [(a, b, product_bonds[(a, b)]) for a, b in sorted(formed_pairs)]
    broken = [(a, b, reactant_bonds[(a, b)]) for a, b in sorted(broken_pairs)]
    changed = [
        (a, b, reactant_bonds[(a, b)], product_bonds[(a, b)])
        for a, b in sorted(common_pairs)
        if reactant_bonds[(a, b)] != product_bonds[(a, b)]
    ]
    reacting_atoms = sorted({atom for bond in formed + broken for atom in bond[:2]} | {a for a, _, _, _ in changed} | {b for _, b, _, _ in changed})

    return ReactionCenterEdit(
        formed_bonds=formed,
        broken_bonds=broken,
        changed_bonds=changed,
        reacting_atoms=reacting_atoms,
        mapped=True,
        reason="ok",
    )


def reaction_center_signature(edit: ReactionCenterEdit) -> str:
    if not edit.mapped:
        return edit.reason
    return (
        f"F:{';'.join(f'{a}-{b}:{o}' for a, b, o in edit.formed_bonds)}|"
        f"B:{';'.join(f'{a}-{b}:{o}' for a, b, o in edit.broken_bonds)}|"
        f"C:{';'.join(f'{a}-{b}:{o1}>{o2}' for a, b, o1, o2 in edit.changed_bonds)}"
    )


def extract_edit_vocabulary(reactions: Iterable[str], min_count: int = 1) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for reaction in reactions:
        try:
            signature = reaction_center_signature(extract_reaction_center(reaction))
        except Exception:
            continue
        if signature in {"missing_atom_mapping", ""}:
            continue
        counts[signature] = counts.get(signature, 0) + 1
    return {signature: count for signature, count in counts.items() if count >= min_count}

