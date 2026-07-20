"""PC-CNG v2 reaction-center boundary counterfactual generator.

The MVP generator mostly used string edits. This module moves the generator
toward the Science Advances negative-data lesson: keep the same reaction
context and generate type-1 negatives, i.e. unexpected but chemically meaningful
alternative products near the reaction center.

The generator is intentionally conservative. It requires atom mapping by
default, extracts reaction-center atoms, and only mutates product atoms/bonds
near that center. RXNMapper is used opportunistically when an unmapped reaction
is provided and the package is installed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .atom_mapped_graph_edit import ReactionCenterEdit, extract_reaction_center, has_atom_mapping
from .chem_utils import (
    atom_balance_score,
    canonicalize_reaction,
    join_reaction,
    molecule_parts,
    split_reaction,
    string_similarity,
    token_jaccard,
)

try:  # pragma: no cover - environment dependent
    from rdkit import Chem, RDLogger  # type: ignore

    RDLogger.DisableLog("rdApp.*")
except Exception:  # pragma: no cover
    Chem = None  # type: ignore


ATOM_TRANSMUTATIONS: Tuple[Tuple[int, int, str], ...] = (
    (17, 35, "center_atom:Cl->Br"),
    (35, 17, "center_atom:Br->Cl"),
    (53, 35, "center_atom:I->Br"),
    (9, 17, "center_atom:F->Cl"),
    (7, 8, "center_atom:N->O"),
    (8, 7, "center_atom:O->N"),
    (8, 16, "center_atom:O->S"),
    (16, 8, "center_atom:S->O"),
)


BOND_ORDER_ALTERNATIVES = {
    "SINGLE": ("DOUBLE", "center_bond:SINGLE->DOUBLE"),
    "DOUBLE": ("SINGLE", "center_bond:DOUBLE->SINGLE"),
}


@dataclass
class BoundaryCandidate:
    source_id: str
    positive_reaction: str
    candidate_reaction: str
    task: str
    failure_type: str
    edit_action: str
    parent_reactants: str
    parent_product: str
    candidate_reactants: str
    candidate_product: str
    valid: float
    atom_balance: float
    locality: float
    closeness: float
    hard_score: float
    false_negative_risk: float
    passes_filter: bool
    mapped: bool
    center_maps: str
    label: int = 0
    provenance: str = "pc_cng_v2_reaction_boundary"

    def to_dict(self) -> Dict[str, str | float | bool | int]:
        return asdict(self)


class RXNMapperAdapter:
    """Lazy wrapper so the core module remains importable without RXNMapper."""

    def __init__(self) -> None:
        self._mapper = None
        self.available = True

    def map_reaction(self, reaction_smiles: str) -> Optional[str]:
        if has_atom_mapping(reaction_smiles):
            return reaction_smiles
        if not self.available:
            return None
        try:
            if self._mapper is None:
                from rxnmapper import RXNMapper  # type: ignore

                self._mapper = RXNMapper()
            result = self._mapper.get_attention_guided_atom_maps([reaction_smiles])[0]
            mapped = result.get("mapped_rxn")
            return str(mapped) if mapped else None
        except Exception:
            self.available = False
            return None


class ReactionBoundaryGenerator:
    """Generate type-1 boundary negatives by local product graph edits."""

    def __init__(
        self,
        min_product_similarity: float = 0.30,
        max_product_similarity: float = 0.97,
        max_false_negative_risk: float = 0.85,
        max_candidates_per_reaction: int = 4,
        allow_unmapped_fallback: bool = False,
        mapper: Optional[RXNMapperAdapter] = None,
    ) -> None:
        self.min_product_similarity = min_product_similarity
        self.max_product_similarity = max_product_similarity
        self.max_false_negative_risk = max_false_negative_risk
        self.max_candidates_per_reaction = max_candidates_per_reaction
        self.allow_unmapped_fallback = allow_unmapped_fallback
        self.mapper = mapper or RXNMapperAdapter()

    def generate_for_reaction(
        self,
        reaction_smiles: str,
        source_id: str,
        include_failed: bool = False,
    ) -> List[BoundaryCandidate]:
        if Chem is None:
            return []

        mapped_reaction = self.mapper.map_reaction(reaction_smiles) or reaction_smiles
        try:
            reactants, agents, product = split_reaction(mapped_reaction)
        except ValueError:
            return []

        positive = join_reaction(reactants, product, agents)
        edit = extract_reaction_center(mapped_reaction)
        center_maps = set(edit.reacting_atoms)
        if not center_maps and not self.allow_unmapped_fallback:
            return []

        raw_edits = self._product_center_edits(product, center_maps, edit)
        seen: Set[str] = set()
        candidates: List[BoundaryCandidate] = []
        for candidate_product, action in raw_edits:
            candidate_rxn = join_reaction(reactants, candidate_product, agents)
            canonical = canonicalize_reaction(candidate_rxn) or candidate_rxn
            if canonical in seen or canonical == (canonicalize_reaction(positive) or positive):
                continue
            seen.add(canonical)

            candidate = self._score_candidate(
                source_id=source_id,
                positive=positive,
                reactants=reactants,
                parent_product=product,
                candidate_product=candidate_product,
                edit_action=action,
                center_maps=center_maps,
                mapped=edit.mapped,
            )
            if include_failed or candidate.passes_filter:
                candidates.append(candidate)

        candidates.sort(key=lambda item: item.hard_score, reverse=True)
        return candidates[: self.max_candidates_per_reaction]

    def _product_center_edits(
        self,
        product: str,
        center_maps: Set[int],
        edit: ReactionCenterEdit,
    ) -> Iterable[Tuple[str, str]]:
        parts = molecule_parts(product)
        if not parts:
            return

        for part_index, part in enumerate(parts):
            mol = Chem.MolFromSmiles(part)
            if mol is None:
                continue
            atom_indices = self._center_atom_indices(mol, center_maps)
            if not atom_indices and not self.allow_unmapped_fallback:
                continue
            if not atom_indices and self.allow_unmapped_fallback:
                atom_indices = list(range(mol.GetNumAtoms()))

            for candidate_mol, action in self._formed_bond_migration_edits(mol, edit):
                yield self._replace_product_part(parts, part_index, candidate_mol), action

            for atom_idx in atom_indices:
                for candidate_mol, action in self._atom_transmutation_edits(mol, atom_idx):
                    yield self._replace_product_part(parts, part_index, candidate_mol), action

            for bond in mol.GetBonds():
                left = bond.GetBeginAtomIdx()
                right = bond.GetEndAtomIdx()
                if left not in atom_indices and right not in atom_indices:
                    continue
                for candidate_mol, action in self._bond_order_edits(mol, bond.GetIdx()):
                    yield self._replace_product_part(parts, part_index, candidate_mol), action

    def _center_atom_indices(self, mol, center_maps: Set[int]) -> List[int]:
        indices: List[int] = []
        for atom in mol.GetAtoms():
            atom_map = atom.GetAtomMapNum()
            if atom_map and atom_map in center_maps:
                indices.append(atom.GetIdx())
        return indices

    def _atom_transmutation_edits(self, mol, atom_idx: int):
        atom = mol.GetAtomWithIdx(atom_idx)
        atomic_num = atom.GetAtomicNum()
        for old_num, new_num, action in ATOM_TRANSMUTATIONS:
            if atomic_num != old_num:
                continue
            rw = Chem.RWMol(mol)
            rw_atom = rw.GetAtomWithIdx(atom_idx)
            rw_atom.SetAtomicNum(new_num)
            rw_atom.SetFormalCharge(0)
            candidate = self._sanitize(rw)
            if candidate is not None:
                yield candidate, action

    def _formed_bond_migration_edits(self, mol, edit: ReactionCenterEdit):
        """Move a newly formed substituent bond to a nearby competing atom.

        This targets the Science Advances type-1 negative pattern: same
        reaction context, but a regio/chemoselective alternative product.
        """
        if not edit.formed_bonds:
            return
        map_to_idx = {
            atom.GetAtomMapNum(): atom.GetIdx()
            for atom in mol.GetAtoms()
            if atom.GetAtomMapNum()
        }
        for left_map, right_map, _order in edit.formed_bonds:
            if left_map not in map_to_idx or right_map not in map_to_idx:
                continue
            left_idx = map_to_idx[left_map]
            right_idx = map_to_idx[right_map]
            left_atom = mol.GetAtomWithIdx(left_idx)
            right_atom = mol.GetAtomWithIdx(right_idx)

            # In alkylation/acylation-like edits, the newly attached fragment is
            # often a low-degree carbon/hetero atom. We move that fragment to a
            # chemically competing anchor atom in the same product graph.
            if self._looks_like_transfer_fragment(left_atom, right_atom):
                fragment_idx, old_anchor_idx = left_idx, right_idx
                fragment_map, old_anchor_map = left_map, right_map
            elif self._looks_like_transfer_fragment(right_atom, left_atom):
                fragment_idx, old_anchor_idx = right_idx, left_idx
                fragment_map, old_anchor_map = right_map, left_map
            else:
                continue

            for new_anchor in self._candidate_anchor_atoms(mol, fragment_idx, old_anchor_idx):
                new_anchor_idx = new_anchor.GetIdx()
                if mol.GetBondBetweenAtoms(fragment_idx, new_anchor_idx) is not None:
                    continue
                rw = Chem.RWMol(mol)
                old_bond = rw.GetBondBetweenAtoms(fragment_idx, old_anchor_idx)
                if old_bond is None:
                    continue
                rw.RemoveBond(fragment_idx, old_anchor_idx)
                rw.AddBond(fragment_idx, new_anchor_idx, Chem.BondType.SINGLE)
                candidate = self._sanitize(rw)
                if candidate is None:
                    continue
                new_anchor_map = new_anchor.GetAtomMapNum()
                action = f"formed_bond_migrate:{fragment_map}:{old_anchor_map}->{new_anchor_map}"
                yield candidate, action

    def _looks_like_transfer_fragment(self, atom, anchor_atom) -> bool:
        if atom.GetDegree() > 1:
            return False
        if atom.GetAtomicNum() not in {6, 7, 8, 16}:
            return False
        # Avoid moving ring atoms or aromatic atoms as "substituents".
        if atom.IsInRing() or atom.GetIsAromatic():
            return False
        return anchor_atom.GetIdx() in [neighbor.GetIdx() for neighbor in atom.GetNeighbors()]

    def _candidate_anchor_atoms(self, mol, fragment_idx: int, old_anchor_idx: int):
        old_anchor = mol.GetAtomWithIdx(old_anchor_idx)
        candidates = []
        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            if idx in {fragment_idx, old_anchor_idx}:
                continue
            if atom.GetAtomicNum() not in {7, 8, 16}:
                continue
            if atom.GetFormalCharge() != 0:
                continue
            try:
                distance = len(Chem.GetShortestPath(mol, old_anchor_idx, idx)) - 1
            except Exception:
                distance = 999
            if distance <= 5:
                candidates.append((distance, atom))
        candidates.sort(key=lambda item: item[0])
        return [atom for _, atom in candidates[:8]]

    def _bond_order_edits(self, mol, bond_idx: int):
        bond = mol.GetBondWithIdx(bond_idx)
        current = str(bond.GetBondType())
        if current not in BOND_ORDER_ALTERNATIVES:
            return
        target_name, action = BOND_ORDER_ALTERNATIVES[current]
        target = getattr(Chem.BondType, target_name)
        rw = Chem.RWMol(mol)
        rw.GetBondWithIdx(bond_idx).SetBondType(target)
        candidate = self._sanitize(rw)
        if candidate is not None:
            yield candidate, action

    def _sanitize(self, rw_mol):
        try:
            mol = rw_mol.GetMol()
            Chem.SanitizeMol(mol)
            return mol
        except Exception:
            return None

    def _replace_product_part(self, parts: Sequence[str], index: int, mol) -> str:
        new_parts = list(parts)
        new_parts[index] = Chem.MolToSmiles(mol, isomericSmiles=True)
        return ".".join(new_parts)

    def _score_candidate(
        self,
        source_id: str,
        positive: str,
        reactants: str,
        parent_product: str,
        candidate_product: str,
        edit_action: str,
        center_maps: Set[int],
        mapped: bool,
    ) -> BoundaryCandidate:
        candidate_rxn = join_reaction(reactants, candidate_product)
        atom_balance = atom_balance_score(reactants, candidate_product)
        product_similarity = 0.5 * token_jaccard(parent_product, candidate_product) + 0.5 * string_similarity(
            parent_product, candidate_product
        )
        valid = 1.0 if Chem.MolFromSmiles(candidate_product) is not None else 0.0
        closeness = max(0.0, min(1.0, product_similarity))

        # Type-1 target: close enough to be an informative alternative product,
        # but not so close that it is probably the same/known positive.
        boundary_band = 1.0 - abs(closeness - 0.62) / 0.62
        hard_score = valid * max(0.0, min(1.0, 0.45 * closeness + 0.30 * atom_balance + 0.25 * boundary_band))
        # Boundary negatives are supposed to be close and often atom-balanced.
        # Treat only near-identical alternatives as high-risk by default.
        false_negative_risk = valid * max(0.0, min(1.0, max(0.0, closeness - 0.88) / 0.12))
        if atom_balance > 0.98 and closeness > 0.92:
            false_negative_risk = min(1.0, false_negative_risk + 0.15)
        passes_filter = (
            bool(valid)
            and self.min_product_similarity <= product_similarity <= self.max_product_similarity
            and false_negative_risk <= self.max_false_negative_risk
            and candidate_rxn != positive
        )

        return BoundaryCandidate(
            source_id=source_id,
            positive_reaction=positive,
            candidate_reaction=candidate_rxn,
            task="forward_outcome",
            failure_type="reaction_center_alternative",
            edit_action=edit_action,
            parent_reactants=reactants,
            parent_product=parent_product,
            candidate_reactants=reactants,
            candidate_product=candidate_product,
            valid=valid,
            atom_balance=atom_balance,
            locality=product_similarity,
            closeness=closeness,
            hard_score=hard_score,
            false_negative_risk=false_negative_risk,
            passes_filter=passes_filter,
            mapped=mapped,
            center_maps=";".join(str(value) for value in sorted(center_maps)),
        )
