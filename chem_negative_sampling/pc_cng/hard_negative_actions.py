"""Hard-negative candidate actions for PC-CNG.

This module expands the candidate generation space beyond simple anchor
migration. It provides four action families requested by the current PC-CNG v3
plan:

- heteroatom: move a newly formed substituent to a competing hetero atom.
- regio: move a newly formed substituent to a nearby regioisomeric anchor.
- class_fallback: no-conversion / partial-conversion same-context products for low-support classes.
- partial_product: atom-map product fragments for class-specific partial reaction failures.
- unreacted_substrate: high-similarity reactant recovery for hydrogenation/Rh gaps.
- tautomer: use product tautomer alternatives as type-1 boundary candidates.
- low_yield_seed: import real low-yield / failed reactions as seed negatives.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, MutableMapping, Optional, Sequence, Set

from .chem_utils import (
    atom_balance_score,
    canonicalize_reaction,
    join_reaction,
    molecule_parts,
    split_reaction,
    string_similarity,
    token_jaccard,
)
from .reaction_boundary_generator import RXNMapperAdapter
from .reaction_center_edit_decoder import build_edit_candidate_groups

try:  # pragma: no cover - environment dependent
    from rdkit import Chem, RDLogger  # type: ignore
    from rdkit.Chem.MolStandardize import rdMolStandardize  # type: ignore

    RDLogger.DisableLog("rdApp.*")
except Exception:  # pragma: no cover
    Chem = None  # type: ignore
    rdMolStandardize = None  # type: ignore


HETERO_ANCHOR_COLUMNS = (
    "candidate_anchor_atom_N",
    "candidate_anchor_atom_O",
    "candidate_anchor_atom_S",
    "candidate_anchor_atom_P",
    "candidate_anchor_atom_F",
    "candidate_anchor_atom_Cl",
    "candidate_anchor_atom_Br",
    "candidate_anchor_atom_I",
)

MIGRATABLE_TERMINAL_ATOMS: Set[int] = {6, 7, 8, 9, 16, 17, 35, 53}
HETERO_ANCHOR_ATOMS: Set[int] = {7, 8, 15, 16}
REGIO_ANCHOR_ATOMS: Set[int] = {6, 7, 8, 15, 16}


@dataclass
class HardNegativeCandidate:
    source_id: str
    positive_reaction: str
    candidate_reaction: str
    task: str
    failure_type: str
    action_family: str
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
    provenance: str = "pc_cng_v3_hard_negative_actions"
    reaction_class: str = ""

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def is_known_positive(reaction_smiles: str, known_positives: Set[str]) -> bool:
    canonical = canonicalize_reaction(reaction_smiles)
    return bool(canonical and canonical in known_positives)


def _float(row: Dict[str, object], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default) or default)
    except (TypeError, ValueError):
        return default


def _candidate_from_row(row: Dict[str, object], action_family: str, score: Optional[float] = None) -> HardNegativeCandidate:
    product_similarity = _float(row, "product_similarity")
    atom_balance = _float(row, "atom_balance")
    hard_score = score
    if hard_score is None:
        distance = _float(row, "candidate_distance_to_true_anchor", 99.0)
        distance_score = 1.0 / (1.0 + max(0.0, distance - 1.0))
        same_atom = _float(row, "candidate_same_atomic_num_as_true")
        hard_score = 0.40 * product_similarity + 0.30 * atom_balance + 0.20 * distance_score + 0.10 * same_atom
    return HardNegativeCandidate(
        source_id=str(row["source_id"]),
        positive_reaction=str(row["positive_reaction"]),
        candidate_reaction=str(row["candidate_reaction"]),
        task="forward_outcome",
        failure_type=f"{action_family}_hard_negative",
        action_family=action_family,
        edit_action=str(row["edit_action"]),
        parent_reactants=str(row["reactants"]),
        parent_product=str(row["parent_product"]),
        candidate_reactants=str(row["reactants"]),
        candidate_product=str(row["candidate_product"]),
        valid=1.0,
        atom_balance=atom_balance,
        locality=product_similarity,
        closeness=product_similarity,
        hard_score=float(hard_score),
        false_negative_risk=max(0.0, min(1.0, max(0.0, product_similarity - 0.90) / 0.10)),
        passes_filter=True,
        mapped=True,
        center_maps=f"{row.get('fragment_map', '')};{row.get('true_anchor_map', '')};{row.get('candidate_anchor_map', '')}",
    )


def anchor_candidate_actions(
    reaction_smiles: str,
    source_id: str,
    split: str,
    label_type: str,
    action_families: Set[str],
    mapper: Optional[RXNMapperAdapter] = None,
    map_unmapped: bool = False,
    known_positives: Optional[Set[str]] = None,
    max_candidates_per_pair: int = 12,
    max_anchor_distance: int = 6,
    min_product_similarity: float = 0.65,
    max_product_similarity: float = 0.98,
    min_atom_balance: float = 0.55,
    diagnostics: Optional[MutableMapping[str, int]] = None,
) -> List[HardNegativeCandidate]:
    known_positives = known_positives or set()
    diagnostics = diagnostics if diagnostics is not None else {}

    def bump(key: str) -> None:
        diagnostics[key] = int(diagnostics.get(key, 0)) + 1

    groups, reason = build_edit_candidate_groups(
        reaction_smiles=reaction_smiles,
        source_id=source_id,
        split=split,
        label_type=label_type,
        mapper=mapper,
        map_unmapped=map_unmapped,
        max_candidates_per_pair=max_candidates_per_pair,
        max_anchor_distance=max_anchor_distance,
    )
    if reason != "ok":
        bump(f"anchor_group_reason:{reason}")
        return []

    out: List[HardNegativeCandidate] = []
    seen: Set[str] = set()
    for group in groups:
        for row in group.rows:
            bump("anchor_raw_rows")
            if int(row.get("is_true_anchor", 0) or 0) == 1:
                bump("skip_true_anchor")
                continue
            candidate_reaction = str(row.get("candidate_reaction", ""))
            if not candidate_reaction:
                bump("skip_empty_candidate_reaction")
                continue
            product_similarity = _float(row, "product_similarity")
            atom_balance = _float(row, "atom_balance")
            distance = _float(row, "candidate_distance_to_true_anchor", 99.0)
            is_hetero = any(_float(row, column) > 0.5 for column in HETERO_ANCHOR_COLUMNS)
            same_atom = _float(row, "candidate_same_atomic_num_as_true") > 0.5
            same_ring = _float(row, "candidate_same_ring_as_true") > 0.5
            family = None
            if "heteroatom" in action_families and is_hetero:
                family = "heteroatom"
            elif "regio" in action_families and (same_atom or same_ring or distance <= 4):
                family = "regio"
            if family is None:
                bump("skip_no_requested_anchor_family")
                continue
            bump(f"raw_family:{family}")

            if is_known_positive(candidate_reaction, known_positives):
                bump(f"skip_known_positive:{family}")
                continue
            if product_similarity < min_product_similarity:
                bump(f"skip_low_product_similarity:{family}")
                continue
            if product_similarity > max_product_similarity:
                bump(f"skip_high_product_similarity:{family}")
                continue
            if atom_balance < min_atom_balance:
                bump(f"skip_low_atom_balance:{family}")
                continue
            if distance > max_anchor_distance:
                bump(f"skip_distance:{family}")
                continue

            canonical = canonicalize_reaction(candidate_reaction) or candidate_reaction
            if canonical in seen:
                bump(f"skip_duplicate:{family}")
                continue
            seen.add(canonical)
            out.append(_candidate_from_row(row, family))
            bump(f"kept_family:{family}")
    return sorted(out, key=lambda item: item.hard_score, reverse=True)


def _same_ring(mol, left_idx: int, right_idx: int) -> bool:
    try:
        for ring in mol.GetRingInfo().AtomRings():
            if left_idx in ring and right_idx in ring:
                return True
    except Exception:
        return False
    return False


def _anchor_has_available_valence(atom) -> bool:
    atomic_num = atom.GetAtomicNum()
    if atom.GetFormalCharge() != 0:
        return False
    if atomic_num == 6:
        return atom.GetTotalNumHs() > 0
    if atomic_num in HETERO_ANCHOR_ATOMS:
        return atom.GetTotalNumHs() > 0 or atom.GetDegree() <= 2
    return False


def _terminal_substituent_pairs(mol) -> List[tuple[int, int]]:
    pairs: List[tuple[int, int]] = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() not in MIGRATABLE_TERMINAL_ATOMS:
            continue
        if atom.GetDegree() != 1:
            continue
        if atom.IsInRing() or atom.GetIsAromatic():
            continue
        neighbor = atom.GetNeighbors()[0]
        if not (neighbor.GetIsAromatic() or neighbor.IsInRing()):
            continue
        pairs.append((atom.GetIdx(), neighbor.GetIdx()))
    return pairs


def _move_terminal_substituent(mol, fragment_idx: int, old_anchor_idx: int, new_anchor_idx: int):
    if mol.GetBondBetweenAtoms(fragment_idx, old_anchor_idx) is None:
        return None
    if mol.GetBondBetweenAtoms(fragment_idx, new_anchor_idx) is not None:
        return None
    rw = Chem.RWMol(mol)
    rw.RemoveBond(fragment_idx, old_anchor_idx)
    rw.AddBond(fragment_idx, new_anchor_idx, Chem.BondType.SINGLE)
    try:
        candidate = rw.GetMol()
        Chem.SanitizeMol(candidate)
        return candidate
    except Exception:
        return None


def _replace_part(parts: Sequence[str], part_index: int, mol) -> str:
    new_parts = list(parts)
    new_parts[part_index] = Chem.MolToSmiles(mol, isomericSmiles=True)
    return ".".join(new_parts)


def diversity_anchor_actions(
    reaction_smiles: str,
    source_id: str,
    action_families: Set[str],
    known_positives: Optional[Set[str]] = None,
    max_candidates_per_reaction: int = 12,
    max_anchor_distance: int = 8,
    min_product_similarity: float = 0.45,
    max_product_similarity: float = 0.995,
    min_atom_balance: float = 0.35,
    diagnostics: Optional[MutableMapping[str, int]] = None,
) -> List[HardNegativeCandidate]:
    """Generate diversity-aware regio/heteroatom shifts without formed-bond maps.

    The earlier anchor generator depends on detecting the newly formed bond.
    Many useful products either fail that extraction or collapse to repeated
    candidates. This fallback searches product scaffolds directly: terminal
    substituents on aromatic/ring anchors are moved to same-ring or nearby
    chemically available anchors. Known-positive filtering remains mandatory.
    """

    known_positives = known_positives or set()
    diagnostics = diagnostics if diagnostics is not None else {}

    def bump(key: str) -> None:
        diagnostics[key] = int(diagnostics.get(key, 0)) + 1

    if Chem is None:
        bump("diverse_anchor_reason:rdkit_unavailable")
        return []
    try:
        reactants, agents, product = split_reaction(reaction_smiles)
    except ValueError:
        bump("diverse_anchor_reason:invalid_reaction")
        return []

    parts = molecule_parts(product)
    if not parts:
        bump("diverse_anchor_reason:empty_product")
        return []

    out: List[HardNegativeCandidate] = []
    seen: Set[str] = set()
    parent_reaction = join_reaction(reactants, product, agents)
    parent_canonical = canonicalize_reaction(parent_reaction) or parent_reaction
    for part_index, part in enumerate(parts):
        mol = Chem.MolFromSmiles(part)
        if mol is None:
            bump("diverse_anchor_reason:invalid_product_part")
            continue
        pairs = _terminal_substituent_pairs(mol)
        if not pairs:
            bump("diverse_anchor_reason:no_terminal_substituent")
            continue
        for fragment_idx, old_anchor_idx in pairs:
            fragment = mol.GetAtomWithIdx(fragment_idx)
            old_anchor = mol.GetAtomWithIdx(old_anchor_idx)
            for atom in mol.GetAtoms():
                bump("diverse_anchor_raw_rows")
                new_anchor_idx = atom.GetIdx()
                if new_anchor_idx in {fragment_idx, old_anchor_idx}:
                    bump("diverse_skip_self_or_old_anchor")
                    continue
                if atom.GetAtomicNum() not in REGIO_ANCHOR_ATOMS:
                    bump("diverse_skip_anchor_atom_type")
                    continue
                if not _anchor_has_available_valence(atom):
                    bump("diverse_skip_no_available_valence")
                    continue
                if mol.GetBondBetweenAtoms(fragment_idx, new_anchor_idx) is not None:
                    bump("diverse_skip_existing_fragment_bond")
                    continue
                try:
                    distance = len(Chem.GetShortestPath(mol, old_anchor_idx, new_anchor_idx)) - 1
                except Exception:
                    distance = 999
                same_ring = _same_ring(mol, old_anchor_idx, new_anchor_idx)
                if distance > max_anchor_distance and not same_ring:
                    bump("diverse_skip_distance")
                    continue

                family = None
                if "heteroatom" in action_families and atom.GetAtomicNum() in HETERO_ANCHOR_ATOMS:
                    family = "heteroatom"
                elif "regio" in action_families and (
                    atom.GetAtomicNum() == old_anchor.GetAtomicNum() or same_ring or distance <= 4
                ):
                    family = "regio"
                if family is None:
                    bump("diverse_skip_no_requested_family")
                    continue
                bump(f"diverse_raw_family:{family}")

                candidate_mol = _move_terminal_substituent(mol, fragment_idx, old_anchor_idx, new_anchor_idx)
                if candidate_mol is None:
                    bump(f"diverse_skip_sanitize:{family}")
                    continue
                candidate_product = _replace_part(parts, part_index, candidate_mol)
                candidate_reaction = join_reaction(reactants, candidate_product, agents)
                canonical = canonicalize_reaction(candidate_reaction) or candidate_reaction
                if canonical == parent_canonical:
                    bump(f"diverse_skip_parent_identity:{family}")
                    continue
                if canonical in seen:
                    bump(f"diverse_skip_local_duplicate:{family}")
                    continue
                if is_known_positive(candidate_reaction, known_positives):
                    bump(f"diverse_skip_known_positive:{family}")
                    continue

                similarity = 0.5 * token_jaccard(product, candidate_product) + 0.5 * string_similarity(
                    product, candidate_product
                )
                atom_balance = atom_balance_score(reactants, candidate_product)
                if similarity < min_product_similarity:
                    bump(f"diverse_skip_low_product_similarity:{family}")
                    continue
                if similarity > max_product_similarity:
                    bump(f"diverse_skip_high_product_similarity:{family}")
                    continue
                if atom_balance < min_atom_balance:
                    bump(f"diverse_skip_low_atom_balance:{family}")
                    continue
                seen.add(canonical)
                hard_score = 0.45 * similarity + 0.35 * atom_balance + 0.20 * (1.0 / (1.0 + max(0, distance - 1)))
                out.append(
                    HardNegativeCandidate(
                        source_id=source_id,
                        positive_reaction=parent_reaction,
                        candidate_reaction=candidate_reaction,
                        task="forward_outcome",
                        failure_type=f"{family}_diverse_anchor_hard_negative",
                        action_family=family,
                        edit_action=(
                            f"diverse_anchor_shift:{fragment.GetSymbol()}{fragment_idx}:"
                            f"{old_anchor.GetSymbol()}{old_anchor_idx}->{atom.GetSymbol()}{new_anchor_idx}"
                        ),
                        parent_reactants=reactants,
                        parent_product=product,
                        candidate_reactants=reactants,
                        candidate_product=candidate_product,
                        valid=1.0,
                        atom_balance=atom_balance,
                        locality=similarity,
                        closeness=similarity,
                        hard_score=float(hard_score),
                        false_negative_risk=max(0.0, min(1.0, max(0.0, similarity - 0.92) / 0.08)),
                        passes_filter=True,
                        mapped=":" in reaction_smiles,
                        center_maps=f"{fragment_idx};{old_anchor_idx};{new_anchor_idx}",
                        provenance="pc_cng_v3_diversity_anchor_actions",
                    )
                )
                bump(f"diverse_kept_family:{family}")
                if len(out) >= max_candidates_per_reaction:
                    return sorted(out, key=lambda item: item.hard_score, reverse=True)
    return sorted(out, key=lambda item: item.hard_score, reverse=True)[:max_candidates_per_reaction]


def tautomer_actions(
    reaction_smiles: str,
    source_id: str,
    known_positives: Optional[Set[str]] = None,
    max_tautomers: int = 8,
) -> List[HardNegativeCandidate]:
    known_positives = known_positives or set()
    if Chem is None or rdMolStandardize is None:
        return []
    try:
        reactants, agents, product = split_reaction(reaction_smiles)
    except ValueError:
        return []
    mol = Chem.MolFromSmiles(product)
    if mol is None:
        return []
    parent_product = Chem.MolToSmiles(mol, isomericSmiles=True)
    enumerator = rdMolStandardize.TautomerEnumerator()
    try:
        tautomers = list(enumerator.Enumerate(mol))
    except Exception:
        return []

    out: List[HardNegativeCandidate] = []
    seen: Set[str] = {canonicalize_reaction(join_reaction(reactants, parent_product, agents)) or ""}
    for idx, tautomer in enumerate(tautomers):
        if len(out) >= max_tautomers:
            break
        tautomer_smiles = Chem.MolToSmiles(tautomer, isomericSmiles=True)
        if tautomer_smiles == parent_product:
            continue
        candidate_reaction = join_reaction(reactants, tautomer_smiles, agents)
        canonical = canonicalize_reaction(candidate_reaction) or candidate_reaction
        if canonical in seen or is_known_positive(candidate_reaction, known_positives):
            continue
        seen.add(canonical)
        similarity = 0.5 * token_jaccard(parent_product, tautomer_smiles) + 0.5 * string_similarity(parent_product, tautomer_smiles)
        atom_balance = atom_balance_score(reactants, tautomer_smiles)
        if similarity < 0.60:
            continue
        out.append(
            HardNegativeCandidate(
                source_id=source_id,
                positive_reaction=join_reaction(reactants, parent_product, agents),
                candidate_reaction=candidate_reaction,
                task="forward_outcome",
                failure_type="tautomer_hard_negative",
                action_family="tautomer",
                edit_action=f"tautomer:{idx}",
                parent_reactants=reactants,
                parent_product=parent_product,
                candidate_reactants=reactants,
                candidate_product=tautomer_smiles,
                valid=1.0,
                atom_balance=atom_balance,
                locality=similarity,
                closeness=similarity,
                hard_score=0.5 * similarity + 0.5 * atom_balance,
                false_negative_risk=max(0.0, min(1.0, max(0.0, similarity - 0.92) / 0.08)),
                passes_filter=True,
                mapped=False,
                center_maps="",
            )
        )
    return sorted(out, key=lambda item: item.hard_score, reverse=True)


def _mapped_atom_set(smiles: str) -> Set[int]:
    if Chem is None:
        return set()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return set()
    return {int(atom.GetAtomMapNum()) for atom in mol.GetAtoms() if int(atom.GetAtomMapNum()) > 0}


def _fragment_smiles_from_maps(product_part: str, selected_maps: Set[int]) -> Optional[str]:
    if Chem is None or not selected_maps:
        return None
    mol = Chem.MolFromSmiles(product_part)
    if mol is None:
        return None
    atom_indices = [
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if int(atom.GetAtomMapNum()) > 0 and int(atom.GetAtomMapNum()) in selected_maps
    ]
    if not atom_indices or len(atom_indices) == mol.GetNumAtoms():
        return None
    try:
        smiles = Chem.MolFragmentToSmiles(
            mol,
            atomsToUse=atom_indices,
            isomericSmiles=True,
            canonical=True,
        )
    except Exception:
        return None
    if not smiles or Chem.MolFromSmiles(smiles) is None:
        return None
    return smiles


def partial_product_actions(
    reaction_smiles: str,
    source_id: str,
    known_positives: Optional[Set[str]] = None,
    max_candidates_per_reaction: int = 8,
    min_product_similarity: float = 0.0,
    max_product_similarity: float = 0.98,
    diagnostics: Optional[MutableMapping[str, int]] = None,
) -> List[HardNegativeCandidate]:
    """Generate atom-map partial-product fragments for weak reaction classes.

    Metal couplings, amide couplings, and hydrogenations often share the same
    unreacted starting-material fragments across many records. Product-side
    mapped fragments provide a more diverse partial-reaction stress test: they
    preserve the observed product context while dropping atoms contributed by
    one reactant partner.
    """

    known_positives = known_positives or set()
    diagnostics = diagnostics if diagnostics is not None else {}

    def bump(key: str) -> None:
        diagnostics[key] = int(diagnostics.get(key, 0)) + 1

    if Chem is None:
        bump("partial_product_reason:rdkit_unavailable")
        return []
    try:
        reactants, agents, product = split_reaction(reaction_smiles)
    except ValueError:
        bump("partial_product_reason:invalid_reaction")
        return []

    reactant_parts = molecule_parts(reactants)
    product_parts = molecule_parts(product)
    if not reactant_parts or not product_parts:
        bump("partial_product_reason:missing_side")
        return []

    reactant_map_sets = [maps for maps in (_mapped_atom_set(part) for part in reactant_parts) if maps]
    if len(reactant_map_sets) < 2:
        bump("partial_product_reason:insufficient_mapped_reactants")
        return []

    parent_reaction = join_reaction(reactants, product, agents)
    parent_canonical = canonicalize_reaction(parent_reaction) or parent_reaction
    candidates: List[HardNegativeCandidate] = []
    seen: Set[str] = set()

    for product_index, product_part in enumerate(product_parts):
        product_maps = _mapped_atom_set(product_part)
        if not product_maps:
            bump("partial_product_reason:unmapped_product_part")
            continue
        for reactant_index, reactant_maps in enumerate(reactant_map_sets):
            overlap = product_maps & reactant_maps
            if not overlap:
                bump("partial_product_skip:no_reactant_overlap")
                continue
            candidate_specs = [
                (f"product_fragment_from_reactant_{reactant_index}", overlap),
                (f"product_complement_without_reactant_{reactant_index}", product_maps - overlap),
            ]
            for edit_name, selected_maps in candidate_specs:
                bump("partial_product_raw_rows")
                fragment = _fragment_smiles_from_maps(product_part, selected_maps)
                if not fragment:
                    bump("partial_product_skip:empty_or_full_fragment")
                    continue
                candidate_reaction = join_reaction(reactants, fragment, agents)
                canonical = canonicalize_reaction(candidate_reaction) or candidate_reaction
                if canonical == parent_canonical:
                    bump("partial_product_skip_parent_identity")
                    continue
                if canonical in seen:
                    bump("partial_product_skip_local_duplicate")
                    continue
                if is_known_positive(candidate_reaction, known_positives):
                    bump("partial_product_skip_known_positive")
                    continue
                similarity = 0.5 * token_jaccard(product, fragment) + 0.5 * string_similarity(product, fragment)
                if similarity < min_product_similarity:
                    bump("partial_product_skip_low_similarity")
                    continue
                if similarity > max_product_similarity:
                    bump("partial_product_skip_high_similarity")
                    continue
                atom_balance = atom_balance_score(reactants, fragment)
                map_fraction = min(1.0, len(selected_maps) / max(1, len(product_maps)))
                hard_score = 0.50 * similarity + 0.30 * atom_balance + 0.20 * map_fraction
                seen.add(canonical)
                candidates.append(
                    HardNegativeCandidate(
                        source_id=source_id,
                        positive_reaction=parent_reaction,
                        candidate_reaction=candidate_reaction,
                        task="forward_outcome",
                        failure_type="partial_product_fragment_hard_negative",
                        action_family="partial_product",
                        edit_action=f"partial_product:{edit_name}:product_part_{product_index}",
                        parent_reactants=reactants,
                        parent_product=product,
                        candidate_reactants=reactants,
                        candidate_product=fragment,
                        valid=1.0,
                        atom_balance=atom_balance,
                        locality=similarity,
                        closeness=similarity,
                        hard_score=float(hard_score),
                        false_negative_risk=max(0.0, min(1.0, max(0.0, similarity - 0.93) / 0.07)),
                        passes_filter=True,
                        mapped=True,
                        center_maps=";".join(str(item) for item in sorted(selected_maps)),
                        provenance="pc_cng_v3_partial_product_actions",
                    )
                )
                bump("partial_product_kept")
                if len(candidates) >= max_candidates_per_reaction:
                    return sorted(candidates, key=lambda item: item.hard_score, reverse=True)
    return sorted(candidates, key=lambda item: item.hard_score, reverse=True)


def class_fallback_actions(
    reaction_smiles: str,
    source_id: str,
    known_positives: Optional[Set[str]] = None,
    max_candidates_per_reaction: int = 8,
    min_product_similarity: float = 0.05,
    max_product_similarity: float = 0.98,
    diagnostics: Optional[MutableMapping[str, int]] = None,
) -> List[HardNegativeCandidate]:
    """Generate no-conversion / partial-conversion candidates.

    Low-support classes such as hydrogenations and metal couplings often fail
    the anchor-shift generator because there is no simple terminal substituent
    to migrate. For these cases, unchanged reactant molecules are meaningful
    same-context negatives: the experimental outcome is the observed product,
    not an unreacted starting material.
    """

    known_positives = known_positives or set()
    diagnostics = diagnostics if diagnostics is not None else {}

    def bump(key: str) -> None:
        diagnostics[key] = int(diagnostics.get(key, 0)) + 1

    try:
        reactants, agents, product = split_reaction(reaction_smiles)
    except ValueError:
        bump("class_fallback_reason:invalid_reaction")
        return []
    reactant_parts = molecule_parts(reactants)
    if not reactant_parts or not product:
        bump("class_fallback_reason:missing_side")
        return []

    parent_reaction = join_reaction(reactants, product, agents)
    parent_canonical = canonicalize_reaction(parent_reaction) or parent_reaction
    candidates: List[HardNegativeCandidate] = []
    seen: Set[str] = set()
    # Prefer larger reactants first; they are usually closer no-conversion products.
    sorted_parts = sorted(set(reactant_parts), key=len, reverse=True)
    for idx, candidate_product in enumerate(sorted_parts):
        candidate_reaction = join_reaction(reactants, candidate_product, agents)
        canonical = canonicalize_reaction(candidate_reaction) or candidate_reaction
        if canonical == parent_canonical:
            bump("class_fallback_skip_parent_identity")
            continue
        if canonical in seen:
            bump("class_fallback_skip_duplicate")
            continue
        if is_known_positive(candidate_reaction, known_positives):
            bump("class_fallback_skip_known_positive")
            continue
        similarity = 0.5 * token_jaccard(product, candidate_product) + 0.5 * string_similarity(product, candidate_product)
        if similarity < min_product_similarity:
            bump("class_fallback_skip_low_similarity")
            continue
        if similarity > max_product_similarity:
            bump("class_fallback_skip_high_similarity")
            continue
        atom_balance = atom_balance_score(reactants, candidate_product)
        hard_score = 0.55 * similarity + 0.35 * atom_balance + 0.10 * (1.0 / (1.0 + idx))
        seen.add(canonical)
        candidates.append(
            HardNegativeCandidate(
                source_id=source_id,
                positive_reaction=parent_reaction,
                candidate_reaction=candidate_reaction,
                task="forward_outcome",
                failure_type="class_fallback_no_conversion_hard_negative",
                action_family="class_fallback",
                edit_action=f"class_fallback:no_conversion_reactant_{idx}",
                parent_reactants=reactants,
                parent_product=product,
                candidate_reactants=reactants,
                candidate_product=candidate_product,
                valid=1.0,
                atom_balance=atom_balance,
                locality=similarity,
                closeness=similarity,
                hard_score=float(hard_score),
                false_negative_risk=max(0.0, min(1.0, max(0.0, similarity - 0.95) / 0.05)),
                passes_filter=True,
                mapped=":" in reaction_smiles,
                center_maps="",
                provenance="pc_cng_v3_class_fallback_actions",
            )
        )
        bump("class_fallback_kept")
        if len(candidates) >= max_candidates_per_reaction:
            break
    return sorted(candidates, key=lambda item: item.hard_score, reverse=True)


def unreacted_substrate_actions(
    reaction_smiles: str,
    source_id: str,
    known_positives: Optional[Set[str]] = None,
    max_candidates_per_reaction: int = 8,
    min_product_similarity: float = 0.0,
    max_product_similarity: float = 0.9999,
    diagnostics: Optional[MutableMapping[str, int]] = None,
) -> List[HardNegativeCandidate]:
    """Keep unreduced/unclosed reactant-side substrates as hard negatives.

    The class-fallback action intentionally filters very high product
    similarities. Hydrogenation and Rh intramolecular reactions often need the
    opposite behavior: the most meaningful failed outcome is the reactant-side
    substrate, which differs from the observed product by only a few hydrogens
    or one ring-closing bond.
    """

    known_positives = known_positives or set()
    diagnostics = diagnostics if diagnostics is not None else {}

    def bump(key: str) -> None:
        diagnostics[key] = int(diagnostics.get(key, 0)) + 1

    try:
        reactants, agents, product = split_reaction(reaction_smiles)
    except ValueError:
        bump("unreacted_substrate_reason:invalid_reaction")
        return []
    reactant_parts = molecule_parts(reactants)
    if not reactant_parts or not product:
        bump("unreacted_substrate_reason:missing_side")
        return []

    parent_reaction = join_reaction(reactants, product, agents)
    parent_canonical = canonicalize_reaction(parent_reaction) or parent_reaction
    candidates: List[HardNegativeCandidate] = []
    seen: Set[str] = set()

    sorted_parts = sorted(set(reactant_parts), key=len, reverse=True)
    for idx, candidate_product in enumerate(sorted_parts):
        bump("unreacted_substrate_raw_rows")
        candidate_reaction = join_reaction(reactants, candidate_product, agents)
        canonical = canonicalize_reaction(candidate_reaction) or candidate_reaction
        if canonical == parent_canonical:
            bump("unreacted_substrate_skip_parent_identity")
            continue
        if canonical in seen:
            bump("unreacted_substrate_skip_duplicate")
            continue
        if is_known_positive(candidate_reaction, known_positives):
            bump("unreacted_substrate_skip_known_positive")
            continue
        similarity = 0.5 * token_jaccard(product, candidate_product) + 0.5 * string_similarity(product, candidate_product)
        if similarity < min_product_similarity:
            bump("unreacted_substrate_skip_low_similarity")
            continue
        if similarity > max_product_similarity:
            bump("unreacted_substrate_skip_high_similarity")
            continue
        atom_balance = atom_balance_score(reactants, candidate_product)
        # Keep this below the generic "very close positive" review threshold.
        hard_score = min(0.84, 0.60 * similarity + 0.30 * atom_balance + 0.10 * (1.0 / (1.0 + idx)))
        seen.add(canonical)
        candidates.append(
            HardNegativeCandidate(
                source_id=source_id,
                positive_reaction=parent_reaction,
                candidate_reaction=candidate_reaction,
                task="forward_outcome",
                failure_type="unreacted_substrate_hard_negative",
                action_family="unreacted_substrate",
                edit_action=f"unreacted_substrate:reactant_{idx}",
                parent_reactants=reactants,
                parent_product=product,
                candidate_reactants=reactants,
                candidate_product=candidate_product,
                valid=1.0,
                atom_balance=atom_balance,
                locality=similarity,
                closeness=similarity,
                hard_score=float(hard_score),
                false_negative_risk=0.0,
                passes_filter=True,
                mapped=":" in reaction_smiles,
                center_maps="",
                provenance="pc_cng_v3_unreacted_substrate_actions",
            )
        )
        bump("unreacted_substrate_kept")
        if len(candidates) >= max_candidates_per_reaction:
            break
    return sorted(candidates, key=lambda item: item.hard_score, reverse=True)


def low_yield_seed_action(row: Dict[str, str], yield_threshold: float = 5.0) -> Optional[HardNegativeCandidate]:
    label_type = row.get("label_type", "")
    raw_yield = row.get("yield", "")
    is_low_yield = label_type == "real_negative"
    try:
        if raw_yield:
            is_low_yield = is_low_yield or float(raw_yield.replace("%", "")) <= yield_threshold
    except ValueError:
        pass
    if not is_low_yield:
        return None
    reaction = row.get("reaction_smiles", "")
    if not reaction:
        return None
    try:
        reactants, agents, product = split_reaction(reaction)
    except ValueError:
        return None
    return HardNegativeCandidate(
        source_id=row.get("source_id", ""),
        positive_reaction="",
        candidate_reaction=reaction,
        task="forward_outcome",
        failure_type="low_yield_seed_hard_negative",
        action_family="low_yield_seed",
        edit_action=f"low_yield_seed:{raw_yield or label_type}",
        parent_reactants=reactants,
        parent_product="",
        candidate_reactants=reactants,
        candidate_product=product,
        valid=1.0,
        atom_balance=atom_balance_score(reactants, product),
        locality=0.0,
        closeness=0.0,
        hard_score=1.0,
        false_negative_risk=0.0,
        passes_filter=True,
        mapped=":" in reaction,
        center_maps="",
        provenance="pc_cng_v3_low_yield_seed",
    )


def output_fieldnames() -> List[str]:
    return list(HardNegativeCandidate.__dataclass_fields__.keys())
