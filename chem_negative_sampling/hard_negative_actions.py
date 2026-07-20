"""Hard-negative candidate actions for PC-CNG.

This module expands the candidate generation space beyond simple anchor
migration. It provides four action families requested by the current PC-CNG v3
plan:

- heteroatom: move a newly formed substituent to a competing hetero atom.
- regio: move a newly formed substituent to a nearby regioisomeric anchor.
- tautomer: use product tautomer alternatives as type-1 boundary candidates.
- low_yield_seed: import real low-yield / failed reactions as seed negatives.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set

from .chem_utils import atom_balance_score, canonicalize_reaction, join_reaction, split_reaction, string_similarity, token_jaccard
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
) -> List[HardNegativeCandidate]:
    known_positives = known_positives or set()
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
        return []

    out: List[HardNegativeCandidate] = []
    seen: Set[str] = set()
    for group in groups:
        for row in group.rows:
            if int(row.get("is_true_anchor", 0) or 0) == 1:
                continue
            candidate_reaction = str(row.get("candidate_reaction", ""))
            if not candidate_reaction or is_known_positive(candidate_reaction, known_positives):
                continue
            product_similarity = _float(row, "product_similarity")
            atom_balance = _float(row, "atom_balance")
            distance = _float(row, "candidate_distance_to_true_anchor", 99.0)
            if not (0.65 <= product_similarity <= 0.98 and atom_balance >= 0.55 and distance <= max_anchor_distance):
                continue

            family = None
            is_hetero = any(_float(row, column) > 0.5 for column in HETERO_ANCHOR_COLUMNS)
            same_atom = _float(row, "candidate_same_atomic_num_as_true") > 0.5
            same_ring = _float(row, "candidate_same_ring_as_true") > 0.5
            if "heteroatom" in action_families and is_hetero:
                family = "heteroatom"
            elif "regio" in action_families and (same_atom or same_ring or distance <= 4):
                family = "regio"
            if family is None:
                continue

            canonical = canonicalize_reaction(candidate_reaction) or candidate_reaction
            if canonical in seen:
                continue
            seen.add(canonical)
            out.append(_candidate_from_row(row, family))
    return sorted(out, key=lambda item: item.hard_score, reverse=True)


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
