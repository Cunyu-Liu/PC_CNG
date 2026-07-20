"""Counterfactual negative reaction generation for the PC-CNG MVP."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple

from .chem_utils import (
    canonicalize_reaction,
    is_valid_smiles,
    join_reaction,
    molecule_parts,
    replace_first,
    split_reaction,
)
from .validator import CounterfactualValidator


PRODUCT_REPLACEMENTS: Tuple[Tuple[str, str], ...] = (
    ("C(=O)N", "C(=O)O"),
    ("C(=O)O", "C(=O)N"),
    ("NC(=O)", "OC(=O)"),
    ("OC(=O)", "NC(=O)"),
    ("Cl", "Br"),
    ("Br", "Cl"),
    ("N", "O"),
    ("O", "N"),
)

REACTANT_REPLACEMENTS: Tuple[Tuple[str, str], ...] = (
    ("Cl", "Br"),
    ("Br", "Cl"),
    ("C(=O)Cl", "C(=O)O"),
    ("C(=O)O", "C(=O)Cl"),
    ("N", "O"),
    ("O", "N"),
)


@dataclass
class CounterfactualCandidate:
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
    label: int = 0
    provenance: str = "pc_cng_mvp_synthetic_counterfactual"

    def to_dict(self) -> Dict[str, str | float | bool | int]:
        return asdict(self)


class CounterfactualGenerator:
    """Generate forward and retrosynthesis-oriented counterfactual negatives."""

    def __init__(self, validator: Optional[CounterfactualValidator] = None):
        self.validator = validator or CounterfactualValidator()

    def generate_for_reaction(
        self,
        reaction_smiles: str,
        source_id: str,
        include_failed: bool = False,
    ) -> List[CounterfactualCandidate]:
        reactants, agents, product = split_reaction(reaction_smiles)
        positive = join_reaction(reactants, product, agents)

        raw: List[Tuple[str, str, str, str, str]] = []
        raw.extend(self._forward_outcome_edits(reactants, product, agents))
        raw.extend(self._retro_precursor_edits(reactants, product, agents))

        seen: Set[str] = set()
        candidates: List[CounterfactualCandidate] = []
        for task, failure_type, edit_action, cand_reactants, cand_product in raw:
            candidate_rxn = join_reaction(cand_reactants, cand_product, agents)
            canonical = canonicalize_reaction(candidate_rxn) or candidate_rxn
            if canonical in seen:
                continue
            seen.add(canonical)

            scores = self.validator.score(positive, candidate_rxn, failure_type, task)
            if not include_failed and not scores.passes_filter:
                continue
            candidates.append(
                CounterfactualCandidate(
                    source_id=source_id,
                    positive_reaction=positive,
                    candidate_reaction=candidate_rxn,
                    task=task,
                    failure_type=failure_type,
                    edit_action=edit_action,
                    parent_reactants=reactants,
                    parent_product=product,
                    candidate_reactants=cand_reactants,
                    candidate_product=cand_product,
                    **scores.to_dict(),
                )
            )
        return sorted(candidates, key=lambda item: item.hard_score, reverse=True)

    def _forward_outcome_edits(
        self, reactants: str, product: str, agents: str
    ) -> Iterable[Tuple[str, str, str, str, str]]:
        # Type 2 negative: starting materials remain unreacted.
        yield ("forward_outcome", "no_reaction", "product:=reactants", reactants, reactants)

        # Type 1 negative: plausible but wrong product through local edit.
        edited = replace_first(product, PRODUCT_REPLACEMENTS)
        if edited:
            cand_product, action = edited
            yield ("forward_outcome", "chemoselectivity_error", action, reactants, cand_product)

        # Under/over-reaction proxies are intentionally simple in the MVP.
        product_parts = molecule_parts(product)
        if len(product_parts) == 1 and is_valid_smiles(product + ".O"):
            yield ("forward_outcome", "side_product", "append:O", reactants, product + ".O")

    def _retro_precursor_edits(
        self, reactants: str, product: str, agents: str
    ) -> Iterable[Tuple[str, str, str, str, str]]:
        reactant_parts = molecule_parts(reactants)
        if not reactant_parts:
            return

        # Retrosynthesis hallucination proxy: no useful disconnection.
        yield ("retro_precursor", "retro_no_disconnection", "reactants:=product", product, product)

        if len(reactant_parts) > 1:
            missing_last = ".".join(reactant_parts[:-1])
            if missing_last:
                yield (
                    "retro_precursor",
                    "retro_missing_reactant",
                    "drop:last_reactant",
                    missing_last,
                    product,
                )

        # Near-miss precursor: alter one functional group in one precursor.
        for index, part in enumerate(reactant_parts):
            edited = replace_first(part, REACTANT_REPLACEMENTS)
            if not edited:
                continue
            edited_part, action = edited
            new_parts = list(reactant_parts)
            new_parts[index] = edited_part
            yield (
                "retro_precursor",
                "retro_wrong_functional_group",
                f"reactant[{index}].{action}",
                ".".join(new_parts),
                product,
            )
            break

