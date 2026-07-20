"""Validation and scoring for generated counterfactual reactions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict

from .chem_utils import (
    atom_balance_score,
    is_valid_reaction,
    split_reaction,
    string_similarity,
    token_jaccard,
)


@dataclass
class ValidationScores:
    valid: float
    atom_balance: float
    locality: float
    closeness: float
    hard_score: float
    false_negative_risk: float
    passes_filter: bool

    def to_dict(self) -> Dict[str, float | bool]:
        return asdict(self)


class CounterfactualValidator:
    """PhysChem-inspired lightweight validator.

    The MVP uses approximate, dependency-light scores. RDKit-backed validity is
    automatically used through chem_utils when RDKit is installed.
    """

    def __init__(self, min_closeness: float = 0.15, max_false_negative_risk: float = 0.92):
        self.min_closeness = min_closeness
        self.max_false_negative_risk = max_false_negative_risk

    def score(
        self,
        positive_reaction: str,
        candidate_reaction: str,
        failure_type: str,
        task: str,
    ) -> ValidationScores:
        valid = 1.0 if is_valid_reaction(candidate_reaction) else 0.0
        try:
            pos_reactants, _, pos_products = split_reaction(positive_reaction)
            cand_reactants, _, cand_products = split_reaction(candidate_reaction)
        except ValueError:
            return ValidationScores(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, False)

        atom_balance = atom_balance_score(cand_reactants, cand_products)

        if task == "retro_precursor":
            # Same target product, altered precursor set. Locality is measured
            # against the original precursor set to prefer near-miss routes.
            locality = 0.5 * token_jaccard(pos_reactants, cand_reactants) + 0.5 * string_similarity(
                pos_reactants, cand_reactants
            )
        else:
            # Same precursor set, altered outcome. Locality is measured against
            # the observed positive product to prefer plausible wrong outcomes.
            locality = 0.5 * token_jaccard(pos_products, cand_products) + 0.5 * string_similarity(
                pos_products, cand_products
            )

        no_reaction_bonus = 0.12 if failure_type in {"no_reaction", "retro_no_disconnection"} else 0.0
        closeness = max(0.0, min(1.0, locality + no_reaction_bonus))

        # Semi-hard target: neither random garbage nor identical to the positive.
        hard_score = valid * (0.55 * closeness + 0.25 * atom_balance + 0.20 * (1.0 - abs(closeness - 0.55)))
        hard_score = max(0.0, min(1.0, hard_score))

        # Close, atom-balanced alternatives are valuable but risky: they may be
        # unreported positives. Keep this explicit instead of hiding it.
        false_negative_risk = valid * (0.65 * max(0.0, closeness - 0.55) / 0.45 + 0.35 * atom_balance)
        if failure_type in {"no_reaction", "retro_missing_reactant", "retro_no_disconnection"}:
            false_negative_risk *= 0.65
        false_negative_risk = max(0.0, min(1.0, false_negative_risk))

        passes_filter = (
            bool(valid)
            and closeness >= self.min_closeness
            and false_negative_risk <= self.max_false_negative_risk
            and candidate_reaction != positive_reaction
        )

        return ValidationScores(
            valid=valid,
            atom_balance=atom_balance,
            locality=locality,
            closeness=closeness,
            hard_score=hard_score,
            false_negative_risk=false_negative_risk,
            passes_filter=passes_filter,
        )

