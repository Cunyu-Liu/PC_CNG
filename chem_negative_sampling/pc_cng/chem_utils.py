"""Lightweight chemistry utilities for the PC-CNG MVP.

RDKit is used when available. The fallback path is deliberately conservative
and only provides approximate token/count checks so the prototype can run in a
minimal Python environment.
"""

from __future__ import annotations

from collections import Counter
import re
from typing import Dict, Iterable, List, Optional, Tuple

try:  # pragma: no cover - depends on environment
    from rdkit import Chem  # type: ignore
except Exception:  # pragma: no cover - expected in the current lightweight env
    Chem = None  # type: ignore


ATOM_RE = re.compile(r"Cl|Br|Si|Na|Li|Mg|Ca|Al|[B-IK-Z][a-z]?|[cnops]")


def split_reaction(reaction_smiles: str) -> Tuple[str, str, str]:
    """Split reaction SMILES into reactants, agents, products."""
    rxn = reaction_smiles.strip()
    if ">>" in rxn:
        reactants, products = rxn.split(">>", 1)
        return reactants.strip(), "", products.strip()
    parts = rxn.split(">")
    if len(parts) == 3:
        return parts[0].strip(), parts[1].strip(), parts[2].strip()
    raise ValueError(f"Invalid reaction SMILES: {reaction_smiles}")


def join_reaction(reactants: str, products: str, agents: str = "") -> str:
    """Join reaction parts using the compact reactants>>products form."""
    reactants = reactants.strip()
    products = products.strip()
    agents = agents.strip()
    if agents:
        return f"{reactants}>{agents}>{products}"
    return f"{reactants}>>{products}"


def molecule_parts(smiles: str) -> List[str]:
    return [part.strip() for part in smiles.split(".") if part.strip()]


def atom_tokens(smiles: str) -> List[str]:
    """Return approximate atom tokens from a SMILES string."""
    tokens = []
    for token in ATOM_RE.findall(smiles):
        if token in {"c", "n", "o", "p", "s"}:
            tokens.append(token.upper())
        else:
            tokens.append(token)
    return tokens


def atom_counts(smiles: str) -> Counter:
    return Counter(atom_tokens(smiles))


def atom_count_distance(left: str, right: str) -> int:
    """L1 distance between approximate atom-count vectors."""
    left_counts = atom_counts(left)
    right_counts = atom_counts(right)
    keys = set(left_counts) | set(right_counts)
    return sum(abs(left_counts.get(key, 0) - right_counts.get(key, 0)) for key in keys)


def atom_balance_score(reactants: str, products: str) -> float:
    """Approximate atom-balance score in [0, 1]."""
    total = sum(atom_counts(reactants).values()) + sum(atom_counts(products).values())
    if total == 0:
        return 0.0
    distance = atom_count_distance(reactants, products)
    return max(0.0, 1.0 - distance / max(total, 1))


def token_jaccard(left: str, right: str) -> float:
    left_set = set(atom_tokens(left))
    right_set = set(atom_tokens(right))
    if not left_set and not right_set:
        return 1.0
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / len(left_set | right_set)


def levenshtein_distance(left: str, right: str) -> int:
    """Small stdlib Levenshtein implementation for string closeness."""
    if left == right:
        return 0
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        for j, right_char in enumerate(right, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + (left_char != right_char)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return previous[-1]


def string_similarity(left: str, right: str) -> float:
    denom = max(len(left), len(right), 1)
    return max(0.0, 1.0 - levenshtein_distance(left, right) / denom)


def canonicalize_smiles(smiles: str) -> Optional[str]:
    """Canonicalize a molecule or multi-molecule SMILES if RDKit is present."""
    parts = molecule_parts(smiles)
    if not parts:
        return None
    if Chem is None:
        return ".".join(parts)

    canonical_parts = []
    for part in parts:
        mol = Chem.MolFromSmiles(part)
        if mol is None:
            return None
        canonical_parts.append(Chem.MolToSmiles(mol))
    return ".".join(sorted(canonical_parts))


def is_valid_smiles(smiles: str) -> bool:
    if not smiles or smiles.count("(") != smiles.count(")"):
        return False
    if Chem is None:
        return bool(atom_tokens(smiles))
    return canonicalize_smiles(smiles) is not None


def is_valid_reaction(reaction_smiles: str) -> bool:
    try:
        reactants, _, products = split_reaction(reaction_smiles)
    except ValueError:
        return False
    return is_valid_smiles(reactants) and is_valid_smiles(products)


def canonicalize_reaction(reaction_smiles: str) -> Optional[str]:
    try:
        reactants, agents, products = split_reaction(reaction_smiles)
    except ValueError:
        return None
    can_reactants = canonicalize_smiles(reactants)
    can_products = canonicalize_smiles(products)
    if can_reactants is None or can_products is None:
        return None
    return join_reaction(can_reactants, can_products, agents)


def replace_first(smiles: str, replacements: Iterable[Tuple[str, str]]) -> Optional[Tuple[str, str]]:
    """Apply the first safe substring replacement and return (new, action)."""
    for old, new in replacements:
        if old in smiles and old != new:
            edited = smiles.replace(old, new, 1)
            if edited != smiles and is_valid_smiles(edited):
                return edited, f"replace:{old}->{new}"
    return None

