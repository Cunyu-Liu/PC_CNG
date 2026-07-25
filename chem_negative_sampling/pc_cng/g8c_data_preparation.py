"""G8-C learned structured proposal — real data preparation.

This module replaces the heuristic supervision used by the original
``p4_g8c_learned_structured_proposal.py`` training driver with REAL targets
extracted from atom-mapped HTE reactions and REAL rule-generator proposals.

It provides:
    * ``extract_real_edit_targets``        - real bond break/form/change
    * ``extract_rule_proposals``           - actual ReactionBoundaryGenerator output
    * ``build_competing_outcome_pairs``    - real HTE competing-outcome pairs
    * ``build_preference_pairs``           - real DPO preference pairs
    * ``load_g8c_training_data``           - end-to-end data assembly

All functions degrade gracefully when a reaction is unmapped or when the rule
generator returns no candidates: they fall back to ``NO_EDIT`` at locus 0 so
that the training loop never crashes on a single bad record.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .atom_mapped_graph_edit import (
    ReactionCenterEdit,
    extract_reaction_center,
    has_atom_mapping,
)
from .chem_utils import join_reaction, molecule_parts, split_reaction
from .reaction_boundary_generator import (
    BoundaryCandidate,
    ReactionBoundaryGenerator,
)

# The EditType enum lives in the proposal module; import lazily to avoid a
# circular import (the proposal module imports data-preparation-free helpers).
try:  # pragma: no cover - import-time guard
    from .p4_g8c_learned_structured_proposal import EditType
    _EDIT_TYPE_RESOLVED = True
except Exception:  # pragma: no cover
    # Fallback integer constants matching the IntEnum in the proposal module.
    class EditType:  # type: ignore
        ATOM_TRANSMUTATION = 0
        BOND_ORDER_CHANGE = 1
        FORMED_BOND_MIGRATE = 2
        NO_EDIT = 3

    _EDIT_TYPE_RESOLVED = False


logger = logging.getLogger(__name__)

# Default HTE parquet location (relative to the pc_cng_research project root).
DEFAULT_HTE_PARQUET = os.environ.get(
    "G8C_HTE_PARQUET",
    "/home/cunyuliu/pc_cng_research/data/processed/p4_hte_normalized.parquet",
)


# ---------------------------------------------------------------------------
# 1. Real edit targets from atom-mapped reactions
# ---------------------------------------------------------------------------

def _edit_type_from_center(center: ReactionCenterEdit) -> int:
    """Map a :class:`ReactionCenterEdit` to an :class:`EditType` value.

    Priority: formed_bonds -> FORMED_BOND_MIGRATE, broken_bonds ->
    BOND_ORDER_CHANGE, changed_bonds -> BOND_ORDER_CHANGE, else NO_EDIT.
    """
    if not center or not center.mapped:
        return int(EditType.NO_EDIT)
    if center.formed_bonds:
        return int(EditType.FORMED_BOND_MIGRATE)
    if center.broken_bonds:
        return int(EditType.BOND_ORDER_CHANGE)
    if center.changed_bonds:
        return int(EditType.BOND_ORDER_CHANGE)
    return int(EditType.NO_EDIT)


def _first_locus_from_center(center: ReactionCenterEdit) -> int:
    """Return the first reacting atom map number (locus), or 0 if absent."""
    if center and center.mapped and center.reacting_atoms:
        try:
            return int(center.reacting_atoms[0])
        except (TypeError, ValueError):
            return 0
    # Fall back to the first atom of the first formed/broken/changed bond.
    if center:
        for bond_list in (center.formed_bonds, center.broken_bonds):
            if bond_list:
                try:
                    return int(bond_list[0][0])
                except (TypeError, ValueError):
                    return 0
        if center.changed_bonds:
            try:
                return int(center.changed_bonds[0][0])
            except (TypeError, ValueError):
                return 0
    return 0


def extract_real_edit_targets(reaction_smiles: str) -> Dict[str, Any]:
    """Extract REAL edit targets (locus, edit_type) from an atom-mapped reaction.

    Uses :func:`extract_reaction_center` to obtain the true formed / broken /
    changed bond sets.  Returns a dict suitable for direct use as Stage-1
    supervision.  When the reaction is unmapped or unparseable, returns
    ``NO_EDIT`` at locus 0 so the caller can still batch the example.
    """
    try:
        center = extract_reaction_center(reaction_smiles)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("extract_reaction_center failed for %r: %s", reaction_smiles, exc)
        center = ReactionCenterEdit(
            formed_bonds=[], broken_bonds=[], changed_bonds=[],
            reacting_atoms=[], mapped=False, reason="exception",
        )

    return {
        "locus": _first_locus_from_center(center),
        "edit_type": _edit_type_from_center(center),
        "formed_bonds": list(getattr(center, "formed_bonds", []) or []),
        "broken_bonds": list(getattr(center, "broken_bonds", []) or []),
        "changed_bonds": list(getattr(center, "changed_bonds", []) or []),
        "reacting_atoms": list(getattr(center, "reacting_atoms", []) or []),
        "mapped": bool(getattr(center, "mapped", False)),
    }


# ---------------------------------------------------------------------------
# 2. Real rule-generator proposals
# ---------------------------------------------------------------------------

def _parse_edit_action(edit_action: str, center: ReactionCenterEdit) -> Tuple[int, int]:
    """Parse a rule-generator ``edit_action`` string into (locus, edit_type).

    Recognised prefixes (see ``reaction_boundary_generator.py``):
        * ``formed_bond_migrate:<frag>:<old>-><new>`` -> FORMED_BOND_MIGRATE
        * ``center_atom:<old>-><new>``                -> ATOM_TRANSMUTATION
        * ``center_bond:<old>-><new>``                -> BOND_ORDER_CHANGE
    """
    action = (edit_action or "").strip()
    locus = _first_locus_from_center(center)
    if action.startswith("formed_bond_migrate"):
        # parse the fragment map number if present
        parts = action.split(":")
        if len(parts) >= 2:
            try:
                locus = int(parts[1])
            except ValueError:
                pass
        return locus, int(EditType.FORMED_BOND_MIGRATE)
    if action.startswith("center_atom"):
        return locus, int(EditType.ATOM_TRANSMUTATION)
    if action.startswith("center_bond"):
        return locus, int(EditType.BOND_ORDER_CHANGE)
    # Unknown action string — keep the reaction-center locus and NO_EDIT.
    return locus, int(EditType.NO_EDIT)


def extract_rule_proposals(
    reaction_smiles: str,
    generator: ReactionBoundaryGenerator,
    source_id: str = "",
) -> List[Dict[str, Any]]:
    """Run the rule generator and return its ACTUAL proposals as plain dicts.

    Each dict mirrors the fields needed by Stage-2 imitation training:
    ``locus``, ``edit_type``, ``edit_action``, ``candidate_product``,
    ``hard_score``.  Returns an empty list when the generator produces no
    candidates (unmapped reaction, no formed bonds, etc.).
    """
    try:
        candidates = generator.generate_for_reaction(
            reaction_smiles, source_id=source_id or "g8c",
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("rule generator failed for %r: %s", reaction_smiles, exc)
        return []

    try:
        center = extract_reaction_center(reaction_smiles)
    except Exception:
        center = ReactionCenterEdit(
            formed_bonds=[], broken_bonds=[], changed_bonds=[],
            reacting_atoms=[], mapped=False, reason="exception",
        )

    out: List[Dict[str, Any]] = []
    for cand in candidates:
        locus, edit_type = _parse_edit_action(cand.edit_action, center)
        out.append({
            "locus": locus,
            "edit_type": edit_type,
            "edit_action": cand.edit_action,
            "candidate_product": cand.candidate_product,
            "hard_score": float(cand.hard_score),
        })
    return out


# ---------------------------------------------------------------------------
# 3. Real competing-outcome pairs from HTE data
# ---------------------------------------------------------------------------

def _reaction_product(reaction_smiles: str) -> str:
    """Return the product side of a reaction SMILES (empty on failure)."""
    try:
        _, _, product = split_reaction(reaction_smiles)
        return product.strip()
    except ValueError:
        try:
            left, right = reaction_smiles.split(">>", 1)
            return right.strip()
        except Exception:
            return ""


def _reaction_reactants(reaction_smiles: str) -> str:
    """Return the reactant side of a reaction SMILES (empty on failure)."""
    try:
        reactants, _, _ = split_reaction(reaction_smiles)
        return reactants.strip()
    except ValueError:
        try:
            left, _ = reaction_smiles.split(">>", 1)
            return left.strip()
        except Exception:
            return ""


def build_competing_outcome_pairs(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Construct REAL competing-outcome pairs from HTE data.

    Strategy:
        * Group rows by ``reactant_1_smiles`` (same reactant scaffold).
        * Within each group that yields multiple distinct products, pair the
          highest-yield product (preferred) with the lowest-yield product
          (competing).
        * For zero-yield records (``measured_yield == 0``), pair the intended
          product (from ``reaction_smiles``) with an empty product ("no
          reaction"), which represents the experimentally observed failure.

    Each returned dict has the schema requested by the task spec.
    """
    pairs: List[Dict[str, Any]] = []
    if df is None or len(df) == 0:
        return pairs

    required = {"reactant_1_smiles", "measured_yield", "reaction_smiles",
                "experimental_group"}
    if not required.issubset(df.columns):
        # Degrade gracefully: nothing to pair.
        logger.warning("build_competing_outcome_pairs: missing columns %s",
                       sorted(required - set(df.columns)))
        return pairs

    # ---- multi-product groups: preferred (high yield) vs competing (low yield)
    grouped = df.groupby("reactant_1_smiles", sort=False)
    for reactant, group in grouped:
        if len(group) < 2:
            continue
        # Use the canonical product string from the `products` column when
        # available, otherwise parse reaction_smiles.
        if "products" in group.columns:
            products = group["products"].fillna("").astype(str).tolist()
        else:
            products = [_reaction_product(r) for r in group["reaction_smiles"]]
        yields = group["measured_yield"].fillna(0.0).astype(float).tolist()
        exp_groups = group["experimental_group"].astype(str).tolist()
        rxns = group["reaction_smiles"].astype(str).tolist()

        unique_products = {p for p in products if p}
        if len(unique_products) < 2:
            continue

        # preferred = highest yield, competing = lowest yield (distinct product)
        order = sorted(range(len(products)),
                       key=lambda i: yields[i], reverse=True)
        pref_idx = order[0]
        comp_idx = None
        for idx in order[1:]:
            if products[idx] and products[idx] != products[pref_idx]:
                comp_idx = idx
                break
        if comp_idx is None:
            continue

        pairs.append({
            "reactants": str(reactant),
            "preferred_product": products[pref_idx],
            "preferred_yield": float(yields[pref_idx]),
            "competing_product": products[comp_idx],
            "competing_yield": float(yields[comp_idx]),
            "experimental_group": exp_groups[pref_idx],
            "reaction_smiles": rxns[pref_idx],
            # Atom-mapped reaction for the competing outcome (so Stage 3 can
            # featurize both sides).  Falls back to the preferred reaction
            # when the competing row has no distinct reaction SMILES.
            "competing_reaction_smiles": rxns[comp_idx] or rxns[pref_idx],
        })

    # ---- zero-yield records: intended product vs "no reaction"
    zero_mask = df["measured_yield"].fillna(0.0).astype(float) <= 0.0
    zero_df = df[zero_mask]
    for _, row in zero_df.iterrows():
        rxn = str(row["reaction_smiles"])
        intended_product = _reaction_product(rxn)
        reactant = str(row.get("reactant_1_smiles", _reaction_reactants(rxn)))
        # "no reaction" competing reaction: same reactants, empty product.
        reactants_side = _reaction_reactants(rxn)
        no_reaction_rxn = join_reaction(reactants_side, "") if reactants_side else rxn
        pairs.append({
            "reactants": reactant,
            "preferred_product": intended_product,
            "preferred_yield": 0.0,
            "competing_product": "",  # "no reaction"
            "competing_yield": 0.0,
            "experimental_group": str(row.get("experimental_group", "")),
            "reaction_smiles": rxn,
            "competing_reaction_smiles": no_reaction_rxn,
        })

    return pairs


# ---------------------------------------------------------------------------
# 4. Real preference pairs for DPO
# ---------------------------------------------------------------------------

def build_preference_pairs(
    df: pd.DataFrame,
    generator: Optional[ReactionBoundaryGenerator] = None,
    max_generator_pairs: int = 2000,
) -> List[Dict[str, Any]]:
    """Construct REAL preference pairs for Stage-4 DPO training.

    Two sources of pairs are combined:
        1. High-yield product (preferred) vs low-yield/zero-yield product
           (dispreferred) from the same reactant scaffold.
        2. Real product (preferred) vs a PC-CNG rule-generated negative
           (dispreferred), when a ``generator`` is supplied.

    Both reactions in a pair are atom-mapped so the model can featurize them.
    """
    pairs: List[Dict[str, Any]] = []
    if df is None or len(df) == 0:
        return pairs

    required = {"reactant_1_smiles", "measured_yield", "reaction_smiles",
                "experimental_group"}
    if not required.issubset(df.columns):
        logger.warning("build_preference_pairs: missing columns %s",
                       sorted(required - set(df.columns)))
        return pairs

    # ---- source 1: yield-contrast pairs from same reactant scaffold
    grouped = df.groupby("reactant_1_smiles", sort=False)
    for reactant, group in grouped:
        if len(group) < 2:
            continue
        yields = group["measured_yield"].fillna(0.0).astype(float).tolist()
        rxns = group["reaction_smiles"].astype(str).tolist()
        exp_groups = group["experimental_group"].astype(str).tolist()
        order = sorted(range(len(rxns)), key=lambda i: yields[i], reverse=True)
        pref_idx = order[0]
        disp_idx = order[-1]
        if pref_idx == disp_idx:
            continue
        if yields[pref_idx] <= yields[disp_idx]:
            continue
        pairs.append({
            "reactants": str(reactant),
            "preferred_reaction": rxns[pref_idx],
            "dispreferred_reaction": rxns[disp_idx],
            "preferred_yield": float(yields[pref_idx]),
            "dispreferred_yield": float(yields[disp_idx]),
            "experimental_group": exp_groups[pref_idx],
        })

    # ---- source 2: real product vs PC-CNG generated negative
    if generator is not None:
        # Use a small subsample of successful reactions to keep this affordable.
        positive_df = df[df["measured_yield"].fillna(0.0).astype(float) > 0.0]
        if len(positive_df) > max_generator_pairs:
            positive_df = positive_df.sample(
                n=max_generator_pairs, random_state=20260724)
        for _, row in positive_df.iterrows():
            rxn = str(row["reaction_smiles"])
            try:
                cands = generator.generate_for_reaction(rxn, source_id="g8c_dpo")
            except Exception:
                cands = []
            if not cands:
                continue
            # dispreferred = highest-scoring rule-generated negative
            cand = max(cands, key=lambda c: c.hard_score)
            disp_rxn = cand.candidate_reaction
            if not disp_rxn or disp_rxn == rxn:
                continue
            pairs.append({
                "reactants": str(row.get("reactant_1_smiles",
                                         _reaction_reactants(rxn))),
                "preferred_reaction": rxn,
                "dispreferred_reaction": disp_rxn,
                "preferred_yield": float(row["measured_yield"]),
                "dispreferred_yield": 0.0,
                "experimental_group": str(row.get("experimental_group", "")),
            })

    return pairs


# ---------------------------------------------------------------------------
# 5. End-to-end data loading
# ---------------------------------------------------------------------------

def _split_by_column(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Return ``{split: [reaction_smiles, ...]}`` from the HTE dataframe."""
    out: Dict[str, List[str]] = {"train": [], "val": [], "test": []}
    if "split" not in df.columns:
        rxns = df["reaction_smiles"].astype(str).tolist()
        out["train"] = rxns
        return out
    for split_name in ("train", "val", "test"):
        sub = df[df["split"].astype(str) == split_name]
        out[split_name] = sub["reaction_smiles"].astype(str).tolist()
    return out


def load_g8c_training_data(
    hte_parquet_path: str = DEFAULT_HTE_PARQUET,
    generator: Optional[ReactionBoundaryGenerator] = None,
    max_rule_reactions: Optional[int] = None,
    use_rule_generator: bool = True,
    df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Load and prepare all G8-C training data in one pass.

    Parameters
    ----------
    hte_parquet_path : str
        Path to the HTE normalized parquet file.
    generator : ReactionBoundaryGenerator, optional
        Pre-constructed generator. A default one is created if absent.
    max_rule_reactions : int, optional
        Cap on the number of reactions for which to run the (slow) rule
        generator.  ``None`` means no cap.
    use_rule_generator : bool
        If ``False``, skip rule generation entirely (useful for smoke tests).
    df : pd.DataFrame, optional
        Pre-loaded dataframe; if supplied, ``hte_parquet_path`` is ignored.

    Returns
    -------
    dict
        ``{
            "reactions": {"train": [...], "val": [...], "test": [...]},
            "edit_targets": {reaction_smiles: {...}},
            "rule_proposals": {reaction_smiles: [{...}, ...]},
            "competing_pairs": [...],
            "preference_pairs": [...],
            "hte_df": df,
        }``
    """
    if df is None:
        df = pd.read_parquet(hte_parquet_path)

    # 1. split reactions by HTE split column
    splits = _split_by_column(df)

    # 2. real edit targets for ALL atom-mapped reactions
    all_rxns = list(dict.fromkeys(
        df["reaction_smiles"].astype(str).tolist()
    ))
    edit_targets: Dict[str, Dict[str, Any]] = {}
    for rxn in all_rxns:
        edit_targets[rxn] = extract_real_edit_targets(rxn)

    # 3. rule-generator proposals (cached).  This is the slow step, so it is
    #    optional and capped.
    rule_proposals: Dict[str, List[Dict[str, Any]]] = {}
    if use_rule_generator:
        if generator is None:
            generator = ReactionBoundaryGenerator(
                max_candidates_per_reaction=4, allow_unmapped_fallback=False)
        rxns_for_rules = all_rxns
        if max_rule_reactions is not None and len(rxns_for_rules) > max_rule_reactions:
            rxns_for_rules = rxns_for_rules[:max_rule_reactions]
        for i, rxn in enumerate(rxns_for_rules):
            rule_proposals[rxn] = extract_rule_proposals(rxn, generator)

    # 4. competing-outcome and preference pairs (built from the FULL df)
    competing_pairs = build_competing_outcome_pairs(df)
    preference_pairs = build_preference_pairs(
        df, generator=generator if use_rule_generator else None)

    return {
        "reactions": splits,
        "edit_targets": edit_targets,
        "rule_proposals": rule_proposals,
        "competing_pairs": competing_pairs,
        "preference_pairs": preference_pairs,
        "hte_df": df,
    }


__all__ = [
    "extract_real_edit_targets",
    "extract_rule_proposals",
    "build_competing_outcome_pairs",
    "build_preference_pairs",
    "load_g8c_training_data",
    "DEFAULT_HTE_PARQUET",
]
