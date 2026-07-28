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

Formal examples never degrade to pseudo-supervision.  Unmapped reactions are
marked ineligible for Stage 1, and rule-generator non-applicability is encoded
as ``NOT_APPLICABLE`` rather than conflated with the chemically meaningful
``NO_EDIT`` action.
"""

from __future__ import annotations

import logging
import hashlib
import json
import os
from pathlib import Path
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
from .g8c_action_schema import EditType

try:  # pragma: no cover - dependency is environment specific
    from rdkit import Chem
except Exception:  # pragma: no cover
    Chem = None


logger = logging.getLogger(__name__)

# Default HTE parquet location (relative to the pc_cng_research project root).
DEFAULT_HTE_PARQUET = os.environ.get(
    "G8C_HTE_PARQUET",
    "/home/cunyuliu/pc_cng_research/data/processed/p4_hte_normalized.parquet",
)
DEFAULT_COLLISION_REVIEW = os.environ.get(
    "G8C_COLLISION_REVIEW",
    "/home/cunyuliu/pc_cng_research/results/v2_boundary_generation/"
    "hitea_full_boundary_negatives_reviewed.csv",
)
DEFAULT_EXPERT_FORMS = (
    "/home/cunyuliu/pc_cng_research/results/expert_review_executed_20260720/"
    "reviewer_forms/reviewer_1_form.csv",
    "/home/cunyuliu/pc_cng_research/results/expert_review_executed_20260720/"
    "reviewer_forms/reviewer_2_form.csv",
)
FORMAL_CONTEXT_COLUMNS = (
    "reactant_1_smiles",
    "reactant_2_smiles",
    "catalyst_1_smiles",
    "catalyst_2_smiles",
    "solvent",
    "temperature",
    "reaction_time_hrs",
)
FORMAL_PAIR_PARTITION_VERSION = "context_hash_80_20_v1"
_BOND_ORDER_TO_INDEX = {
    "SINGLE": 0,
    "DOUBLE": 1,
    "TRIPLE": 2,
}


# ---------------------------------------------------------------------------
# 1. Real edit targets from atom-mapped reactions
# ---------------------------------------------------------------------------

def _bond_action(
    edit_type: EditType,
    left_map: int,
    right_map: int,
    *,
    bond_order: Optional[str] = None,
) -> Dict[str, Any]:
    bond_order_index = (
        _BOND_ORDER_TO_INDEX.get(str(bond_order), -100)
        if edit_type == EditType.BOND_ORDER_CHANGE
        else -100
    )
    return {
        "locus_map": int(left_map),
        "partner_map": int(right_map),
        "edit_type": int(edit_type),
        "argument_kind": (
            "bond_order"
            if edit_type == EditType.BOND_ORDER_CHANGE
            else "partner_atom"
        ),
        "bond_order_index": bond_order_index,
    }


def _real_actions(center: ReactionCenterEdit) -> List[Dict[str, Any]]:
    actions: List[Dict[str, Any]] = []
    for left, right, order in center.formed_bonds:
        actions.append(
            _bond_action(EditType.BOND_FORM, left, right, bond_order=order)
        )
    for left, right, order in center.broken_bonds:
        actions.append(
            _bond_action(EditType.BOND_BREAK, left, right, bond_order=order)
        )
    for left, right, _old_order, new_order in center.changed_bonds:
        actions.append(
            _bond_action(
                EditType.BOND_ORDER_CHANGE,
                left,
                right,
                bond_order=new_order,
            )
        )
    if not actions:
        actions.append(
            {
                "locus_map": 0,
                "partner_map": 0,
                "edit_type": int(EditType.NO_EDIT),
                "argument_kind": "none",
                "bond_order_index": -100,
            }
        )
    return actions


def extract_real_edit_targets(reaction_smiles: str) -> Dict[str, Any]:
    """Extract all real bond-form/break/change actions.

    Atom-map numbers are retained until graph collation, where they are
    translated through each graph's ``atom_map_to_idx`` mapping.  Unmapped or
    unparseable reactions are explicitly ineligible for formal supervision;
    they are never converted to a synthetic ``NO_EDIT`` target.
    """
    try:
        center = extract_reaction_center(reaction_smiles)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("extract_reaction_center failed for %r: %s", reaction_smiles, exc)
        center = ReactionCenterEdit(
            formed_bonds=[], broken_bonds=[], changed_bonds=[],
            reacting_atoms=[], mapped=False, reason="exception",
        )

    actions = _real_actions(center) if center.mapped else []
    primary = actions[0] if actions else {
        "locus_map": 0,
        "partner_map": 0,
        "edit_type": int(EditType.NOT_APPLICABLE),
        "argument_kind": "none",
        "bond_order_index": -100,
    }
    return {
        "locus": int(primary["locus_map"]),
        "locus_map": int(primary["locus_map"]),
        "partner_map": int(primary["partner_map"]),
        "edit_type": int(primary["edit_type"]),
        "actions": actions,
        "formed_bonds": list(getattr(center, "formed_bonds", []) or []),
        "broken_bonds": list(getattr(center, "broken_bonds", []) or []),
        "changed_bonds": list(getattr(center, "changed_bonds", []) or []),
        "reacting_atoms": list(getattr(center, "reacting_atoms", []) or []),
        "mapped": bool(getattr(center, "mapped", False)),
        "valid_for_formal": bool(getattr(center, "mapped", False)),
        "formal_exclusion_reason": (
            "" if getattr(center, "mapped", False)
            else str(getattr(center, "reason", "unmapped_or_unparseable"))
        ),
    }


# ---------------------------------------------------------------------------
# 2. Real rule-generator proposals
# ---------------------------------------------------------------------------

def _mapped_atom_numbers(smiles: str) -> Dict[int, int]:
    if Chem is None:
        return {}
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}
    return {
        int(atom.GetAtomMapNum()): int(atom.GetAtomicNum())
        for atom in mol.GetAtoms()
        if atom.GetAtomMapNum()
    }


def _mapped_bond_orders(smiles: str) -> Dict[Tuple[int, int], str]:
    if Chem is None:
        return {}
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}
    out: Dict[Tuple[int, int], str] = {}
    for bond in mol.GetBonds():
        left = int(bond.GetBeginAtom().GetAtomMapNum())
        right = int(bond.GetEndAtom().GetAtomMapNum())
        if left and right:
            out[tuple(sorted((left, right)))] = str(bond.GetBondType())
    return out


def _parse_edit_action(
    edit_action: str,
    parent_product: str,
    candidate_product: str,
) -> Dict[str, Any]:
    """Parse an actual rule proposal into locus, type and argument targets."""
    action = (edit_action or "").strip()
    base = {
        "locus_map": 0,
        "partner_map": 0,
        "atom_target_atomic_num": -100,
        "bond_order_index": -100,
        "applicable": True,
    }
    if action.startswith("formed_bond_migrate"):
        parts = action.split(":")
        if len(parts) >= 3:
            try:
                base["locus_map"] = int(parts[1])
                base["partner_map"] = int(parts[2].split("->", 1)[1])
            except ValueError:
                pass
        return {**base, "edit_type": int(EditType.FORMED_BOND_MIGRATE)}
    if action.startswith("center_atom"):
        before = _mapped_atom_numbers(parent_product)
        after = _mapped_atom_numbers(candidate_product)
        changed = [
            atom_map for atom_map in sorted(set(before) & set(after))
            if before[atom_map] != after[atom_map]
        ]
        if changed:
            base["locus_map"] = int(changed[0])
            base["atom_target_atomic_num"] = int(after[changed[0]])
        return {**base, "edit_type": int(EditType.ATOM_TRANSMUTATION)}
    if action.startswith("center_bond"):
        before_bonds = _mapped_bond_orders(parent_product)
        after_bonds = _mapped_bond_orders(candidate_product)
        changed = [
            pair for pair in sorted(set(before_bonds) & set(after_bonds))
            if before_bonds[pair] != after_bonds[pair]
        ]
        if changed:
            left, right = changed[0]
            base["locus_map"] = int(left)
            base["partner_map"] = int(right)
            base["bond_order_index"] = _BOND_ORDER_TO_INDEX.get(
                after_bonds[(left, right)],
                -100,
            )
        return {**base, "edit_type": int(EditType.BOND_ORDER_CHANGE)}
    return {
        **base,
        "edit_type": int(EditType.NOT_APPLICABLE),
        "applicable": False,
    }


def extract_rule_proposals(
    reaction_smiles: str,
    generator: ReactionBoundaryGenerator,
    source_id: str = "",
) -> List[Dict[str, Any]]:
    """Run the rule generator and return its ACTUAL proposals as plain dicts.

    Each dict mirrors the fields needed by Stage-2 imitation training:
    ``locus``, ``edit_type``, ``edit_action``, ``candidate_product``,
    ``hard_score``.  When the rule is inapplicable, returns one explicit
    ``NOT_APPLICABLE`` target rather than a ``NO_EDIT`` pseudo-action.
    """
    try:
        candidates = generator.generate_for_reaction(
            reaction_smiles, source_id=source_id or "g8c",
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("rule generator failed for %r: %s", reaction_smiles, exc)
        candidates = []

    out: List[Dict[str, Any]] = []
    for cand in candidates:
        target = _parse_edit_action(
            cand.edit_action,
            cand.parent_product,
            cand.candidate_product,
        )
        out.append({
            "locus": int(target["locus_map"]),
            **target,
            "edit_action": cand.edit_action,
            "candidate_product": cand.candidate_product,
            "candidate_reaction": cand.candidate_reaction,
            "hard_score": float(cand.hard_score),
        })
    if out:
        return out
    return [{
        "locus": 0,
        "locus_map": 0,
        "partner_map": 0,
        "edit_type": int(EditType.NOT_APPLICABLE),
        "atom_target_atomic_num": -100,
        "bond_order_index": -100,
        "applicable": False,
        "edit_action": "NOT_APPLICABLE",
        "candidate_product": "",
        "candidate_reaction": "",
        "hard_score": 1.0,
    }]


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

    A pair is valid only when reactants, catalysts, solvent, temperature and
    reaction time match exactly and two distinct products were actually
    recorded.  Zero-yield intended products are not converted into invented
    "empty product" observations.
    """
    pairs: List[Dict[str, Any]] = []
    if df is None or len(df) == 0:
        return pairs

    required = {
        "reactant_1_smiles",
        "reactant_2_smiles",
        "measured_yield",
        "reaction_smiles",
        "experimental_group",
        "products",
        "split",
    }
    if not required.issubset(df.columns):
        # Degrade gracefully: nothing to pair.
        logger.warning("build_competing_outcome_pairs: missing columns %s",
                       sorted(required - set(df.columns)))
        return pairs

    context_columns = [
        column for column in FORMAL_CONTEXT_COLUMNS if column in df.columns
    ]
    grouped = df.groupby(["split", *context_columns], sort=False, dropna=False)
    for group_key, group in grouped:
        if len(group) < 2:
            continue
        products = group["products"].fillna("").astype(str).tolist()
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

        split_name = str(group_key[0])
        context_values = group_key[1:]
        context = {
            column: ("" if pd.isna(value) else str(value))
            for column, value in zip(context_columns, context_values)
        }
        context_key = json.dumps(context, sort_keys=True, separators=(",", ":"))
        pairs.append({
            "reactants": ".".join(
                value for value in (
                    context.get("reactant_1_smiles", ""),
                    context.get("reactant_2_smiles", ""),
                )
                if value
            ),
            "preferred_product": products[pref_idx],
            "preferred_yield": float(yields[pref_idx]),
            "competing_product": products[comp_idx],
            "competing_yield": float(yields[comp_idx]),
            "experimental_group": exp_groups[pref_idx],
            "split": split_name,
            "context": context,
            "context_key": context_key,
            "pair_provenance": "observed_multi_product_same_full_context",
            "reaction_smiles": rxns[pref_idx],
            "competing_reaction_smiles": rxns[comp_idx],
        })

    return pairs


# ---------------------------------------------------------------------------
# 4. Real preference pairs for DPO
# ---------------------------------------------------------------------------

def build_preference_pairs(
    df: pd.DataFrame,
    generator: Optional[ReactionBoundaryGenerator] = None,
    max_generator_pairs: int = 2000,
    *,
    formal: bool = True,
) -> List[Dict[str, Any]]:
    """Construct REAL preference pairs for Stage-4 DPO training.

    Formal pairs come only from observed same-full-context outcomes.  The
    optional generated comparison is retained for exploratory mode and is
    never admitted when ``formal=True``.
    """
    pairs: List[Dict[str, Any]] = []
    if df is None or len(df) == 0:
        return pairs

    for observed in build_competing_outcome_pairs(df):
        pairs.append({
            "reactants": observed["reactants"],
            "preferred_reaction": observed["reaction_smiles"],
            "dispreferred_reaction": observed["competing_reaction_smiles"],
            "preferred_yield": observed["preferred_yield"],
            "dispreferred_yield": observed["competing_yield"],
            "experimental_group": observed["experimental_group"],
            "split": observed["split"],
            "context": observed["context"],
            "context_key": observed["context_key"],
            "pair_provenance": "observed_yield_preference_same_full_context",
        })

    # ---- source 2: real product vs PC-CNG generated negative
    if generator is not None and not formal:
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
                "split": str(row.get("split", "")),
                "pair_provenance": "exploratory_rule_generated_preference",
            })

    return pairs


def assign_formal_pair_partition(
    pairs: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Assign an immutable context-group train/validation partition.

    The source HTE split places every observed multi-product context in the
    training split.  A deterministic context hash therefore creates the
    validation partition needed for Stage-3/4 model selection.  All outcomes
    from one complete reaction context remain together.  Any future source
    ``test`` rows remain sealed and are never reassigned.
    """
    assigned: List[Dict[str, Any]] = []
    for raw_pair in pairs:
        pair = dict(raw_pair)
        source_split = str(pair.get("split", "train"))
        context_key = str(pair.get("context_key", ""))
        if source_split == "test":
            formal_split = "test"
        elif source_split == "val":
            formal_split = "val"
        else:
            token = (
                f"{FORMAL_PAIR_PARTITION_VERSION}|{context_key}"
            ).encode("utf-8")
            bucket = int(hashlib.sha256(token).hexdigest()[:8], 16) % 5
            formal_split = "val" if bucket == 0 else "train"
        pair["source_split"] = source_split
        pair["formal_split"] = formal_split
        pair["formal_partition_version"] = FORMAL_PAIR_PARTITION_VERSION
        assigned.append(pair)
    return assigned


# ---------------------------------------------------------------------------
# 5. Candidate-level false-negative-risk supervision
# ---------------------------------------------------------------------------

def _hte_outcome_risk_examples(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Build context-specific held-out HTE risk labels.

    Measured yield >=10% is positive evidence that treating the candidate as a
    negative would be a false negative.  A reported zero that is neither
    missing nor below a detection-limit placeholder is negative evidence.
    Ambiguous rows are excluded.
    """
    examples: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        measured = float(row.get("measured_yield", 0.0) or 0.0)
        reported_zero = bool(row.get("reported_zero", False))
        missing = bool(row.get("missing_measurement", False))
        if measured >= 10.0:
            label = 1
        elif reported_zero and not missing:
            label = 0
        else:
            continue
        examples.append({
            "reaction_smiles": str(row["reaction_smiles"]),
            "risk_label": label,
            "risk_source": "heldout_hte_outcome",
            "split": str(row.get("split", "")),
            "record_id": str(row.get("record_id", "")),
            "experimental_group": str(row.get("experimental_group", "")),
        })
    return examples


def _collision_risk_examples(
    path: str,
    split_lookup: Dict[str, str],
) -> List[Dict[str, Any]]:
    review_path = Path(path)
    if not review_path.exists():
        return []
    frame = pd.read_csv(review_path)
    required = {"source_id", "candidate_reaction", "review_status"}
    if not required.issubset(frame.columns):
        return []
    examples: List[Dict[str, Any]] = []
    for _, row in frame.iterrows():
        status = str(row.get("review_status", ""))
        if status == "discard_known_positive":
            label = 1
            source = "known_positive_collision"
        elif status == "keep_synthetic_negative":
            label = 0
            source = "reviewed_synthetic_negative"
        else:
            continue
        reaction = str(row.get("candidate_reaction", ""))
        if not reaction:
            continue
        source_id = str(row.get("source_id", ""))
        examples.append({
            "reaction_smiles": reaction,
            "risk_label": label,
            "risk_source": source,
            "split": split_lookup.get(source_id, "train"),
            "record_id": source_id,
            "experimental_group": source_id,
        })
    return examples


def _expert_risk_examples(paths: Tuple[str, ...]) -> List[Dict[str, Any]]:
    """Load only genuinely completed human-review rows."""
    examples: List[Dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        for _, row in frame.iterrows():
            timestamp = row.get("review_timestamp")
            feasibility = row.get("feasibility")
            verdict = row.get("overall_verdict")
            if pd.isna(timestamp) or (pd.isna(feasibility) and pd.isna(verdict)):
                continue
            numeric = None
            if not pd.isna(feasibility):
                try:
                    numeric = float(feasibility)
                except (TypeError, ValueError):
                    numeric = None
            if numeric is not None:
                label = int(numeric >= 4.0)
            else:
                normalized = str(verdict).strip().lower()
                if normalized in {"positive", "feasible", "accept", "1", "true"}:
                    label = 1
                elif normalized in {"negative", "infeasible", "reject", "0", "false"}:
                    label = 0
                else:
                    continue
            reaction = str(row.get("candidate_reaction", ""))
            if not reaction:
                continue
            examples.append({
                "reaction_smiles": reaction,
                "risk_label": label,
                "risk_source": "expert_label",
                "split": "train",
                "record_id": str(row.get("sample_id", "")),
                "experimental_group": str(row.get("reviewer_id", "expert")),
            })
    return examples


def build_risk_supervision(
    df: pd.DataFrame,
    competing_pairs: List[Dict[str, Any]],
    *,
    collision_review_path: str = DEFAULT_COLLISION_REVIEW,
    expert_form_paths: Tuple[str, ...] = DEFAULT_EXPERT_FORMS,
) -> Dict[str, Any]:
    split_lookup = {
        str(row["record_id"]): str(row["split"])
        for _, row in df[["record_id", "split"]].drop_duplicates().iterrows()
    }
    examples = _hte_outcome_risk_examples(df)
    examples.extend(_collision_risk_examples(collision_review_path, split_lookup))
    for pair in competing_pairs:
        for key in ("reaction_smiles", "competing_reaction_smiles"):
            examples.append({
                "reaction_smiles": str(pair[key]),
                "risk_label": 1,
                "risk_source": "observed_competing_product",
                "split": str(pair.get("formal_split", pair["split"])),
                "record_id": str(pair["context_key"]),
                "experimental_group": str(pair["experimental_group"]),
            })
    expert = _expert_risk_examples(expert_form_paths)
    examples.extend(expert)

    deduplicated: Dict[Tuple[str, str, int, str], Dict[str, Any]] = {}
    for example in examples:
        key = (
            str(example["reaction_smiles"]),
            str(example["risk_source"]),
            int(example["risk_label"]),
            str(example["split"]),
        )
        deduplicated[key] = example
    examples = list(deduplicated.values())
    by_split = {
        split: [example for example in examples if example["split"] == split]
        for split in ("train", "val", "test")
    }
    sources = (
        "known_positive_collision",
        "observed_competing_product",
        "expert_label",
        "heldout_hte_outcome",
    )
    availability = {
        source: sum(example["risk_source"] == source for example in examples)
        for source in sources
    }
    return {
        "by_split": by_split,
        "source_availability": availability,
        "expert_labels_available": availability["expert_label"] > 0,
    }


# ---------------------------------------------------------------------------
# 6. End-to-end data loading
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
    formal: bool = True,
    collision_review_path: str = DEFAULT_COLLISION_REVIEW,
    expert_form_paths: Tuple[str, ...] = DEFAULT_EXPERT_FORMS,
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

    # 3. rule-generator proposals for training reactions. Non-applicability is
    #    an explicit target, so every processed reaction has a non-empty cache.
    rule_proposals: Dict[str, List[Dict[str, Any]]] = {}
    if use_rule_generator:
        if generator is None:
            generator = ReactionBoundaryGenerator(
                max_candidates_per_reaction=4, allow_unmapped_fallback=False)
        rxns_for_rules = list(dict.fromkeys(splits["train"]))
        if max_rule_reactions is not None and len(rxns_for_rules) > max_rule_reactions:
            rxns_for_rules = rxns_for_rules[:max_rule_reactions]
        for i, rxn in enumerate(rxns_for_rules):
            rule_proposals[rxn] = extract_rule_proposals(rxn, generator)

    # 4. observed same-full-context pairs, preserving split boundaries.
    competing_pairs = assign_formal_pair_partition(
        build_competing_outcome_pairs(df)
    )
    preference_pairs = assign_formal_pair_partition(build_preference_pairs(
        df,
        generator=generator if use_rule_generator else None,
        formal=formal,
    ))
    competing_by_split = {
        split: [
            pair for pair in competing_pairs
            if pair["formal_split"] == split
        ]
        for split in ("train", "val", "test")
    }
    preference_by_split = {
        split: [
            pair for pair in preference_pairs
            if pair["formal_split"] == split
        ]
        for split in ("train", "val", "test")
    }
    risk_supervision = build_risk_supervision(
        df,
        competing_pairs,
        collision_review_path=collision_review_path,
        expert_form_paths=expert_form_paths,
    )
    valid_edit_targets = sum(
        bool(target.get("valid_for_formal")) for target in edit_targets.values()
    )

    return {
        "reactions": splits,
        "edit_targets": edit_targets,
        "rule_proposals": rule_proposals,
        "competing_pairs": competing_by_split["train"],
        "preference_pairs": preference_by_split["train"],
        "competing_pairs_by_split": competing_by_split,
        "preference_pairs_by_split": preference_by_split,
        "risk_supervision": risk_supervision,
        "data_audit": {
            "n_unique_reactions": len(all_rxns),
            "n_valid_formal_edit_targets": valid_edit_targets,
            "n_invalid_formal_edit_targets": len(all_rxns) - valid_edit_targets,
            "n_train_rule_targets": len(rule_proposals),
            "n_same_context_competing_pairs": len(competing_pairs),
            "n_train_competing_pairs": len(competing_by_split["train"]),
            "n_val_competing_pairs": len(competing_by_split["val"]),
            "n_test_competing_pairs": len(competing_by_split["test"]),
            "formal_pair_partition_version": FORMAL_PAIR_PARTITION_VERSION,
            "risk_source_availability": risk_supervision["source_availability"],
        },
        "hte_df": df,
    }


__all__ = [
    "extract_real_edit_targets",
    "extract_rule_proposals",
    "build_competing_outcome_pairs",
    "build_preference_pairs",
    "assign_formal_pair_partition",
    "build_risk_supervision",
    "load_g8c_training_data",
    "DEFAULT_HTE_PARQUET",
    "DEFAULT_COLLISION_REVIEW",
    "DEFAULT_EXPERT_FORMS",
    "FORMAL_CONTEXT_COLUMNS",
    "FORMAL_PAIR_PARTITION_VERSION",
]
