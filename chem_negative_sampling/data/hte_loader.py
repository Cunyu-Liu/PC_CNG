"""HTE (High-Throughput Experimentation) data loader for P3-05.

This module is part of the P3-05 task: HTE data evaluation via leave-one-out
stratified by ``reaction_class``.  It loads the HTEa CSV located at
``data/processed/hitea_full_normalized.csv`` on the remote server, groups
reactions by their ``reaction_class`` field (e.g. ``Alkylation``,
``Acylation``), and provides a stratified sampler used by
``chem_negative_sampling/evaluation/hte_eval.py``.

Public API
----------
- :class:`HTEGroup` -- dataclass summarising one ``reaction_class`` bucket.
- :func:`load_hte_by_class` -- load + group a HTEa CSV file.
- :func:`stratified_sample` -- stratified sample of reaction SMILES across
  classes for leave-one-out evaluation.

The module depends only on Python 3.10 stdlib + ``pandas`` (already used by
the rest of the project) and is intentionally lazy about RDKit -- validity
checks happen in :mod:`hte_eval` to keep this loader cheap.
"""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

__all__ = [
    "HTEGroup",
    "load_hte_by_class",
    "stratified_sample",
    "iter_reactions",
]

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class HTEGroup:
    """Summary of one ``reaction_class`` bucket from the HTEa CSV.

    Attributes
    ----------
    reaction_class:
        The class label (e.g. ``"Alkylation"``).
    n_reactions:
        Total number of reactions in this class.
    n_unique_products:
        Number of unique product SMILES (taken from the ``products`` column
        when available, otherwise parsed from ``reaction_smiles``).
    yield_mean:
        Mean of the ``yield`` column (``NaN`` if no yields recorded).
    yield_std:
        Sample standard deviation of the ``yield`` column (``NaN`` if <2
        valid yields).
    reaction_smiles_list:
        Ordered list of reaction SMILES strings belonging to this class.
    source_ids:
        Parallel list of ``source_id`` values (may contain ``None``).
    """

    reaction_class: str
    n_reactions: int
    n_unique_products: int
    yield_mean: float
    yield_std: float
    reaction_smiles_list: List[str] = field(default_factory=list)
    source_ids: List[Optional[str]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------


def _parse_yield(raw: str) -> Optional[float]:
    """Parse a yield cell, returning ``None`` for empty / non-numeric values.

    The HTEa normalised CSV uses empty strings for missing yields.  Some
    legacy rows may use ``"nan"`` / ``"NaN"`` strings; we treat all of them
    as missing.
    """
    if raw is None:
        return None
    s = raw.strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _extract_product_smiles(row: Dict[str, str], reaction_smiles: str) -> str:
    """Get the product SMILES for a row, preferring the ``products`` column."""
    prod = row.get("products", "").strip()
    if prod:
        # products may itself be a dotted list of molecules; take the whole
        # canonical string for uniqueness purposes.
        return prod
    # Fallback: parse reaction_smiles "reactants>>products"
    if ">>" in reaction_smiles:
        return reaction_smiles.rsplit(">>", 1)[1].strip()
    return ""


def iter_reactions(csv_path: str) -> List[Dict[str, str]]:
    """Read the HTEa CSV into a list of row dictionaries.

    Uses the stdlib ``csv`` module to avoid a hard dependency on pandas in
    unit tests (which run on the CI host without GPU/remote access).

    Parameters
    ----------
    csv_path:
        Path to ``hitea_full_normalized.csv`` (or compatible).

    Returns
    -------
    list of dict
        One dict per row.  Missing columns are returned as the empty string.
    """
    rows: List[Dict[str, str]] = []
    with open(csv_path, "r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            return rows
        for raw in reader:
            # Normalise None -> "" so downstream code can call .strip() safely
            row = {k: (v if v is not None else "") for k, v in raw.items()}
            rows.append(row)
    return rows


def load_hte_by_class(
    csv_path: str,
    min_class_size: int = 20,
) -> Dict[str, HTEGroup]:
    """Load a HTEa CSV and group reactions by ``reaction_class``.

    Parameters
    ----------
    csv_path:
        Path to ``data/processed/hitea_full_normalized.csv``.
    min_class_size:
        Minimum number of reactions required to keep a class.  Classes with
        fewer reactions are silently dropped (default 20, matching the
        P3-05 task spec).

    Returns
    -------
    dict
        Mapping ``reaction_class -> HTEGroup``.  Iteration order matches the
        first appearance of each class in the CSV (Python dicts preserve
        insertion order).

    Raises
    ------
    ValueError
        If the CSV is missing the ``reaction_class`` or ``reaction_smiles``
        columns.
    """
    if min_class_size < 1:
        raise ValueError(f"min_class_size must be >= 1, got {min_class_size}")

    rows = iter_reactions(csv_path)
    if not rows:
        return {}

    required = {"reaction_class", "reaction_smiles"}
    missing = required - set(rows[0].keys())
    if missing:
        raise ValueError(
            f"CSV missing required columns: {sorted(missing)}; "
            f"found columns: {sorted(rows[0].keys())}"
        )

    # Accumulate per-class raw data
    classes: Dict[str, Dict[str, object]] = {}
    for row in rows:
        cls = (row.get("reaction_class") or "").strip()
        if not cls:
            continue
        rxn = (row.get("reaction_smiles") or "").strip()
        if not rxn:
            continue
        bucket = classes.setdefault(
            cls,
            {
                "smiles": [],
                "source_ids": [],
                "yields": [],
                "products": set(),
            },
        )
        bucket["smiles"].append(rxn)
        bucket["source_ids"].append((row.get("source_id") or "").strip() or None)
        y = _parse_yield(row.get("yield", ""))
        if y is not None:
            bucket["yields"].append(y)
        prod = _extract_product_smiles(row, rxn)
        if prod:
            bucket["products"].add(prod)

    result: Dict[str, HTEGroup] = {}
    for cls, data in classes.items():
        n = len(data["smiles"])
        if n < min_class_size:
            continue
        yields: List[float] = data["yields"]  # type: ignore[assignment]
        if yields:
            mean_y = float(sum(yields) / len(yields))
            if len(yields) >= 2:
                # Sample std (ddof=1) to match numpy's default
                var = sum((y - mean_y) ** 2 for y in yields) / (len(yields) - 1)
                std_y = float(var ** 0.5)
            else:
                std_y = float("nan")
        else:
            mean_y = float("nan")
            std_y = float("nan")
        result[cls] = HTEGroup(
            reaction_class=cls,
            n_reactions=n,
            n_unique_products=len(data["products"]),
            yield_mean=mean_y,
            yield_std=std_y,
            reaction_smiles_list=list(data["smiles"]),
            source_ids=list(data["source_ids"]),  # type: ignore[arg-type]
        )
    return result


# ---------------------------------------------------------------------------
# Stratified sampling
# ---------------------------------------------------------------------------


def stratified_sample(
    hte_groups: Dict[str, HTEGroup],
    n_per_class: int = 50,
    seed: int = 42,
) -> List[str]:
    """Stratified sample of reaction SMILES for leave-one-out evaluation.

    For each class in ``hte_groups``, sample ``min(n_per_class, n_reactions)``
    reaction SMILES without replacement.  The concatenation across classes is
    returned (class order preserved).

    Parameters
    ----------
    hte_groups:
        Output of :func:`load_hte_by_class`.
    n_per_class:
        Maximum number of reactions to sample per class (default 50).
    seed:
        RNG seed for reproducibility (default 42).

    Returns
    -------
    list of str
        Sampled reaction SMILES, length ``<= n_per_class * n_classes``.
    """
    if n_per_class < 1:
        raise ValueError(f"n_per_class must be >= 1, got {n_per_class}")

    rng = random.Random(seed)
    sampled: List[str] = []
    for cls, group in hte_groups.items():
        smiles = group.reaction_smiles_list
        if not smiles:
            continue
        k = min(n_per_class, len(smiles))
        # ``rng.sample`` returns a new list (no mutation of input).  We sort
        # the population first so the result is deterministic w.r.t. input
        # order regardless of Python's hash randomisation.
        population = sorted(smiles)
        sampled.extend(rng.sample(population, k))
    return sampled


def family_cluster_bootstrap(
    hte_groups: Dict[str, HTEGroup],
    seed: int = 0,
    n_bootstrap: int = 1000,
) -> List[List[str]]:
    """Family-cluster bootstrap: resample *classes* with replacement.

    Used by :mod:`hte_eval` to compute family-cluster bootstrap CIs of the
    leave-one-out metrics.  Returns a list of bootstrap replicates, where
    each replicate is the list of reaction SMILES obtained by sampling
    ``len(hte_groups)`` classes with replacement and concatenating their
    full reaction lists.

    Parameters
    ----------
    hte_groups:
        Output of :func:`load_hte_by_class`.
    seed:
        RNG seed.
    n_bootstrap:
        Number of bootstrap replicates (default 1000).

    Returns
    -------
    list of list of str
        ``n_bootstrap`` replicates.
    """
    if n_bootstrap < 1:
        raise ValueError(f"n_bootstrap must be >= 1, got {n_bootstrap}")
    rng = random.Random(seed)
    class_names = list(hte_groups.keys())
    if not class_names:
        return []
    replicates: List[List[str]] = []
    for _ in range(n_bootstrap):
        rep: List[str] = []
        for _ in range(len(class_names)):
            cls = rng.choice(class_names)
            rep.extend(hte_groups[cls].reaction_smiles_list)
        replicates.append(rep)
    return replicates
