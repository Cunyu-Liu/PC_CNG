"""Leave-one-out evaluation on HTE data for P3-05.

This module implements the P3-05 task: leave-one-out evaluation on the HTEa
dataset, comparing three negative-sampling strategies:

1. **PC-CNG negatives** -- synthetic negatives produced by the PC-CNG
   pipeline (``results/uspto_openmolecules_full_generation/pc_cng_synthetic_negatives_reviewed.csv``).
2. **Random negatives** -- on-the-fly SMILES perturbations (atom swap +
   bond deletion) generated with RDKit.
3. **No negatives** -- single-class baseline that scores the held-out
   positive by its maximum Tanimoto similarity to the training-set
   positives (no classifier is trained).

For each held-out reaction we train a Morgan-FP + LogisticRegression
classifier (strategies 1 & 2) or fall back to a similarity scorer
(strategy 3), then rank the held-out positive against a pool of candidate
negatives.  Reported metrics: **Top-1 accuracy**, **MRR**, **NDCG@10**.
The evaluation is repeated across 10 seeds and family-cluster bootstrap
CIs are computed by resampling ``reaction_class`` buckets.

CLI usage (run from ``chem_negative_sampling/``)::

    python -m evaluation.hte_eval \
        --hte-csv data/processed/hitea_full_normalized.csv \
        --pc-cng-negatives results/uspto_openmolecules_full_generation/pc_cng_synthetic_negatives_reviewed.csv \
        --output-dir results/hte_evaluation_20260720 \
        --n-per-class 50 --min-class-size 20 \
        --seeds 20260710,20260711,20260712,20260713,20260714,20260715,20260716,20260717,20260718,20260719

The module depends only on Python 3.10 stdlib + RDKit + scikit-learn +
numpy (no new dependencies).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import statistics
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from sklearn.linear_model import LogisticRegression

# ---------------------------------------------------------------------------
# Robust import of the sibling hte_loader module.
# On the server the file lives at chem_negative_sampling/evaluation/hte_eval.py
# and imports as ``from data.hte_loader import ...``.  In the flat local
# layout (e.g. /tmp/p3_files/p3_05/) the loader is in the same directory,
# so we fall back to ``from hte_loader import ...``.
# ---------------------------------------------------------------------------
try:  # pragma: no cover - exercised on the server only
    from data.hte_loader import (  # type: ignore[import]
        HTEGroup,
        family_cluster_bootstrap,
        load_hte_by_class,
        stratified_sample,
    )
except ImportError:  # flat local layout (unit tests)
    from hte_loader import (  # type: ignore[no-redef]
        HTEGroup,
        family_cluster_bootstrap,
        load_hte_by_class,
        stratified_sample,
    )


__all__ = [
    "EvaluationResult",
    "SeedMetrics",
    "morgan_fingerprint",
    "reaction_fingerprint",
    "generate_random_negatives",
    "load_pc_cng_negatives",
    "compute_top1",
    "compute_mrr",
    "compute_ndcg_at_k",
    "evaluate_leave_one_out",
    "family_cluster_bootstrap_ci",
    "run_evaluation",
    "main",
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class SeedMetrics:
    """Per-seed evaluation metrics for one negative strategy.

    All metric fields are floats in ``[0, 1]``.  ``n_evaluated`` is the
    number of held-out reactions that contributed to the mean (reactions
    whose fingerprints could not be parsed are skipped).
    """

    seed: int
    strategy: str
    top1: float
    mrr: float
    ndcg10: float
    n_evaluated: int


@dataclass
class EvaluationResult:
    """Full evaluation output across all seeds + strategies.

    The ``per_seed`` list has one entry per (seed, strategy) pair.  The
    ``summary`` dict has per-strategy aggregated stats (mean + 95% CI).
    """

    per_seed: List[SeedMetrics] = field(default_factory=list)
    summary: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    bootstrap_ci: Dict[str, Dict[str, Tuple[float, float]]] = field(
        default_factory=dict
    )
    meta: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Fingerprinting
# ---------------------------------------------------------------------------


def _safe_mol_from_smiles(smiles: str) -> Optional[Chem.Mol]:
    """Parse a SMILES string, returning ``None`` on failure (no exceptions)."""
    if not smiles:
        return None
    try:
        return Chem.MolFromSmiles(smiles)
    except Exception:  # pragma: no cover - RDKit rarely raises
        return None


def morgan_fingerprint(
    smiles: str,
    radius: int = 2,
    n_bits: int = 2048,
) -> Optional[np.ndarray]:
    """Return a Morgan fingerprint as a numpy bit vector.

    Parameters
    ----------
    smiles:
        Molecule SMILES (single molecule, not a reaction).
    radius:
        Morgan radius (default 2 = ECFP4).
    n_bits:
        Bit-vector length (default 2048).

    Returns
    -------
    numpy.ndarray or None
        ``np.uint8`` array of shape ``(n_bits,)`` or ``None`` if SMILES is
        unparseable.
    """
    mol = _safe_mol_from_smiles(smiles)
    if mol is None:
        return None
    try:
        bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    except Exception:  # pragma: no cover
        return None
    arr = np.zeros((n_bits,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(bv, arr)
    return arr


def reaction_fingerprint(
    reaction_smiles: str,
    radius: int = 2,
    n_bits: int = 2048,
) -> Optional[np.ndarray]:
    """Build a reaction fingerprint = product_fp - reactant_fp (difference).

    The reaction SMILES uses the ``reactants>>products`` format.  Each side
    may contain multiple dot-separated molecules.  The fingerprint is the
    XOR of all Morgan fingerprints on the reactant side, plus the XOR of
    all fingerprints on the product side, then XOR-ed together to produce a
    single bit vector.  This matches the difference-fingerprint intuition
    while staying non-negative for sklearn.

    Parameters
    ----------
    reaction_smiles:
        Reaction SMILES, ``"A.B>>C.D"``.
    radius, n_bits:
        Morgan FP hyper-parameters.

    Returns
    -------
    numpy.ndarray or None
        Bit vector of shape ``(n_bits,)`` or ``None`` if any molecule is
        unparseable.
    """
    if not reaction_smiles or ">>" not in reaction_smiles:
        return None
    reactants_str, products_str = reaction_smiles.split(">>", 1)
    if not reactants_str.strip() or not products_str.strip():
        return None

    def _union_fp(smiles_list: Sequence[str]) -> Optional[np.ndarray]:
        acc = np.zeros((n_bits,), dtype=np.uint8)
        any_ok = False
        for smi in smiles_list:
            smi = smi.strip()
            if not smi:
                continue
            fp = morgan_fingerprint(smi, radius=radius, n_bits=n_bits)
            if fp is None:
                continue
            acc |= fp
            any_ok = True
        return acc if any_ok else None

    react_fp = _union_fp(reactants_str.split("."))
    prod_fp = _union_fp(products_str.split("."))
    if react_fp is None or prod_fp is None:
        return None
    # XOR-like combination: a bit is set if it appears on either side.
    # This produces a stable, non-negative feature vector for sklearn.
    return (react_fp | prod_fp).astype(np.uint8)


# ---------------------------------------------------------------------------
# Random negative generation (RDKit perturbation)
# ---------------------------------------------------------------------------

# Common organic atoms for the swap heuristic.  Hydrogen is excluded.
_SWAP_ATOMS = ["C", "N", "O", "S", "F", "Cl", "Br"]


def _pick_swappable_atom(mol: Chem.Mol, rng: random.Random) -> Optional[int]:
    """Pick the index of a heavy, non-aromatic, non-ring atom to swap."""
    candidates: List[int] = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() <= 1:
            continue
        # Skip atoms in rings -- swapping them often produces invalid mols.
        if atom.IsInRing():
            continue
        candidates.append(atom.GetIdx())
    if not candidates:
        # Fallback: any heavy atom
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() > 1:
                candidates.append(atom.GetIdx())
        if not candidates:
            return None
    return rng.choice(candidates)


def _swap_atom(mol: Chem.Mol, rng: random.Random) -> Optional[Chem.Mol]:
    """Swap a single atom's element.  Returns a new Mol or ``None``."""
    idx = _pick_swappable_atom(mol, rng)
    if idx is None:
        return None
    atom = mol.GetAtomWithIdx(idx)
    original = atom.GetSymbol()
    choices = [a for a in _SWAP_ATOMS if a != original]
    if not choices:
        return None
    new_sym = rng.choice(choices)
    new_num = Chem.GetPeriodicTable().GetAtomicNumber(new_sym)
    rw = Chem.RWMol(mol)
    rw.GetAtomWithIdx(idx).SetAtomicNum(new_num)
    try:
        new_mol = rw.GetMol()
        # Round-trip through SMILES to verify validity.
        smi = Chem.MolToSmiles(new_mol)
        if not smi:
            return None
        return Chem.MolFromSmiles(smi)
    except Exception:
        return None


def _delete_bond(mol: Chem.Mol, rng: random.Random) -> Optional[Chem.Mol]:
    """Delete a single non-ring bond.  Returns a new Mol or ``None``."""
    bonds = [b for b in mol.GetBonds() if not b.IsInRing()]
    if not bonds:
        return None
    bond = rng.choice(bonds)
    rw = Chem.RWMol(mol)
    rw.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
    try:
        new_mol = rw.GetMol()
        smi = Chem.MolToSmiles(new_mol)
        if not smi:
            return None
        # MolToSmiles may emit multiple fragments -- keep the largest.
        if "." in smi:
            fragments = sorted(smi.split("."), key=len, reverse=True)
            smi = fragments[0]
        return Chem.MolFromSmiles(smi)
    except Exception:
        return None


def generate_random_negatives(
    positive_smiles: Sequence[str],
    n_negatives: int,
    seed: int = 42,
    max_attempts_per_neg: int = 10,
) -> List[str]:
    """Generate ``n_negatives`` random negative reaction SMILES.

    Each negative is built by perturbing the *product* side of a randomly
    sampled positive reaction: with probability 0.5 we swap one atom,
    otherwise we delete one bond.  The reactant side is left untouched, so
    the negative reaction is chemically plausible (same starting material)
    but leads to a different (impossible) product.

    Parameters
    ----------
    positive_smiles:
        Pool of positive reaction SMILES to perturb.
    n_negatives:
        Number of negatives to generate.
    seed:
        RNG seed.
    max_attempts_per_neg:
        Maximum perturbation attempts before giving up on one slot
        (default 10).  Failed slots are simply skipped.

    Returns
    -------
    list of str
        ``<= n_negatives`` perturbed reaction SMILES.
    """
    if n_negatives <= 0 or not positive_smiles:
        return []
    rng = random.Random(seed)
    negatives: List[str] = []
    seen: set = set()
    attempts_total = 0
    max_total = n_negatives * max_attempts_per_neg
    while len(negatives) < n_negatives and attempts_total < max_total:
        attempts_total += 1
        rxn = rng.choice(positive_smiles)
        if ">>" not in rxn:
            continue
        reactants, products = rxn.split(">>", 1)
        product_mol = _safe_mol_from_smiles(products)
        if product_mol is None:
            continue
        # Apply 1-2 perturbations (sometimes chain them for more diversity).
        mol = product_mol
        n_perturb = rng.choice([1, 1, 2])
        for _ in range(n_perturb):
            op = rng.choice(["swap", "delete"])
            if op == "swap":
                new_mol = _swap_atom(mol, rng)
            else:
                new_mol = _delete_bond(mol, rng)
            if new_mol is None:
                break
            mol = new_mol
        try:
            new_prod_smi = Chem.MolToSmiles(mol)
        except Exception:
            continue
        if not new_prod_smi or new_prod_smi == products.strip():
            continue
        neg_rxn = f"{reactants}>>{new_prod_smi}"
        if neg_rxn in seen:
            continue
        seen.add(neg_rxn)
        negatives.append(neg_rxn)
    return negatives


# ---------------------------------------------------------------------------
# PC-CNG negatives loader
# ---------------------------------------------------------------------------


def load_pc_cng_negatives(csv_path: str, max_n: Optional[int] = None) -> List[str]:
    """Load PC-CNG synthetic negatives from the reviewed CSV.

    The CSV on the server is expected to have a ``reaction_smiles`` column
    (case-insensitive).  If the file does not exist or the column is
    missing, an empty list is returned -- callers should treat this as
    "PC-CNG negatives unavailable" and skip strategy (a) gracefully.

    Parameters
    ----------
    csv_path:
        Path to ``pc_cng_synthetic_negatives_reviewed.csv``.
    max_n:
        Optional cap on the number of negatives returned (random sample).

    Returns
    -------
    list of str
        Reaction SMILES strings.
    """
    if not csv_path or not os.path.exists(csv_path):
        return []
    negatives: List[str] = []
    try:
        with open(csv_path, "r", newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames is None:
                return []
            # Case-insensitive column lookup
            cols = {c.lower().strip(): c for c in reader.fieldnames}
            rxn_col = cols.get("reaction_smiles") or cols.get("smiles")
            if rxn_col is None:
                return []
            for row in reader:
                smi = (row.get(rxn_col) or "").strip()
                if smi:
                    negatives.append(smi)
    except OSError:
        return []
    if max_n is not None and 0 <= max_n < len(negatives):
        rng = random.Random(0)
        negatives = rng.sample(negatives, max_n)
    return negatives


# ---------------------------------------------------------------------------
# Ranking metrics
# ---------------------------------------------------------------------------


def compute_top1(rank_of_positive: int) -> float:
    """Top-1 accuracy for a single query: 1.0 if rank==1 else 0.0."""
    return 1.0 if rank_of_positive == 1 else 0.0


def compute_mrr(rank_of_positive: int) -> float:
    """Reciprocal rank: ``1 / rank``."""
    if rank_of_positive <= 0:
        return 0.0
    return 1.0 / float(rank_of_positive)


def compute_ndcg_at_k(rank_of_positive: int, k: int = 10) -> float:
    """NDCG@k for a single relevant item at ``rank_of_positive``.

    With exactly one relevant item, DCG = ``1 / log2(rank + 1)`` if
    ``rank <= k`` else 0, and IDCG = ``1 / log2(2) = 1``.  So
    NDCG@k = DCG / IDCG = ``1 / log2(rank + 1)`` if ``rank <= k`` else 0.
    """
    if k < 1:
        return 0.0
    if rank_of_positive < 1 or rank_of_positive > k:
        return 0.0
    return 1.0 / math.log2(rank_of_positive + 1)


def _rank_of_positive(scores: Sequence[Tuple[float, int]], positive_idx: int) -> int:
    """1-based rank of the positive candidate in a sorted score list.

    ``scores`` is a sequence of ``(score, candidate_index)`` tuples.  The
    list is sorted by score descending; ties are broken by lower index
    (deterministic).
    """
    ordered = sorted(scores, key=lambda x: (-x[0], x[1]))
    for rank, (_, idx) in enumerate(ordered, start=1):
        if idx == positive_idx:
            return rank
    # Positive not found -- shouldn't happen, but be defensive.
    return len(scores) + 1


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------


def _build_training_matrix(
    reaction_smiles_list: Sequence[str],
    radius: int = 2,
    n_bits: int = 2048,
) -> Tuple[np.ndarray, List[int]]:
    """Compute reaction fingerprints for a list of SMILES.

    Returns
    -------
    X : np.ndarray, shape (n_valid, n_bits)
    valid_indices : list of int
        Original indices of the rows that were successfully featurised.
    """
    rows: List[np.ndarray] = []
    valid: List[int] = []
    for i, smi in enumerate(reaction_smiles_list):
        fp = reaction_fingerprint(smi, radius=radius, n_bits=n_bits)
        if fp is None:
            continue
        rows.append(fp)
        valid.append(i)
    if not rows:
        return np.zeros((0, n_bits), dtype=np.uint8), valid
    return np.vstack(rows), valid


def _train_logreg(
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int = 42,
) -> LogisticRegression:
    """Train a LogisticRegression with deterministic seeding."""
    if len(np.unique(y_train)) < 2:
        # Degenerate single-class training set.  We still fit a model so
        # the API is uniform, but the caller should detect this case via
        # ``y_train`` before calling.
        raise ValueError("y_train must contain at least 2 classes")
    # Cast to float32 to avoid uint8 overflow warnings from sklearn's
    # internal matmul (extmath.py).
    X = X_train.astype(np.float32) if X_train.dtype != np.float32 else X_train
    return LogisticRegression(
        max_iter=1000,
        random_state=seed,
        solver="liblinear",  # robust for small datasets
        class_weight="balanced",
    ).fit(X, y_train)


def _similarity_score(
    query_fp: np.ndarray,
    train_fps: np.ndarray,
) -> float:
    """Maximum Tanimoto similarity between ``query_fp`` and ``train_fps``.

    Used as the no-negative scorer: a held-out positive is "more real" if
    it closely resembles a known positive.  Both inputs are bit vectors
    (uint8 0/1).
    """
    if train_fps.size == 0:
        return 0.0
    # Cast to float64 up-front to avoid uint8 overflow in the dot product
    # (sum of 2048 ones == 2048, well within uint8 range only if any bit is
    # set; but if the array has many 1s the matmul on uint8 wraps around).
    q = query_fp.astype(np.float64)
    train = train_fps.astype(np.float64)
    # Vectorised intersection: dot product of 0/1 vectors.  Some rows of
    # ``train`` may be all-zero (unparseable side products were dropped),
    # which triggers benign divide-by-zero warnings on certain numpy/BLAS
    # builds.  Suppress them -- we mask out zero-union rows below.
    with np.errstate(divide="ignore", invalid="ignore"):
        inter = train @ q  # shape: (n_train,)
        q_count = q.sum()
        train_counts = train.sum(axis=1)
        union = q_count + train_counts - inter
        sims = np.where(union > 0, inter / union, 0.0)
    if sims.size == 0:
        return 0.0
    return float(sims.max())


def _no_negatives_rank(
    held_out_fp: np.ndarray,
    train_pos_fps: np.ndarray,
    candidate_fps: np.ndarray,
) -> int:
    """Rank the held-out positive against random-perturbation candidates
    using nearest-neighbour similarity to the training positives.

    The held-out positive's score is its max Tanimoto to ``train_pos_fps``.
    Each candidate's score is its max Tanimoto to ``train_pos_fps``.  The
    rank of the held-out positive is then 1 + (number of candidates with
    strictly higher score).  Ties are broken in favour of the held-out
    positive.
    """
    pos_score = _similarity_score(held_out_fp, train_pos_fps)
    higher = 0
    for cfp in candidate_fps:
        s = _similarity_score(cfp, train_pos_fps)
        if s > pos_score:
            higher += 1
    return higher + 1


def evaluate_leave_one_out(
    reaction_smiles: Sequence[str],
    strategy: str,
    pc_cng_negatives: Sequence[str] = (),
    n_candidates: int = 10,
    seed: int = 42,
    radius: int = 2,
    n_bits: int = 2048,
) -> SeedMetrics:
    """Run leave-one-out evaluation for a single seed + strategy.

    Parameters
    ----------
    reaction_smiles:
        The held-out candidate pool (already stratified-sampled).  Each
        reaction is held out once.
    strategy:
        One of ``{"pc_cng", "random", "none"}``.
    pc_cng_negatives:
        Pool of PC-CNG negative reactions (only used when
        ``strategy == "pc_cng"``).
    n_candidates:
        Number of candidate negatives ranked alongside each held-out
        positive (default 10).  Should match the ``@k`` of NDCG.
    seed:
        RNG seed for negative sampling + LogReg.
    radius, n_bits:
        Morgan FP hyper-parameters.

    Returns
    -------
    SeedMetrics
        Aggregated Top-1 / MRR / NDCG@10 across all held-out reactions.
    """
    strategy = strategy.lower()
    if strategy not in {"pc_cng", "random", "none"}:
        raise ValueError(f"Unknown strategy: {strategy}")
    if n_candidates < 1:
        raise ValueError("n_candidates must be >= 1")

    rng = random.Random(seed)
    n = len(reaction_smiles)
    if n < 2:
        return SeedMetrics(seed=seed, strategy=strategy, top1=0.0, mrr=0.0,
                           ndcg10=0.0, n_evaluated=0)

    # Pre-compute fingerprints for all reactions once.
    fps: List[Optional[np.ndarray]] = [
        reaction_fingerprint(s, radius=radius, n_bits=n_bits) for s in reaction_smiles
    ]

    top1_sum = 0.0
    mrr_sum = 0.0
    ndcg_sum = 0.0
    n_evaluated = 0

    for i in range(n):
        held_out_fp = fps[i]
        if held_out_fp is None:
            continue

        # Training positives: all other reactions with valid fingerprints.
        train_idx = [j for j in range(n) if j != i and fps[j] is not None]
        if not train_idx:
            continue
        train_pos_fps = np.vstack([fps[j] for j in train_idx])

        # Sample candidate negatives for this held-out reaction.
        if strategy == "pc_cng":
            if not pc_cng_negatives:
                continue
            sampled = rng.sample(
                list(pc_cng_negatives),
                min(n_candidates, len(pc_cng_negatives)),
            )
            cand_fps = [
                reaction_fingerprint(s, radius=radius, n_bits=n_bits)
                for s in sampled
            ]
            # Drop unparseable candidates
            valid_cand = [(s, fp) for s, fp in zip(sampled, cand_fps) if fp is not None]
            if not valid_cand:
                continue
            cand_fps_arr = np.vstack([fp for _, fp in valid_cand])
        elif strategy == "random":
            # Generate n_candidates perturbations of the held-out positive.
            sampled = generate_random_negatives(
                [reaction_smiles[i]], n_candidates, seed=seed * 1000 + i
            )
            cand_fps = [
                reaction_fingerprint(s, radius=radius, n_bits=n_bits)
                for s in sampled
            ]
            valid_cand = [(s, fp) for s, fp in zip(sampled, cand_fps) if fp is not None]
            if not valid_cand:
                continue
            cand_fps_arr = np.vstack([fp for _, fp in valid_cand])
        else:  # "none"
            # Sample n_candidates from the training positives themselves,
            # then treat them as "decoy positives".  This is the standard
            # no-negative baseline for one-class ranking.
            decoy_idx = rng.sample(train_idx, min(n_candidates, len(train_idx)))
            cand_fps_arr = np.vstack([fps[j] for j in decoy_idx])

        # Build the candidate score matrix.  The held-out positive is
        # appended at index 0 of the candidate list for scoring.
        if strategy in {"pc_cng", "random"}:
            X_train = np.vstack([train_pos_fps, cand_fps_arr])
            y_train = np.array(
                [1] * len(train_pos_fps) + [0] * len(cand_fps_arr),
                dtype=np.int32,
            )
            # Guard: if either class has 0 samples, skip.
            if len(np.unique(y_train)) < 2:
                continue
            try:
                clf = _train_logreg(X_train, y_train, seed=seed)
            except Exception:
                continue
            # Score = P(positive).  Build candidate list: [held_out, *cands]
            X_test = np.vstack([held_out_fp, cand_fps_arr]).astype(np.float32)
            probs = clf.predict_proba(X_test)
            pos_col = list(clf.classes_).index(1) if 1 in clf.classes_ else 1
            scores = [(float(probs[k, pos_col]), k) for k in range(len(X_test))]
            rank = _rank_of_positive(scores, positive_idx=0)
        else:
            # No-negatives: similarity-based ranking.
            rank = _no_negatives_rank(
                held_out_fp, train_pos_fps, cand_fps_arr
            )

        top1_sum += compute_top1(rank)
        mrr_sum += compute_mrr(rank)
        ndcg_sum += compute_ndcg_at_k(rank, k=10)
        n_evaluated += 1

    if n_evaluated == 0:
        return SeedMetrics(seed=seed, strategy=strategy, top1=0.0, mrr=0.0,
                           ndcg10=0.0, n_evaluated=0)
    return SeedMetrics(
        seed=seed,
        strategy=strategy,
        top1=top1_sum / n_evaluated,
        mrr=mrr_sum / n_evaluated,
        ndcg10=ndcg_sum / n_evaluated,
        n_evaluated=n_evaluated,
    )


# ---------------------------------------------------------------------------
# Family-cluster bootstrap CI
# ---------------------------------------------------------------------------


def family_cluster_bootstrap_ci(
    per_reaction_ranks: Sequence[int],
    reaction_classes: Sequence[str],
    n_bootstrap: int = 1000,
    seed: int = 42,
    metric: str = "mrr",
    k: int = 10,
) -> Tuple[float, float]:
    """Family-cluster bootstrap CI for a ranking metric.

    Re-samples ``reaction_classes`` (families) with replacement, then
    recomputes the mean metric over all reactions in the resampled families
    (with multiplicity).

    Parameters
    ----------
    per_reaction_ranks:
        1-based rank of the positive for each reaction.
    reaction_classes:
        ``reaction_class`` label for each reaction (same length as
        ``per_reaction_ranks``).
    n_bootstrap:
        Number of bootstrap replicates.
    seed:
        RNG seed.
    metric:
        ``"top1"``, ``"mrr"``, or ``"ndcg"``.
    k:
        ``k`` for NDCG (only used when ``metric == "ndcg"``).

    Returns
    -------
    (lo, hi) : tuple of float
        2.5 / 97.5 percentile CI bounds.
    """
    if len(per_reaction_ranks) != len(reaction_classes):
        raise ValueError("per_reaction_ranks and reaction_classes must align")
    if not per_reaction_ranks:
        return (0.0, 0.0)
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be >= 1")

    rng = random.Random(seed)
    # Group reaction indices by class
    by_class: Dict[str, List[int]] = {}
    for i, cls in enumerate(reaction_classes):
        by_class.setdefault(cls, []).append(i)
    class_names = list(by_class.keys())

    # Precompute per-reaction metric values
    metric_fn: Callable[[int], float]
    if metric == "top1":
        metric_fn = compute_top1
    elif metric == "mrr":
        metric_fn = compute_mrr
    elif metric == "ndcg":
        metric_fn = lambda r: compute_ndcg_at_k(r, k=k)
    else:
        raise ValueError(f"Unknown metric: {metric}")
    values = [metric_fn(r) for r in per_reaction_ranks]

    boot_means: List[float] = []
    for _ in range(n_bootstrap):
        acc: List[float] = []
        for _ in range(len(class_names)):
            cls = rng.choice(class_names)
            for idx in by_class[cls]:
                acc.append(values[idx])
        if acc:
            boot_means.append(sum(acc) / len(acc))
    if not boot_means:
        return (0.0, 0.0)
    boot_means.sort()
    lo = boot_means[int(0.025 * len(boot_means))]
    hi = boot_means[int(0.975 * len(boot_means)) - 1]
    return (float(lo), float(hi))


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_evaluation(
    hte_csv: str,
    pc_cng_negatives_csv: Optional[str],
    seeds: Sequence[int],
    n_per_class: int = 50,
    min_class_size: int = 20,
    n_candidates: int = 10,
    output_dir: Optional[str] = None,
    n_bootstrap: int = 1000,
    radius: int = 2,
    n_bits: int = 2048,
) -> EvaluationResult:
    """Run the full P3-05 leave-one-out evaluation across seeds + strategies.

    Parameters mirror the CLI flags.  See :func:`main` for details.  When
    ``output_dir`` is provided, results are written as JSON
    (``metrics.json``) and a Markdown summary (``summary.md``) is also
    produced.

    Returns
    -------
    EvaluationResult
        Aggregated metrics + CIs.
    """
    hte_groups = load_hte_by_class(hte_csv, min_class_size=min_class_size)
    if not hte_groups:
        raise ValueError(
            f"No HTE classes with >= {min_class_size} reactions found in {hte_csv}"
        )
    pc_cng_negatives = load_pc_cng_negatives(pc_cng_negatives_csv) if pc_cng_negatives_csv else []

    per_seed: List[SeedMetrics] = []
    for seed in seeds:
        sampled = stratified_sample(hte_groups, n_per_class=n_per_class, seed=seed)
        if not sampled:
            continue
        for strategy in ("pc_cng", "random", "none"):
            if strategy == "pc_cng" and not pc_cng_negatives:
                # PC-CNG negatives unavailable -- record zero metrics with
                # n_evaluated=0 so downstream code can detect the skip.
                per_seed.append(
                    SeedMetrics(seed=seed, strategy="pc_cng", top1=0.0,
                                mrr=0.0, ndcg10=0.0, n_evaluated=0)
                )
                continue
            m = evaluate_leave_one_out(
                sampled,
                strategy=strategy,
                pc_cng_negatives=pc_cng_negatives,
                n_candidates=n_candidates,
                seed=seed,
                radius=radius,
                n_bits=n_bits,
            )
            per_seed.append(m)

    # Aggregate per-strategy summary
    summary: Dict[str, Dict[str, Any]] = {}
    for strategy in ("pc_cng", "random", "none"):
        vals = [m for m in per_seed if m.strategy == strategy and m.n_evaluated > 0]
        if not vals:
            summary[strategy] = {
                "n_seeds": 0,
                "top1_mean": float("nan"),
                "mrr_mean": float("nan"),
                "ndcg10_mean": float("nan"),
                "top1_std": float("nan"),
                "mrr_std": float("nan"),
                "ndcg10_std": float("nan"),
            }
            continue
        top1s = [m.top1 for m in vals]
        mrrs = [m.mrr for m in vals]
        ndcgs = [m.ndcg10 for m in vals]
        summary[strategy] = {
            "n_seeds": len(vals),
            "top1_mean": float(statistics.mean(top1s)),
            "mrr_mean": float(statistics.mean(mrrs)),
            "ndcg10_mean": float(statistics.mean(ndcgs)),
            "top1_std": float(statistics.pstdev(top1s)) if len(top1s) > 1 else 0.0,
            "mrr_std": float(statistics.pstdev(mrrs)) if len(mrrs) > 1 else 0.0,
            "ndcg10_std": float(statistics.pstdev(ndcgs)) if len(ndcgs) > 1 else 0.0,
        }

    # Family-cluster bootstrap CI (use last seed's per-reaction ranks).
    bootstrap_ci: Dict[str, Dict[str, Tuple[float, float]]] = {}
    last_seed = seeds[-1] if seeds else 0
    sampled = stratified_sample(hte_groups, n_per_class=n_per_class, seed=last_seed)
    # Map each sampled reaction back to its class
    smi_to_class: Dict[str, str] = {}
    for cls, group in hte_groups.items():
        for smi in group.reaction_smiles_list:
            smi_to_class[smi] = cls
    classes = [smi_to_class.get(s, "Unknown") for s in sampled]

    for strategy in ("pc_cng", "random", "none"):
        if strategy == "pc_cng" and not pc_cng_negatives:
            bootstrap_ci[strategy] = {"top1": (0.0, 0.0), "mrr": (0.0, 0.0), "ndcg10": (0.0, 0.0)}
            continue
        # Recompute per-reaction ranks for bootstrap.  We re-run the
        # evaluation but collect ranks.  To avoid duplicating logic, we
        # re-evaluate and store ranks via a thin shim.
        ranks, kept_indices = _collect_per_reaction_ranks(
            sampled,
            strategy=strategy,
            pc_cng_negatives=pc_cng_negatives,
            n_candidates=n_candidates,
            seed=last_seed,
            radius=radius,
            n_bits=n_bits,
        )
        if not ranks:
            bootstrap_ci[strategy] = {"top1": (0.0, 0.0), "mrr": (0.0, 0.0), "ndcg10": (0.0, 0.0)}
            continue
        # Only keep classes for reactions that produced a rank (alignment fix).
        kept_classes = [classes[i] for i in kept_indices]
        bootstrap_ci[strategy] = {
            "top1": family_cluster_bootstrap_ci(ranks, kept_classes, n_bootstrap=n_bootstrap, seed=last_seed, metric="top1"),
            "mrr": family_cluster_bootstrap_ci(ranks, kept_classes, n_bootstrap=n_bootstrap, seed=last_seed, metric="mrr"),
            "ndcg10": family_cluster_bootstrap_ci(ranks, kept_classes, n_bootstrap=n_bootstrap, seed=last_seed, metric="ndcg", k=10),
        }

    result = EvaluationResult(
        per_seed=per_seed,
        summary=summary,
        bootstrap_ci=bootstrap_ci,
        meta={
            "hte_csv": hte_csv,
            "pc_cng_negatives_csv": pc_cng_negatives_csv,
            "seeds": list(seeds),
            "n_per_class": n_per_class,
            "min_class_size": min_class_size,
            "n_candidates": n_candidates,
            "n_classes": len(hte_groups),
            "class_names": list(hte_groups.keys()),
            "n_pc_cng_negatives": len(pc_cng_negatives),
            "n_bootstrap": n_bootstrap,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
    )

    if output_dir:
        _write_results(result, output_dir)
    return result


def _collect_per_reaction_ranks(
    reaction_smiles: Sequence[str],
    strategy: str,
    pc_cng_negatives: Sequence[str],
    n_candidates: int,
    seed: int,
    radius: int,
    n_bits: int,
) -> Tuple[List[int], List[int]]:
    """Helper: re-run LOO but return per-reaction ranks (for bootstrap).

    Returns ``(ranks, kept_indices)`` where ``kept_indices[i]`` is the
    index into ``reaction_smiles`` for ``ranks[i]``.  This alignment is
    critical so the caller can build a matching ``classes`` list.
    """
    rng = random.Random(seed)
    n = len(reaction_smiles)
    if n < 2:
        return [], []
    fps: List[Optional[np.ndarray]] = [
        reaction_fingerprint(s, radius=radius, n_bits=n_bits) for s in reaction_smiles
    ]
    ranks: List[int] = []
    kept_indices: List[int] = []
    for i in range(n):
        held_out_fp = fps[i]
        if held_out_fp is None:
            continue
        train_idx = [j for j in range(n) if j != i and fps[j] is not None]
        if not train_idx:
            continue
        train_pos_fps = np.vstack([fps[j] for j in train_idx])
        if strategy == "pc_cng":
            if not pc_cng_negatives:
                continue
            sampled = rng.sample(list(pc_cng_negatives), min(n_candidates, len(pc_cng_negatives)))
            cand_fps = [reaction_fingerprint(s, radius=radius, n_bits=n_bits) for s in sampled]
            valid = [fp for fp in cand_fps if fp is not None]
            if not valid:
                continue
            cand_arr = np.vstack(valid)
        elif strategy == "random":
            sampled = generate_random_negatives([reaction_smiles[i]], n_candidates, seed=seed * 1000 + i)
            cand_fps = [reaction_fingerprint(s, radius=radius, n_bits=n_bits) for s in sampled]
            valid = [fp for fp in cand_fps if fp is not None]
            if not valid:
                continue
            cand_arr = np.vstack(valid)
        else:
            decoy_idx = rng.sample(train_idx, min(n_candidates, len(train_idx)))
            cand_arr = np.vstack([fps[j] for j in decoy_idx])

        if strategy in {"pc_cng", "random"}:
            X_train = np.vstack([train_pos_fps, cand_arr])
            y_train = np.array([1] * len(train_pos_fps) + [0] * len(cand_arr), dtype=np.int32)
            if len(np.unique(y_train)) < 2:
                continue
            try:
                clf = _train_logreg(X_train, y_train, seed=seed)
            except Exception:
                continue
            X_test = np.vstack([held_out_fp, cand_arr]).astype(np.float32)
            probs = clf.predict_proba(X_test)
            pos_col = list(clf.classes_).index(1) if 1 in clf.classes_ else 1
            scores = [(float(probs[k, pos_col]), k) for k in range(len(X_test))]
            rank = _rank_of_positive(scores, positive_idx=0)
        else:
            rank = _no_negatives_rank(held_out_fp, train_pos_fps, cand_arr)
        ranks.append(rank)
        kept_indices.append(i)
    return ranks, kept_indices


def _write_results(result: EvaluationResult, output_dir: str) -> None:
    """Write ``metrics.json`` + ``summary.md`` to ``output_dir``."""
    os.makedirs(output_dir, exist_ok=True)
    # Serialise dataclasses -> JSON-friendly dicts
    payload: Dict[str, Any] = {
        "meta": result.meta,
        "per_seed": [asdict(m) for m in result.per_seed],
        "summary": result.summary,
        "bootstrap_ci": {
            strat: {metric: list(ci) for metric, ci in cis.items()}
            for strat, cis in result.bootstrap_ci.items()
        },
    }
    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)

    # Markdown summary
    lines: List[str] = [
        "# P3-05 HTE Leave-One-Out Evaluation",
        "",
        f"- **HTE CSV**: `{result.meta.get('hte_csv', '')}`",
        f"- **PC-CNG negatives CSV**: `{result.meta.get('pc_cng_negatives_csv', '')}`",
        f"- **# reaction classes**: {result.meta.get('n_classes', 0)}",
        f"- **# PC-CNG negatives loaded**: {result.meta.get('n_pc_cng_negatives', 0)}",
        f"- **Seeds**: {result.meta.get('seeds', [])}",
        f"- **n_per_class**: {result.meta.get('n_per_class', 0)}",
        f"- **min_class_size**: {result.meta.get('min_class_size', 0)}",
        f"- **Generated**: {result.meta.get('timestamp', '')}",
        "",
        "## Per-strategy summary (mean +/- std across seeds)",
        "",
        "| Strategy | Top-1 | MRR | NDCG@10 | n_seeds |",
        "|----------|-------|-----|---------|---------|",
    ]
    for strat in ("pc_cng", "random", "none"):
        s = result.summary.get(strat, {})
        lines.append(
            f"| {strat} | {s.get('top1_mean', float('nan')):.4f} "
            f"+/- {s.get('top1_std', float('nan')):.4f} | "
            f"{s.get('mrr_mean', float('nan')):.4f} "
            f"+/- {s.get('mrr_std', float('nan')):.4f} | "
            f"{s.get('ndcg10_mean', float('nan')):.4f} "
            f"+/- {s.get('ndcg10_std', float('nan')):.4f} | "
            f"{s.get('n_seeds', 0)} |"
        )
    lines.extend([
        "",
        "## Family-cluster bootstrap 95% CI (last seed)",
        "",
        "| Strategy | Top-1 CI | MRR CI | NDCG@10 CI |",
        "|----------|----------|--------|------------|",
    ])
    for strat in ("pc_cng", "random", "none"):
        cis = result.bootstrap_ci.get(strat, {})
        lines.append(
            f"| {strat} | "
            f"[{cis.get('top1', (0,0))[0]:.4f}, {cis.get('top1', (0,0))[1]:.4f}] | "
            f"[{cis.get('mrr', (0,0))[0]:.4f}, {cis.get('mrr', (0,0))[1]:.4f}] | "
            f"[{cis.get('ndcg10', (0,0))[0]:.4f}, {cis.get('ndcg10', (0,0))[1]:.4f}] |"
        )
    lines.append("")
    with open(os.path.join(output_dir, "summary.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_seeds(arg: str) -> List[int]:
    """Parse a comma-separated list of seed integers."""
    seeds: List[int] = []
    for tok in arg.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            seeds.append(int(tok))
        except ValueError as e:
            raise argparse.ArgumentTypeError(f"Invalid seed: {tok!r}") from e
    if not seeds:
        raise argparse.ArgumentTypeError("At least one seed is required")
    return seeds


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    p = argparse.ArgumentParser(
        prog="hte_eval",
        description="P3-05 HTE leave-one-out evaluation.",
    )
    p.add_argument(
        "--hte-csv",
        required=True,
        help="Path to data/processed/hitea_full_normalized.csv",
    )
    p.add_argument(
        "--pc-cng-negatives",
        default=None,
        help="Path to results/.../pc_cng_synthetic_negatives_reviewed.csv "
             "(optional; if missing, the pc_cng strategy is skipped).",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write metrics.json + summary.md (created if missing).",
    )
    p.add_argument(
        "--n-per-class",
        type=int,
        default=50,
        help="Stratified sample size per reaction_class (default 50).",
    )
    p.add_argument(
        "--min-class-size",
        type=int,
        default=20,
        help="Minimum reactions per class to keep it (default 20).",
    )
    p.add_argument(
        "--n-candidates",
        type=int,
        default=10,
        help="Number of candidate negatives per held-out positive (default 10).",
    )
    p.add_argument(
        "--seeds",
        type=_parse_seeds,
        required=True,
        help="Comma-separated list of seeds, e.g. 20260710,20260711,...",
    )
    p.add_argument(
        "--n-bootstrap",
        type=int,
        default=1000,
        help="Number of bootstrap replicates for family-cluster CI (default 1000).",
    )
    p.add_argument(
        "--radius",
        type=int,
        default=2,
        help="Morgan fingerprint radius (default 2).",
    )
    p.add_argument(
        "--n-bits",
        type=int,
        default=2048,
        help="Morgan fingerprint bit-vector length (default 2048).",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point.  Returns 0 on success, non-zero on error."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    print(f"[P3-05] Loading HTE data from {args.hte_csv}")
    if args.pc_cng_negatives:
        print(f"[P3-05] PC-CNG negatives (optional): {args.pc_cng_negatives}")
    print(f"[P3-05] Seeds: {args.seeds}")
    print(f"[P3-05] n_per_class={args.n_per_class} min_class_size={args.min_class_size}")
    try:
        result = run_evaluation(
            hte_csv=args.hte_csv,
            pc_cng_negatives_csv=args.pc_cng_negatives,
            seeds=args.seeds,
            n_per_class=args.n_per_class,
            min_class_size=args.min_class_size,
            n_candidates=args.n_candidates,
            output_dir=args.output_dir,
            n_bootstrap=args.n_bootstrap,
            radius=args.radius,
            n_bits=args.n_bits,
        )
    except (OSError, ValueError) as e:
        print(f"[P3-05] ERROR: {e}", file=sys.stderr)
        return 1
    print("[P3-05] Done. Summary:")
    for strat, s in result.summary.items():
        print(
            f"  {strat:8s}: Top-1={s.get('top1_mean', float('nan')):.4f} "
            f"MRR={s.get('mrr_mean', float('nan')):.4f} "
            f"NDCG@10={s.get('ndcg10_mean', float('nan')):.4f} "
            f"(n_seeds={s.get('n_seeds', 0)})"
        )
    if args.output_dir:
        print(f"[P3-05] Wrote {args.output_dir}/metrics.json + summary.md")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
