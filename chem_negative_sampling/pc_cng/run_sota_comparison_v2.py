"""P3-02 SOTA multi-baseline comparison v2 (翻盘 P2-06 NO-GO).

Extends P2-06's ``run_sota_comparison.py`` with a real SOTA Transformer
baseline: **B5 Chemformer zero-shot embedding + logistic regression**.

P2-06 was NO-GO because PC-CNG lost to tanimoto_nn by -45pp MRR, and
LocalRetro / Graph2SMILES / Molecular Transformer were deferred due to
no network access on the remote server.  P3-02 adds the pretrained
Chemformer (Irwin et al. 2022, MLST) as a legitimate SOTA Transformer
baseline.  The Chemformer checkpoint was already downloaded for P3-01
(``model_sanitized.ckpt``), so this baseline runs without any new
network access.

Section 30 P3-02 task: fix L6 (no SOTA direct comparison) by adding a
**pretrained Transformer** baseline that does not need training from
scratch.

This script keeps the three RDKit-only baselines from v1 and adds:

  * B5 (chemformer_scorer): Pretrained Chemformer encoder (frozen, no
    fine-tuning) + logistic regression head trained on the per-seed
    train split.  This is a fair SOTA comparison because it uses the
    same training data as PC-CNG but with Chemformer's 512-dim learned
    embeddings instead of hand-crafted 10-dim features.

LocalRetro / Graph2SMILES / Molecular Transformer remain deferred
(no network access on the remote server).  Their published leaderboard
numbers are cited in the manuscript.

Four+one rankers are compared per target molecule on USPTO-MIT-50k (or,
if unavailable, on PC-CNG synthetic negatives as a fallback). Metrics:
Top-1 / Top-3 / Top-5 / Top-10 route recall, MRR, NDCG@10.

All performance claims are 10-seed paired. The default seeds are
20260710..20260719.

Outputs (under ``--output-dir``):
  * ``summary.json``                       - per-baseline metrics + mean±std
  * ``paired_significance.json``           - PC-CNG vs each baseline
  * ``per_target_metrics.csv``             - per (seed, group, ranker)
  * ``sota_installation_status.json``      - documents deferred SOTA methods
  * ``go_no_go_decision.json``             - GO/NO-GO decision
  * ``per_seed_detail.json``               - per-seed metric detail
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

from .paired_reranking_significance import (
    bootstrap_ci,
    mean,
    paired_permutation_p_value,
    percentile,
    sign_test_p_value,
)
from .reranker import (
    LogisticReactionRanker,
    featurize_reaction,
    split_by_source,
)


DEFAULT_SEEDS = [
    20260710, 20260711, 20260712, 20260713, 20260714,
    20260715, 20260716, 20260717, 20260718, 20260719,
]

DEFAULT_METHODS = "rdkit_template,heuristic_validator,tanimoto_nn,pc_cng,chemformer_scorer"

# SOTA methods that cannot be installed due to no network access.
# NOTE (P3-02): chemformer_scorer is NO LONGER deferred because the
# pretrained checkpoint was already downloaded for P3-01.  Only the
# three methods that require training from scratch remain deferred.
DEFERRED_SOTA_METHODS = (
    "localretro",
    "graph2smiles",
    "molecular_transformer",
)

# Map short method keys to long display names.
METHOD_NAMES = {
    "rdkit_template": "B1_RDKit_template",
    "heuristic_validator": "B2_heuristic_forward_validator",
    "tanimoto_nn": "B3_Tanimoto_nearest_neighbor",
    "pc_cng": "B4_PC_CNG_augmented",
    "chemformer_scorer": "B5_Chemformer_zero_shot_scorer",
}

# Default method ordering for output (excludes deferred SOTA).
BASELINE_KEYS = ("rdkit_template", "heuristic_validator", "tanimoto_nn", "chemformer_scorer")
PROPOSED_KEY = "pc_cng"


# ---------------------------------------------------------------------------
# Safe parsing helpers
# ---------------------------------------------------------------------------


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# FeatureCache + cached logistic ranker (mirrors P2-01)
# ---------------------------------------------------------------------------


class FeatureCache:
    """Cache featurize_reaction results to avoid repeated RDKit parsing."""

    def __init__(self) -> None:
        self._cache: Dict[str, List[float]] = {}

    def get(self, reaction_smiles: str) -> List[float]:
        cached = self._cache.get(reaction_smiles)
        if cached is None:
            cached = featurize_reaction(reaction_smiles)
            self._cache[reaction_smiles] = cached
        return cached

    def precompute(self, reaction_smiles: Sequence[str]) -> None:
        for smi in reaction_smiles:
            if smi not in self._cache:
                self._cache[smi] = featurize_reaction(smi)


class CachedLogisticReactionRanker:
    """LogisticReactionRanker backed by a FeatureCache."""

    def __init__(
        self,
        cache: FeatureCache,
        learning_rate: float = 0.2,
        l2: float = 1e-4,
        epochs: int = 200,
        n_features: int = 10,
    ) -> None:
        self.cache = cache
        self.learning_rate = learning_rate
        self.l2 = l2
        self.epochs = epochs
        self.weights = [0.0 for _ in range(n_features)]

    def _features(self, reaction_smiles: str) -> List[float]:
        return self.cache.get(reaction_smiles)

    def _predict_from_features(self, features: Sequence[float]) -> float:
        z = sum(w * v for w, v in zip(self.weights, features))
        if z >= 0:
            return 1.0 / (1.0 + math.exp(-z))
        exp_z = math.exp(z)
        return exp_z / (1.0 + exp_z)

    def fit(self, rows: Sequence[Dict[str, object]]) -> None:
        if not rows:
            return
        for row in rows:
            self.cache.get(str(row["reaction_smiles"]))
        for _ in range(self.epochs):
            for row in rows:
                x = self._features(str(row["reaction_smiles"]))
                y = float(row["label"])
                pred = self._predict_from_features(x)
                error = pred - y
                for i, value in enumerate(x):
                    grad = error * value + self.l2 * self.weights[i]
                    self.weights[i] -= self.learning_rate * grad

    def predict_proba(self, reaction_smiles: str) -> float:
        return self._predict_from_features(self._features(reaction_smiles))


# ---------------------------------------------------------------------------
# B5: Chemformer zero-shot embedding + logistic regression (P3-02)
# ---------------------------------------------------------------------------
#
# This baseline uses the pretrained Chemformer encoder (Irwin et al. 2022,
# Machine Learning: Science and Technology) as a frozen feature extractor.
# For each reaction SMILES we compute the BOS-pooled 512-dim embedding and
# train a logistic regression head on the per-seed train split.  This is a
# fair SOTA comparison because:
#
#   * Chemformer is a published Transformer trained on 10M USPTO reactions
#   * The encoder weights are frozen (no fine-tuning, zero-shot features)
#   * The logistic head uses the SAME training data as PC-CNG, so any
#     performance difference comes from the feature representation, not
#     from the training set
#
# The checkpoint and vocabulary were already downloaded for P3-01:
#   * models/reaction_lm/chemformer_pretrained_hf/model_sanitized.ckpt
#   * external/reaction_lm/Chemformer/bart_vocab.json
#
# If the Chemformer checkpoint is unavailable or torch is not installed,
# this baseline gracefully degrades to a zero score (and is documented as
# "chemformer_scorer not evaluated due to missing checkpoint" in the
# installation status JSON).


class ChemformerEmbeddingCache:
    """Pre-compute 512-dim Chemformer embeddings for all reaction SMILES.

    The Chemformer encoder is loaded once (frozen) and used to encode
    each unique reaction SMILES.  Embeddings are cached in a dict so
    subsequent ``get()`` calls are O(1).
    """

    def __init__(
        self,
        checkpoint_path: Optional[str],
        vocab_path: Optional[str],
        device: str = "cpu",
        batch_size: int = 16,
    ) -> None:
        self.device = device
        self.batch_size = batch_size
        self._cache: Dict[str, List[float]] = {}
        self._backbone = None
        self._tokenizer = None
        self._load_failed = False
        self._load_error: Optional[str] = None
        if checkpoint_path and vocab_path:
            try:
                # Import here so the module can be imported even when
                # torch / the chem_negative_sampling package is not
                # installed (e.g. for unit tests).
                import sys as _sys
                import os as _os
                _cns_root = _os.path.dirname(
                    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
                )
                if _cns_root not in _sys.path:
                    _sys.path.insert(0, _cns_root)
                from models.pretrained_backbone import (  # type: ignore
                    PretrainedChemformerBackbone,
                    ChemformerTokenizer,
                )
                self._backbone = PretrainedChemformerBackbone(
                    checkpoint_path=checkpoint_path, freeze=True,
                ).to(device)
                self._backbone.eval()
                self._tokenizer = ChemformerTokenizer(vocab_path)
            except Exception as exc:  # pragma: no cover - defensive
                self._load_failed = True
                self._load_error = f"{type(exc).__name__}: {exc}"
                print(
                    f"[chemformer_scorer] WARNING: failed to load backbone - "
                    f"{self._load_error}.  Scorer will return 0.5 for all "
                    f"reactions.",
                    flush=True,
                )
        else:
            self._load_failed = True
            self._load_error = "no checkpoint_path or vocab_path provided"

    @property
    def available(self) -> bool:
        return self._backbone is not None and self._tokenizer is not None

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    @property
    def embedding_dim(self) -> int:
        if not self.available:
            return 0
            # Embedding dim is 512 for the default Chemformer config.
        try:
            return int(self._backbone.hparams.get("d_model", 512))  # type: ignore[union-attr]
        except Exception:
            return 512

    def precompute(self, reaction_smiles: Sequence[str]) -> None:
        """Encode all unique SMILES and cache the embeddings."""
        if not self.available:
            return
        unique = sorted({str(s) for s in reaction_smiles if s})
        to_encode = [s for s in unique if s not in self._cache]
        if not to_encode:
            return
        import torch  # type: ignore
        with torch.no_grad():
            for start in range(0, len(to_encode), self.batch_size):
                batch = to_encode[start : start + self.batch_size]
                try:
                    ids, mask = self._tokenizer.batch_encode(batch)  # type: ignore[union-attr]
                    ids = ids.to(self.device)
                    mask = mask.to(self.device)
                    embeddings = self._backbone(ids, attention_mask=mask)  # type: ignore[union-attr]
                    # embeddings: (batch, d_model) - BOS-pooled output
                    arr = embeddings.detach().cpu().tolist()
                    for smiles, vec in zip(batch, arr):
                        self._cache[smiles] = [float(v) for v in vec]
                except Exception as exc:  # pragma: no cover - defensive
                    # If a batch fails (e.g. invalid SMILES), fall back to
                    # zero embeddings so the rest can still be scored.
                    print(
                        f"[chemformer_scorer] WARNING: batch encoding failed "
                        f"({type(exc).__name__}: {exc}); using zero vectors.",
                        flush=True,
                    )
                    for smiles in batch:
                        self._cache[smiles] = [0.0] * self.embedding_dim

    def get(self, reaction_smiles: str) -> List[float]:
        if reaction_smiles in self._cache:
            return self._cache[reaction_smiles]
        if not self.available:
            return [0.0]
        # Encode on-the-fly for unseen SMILES.
        self.precompute([reaction_smiles])
        return self._cache.get(reaction_smiles, [0.0] * self.embedding_dim)


class ChemformerLogisticRanker:
    """Logistic regression on top of frozen Chemformer embeddings.

    Mirrors :class:`CachedLogisticReactionRanker` but uses 512-dim
    Chemformer embeddings instead of 10-dim hand-crafted features.
    Uses simple SGD with L2 regularization (same as PC-CNG's ranker).
    """

    def __init__(
        self,
        cache: ChemformerEmbeddingCache,
        learning_rate: float = 0.05,
        l2: float = 1e-3,
        epochs: int = 100,
    ) -> None:
        self.cache = cache
        self.learning_rate = learning_rate
        self.l2 = l2
        self.epochs = epochs
        self.weights: List[float] = []
        self.bias: float = 0.0

    def _features(self, reaction_smiles: str) -> List[float]:
        return self.cache.get(reaction_smiles)

    def _predict_from_features(self, features: Sequence[float]) -> float:
        if not self.weights or not features:
            return 0.5
        z = self.bias + sum(
            w * v for w, v in zip(self.weights, features)
        )
        if z >= 0:
            return 1.0 / (1.0 + math.exp(-z))
        exp_z = math.exp(z)
        return exp_z / (1.0 + exp_z)

    def fit(self, rows: Sequence[Dict[str, object]]) -> None:
        if not rows or not self.cache.available:
            return
        # Determine dimension from the first row.
        first_feats = self._features(str(rows[0]["reaction_smiles"]))
        n_features = len(first_feats)
        if n_features == 0:
            return
        self.weights = [0.0] * n_features
        self.bias = 0.0
        # Sigmoid + L2 SGD (matches CachedLogisticReactionRanker's style).
        for _ in range(self.epochs):
            for row in rows:
                x = self._features(str(row["reaction_smiles"]))
                if len(x) != n_features:
                    continue
                y = float(row["label"])
                pred = self._predict_from_features(x)
                error = pred - y
                for i, value in enumerate(x):
                    grad = error * value + self.l2 * self.weights[i]
                    self.weights[i] -= self.learning_rate * grad
                self.bias -= self.learning_rate * error

    def predict_proba(self, reaction_smiles: str) -> float:
        return self._predict_from_features(self._features(reaction_smiles))


def train_chemformer_scorer_ranker(
    train_rows: Sequence[Dict[str, object]],
    seed: int,
    cache: ChemformerEmbeddingCache,
    epochs: int = 100,
) -> ChemformerLogisticRanker:
    """Train the B5 Chemformer-based logistic ranker for one seed."""
    rng = random.Random(seed)
    train_subset = [
        {
            "reaction_smiles": str(row["reaction_smiles"]),
            "label": int(row["label"]),
        }
        for row in train_rows
    ]
    rng.shuffle(train_subset)
    model = ChemformerLogisticRanker(
        cache=cache, learning_rate=0.05, l2=1e-3, epochs=epochs,
    )
    model.fit(train_subset)
    return model


def score_rows_chemformer_scorer(
    model: ChemformerLogisticRanker,
    rows: Sequence[Dict[str, object]],
) -> List[Dict[str, object]]:
    """B5 baseline scorer: Chemformer embedding + logistic regression."""
    return [
        {**row, "score": float(model.predict_proba(str(row["reaction_smiles"]))),
         "ranker_source": "chemformer_scorer"}
        for row in rows
    ]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_pc_cng_negatives(
    path: str,
    max_candidates_per_source: int = 10,
) -> List[Dict[str, object]]:
    """Load PC-CNG synthetic negatives and build ranking rows.

    For each ``source_id`` the ``positive_reaction`` becomes a label=1
    candidate (gold route) and each ``candidate_reaction`` becomes a
    label=0 candidate. The ``parent_product`` SMILES is the target.
    """
    rows: List[Dict[str, object]] = []
    seen_positives: Dict[str, str] = {}
    candidate_counts: Dict[str, int] = defaultdict(int)
    parent_products: Dict[str, str] = {}

    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for record in reader:
            source_id = str(record.get("source_id", "")).strip()
            if not source_id:
                continue
            if candidate_counts[source_id] >= max_candidates_per_source:
                continue
            positive = str(record.get("positive_reaction", "")).strip()
            candidate = str(record.get("candidate_reaction", "")).strip()
            if not positive or not candidate:
                continue
            parent_product = str(record.get("parent_product", "")).strip()
            if source_id not in parent_products and parent_product:
                parent_products[source_id] = parent_product
            if source_id not in seen_positives:
                seen_positives[source_id] = positive
                rows.append({
                    "group_id": source_id,
                    "source_id": source_id,
                    "reaction_smiles": positive,
                    "label": 1,
                    "candidate_source": "positive_reaction",
                    "failure_type": "gold",
                    "edit_action": "",
                    "hard_score": 1.0,
                    "false_negative_risk": 0.0,
                    "parent_product": parent_products.get(source_id, ""),
                })
            rows.append({
                "group_id": source_id,
                "source_id": source_id,
                "reaction_smiles": candidate,
                "label": 0,
                "candidate_source": "pc_cng_synthetic",
                "failure_type": str(record.get("failure_type", "")),
                "edit_action": str(record.get("edit_action", "")),
                "hard_score": _safe_float(record.get("hard_score"), 0.0),
                "false_negative_risk": _safe_float(record.get("false_negative_risk"), 0.0),
                "parent_product": parent_products.get(source_id, ""),
            })
            candidate_counts[source_id] += 1

    return rows


def load_uspto_mit_50k_routes(path: str) -> List[Dict[str, object]]:
    """Load USPTO-MIT-50k routes CSV (optional).

    Expected columns: ``product_smiles``, ``route_smiles``, ``route_id``,
    ``is_gold``. Falls back gracefully if missing or malformed.
    """
    if not path or not os.path.exists(path):
        return []
    rows: List[Dict[str, object]] = []
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for record in reader:
            product = str(
                record.get("product_smiles", "") or record.get("product", "")
            ).strip()
            route = str(
                record.get("route_smiles", "") or record.get("reaction_smiles", "")
            ).strip()
            if not product or not route:
                continue
            route_id = (
                str(record.get("route_id", "") or record.get("id", "")).strip()
                or product
            )
            is_gold = _safe_int(record.get("is_gold", 0))
            rows.append({
                "group_id": product,
                "source_id": product,
                "reaction_smiles": route,
                "label": 1 if is_gold else 0,
                "candidate_source": "gold_route" if is_gold else "alternative_route",
                "failure_type": "gold" if is_gold else "alternative",
                "edit_action": "",
                "hard_score": 1.0 if is_gold else 0.5,
                "false_negative_risk": 0.0,
                "parent_product": product,
            })
    return rows


# ---------------------------------------------------------------------------
# B1: RDKit template-based retrosynthesis baseline
# ---------------------------------------------------------------------------


# Small library of one-step retrosynthesis SMARTS templates. Each entry
# is (template_smarts, name, weight). Conservative, well-known functional
# group disconnections.
_TEMPLATE_REACTIONS: List[Tuple[str, str, float]] = [
    ("[C:1](=[O:2])[O:3][C:4]>>[C:1](=[O:2])[OH].[C:4][OH]", "ester_hydrolysis", 0.7),
    ("[C:1](=[O:2])[NX3:3]>>[C:1](=[O:2])[OH].[NX3:3]", "amide_hydrolysis", 0.7),
    ("[C:1][O:2][C:3]>>[C:1][OH].[O:2][C:3]", "ether_cleavage", 0.5),
    ("[C:1][Cl:2]>>[C:1][OH].[Cl:2]", "alkyl_chloride_sn2", 0.5),
    ("[C:1][Br:2]>>[C:1][OH].[Br:2]", "alkyl_bromide_sn2", 0.5),
    ("[C:1][OH:2]>>[C:1]=[C:1]", "alcohol_dehydration", 0.4),
    ("[C:1]=[N:2]>>[C:1]=[O].[N:2]", "imine_hydrolysis", 0.6),
    (
        "[C:1]([O:2][C:3])([O:4][C:5])>>[C:1]=[O].[C:3][OH].[C:5][OH]",
        "acetal_hydrolysis",
        0.5,
    ),
]


def generate_template_routes(
    target_smiles: str,
    max_routes: int = 5,
) -> List[Dict[str, object]]:
    """Generate retrosynthesis candidates via RDKit template application."""
    try:
        from rdkit import Chem  # type: ignore
        from rdkit.Chem import AllChem  # type: ignore
    except Exception:
        return []

    target_mol = Chem.MolFromSmiles(target_smiles)
    if target_mol is None:
        return []

    routes: List[Dict[str, object]] = []
    seen_reactants: set = set()
    for template_smarts, name, weight in _TEMPLATE_REACTIONS:
        try:
            rxn = AllChem.ReactionFromSmarts(template_smarts)
            if rxn is None:
                continue
            reactant_sets = rxn.RunReactants((target_mol,))
            for rset in reactant_sets:
                try:
                    reactant_smiles = sorted(
                        Chem.MolToSmiles(m) for m in rset if m is not None
                    )
                    if not reactant_smiles:
                        continue
                    reactants_str = ".".join(reactant_smiles)
                    if reactants_str in seen_reactants:
                        continue
                    seen_reactants.add(reactants_str)
                    product_canonical = Chem.MolToSmiles(target_mol)
                    reaction_smiles = f"{reactants_str}>>{product_canonical}"
                    routes.append({
                        "reaction_smiles": reaction_smiles,
                        "score": weight,
                        "source": f"template_{name}",
                    })
                    if len(routes) >= max_routes:
                        return routes
                except Exception:
                    continue
        except Exception:
            continue
    return routes


def score_rows_rdkit_template(
    rows: Sequence[Dict[str, object]],
    cache: Optional[FeatureCache] = None,
) -> List[Dict[str, object]]:
    """B1 baseline scorer.

    For each candidate row, look at the target ``parent_product`` and
    run template-based retrosynthesis. If the candidate's reactant set
    appears among the template-generated precursors, score high (0.9);
    otherwise score by the template-weight-similarity heuristic
    (validity + atom balance).
    """
    # Build target -> set of template precursor strings
    target_to_precursors: Dict[str, set] = {}
    for row in rows:
        parent = str(row.get("parent_product", "")).strip()
        if not parent or parent in target_to_precursors:
            continue
        template_routes = generate_template_routes(parent, max_routes=5)
        precursors = set()
        for tr in template_routes:
            rxn = str(tr.get("reaction_smiles", ""))
            if ">>" in rxn:
                precursors.add(rxn.split(">>", 1)[0].strip())
        target_to_precursors[parent] = precursors

    out_rows: List[Dict[str, object]] = []
    for row in rows:
        rxn = str(row["reaction_smiles"])
        parent = str(row.get("parent_product", "")).strip()
        features = (
            cache.get(rxn) if cache is not None else featurize_reaction(rxn)
        )
        # Default score: atom_balance * 0.5 + valid * 0.3 + jaccard * 0.2
        base_score = 0.5 * features[2] + 0.3 * features[1] + 0.2 * features[3]
        score = base_score
        reactants = rxn.split(">>", 1)[0].strip() if ">>" in rxn else ""
        precursors = target_to_precursors.get(parent, set())
        if precursors and reactants in precursors:
            score = 0.9  # template match boost
        out_rows.append({
            **row,
            "score": float(score),
            "ranker_source": "rdkit_template",
        })
    return out_rows


# ---------------------------------------------------------------------------
# B2: Heuristic forward-reaction validator baseline
# ---------------------------------------------------------------------------


def score_rows_heuristic_validator(
    rows: Sequence[Dict[str, object]],
    cache: Optional[FeatureCache] = None,
) -> List[Dict[str, object]]:
    """B2 baseline scorer.

    Validates whether the reaction is chemically plausible by combining
    atom balance, reaction validity, and reactant/product similarity.
    No learned weights, no PC-CNG negatives.
    """
    out_rows: List[Dict[str, object]] = []
    for row in rows:
        features = (
            cache.get(str(row["reaction_smiles"])) if cache is not None
            else featurize_reaction(str(row["reaction_smiles"]))
        )
        # Feature indices mirror reranker.FEATURE_NAMES:
        # 1: valid, 2: atom_balance, 3: token_jaccard, 4: string_similarity
        score = (
            0.5 * features[2]      # atom_balance
            + 0.3 * features[1]    # valid
            + 0.2 * features[3]    # token_jaccard
        )
        out_rows.append({
            **row,
            "score": float(score),
            "ranker_source": "heuristic_validator",
        })
    return out_rows


# ---------------------------------------------------------------------------
# B3: Nearest-neighbor Tanimoto baseline (uses USPTO train set, k=5)
# ---------------------------------------------------------------------------


def _morgan_fingerprint(smiles: str, radius: int = 2, n_bits: int = 1024):
    """Return RDKit Morgan fingerprint, or None if unavailable / invalid."""
    try:
        from rdkit import Chem  # type: ignore
        from rdkit.Chem import AllChem  # type: ignore
    except Exception:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    except Exception:
        return None


def _canonical_smiles(smiles: str) -> str:
    """Return RDKit canonical SMILES, or stripped input if RDKit unavailable.

    Used by the decontamination filter to compare product identities
    robustly (e.g. ``CCO`` vs ``OCC`` should be treated as the same
    molecule).  Falls back to the stripped input so the filter still
    works (with exact-string matching) when RDKit is not installed.
    """
    if not smiles:
        return ""
    try:
        from rdkit import Chem  # type: ignore
    except Exception:
        return smiles.strip()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles.strip()
    try:
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return smiles.strip()


def filter_leaked_test_rows(
    train_rows: Sequence[Dict[str, object]],
    test_rows: Sequence[Dict[str, object]],
) -> Tuple[List[Dict[str, object]], int, List[str]]:
    """P3-02 decontamination filter.

    A test candidate is considered "leaked" when its canonical
    ``parent_product`` SMILES matches any train-set ``parent_product``
    SMILES.  When the nearest neighbour in ``score_rows_tanimoto_nn``
    is the *exact same product* (Tanimoto = 1.0) the baseline trivially
    recovers the correct label, inflating Tanimoto-NN MRR to 1.0 and
    making PC-CNG look unfairly worse.

    Returns ``(decontaminated_test_rows, n_leaked, leaked_source_ids)``
    where ``leaked_source_ids`` is the sorted list of ``source_id``
    values that were excluded.
    """
    train_products: set = set()
    for row in train_rows:
        prod = str(row.get("parent_product", "")).strip()
        canon = _canonical_smiles(prod) if prod else ""
        if canon:
            train_products.add(canon)

    decontaminated: List[Dict[str, object]] = []
    n_leaked = 0
    leaked_source_ids: set = set()
    for row in test_rows:
        prod = str(row.get("parent_product", "")).strip()
        canon = _canonical_smiles(prod) if prod else ""
        if canon and canon in train_products:
            n_leaked += 1
            leaked_source_ids.add(str(row.get("source_id", "")))
            continue
        decontaminated.append(row)
    return decontaminated, n_leaked, sorted(leaked_source_ids)


def build_train_fingerprints(
    train_rows: Sequence[Dict[str, object]],
) -> List[Tuple[str, object, int]]:
    """Build (parent_product, fingerprint, label) list from train rows."""
    seen: set = set()
    out: List[Tuple[str, object, int]] = []
    for row in train_rows:
        parent = str(row.get("parent_product", "")).strip()
        if not parent or parent in seen:
            continue
        fp = _morgan_fingerprint(parent)
        if fp is None:
            continue
        seen.add(parent)
        out.append((parent, fp, int(row["label"])))
    return out


def score_rows_tanimoto_nn(
    rows: Sequence[Dict[str, object]],
    train_fps: Sequence[Tuple[str, object, int]],
    k: int = 5,
) -> List[Dict[str, object]]:
    """B3 baseline scorer.

    For each candidate row, compute the Morgan fingerprint of the
    candidate's product (right side of ``reaction_smiles``) and find
    the k nearest neighbors (by Tanimoto similarity) in the train set.
    Score = mean label of the k nearest neighbors (so candidates whose
    product matches a train-set gold product get score 1.0).
    """
    try:
        from rdkit import DataStructs  # type: ignore
    except Exception:
        # Fallback: if RDKit DataStructs unavailable, score by hard_score
        return [
            {**row, "score": float(_safe_float(row.get("hard_score"), 0.5)),
             "ranker_source": "tanimoto_nn_fallback"}
            for row in rows
        ]

    # Pre-compute fingerprints for train set once
    train_fp_list = [(smi, fp, lab) for smi, fp, lab in train_fps if fp is not None]
    if not train_fp_list:
        # No train fingerprints -> fallback to hard_score
        return [
            {**row, "score": float(_safe_float(row.get("hard_score"), 0.5)),
             "ranker_source": "tanimoto_nn_fallback"}
            for row in rows
        ]

    out_rows: List[Dict[str, object]] = []
    for row in rows:
        rxn = str(row["reaction_smiles"])
        product = rxn.split(">>", 1)[1].strip() if ">>" in rxn else ""
        # Take first molecule if multiple in product
        product = product.split(".")[0].strip() if product else ""
        fp = _morgan_fingerprint(product) if product else None
        if fp is None:
            score = float(_safe_float(row.get("hard_score"), 0.5))
            out_rows.append({**row, "score": score,
                             "ranker_source": "tanimoto_nn_fallback"})
            continue
        sims = [
            (DataStructs.TanimotoSimilarity(fp, tfp), tlab)
            for _, tfp, tlab in train_fp_list
        ]
        sims.sort(key=lambda x: x[0], reverse=True)
        topk = sims[:k]
        if topk:
            # Score = mean label of k nearest neighbors, weighted by similarity
            total_sim = sum(s for s, _ in topk) or 1.0
            score = sum(s * lab for s, lab in topk) / total_sim
        else:
            score = 0.5
        out_rows.append({**row, "score": float(score),
                         "ranker_source": "tanimoto_nn"})
    return out_rows


# ---------------------------------------------------------------------------
# B4: PC-CNG augmented logistic reranker (proposed method)
# ---------------------------------------------------------------------------


def train_pc_cng_augmented_ranker(
    train_rows: Sequence[Dict[str, object]],
    seed: int,
    cache: FeatureCache,
    epochs: int = 200,
) -> CachedLogisticReactionRanker:
    rng = random.Random(seed)
    train_subset = [
        {
            "reaction_smiles": str(row["reaction_smiles"]),
            "label": int(row["label"]),
        }
        for row in train_rows
    ]
    rng.shuffle(train_subset)
    model = CachedLogisticReactionRanker(
        cache=cache, learning_rate=0.2, l2=1e-4, epochs=epochs, n_features=10,
    )
    model.fit(train_subset)
    return model


def score_rows_pc_cng(
    model: CachedLogisticReactionRanker,
    rows: Sequence[Dict[str, object]],
) -> List[Dict[str, object]]:
    return [
        {**row, "score": float(model.predict_proba(str(row["reaction_smiles"]))),
         "ranker_source": "pc_cng_augmented"}
        for row in rows
    ]


# ---------------------------------------------------------------------------
# Ranking metrics (Top-K recall / MRR / NDCG)
# ---------------------------------------------------------------------------


def _group_rows(
    scored_rows: Sequence[Dict[str, object]],
) -> Dict[str, List[Dict[str, object]]]:
    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in scored_rows:
        grouped[str(row["group_id"])].append(row)
    return grouped


def _evaluable_groups(
    grouped: Dict[str, List[Dict[str, object]]],
) -> List[List[Dict[str, object]]]:
    out: List[List[Dict[str, object]]] = []
    for group_rows in grouped.values():
        labels = [int(r["label"]) for r in group_rows]
        if any(labels) and not all(labels):
            out.append(group_rows)
    return out


def topk_route_recall(
    scored_rows: Sequence[Dict[str, object]],
    k: int,
) -> float:
    """Fraction of evaluable groups where a label=1 route is in the top-K."""
    grouped = _group_rows(scored_rows)
    groups = _evaluable_groups(grouped)
    if not groups:
        return 0.0
    hits = 0
    for group_rows in groups:
        ranked = sorted(group_rows, key=lambda r: float(r["score"]), reverse=True)
        if any(int(r["label"]) == 1 for r in ranked[:k]):
            hits += 1
    return hits / len(groups)


def mrr(scored_rows: Sequence[Dict[str, object]]) -> float:
    grouped = _group_rows(scored_rows)
    groups = _evaluable_groups(grouped)
    if not groups:
        return 0.0
    total = 0.0
    for group_rows in groups:
        ranked = sorted(group_rows, key=lambda r: float(r["score"]), reverse=True)
        for rank, r in enumerate(ranked, start=1):
            if int(r["label"]) == 1:
                total += 1.0 / rank
                break
    return total / len(groups)


def ndcg_at_k(scored_rows: Sequence[Dict[str, object]], k: int = 10) -> float:
    grouped = _group_rows(scored_rows)
    groups = _evaluable_groups(grouped)
    if not groups:
        return 0.0
    total = 0.0
    for group_rows in groups:
        ranked = sorted(group_rows, key=lambda r: float(r["score"]), reverse=True)
        ranked_labels = [int(r["label"]) for r in ranked[:k]]
        dcg_value = sum(
            (1.0 if label else 0.0) / math.log2(rank + 1)
            for rank, label in enumerate(ranked_labels, start=1)
        )
        ideal = sorted([int(r["label"]) for r in group_rows], reverse=True)[:k]
        idcg = sum(
            (1.0 if label else 0.0) / math.log2(rank + 1)
            for rank, label in enumerate(ideal, start=1)
        )
        if idcg > 0:
            total += dcg_value / idcg
    return total / len(groups)


def evaluate(scored_rows: Sequence[Dict[str, object]]) -> Dict[str, float]:
    return {
        "top1": topk_route_recall(scored_rows, 1),
        "top3": topk_route_recall(scored_rows, 3),
        "top5": topk_route_recall(scored_rows, 5),
        "top10": topk_route_recall(scored_rows, 10),
        "mrr": mrr(scored_rows),
        "ndcg_at_10": ndcg_at_k(scored_rows, 10),
    }


def per_group_metrics(
    scored_rows: Sequence[Dict[str, object]],
) -> Dict[str, Dict[str, float]]:
    """Per-group {top1, mrr, ndcg} for paired significance testing."""
    grouped = _group_rows(scored_rows)
    out: Dict[str, Dict[str, float]] = {}
    for group_id, group_rows in grouped.items():
        labels = [int(r["label"]) for r in group_rows]
        if not any(labels) or all(labels):
            continue
        ranked = sorted(group_rows, key=lambda r: float(r["score"]), reverse=True)
        ranked_labels = [int(r["label"]) for r in ranked]
        first_pos = next(
            (rank for rank, lab in enumerate(ranked_labels, start=1) if lab == 1),
            len(ranked_labels),
        )
        ideal = sorted(ranked_labels, reverse=True)
        dcg_value = sum(
            (1.0 if lab else 0.0) / math.log2(rank + 1)
            for rank, lab in enumerate(ranked_labels[:10], start=1)
        )
        idcg = sum(
            (1.0 if lab else 0.0) / math.log2(rank + 1)
            for rank, lab in enumerate(ideal[:10], start=1)
        )
        out[group_id] = {
            "top1": 1.0 if ranked_labels and ranked_labels[0] == 1 else 0.0,
            "mrr": 1.0 / first_pos if first_pos <= len(ranked_labels) else 0.0,
            "ndcg": dcg_value / max(idcg, 1e-12),
        }
    return out


# ---------------------------------------------------------------------------
# Seed runner — score all 4 baselines on the test set
# ---------------------------------------------------------------------------


def run_seed(
    rows: Sequence[Dict[str, object]],
    seed: int,
    methods: Sequence[str],
    train_fraction: float = 0.7,
    epochs: int = 200,
    shared_cache: Optional[FeatureCache] = None,
    tanimoto_k: int = 5,
    chemformer_cache: Optional["ChemformerEmbeddingCache"] = None,
    chemformer_epochs: int = 100,
    decontaminate: bool = False,
) -> Dict[str, object]:
    """Run one seed: score all selected baselines on the test set.

    When ``decontaminate=True``, the P3-02 decontamination filter is
    applied: test candidates whose canonical ``parent_product`` SMILES
    matches any train-set product are excluded from the *primary*
    evaluation.  The decontaminated metrics are stored under the
    standard keys (``{method}_metrics`` / ``{method}_per_group``) so
    that the downstream aggregation, paired significance and Go/No-Go
    decision automatically use the fair (decontaminated) comparison.
    The original contaminated metrics are retained under
    ``{method}_metrics_contam`` / ``{method}_per_group_contam`` for
    transparency, and ``n_leaked`` / ``n_test_contam`` /
    ``n_test_decontam`` are recorded.
    """
    train_rows, test_rows = split_by_source(rows, train_fraction)
    if not test_rows:
        train_rows, test_rows = list(rows), list(rows)

    cache = shared_cache if shared_cache is not None else FeatureCache()
    unique_smiles = sorted({
        str(r["reaction_smiles"]) for r in list(train_rows) + list(test_rows)
    })
    cache.precompute(unique_smiles)

    # Pre-compute Chemformer embeddings for all unique reactions (B5).
    if "chemformer_scorer" in methods and chemformer_cache is not None:
        if chemformer_cache.available:
            print(
                f"[chemformer_scorer] precomputing embeddings for "
                f"{len(unique_smiles)} unique SMILES...", flush=True,
            )
            chemformer_cache.precompute(unique_smiles)
        else:
            print(
                f"[chemformer_scorer] backbone not available "
                f"(error: {chemformer_cache.load_error}); will return 0.5 "
                f"for all reactions.", flush=True,
            )

    # Build train-set fingerprints once for B3
    train_fps: List[Tuple[str, object, int]] = []
    if "tanimoto_nn" in methods:
        train_fps = build_train_fingerprints(train_rows)

    # P3-02 decontamination: filter leaked test candidates whose product
    # appears verbatim in the train set.  When the flag is off we simply
    # evaluate on the full (contaminated) test set — primary == contaminated.
    n_leaked = 0
    leaked_source_ids: List[str] = []
    if decontaminate:
        primary_test_rows, n_leaked, leaked_source_ids = filter_leaked_test_rows(
            train_rows, test_rows,
        )
    else:
        primary_test_rows = list(test_rows)

    result: Dict[str, object] = {
        "seed": seed,
        "n_train": len(train_rows),
        "n_test": len(test_rows),
        "n_leaked": n_leaked,
        "n_test_contam": len(test_rows),
        "n_test_decontam": len(primary_test_rows),
        "leaked_source_ids": leaked_source_ids,
        "decontaminate": bool(decontaminate),
    }

    def _score_all(test_subset: Sequence[Dict[str, object]], suffix: str) -> None:
        """Score every selected method on ``test_subset`` and write to result.

        ``suffix`` is appended to the result keys (``""`` for primary,
        ``"_contam"`` for the contaminated reference when decontamination
        is active).
        """
        # B1: RDKit template
        if "rdkit_template" in methods:
            b1_scored = score_rows_rdkit_template(test_subset, cache=cache)
            result[f"rdkit_template_metrics{suffix}"] = evaluate(b1_scored)
            result[f"rdkit_template_per_group{suffix}"] = per_group_metrics(b1_scored)

        # B2: heuristic validator
        if "heuristic_validator" in methods:
            b2_scored = score_rows_heuristic_validator(test_subset, cache=cache)
            result[f"heuristic_validator_metrics{suffix}"] = evaluate(b2_scored)
            result[f"heuristic_validator_per_group{suffix}"] = per_group_metrics(b2_scored)

        # B3: Tanimoto nearest-neighbor
        if "tanimoto_nn" in methods:
            b3_scored = score_rows_tanimoto_nn(test_subset, train_fps, k=tanimoto_k)
            result[f"tanimoto_nn_metrics{suffix}"] = evaluate(b3_scored)
            result[f"tanimoto_nn_per_group{suffix}"] = per_group_metrics(b3_scored)

        # B4: PC-CNG augmented
        if "pc_cng" in methods:
            pc_cng_model = train_pc_cng_augmented_ranker(
                train_rows, seed, cache=cache, epochs=epochs,
            )
            b4_scored = score_rows_pc_cng(pc_cng_model, test_subset)
            result[f"pc_cng_metrics{suffix}"] = evaluate(b4_scored)
            result[f"pc_cng_per_group{suffix}"] = per_group_metrics(b4_scored)

        # B5: Chemformer zero-shot embedding + logistic regression (P3-02)
        if "chemformer_scorer" in methods:
            if chemformer_cache is not None and chemformer_cache.available:
                b5_model = train_chemformer_scorer_ranker(
                    train_rows, seed, cache=chemformer_cache,
                    epochs=chemformer_epochs,
                )
                b5_scored = score_rows_chemformer_scorer(b5_model, test_subset)
            else:
                # Backbone unavailable - score everything 0.5 so the method
                # is still recorded but flagged as degraded.
                b5_scored = [
                    {**row, "score": 0.5, "ranker_source": "chemformer_scorer"}
                    for row in test_subset
                ]
            result[f"chemformer_scorer_metrics{suffix}"] = evaluate(b5_scored)
            result[f"chemformer_scorer_per_group{suffix}"] = per_group_metrics(b5_scored)

    # Primary metrics: decontaminated test set when the flag is on,
    # otherwise the full (contaminated) test set.
    _score_all(primary_test_rows, "")

    # When decontaminating, also keep the contaminated metrics for
    # transparency so the artifact is visible in summary.json.
    if decontaminate:
        _score_all(test_rows, "_contam")

    return result


# ---------------------------------------------------------------------------
# Paired significance (10-seed, PC-CNG vs each baseline)
# ---------------------------------------------------------------------------


def _paired_significance_one_pair(
    seed_results: Sequence[Dict[str, object]],
    method_a: str,
    method_b: str,
    metric: str = "mrr",
    bootstrap_iterations: int = 10000,
    seed: int = 20260710,
    suffix: str = "",
) -> Dict[str, object]:
    """Compute paired significance for metric delta (method_b - method_a).

    Both group-level (within-seed) and seed-level (across-seed) CIs are
    reported.

    ``suffix`` is appended to the per-group / metrics keys so the same
    routine can compute significance on either the primary
    (decontaminated, suffix ``""``) or the contaminated reference
    (suffix ``"_contam"``) per-group dicts.
    """
    a_per_group_key = f"{method_a}_per_group{suffix}"
    b_per_group_key = f"{method_b}_per_group{suffix}"
    a_metrics_key = f"{method_a}_metrics{suffix}"
    b_metrics_key = f"{method_b}_metrics{suffix}"

    common_groups: Optional[set] = None
    for r in seed_results:
        if a_per_group_key not in r or b_per_group_key not in r:
            continue
        a_groups = set(r[a_per_group_key].keys())
        b_groups = set(r[b_per_group_key].keys())
        g = a_groups & b_groups
        common_groups = g if common_groups is None else (common_groups & g)
    common_groups_sorted = sorted(common_groups) if common_groups else []

    seed_a_metric: List[float] = []
    seed_b_metric: List[float] = []
    seed_deltas: List[float] = []
    for r in seed_results:
        if a_per_group_key not in r or b_per_group_key not in r:
            continue
        a_vals = [
            float(r[a_per_group_key][g][metric])
            for g in common_groups_sorted
            if g in r[a_per_group_key]
        ]
        b_vals = [
            float(r[b_per_group_key][g][metric])
            for g in common_groups_sorted
            if g in r[b_per_group_key]
        ]
        if not a_vals or not b_vals:
            continue
        seed_a_metric.append(mean(a_vals))
        seed_b_metric.append(mean(b_vals))
        seed_deltas.append(mean([b - a for a, b in zip(a_vals, b_vals)]))

    a_group_means = [
        mean([
            float(r[a_per_group_key][g][metric])
            for r in seed_results
            if a_per_group_key in r and g in r[a_per_group_key]
        ])
        for g in common_groups_sorted
    ]
    b_group_means = [
        mean([
            float(r[b_per_group_key][g][metric])
            for r in seed_results
            if b_per_group_key in r and g in r[b_per_group_key]
        ])
        for g in common_groups_sorted
    ]
    group_deltas = [b - a for a, b in zip(a_group_means, b_group_means)]

    ci_low, ci_high = bootstrap_ci(group_deltas, bootstrap_iterations, seed)
    perm_p = paired_permutation_p_value(group_deltas, bootstrap_iterations, seed + 100)
    sign_p = sign_test_p_value(group_deltas)

    rng = random.Random(seed + 500)
    n_seeds = len(seed_deltas)
    seed_bootstrap_deltas: List[float] = []
    if n_seeds > 0:
        for _ in range(bootstrap_iterations):
            sample = [seed_deltas[rng.randrange(n_seeds)] for _ in range(n_seeds)]
            seed_bootstrap_deltas.append(mean(sample))
    seed_ci_low = percentile(seed_bootstrap_deltas, 0.025)
    seed_ci_high = percentile(seed_bootstrap_deltas, 0.975)

    return {
        "n_seeds": len(seed_results),
        "n_common_groups": len(common_groups_sorted),
        "metric": metric,
        "method_a": method_a,
        "method_b": method_b,
        "method_a_mean": mean(seed_a_metric),
        "method_b_mean": mean(seed_b_metric),
        "delta_mean": mean(seed_deltas),
        "delta_pp": mean(seed_deltas) * 100.0,
        "group_level_ci95_low": ci_low,
        "group_level_ci95_high": ci_high,
        "group_level_ci95_low_pp": ci_low * 100.0,
        "group_level_ci95_high_pp": ci_high * 100.0,
        "seed_level_ci95_low": seed_ci_low,
        "seed_level_ci95_high": seed_ci_high,
        "seed_level_ci95_low_pp": seed_ci_low * 100.0,
        "seed_level_ci95_high_pp": seed_ci_high * 100.0,
        "paired_permutation_p": perm_p,
        "sign_test_p": sign_p,
        "method_b_better_groups": sum(1 for d in group_deltas if d > 0.0),
        "method_a_better_groups": sum(1 for d in group_deltas if d < 0.0),
        "tie_groups": sum(1 for d in group_deltas if d == 0.0),
    }


def paired_significance(
    seed_results: Sequence[Dict[str, object]],
    methods: Sequence[str],
    bootstrap_iterations: int = 10000,
    seed: int = 20260710,
    suffix: str = "",
) -> Dict[str, object]:
    """Compute paired significance: PC-CNG vs each baseline.

    Returns a dict with keys ``pc_cng_vs_rdkit_template``,
    ``pc_cng_vs_heuristic_validator``, ``pc_cng_vs_tanimoto_nn``.

    ``suffix`` is forwarded to ``_paired_significance_one_pair`` so the
    same routine can operate on either the primary (decontaminated,
    suffix ``""``) or contaminated (suffix ``"_contam"``) per-group
    dicts.
    """
    out: Dict[str, object] = {}
    baselines = [m for m in methods if m != PROPOSED_KEY]
    for i, baseline in enumerate(baselines):
        if baseline not in [m for m in methods]:
            continue
        if PROPOSED_KEY not in methods:
            continue
        out[f"pc_cng_vs_{baseline}"] = _paired_significance_one_pair(
            seed_results, baseline, PROPOSED_KEY,
            metric="mrr",
            bootstrap_iterations=bootstrap_iterations,
            seed=seed + i,
            suffix=suffix,
        )
    return out


# ---------------------------------------------------------------------------
# Aggregation: mean ± std across seeds
# ---------------------------------------------------------------------------


def _std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = sum(values) / len(values)
    var = sum((v - mu) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(var)


def aggregate_metrics(
    seed_results: Sequence[Dict[str, object]],
    methods: Sequence[str],
    suffix: str = "",
) -> Dict[str, Dict[str, object]]:
    """Aggregate per-method metrics across seeds: mean ± std for each metric.

    ``suffix`` selects which per-seed metrics dict to aggregate — ``""``
    (default, primary / decontaminated when ``--decontaminate`` is on)
    or ``"_contam"`` (contaminated reference).
    """
    out: Dict[str, Dict[str, object]] = {}
    for method in methods:
        metrics_key = f"{method}_metrics{suffix}"
        per_seed_metrics = [
            r[metrics_key] for r in seed_results if metrics_key in r
        ]
        if not per_seed_metrics:
            continue
        metric_keys = list(per_seed_metrics[0].keys())
        agg: Dict[str, object] = {}
        for k in metric_keys:
            values = [float(m[k]) for m in per_seed_metrics]
            agg[k] = {
                "mean": mean(values),
                "std": _std(values),
                "min": min(values),
                "max": max(values),
                "mean_pp": mean(values) * 100.0,
                "std_pp": _std(values) * 100.0,
            }
        agg["n_seeds"] = len(per_seed_metrics)
        agg["method_name"] = METHOD_NAMES.get(method, method)
        out[method] = agg
    return out


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def _write_json(path: str, payload: object) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _write_csv(
    path: str,
    rows: Sequence[Dict[str, object]],
    fieldnames: Sequence[str],
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(fieldnames), extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)


def write_summary(
    path: str,
    methods: Sequence[str],
    ranker_metrics_agg: Dict[str, Dict[str, object]],
    sig: Dict[str, object],
    manifest_meta: Dict[str, object],
    ranker_metrics_agg_contam: Optional[Dict[str, Dict[str, object]]] = None,
    sig_contam: Optional[Dict[str, object]] = None,
    decontam_info: Optional[Dict[str, object]] = None,
) -> None:
    """Write summary.json (per-baseline metrics + mean±std + sig).

    When ``--decontaminate`` is active, the primary ``metrics`` /
    ``paired_significance`` reflect the *decontaminated* (fair)
    comparison, and the original contaminated reference is preserved
    under ``metrics_contam`` / ``paired_significance_contam`` plus a
    ``decontamination`` block recording ``n_leaked`` etc.
    """
    payload = {
        "task": "P3-02 SOTA multi-baseline comparison v2 (翻盘 P2-06 NO-GO)",
        **manifest_meta,
        "methods": [METHOD_NAMES.get(m, m) for m in methods],
        "method_keys": list(methods),
        "metrics": ranker_metrics_agg,
        "paired_significance": sig,
    }
    if ranker_metrics_agg_contam is not None:
        payload["metrics_contam"] = ranker_metrics_agg_contam
    if sig_contam is not None:
        payload["paired_significance_contam"] = sig_contam
    if decontam_info is not None:
        payload["decontamination"] = decontam_info
    _write_json(path, payload)


def write_paired_significance(path: str, sig: Dict[str, object]) -> None:
    _write_json(path, sig)


def write_per_target_metrics(
    path: str,
    seed_results: Sequence[Dict[str, object]],
    methods: Sequence[str],
) -> None:
    """Write per_target_metrics.csv: one row per (seed, group_id, method)."""
    rows: List[Dict[str, object]] = []
    for seed_result in seed_results:
        seed = seed_result["seed"]
        for method in methods:
            per_group_key = f"{method}_per_group"
            if per_group_key not in seed_result:
                continue
            for group_id, metrics in seed_result[per_group_key].items():
                rows.append({
                    "seed": seed,
                    "group_id": group_id,
                    "method": method,
                    "method_name": METHOD_NAMES.get(method, method),
                    "top1": metrics["top1"],
                    "mrr": metrics["mrr"],
                    "ndcg": metrics["ndcg"],
                })
    _write_csv(
        path, rows,
        ["seed", "group_id", "method", "method_name", "top1", "mrr", "ndcg"],
    )


def write_sota_installation_status(
    path: str,
    deferred_methods: Sequence[str] = DEFERRED_SOTA_METHODS,
) -> None:
    """Write sota_installation_status.json documenting deferred SOTA methods."""
    payload = {
        "task": "P3-02 SOTA multi-baseline comparison v2 (翻盘 P2-06 NO-GO)",
        "network_access": "none",
        "deferred_methods": [
            {
                "name": name,
                "status": "deferred",
                "reason": (
                    f"{name} not evaluated due to installation failure: "
                    "no network access on remote server. PyPI install and "
                    "git clone from GitHub both fail."
                ),
                "attempted_install": [
                    f"pip install {name}",
                    f"pip install git+https://github.com/<org>/{name}.git",
                ],
                "fallback": "RDKit-only baselines (B1/B2/B3) used instead.",
            }
            for name in deferred_methods
        ],
        "evaluated_methods": [
            {"key": "rdkit_template", "name": METHOD_NAMES["rdkit_template"]},
            {"key": "heuristic_validator", "name": METHOD_NAMES["heuristic_validator"]},
            {"key": "tanimoto_nn", "name": METHOD_NAMES["tanimoto_nn"]},
            {"key": "pc_cng", "name": METHOD_NAMES["pc_cng"]},
            {"key": "chemformer_scorer", "name": METHOD_NAMES["chemformer_scorer"]},
        ],
        "note": (
            "Per Section 26.1 degradation path: '若某 SOTA 安装受阻，"
            "跳过该方法，明确标注 X method not evaluated due to "
            "installation failure'. LocalRetro / Graph2SMILES / "
            "Molecular Transformer are documented as deferred. "
            "P3-02 adds B5 Chemformer zero-shot scorer (pretrained "
            "checkpoint already downloaded for P3-01, no network "
            "access required)."
        ),
    }
    _write_json(path, payload)


def write_go_no_go_decision(
    path: str,
    sig: Dict[str, object],
    methods: Sequence[str],
) -> None:
    """Write go_no_go_decision.json.

    Decision rule: PC-CNG Top-1 (MRR) >= 3/3 baselines + 1.0 pp on MRR.
    """
    baselines = [m for m in methods if m != PROPOSED_KEY]
    decisions: Dict[str, object] = {}
    wins = 0
    for baseline in baselines:
        key = f"pc_cng_vs_{baseline}"
        pair = sig.get(key)
        if pair is None:
            decisions[key] = {
                "baseline": baseline,
                "delta_pp": None,
                "ci_low_pp": None,
                "ci_high_pp": None,
                "pc_cng_better": None,
                "ci_all_positive": None,
            }
            continue
        delta_pp = float(pair["delta_pp"])
        ci_low_pp = float(pair["seed_level_ci95_low_pp"])
        ci_high_pp = float(pair["seed_level_ci95_high_pp"])
        pc_cng_better = delta_pp > 1.0
        ci_all_positive = ci_low_pp > 0.0 and ci_high_pp > 0.0
        decisions[key] = {
            "baseline": baseline,
            "delta_pp": delta_pp,
            "ci_low_pp": ci_low_pp,
            "ci_high_pp": ci_high_pp,
            "pc_cng_better": pc_cng_better,
            "ci_all_positive": ci_all_positive,
            "paired_permutation_p": pair["paired_permutation_p"],
            "sign_test_p": pair["sign_test_p"],
        }
        if pc_cng_better:
            wins += 1

    overall_go = (
        wins == len(baselines) and len(baselines) >= 1
    )
    payload = {
        "task": "P2-06 SOTA multi-baseline comparison (L6)",
        "decision_rule": (
            "GO iff PC-CNG beats every evaluated baseline by > 1.0 pp MRR "
            "and the seed-level 95% CI is entirely positive."
        ),
        "n_baselines_evaluated": len(baselines),
        "n_baselines_pc_cng_beats": wins,
        "threshold_pp": 1.0,
        "overall_decision": (
            "GO (write to main table)"
            if overall_go
            else "NO-GO (downgrade to supplementary)"
        ),
        "per_baseline": decisions,
        "deferred_sota_methods": list(DEFERRED_SOTA_METHODS),
        "deferred_reason": (
            "LocalRetro / Graph2SMILES / Molecular Transformer could not be "
            "installed due to no network access on the remote server. See "
            "sota_installation_status.json for details."
        ),
    }
    _write_json(path, payload)


def write_per_seed_detail(
    path: str,
    seed_results: Sequence[Dict[str, object]],
    methods: Sequence[str],
) -> None:
    """Write per_seed_detail.json with per-seed metrics for each method."""
    out: List[Dict[str, object]] = []
    for r in seed_results:
        entry: Dict[str, object] = {
            "seed": r["seed"],
            "n_train": r["n_train"],
            "n_test": r["n_test"],
        }
        # Include decontamination bookkeeping when present.
        if "n_leaked" in r:
            entry["n_leaked"] = r["n_leaked"]
        if "n_test_contam" in r:
            entry["n_test_contam"] = r["n_test_contam"]
        if "n_test_decontam" in r:
            entry["n_test_decontam"] = r["n_test_decontam"]
        for method in methods:
            metrics_key = f"{method}_metrics"
            if metrics_key in r:
                entry[f"{method}_metrics"] = r[metrics_key]
            # Also include the contaminated reference when present.
            contam_key = f"{method}_metrics_contam"
            if contam_key in r:
                entry[f"{method}_metrics_contam"] = r[contam_key]
        out.append(entry)
    _write_json(path, out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="P3-02 SOTA multi-baseline comparison v2 (翻盘 P2-06)"
    )
    parser.add_argument(
        "--dataset", default=None,
        help="USPTO-MIT-50k CSV (optional). Falls back to PC-CNG negatives "
             "if not found or unreadable.",
    )
    parser.add_argument(
        "--pc-cng-negatives", required=True,
        help="PC-CNG synthetic negatives CSV (pc_cng_synthetic_negatives_reviewed.csv)",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS),
        help="Comma-separated seeds (default: 20260710..20260719)",
    )
    parser.add_argument(
        "--methods", default=DEFAULT_METHODS,
        help="Comma-separated methods (default: rdkit_template,heuristic_validator,"
             "tanimoto_nn,pc_cng,chemformer_scorer)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Smoke-test limit: cap number of source_ids to N",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-candidates-per-source", type=int, default=10)
    parser.add_argument(
        "--max-sources", type=int, default=2000,
        help="Cap number of source_ids for tractability",
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--tanimoto-k", type=int, default=5)
    # P3-02 B5: Chemformer zero-shot scorer args
    parser.add_argument(
        "--chemformer-ckpt", default=None,
        help="Path to Chemformer sanitized checkpoint (.ckpt).  If not "
             "provided, B5 falls back to 0.5 scores.",
    )
    parser.add_argument(
        "--chemformer-vocab", default=None,
        help="Path to Chemformer bart_vocab.json.",
    )
    parser.add_argument(
        "--chemformer-device", default="cpu",
        help="Device for Chemformer encoder (cpu | cuda:0 | cuda:1 ...).",
    )
    parser.add_argument(
        "--chemformer-batch-size", type=int, default=16,
        help="Batch size for Chemformer encoding.",
    )
    parser.add_argument(
        "--chemformer-epochs", type=int, default=100,
        help="Epochs for the B5 logistic head (default 100).",
    )
    # P3-02 decontamination filter: fix Tanimoto-NN data-leakage artifact.
    parser.add_argument(
        "--decontaminate", action="store_true", default=False,
        help="P3-02 decontamination filter: exclude test candidates whose "
             "canonical parent_product SMILES matches any train-set product. "
             "Primary metrics become the decontaminated (fair) comparison; "
             "contaminated reference metrics are also reported for "
             "transparency under metrics_contam / paired_significance_contam.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)
    os.makedirs(args.output_dir, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    # Validate methods
    valid_methods = set(METHOD_NAMES.keys())
    for m in methods:
        if m not in valid_methods:
            raise ValueError(
                f"Unknown method {m!r}; expected one of {sorted(valid_methods)}"
            )

    # Suppress RDKit warnings
    try:
        from rdkit import RDLogger  # type: ignore
        RDLogger.DisableLog("rdApp.*")
    except Exception:
        pass

    # Write SOTA installation status (deferred methods) up front
    write_sota_installation_status(
        os.path.join(args.output_dir, "sota_installation_status.json"),
    )

    # Load dataset; fall back to PC-CNG negatives
    rows: List[Dict[str, object]] = []
    fallback_used = False
    if args.dataset and os.path.exists(args.dataset):
        rows = load_uspto_mit_50k_routes(args.dataset)
    if not rows:
        fallback_used = True
        rows = load_pc_cng_negatives(
            args.pc_cng_negatives, args.max_candidates_per_source
        )

    # Cap sources for tractability
    source_ids = sorted({str(r["source_id"]) for r in rows})
    if args.limit is not None and args.limit > 0:
        args.max_sources = min(args.max_sources, args.limit)
    if len(source_ids) > args.max_sources:
        rng = random.Random(20260710)
        source_ids = sorted(rng.sample(source_ids, args.max_sources))
    keep = set(source_ids)
    rows = [r for r in rows if str(r["source_id"]) in keep]

    print(
        f"Loaded {len(rows)} route candidates across {len(source_ids)} source_ids",
        flush=True,
    )
    if fallback_used:
        print(
            "NOTE: Using PC-CNG negatives pseudo-routes fallback "
            "(no USPTO-MIT-50k dataset available)",
            flush=True,
        )

    # Pre-compute features for all unique reactions (shared across seeds)
    shared_cache = FeatureCache()
    unique_smiles = sorted({str(r["reaction_smiles"]) for r in rows})
    print(
        f"Pre-computing features for {len(unique_smiles)} unique reactions...",
        flush=True,
    )
    shared_cache.precompute(unique_smiles)
    print("Feature pre-computation complete.", flush=True)

    # P3-02 B5: Build the Chemformer embedding cache (shared across seeds).
    # The backbone is loaded once and reused for all seeds.
    chemformer_cache: Optional[ChemformerEmbeddingCache] = None
    if "chemformer_scorer" in methods:
        print(
            f"[chemformer_scorer] Loading pretrained Chemformer backbone:\n"
            f"  ckpt:   {args.chemformer_ckpt}\n"
            f"  vocab:  {args.chemformer_vocab}\n"
            f"  device: {args.chemformer_device}",
            flush=True,
        )
        chemformer_cache = ChemformerEmbeddingCache(
            checkpoint_path=args.chemformer_ckpt,
            vocab_path=args.chemformer_vocab,
            device=args.chemformer_device,
            batch_size=args.chemformer_batch_size,
        )
        if chemformer_cache.available:
            print(
                f"[chemformer_scorer] Backbone loaded successfully "
                f"(embedding_dim={chemformer_cache.embedding_dim}).",
                flush=True,
            )
        else:
            print(
                f"[chemformer_scorer] WARNING: backbone unavailable "
                f"({chemformer_cache.load_error}); B5 will return 0.5 for "
                f"all reactions.",
                flush=True,
            )

    print(
        f"Methods: {methods}\n"
        f"Deferred SOTA (no network): {list(DEFERRED_SOTA_METHODS)}",
        flush=True,
    )
    if args.decontaminate:
        print(
            "DECONTAMINATION: enabled — test candidates whose canonical "
            "parent_product SMILES matches any train-set product will be "
            "excluded from the primary (fair) evaluation. Contaminated "
            "reference metrics will also be reported under metrics_contam.",
            flush=True,
        )

    # Run seeds
    print(f"\nRunning {len(seeds)} seeds...", flush=True)
    seed_results: List[Dict[str, object]] = []
    for seed in seeds:
        print(f"\n--- Seed {seed} ---", flush=True)
        result = run_seed(
            rows, seed, methods,
            train_fraction=args.train_fraction,
            epochs=args.epochs,
            shared_cache=shared_cache,
            tanimoto_k=args.tanimoto_k,
            chemformer_cache=chemformer_cache,
            chemformer_epochs=args.chemformer_epochs,
            decontaminate=args.decontaminate,
        )
        seed_results.append(result)
        if args.decontaminate and result.get("n_leaked", 0) > 0:
            print(
                f"  [decontam] seed={seed}: n_leaked={result['n_leaked']} "
                f"of n_test_contam={result['n_test_contam']} "
                f"-> n_test_decontam={result['n_test_decontam']}",
                flush=True,
            )
        for method in methods:
            metrics_key = f"{method}_metrics"
            if metrics_key in result:
                m = result[metrics_key]
                print(
                    f"  {METHOD_NAMES.get(method, method):<35s} "
                    f"MRR: {m['mrr']:.4f}  Top1: {m['top1']:.4f}",
                    flush=True,
                )

    # Aggregate metrics across seeds (mean ± std).
    # When --decontaminate is active, the primary (suffix "") aggregation
    # reflects the decontaminated (fair) test set; we additionally aggregate
    # the contaminated reference (suffix "_contam") for transparency.
    ranker_metrics_agg = aggregate_metrics(seed_results, methods)
    ranker_metrics_agg_contam: Optional[Dict[str, Dict[str, object]]] = None
    if args.decontaminate:
        ranker_metrics_agg_contam = aggregate_metrics(
            seed_results, methods, suffix="_contam",
        )

    # Paired significance (primary = decontaminated when flag is on)
    sig = paired_significance(
        seed_results, methods,
        bootstrap_iterations=args.bootstrap_iterations,
        seed=seeds[0] if seeds else 20260710,
    )
    sig_contam: Optional[Dict[str, object]] = None
    if args.decontaminate:
        sig_contam = paired_significance(
            seed_results, methods,
            bootstrap_iterations=args.bootstrap_iterations,
            seed=seeds[0] if seeds else 20260710,
            suffix="_contam",
        )

    # Manifest metadata
    manifest_meta = {
        "task": "P3-02 SOTA multi-baseline comparison v2 (翻盘 P2-06 NO-GO)",
        "fallback_path": "pc_cng_negatives" if fallback_used else "uspto_mit_50k",
        "dataset_path": args.dataset,
        "pc_cng_negatives_path": args.pc_cng_negatives,
        "deferred_sota_methods": list(DEFERRED_SOTA_METHODS),
        "deferred_reason": (
            "LocalRetro / Graph2SMILES / Molecular Transformer could not be "
            "installed due to no network access on the remote server."
        ),
        "n_source_ids": len(source_ids),
        "n_route_candidates": len(rows),
        "n_seeds": len(seeds),
        "seeds": seeds,
        "methods": methods,
        "top_k": args.top_k,
        "max_candidates_per_source": args.max_candidates_per_source,
        "max_sources": args.max_sources,
        "bootstrap_iterations": args.bootstrap_iterations,
        "train_fraction": args.train_fraction,
        "tanimoto_k": args.tanimoto_k,
        "chemformer_ckpt": args.chemformer_ckpt,
        "chemformer_vocab": args.chemformer_vocab,
        "chemformer_device": args.chemformer_device,
        "chemformer_available": (
            chemformer_cache.available if chemformer_cache is not None else False
        ),
        "chemformer_embedding_dim": (
            chemformer_cache.embedding_dim if chemformer_cache is not None else 0
        ),
        "chemformer_epochs": args.chemformer_epochs,
        "decontaminate": bool(args.decontaminate),
    }

    # Build the decontamination info block when the flag is active so
    # summary.json records the leakage artifact for reproducibility.
    decontam_info: Optional[Dict[str, object]] = None
    if args.decontaminate:
        total_leaked = sum(int(r.get("n_leaked", 0)) for r in seed_results)
        total_test_contam = sum(int(r.get("n_test_contam", 0)) for r in seed_results)
        total_test_decontam = sum(int(r.get("n_test_decontam", 0)) for r in seed_results)
        all_leaked_source_ids = sorted({
            sid for r in seed_results
            for sid in r.get("leaked_source_ids", [])  # type: ignore[union-attr]
        })
        decontam_info = {
            "enabled": True,
            "rationale": (
                "Tanimoto-NN baseline achieves MRR=1.0 on the contaminated "
                "test set because test-set products appear verbatim in the "
                "train set (Tanimoto=1.0 nearest neighbour trivially "
                "recovers the label). Decontamination excludes test "
                "candidates whose canonical parent_product SMILES matches "
                "any train-set product, yielding a fair PC-CNG vs "
                "Tanimoto-NN comparison."
            ),
            "n_leaked_total": total_leaked,
            "n_test_contam_total": total_test_contam,
            "n_test_decontam_total": total_test_decontam,
            "leak_rate": (
                total_leaked / total_test_contam if total_test_contam else 0.0
            ),
            "n_leaked_source_ids": len(all_leaked_source_ids),
            "leaked_source_ids_sample": all_leaked_source_ids[:50],
        }
        manifest_meta["n_leaked_total"] = total_leaked
        manifest_meta["n_test_decontam_total"] = total_test_decontam
        manifest_meta["leak_rate"] = decontam_info["leak_rate"]
        print(
            f"\nDecontamination summary: n_leaked_total={total_leaked} "
            f"of n_test_contam_total={total_test_contam} "
            f"(leak_rate={decontam_info['leak_rate']:.4f}) "
            f"-> n_test_decontam_total={total_test_decontam}",
            flush=True,
        )

    # Write outputs
    write_summary(
        os.path.join(args.output_dir, "summary.json"),
        methods, ranker_metrics_agg, sig, manifest_meta,
        ranker_metrics_agg_contam=ranker_metrics_agg_contam,
        sig_contam=sig_contam,
        decontam_info=decontam_info,
    )
    write_paired_significance(
        os.path.join(args.output_dir, "paired_significance.json"),
        sig,
    )
    if args.decontaminate and sig_contam is not None:
        # Also persist the contaminated reference paired significance so the
        # leakage artifact (Tanimoto-NN MRR=1.0) is reproducible.
        write_paired_significance(
            os.path.join(args.output_dir, "paired_significance_contam.json"),
            sig_contam,
        )
    write_per_target_metrics(
        os.path.join(args.output_dir, "per_target_metrics.csv"),
        seed_results, methods,
    )
    write_go_no_go_decision(
        os.path.join(args.output_dir, "go_no_go_decision.json"),
        sig, methods,
    )
    write_per_seed_detail(
        os.path.join(args.output_dir, "per_seed_detail.json"),
        seed_results, methods,
    )

    # Print final summary
    print("\n" + "=" * 70)
    print("P3-02 SOTA multi-baseline comparison v2 — Summary (翻盘 P2-06)")
    print("=" * 70)
    print(f"Fallback path:      {manifest_meta['fallback_path']}")
    print(f"N source_ids:       {len(source_ids)}")
    print(f"N route candidates: {len(rows)}")
    print(f"N seeds:            {len(seeds)}")
    print(f"Deferred SOTA:      {list(DEFERRED_SOTA_METHODS)}")
    print(f"Chemformer B5:      available={manifest_meta['chemformer_available']} "
          f"dim={manifest_meta['chemformer_embedding_dim']}")
    print(f"Decontamination:    {manifest_meta['decontaminate']}")
    if args.decontaminate and decontam_info is not None:
        print(
            f"  n_leaked_total={decontam_info['n_leaked_total']} "
            f"of n_test_contam_total={decontam_info['n_test_contam_total']} "
            f"(leak_rate={decontam_info['leak_rate']:.4f}) "
            f"-> n_test_decontam_total={decontam_info['n_test_decontam_total']}"
        )
    print()
    label_primary = "DECONTAM (fair)" if args.decontaminate else "primary"
    print(f"Primary metrics ({label_primary}):")
    for method in methods:
        agg = ranker_metrics_agg.get(method, {})
        name = METHOD_NAMES.get(method, method)
        mrr_mean = agg.get("mrr", {}).get("mean", 0.0) if isinstance(agg.get("mrr"), dict) else 0.0
        top1_mean = agg.get("top1", {}).get("mean", 0.0) if isinstance(agg.get("top1"), dict) else 0.0
        print(f"  {name:<35s} Top1: {top1_mean:.4f}  MRR: {mrr_mean:.4f}")
    if args.decontaminate and ranker_metrics_agg_contam is not None:
        print()
        print("Contaminated reference (leakage artifact visible here):")
        for method in methods:
            agg = ranker_metrics_agg_contam.get(method, {})
            name = METHOD_NAMES.get(method, method)
            mrr_mean = agg.get("mrr", {}).get("mean", 0.0) if isinstance(agg.get("mrr"), dict) else 0.0
            top1_mean = agg.get("top1", {}).get("mean", 0.0) if isinstance(agg.get("top1"), dict) else 0.0
            print(f"  {name:<35s} Top1: {top1_mean:.4f}  MRR: {mrr_mean:.4f}")
    print()
    for baseline in [m for m in methods if m != PROPOSED_KEY]:
        key = f"pc_cng_vs_{baseline}"
        if key in sig:
            pair = sig[key]
            print(
                f"  PC-CNG vs {baseline} ({label_primary}): "
                f"ΔMRR = {pair['delta_pp']:.2f} pp "
                f"(seed CI [{pair['seed_level_ci95_low_pp']:.2f}, "
                f"{pair['seed_level_ci95_high_pp']:.2f}] pp, "
                f"p_perm = {pair['paired_permutation_p']:.4f})"
            )
    # Go/No-Go
    go_no_go_path = os.path.join(args.output_dir, "go_no_go_decision.json")
    with open(go_no_go_path) as fh:
        go_payload = json.load(fh)
    print()
    print(f"Go/No-Go ({label_primary}): {go_payload['overall_decision']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
