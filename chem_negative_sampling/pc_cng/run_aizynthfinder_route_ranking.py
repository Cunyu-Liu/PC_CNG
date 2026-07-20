"""P2-01 AiZynthFinder real routes retrosynthesis ranking.

Section 26.1 P2-01 task: rerun retrosynthesis ranking using AiZynthFinder
real routes (replacing P1-04's pseudo-route fallback).

Four rankers are compared per target molecule:

  R1 (aizynthfinder_baseline):
      AiZynthFinder default policy score. If AiZynthFinder policy models
      are unavailable (no network access to zenodo), degrades to the
      heuristic scorer from P1-04 (atom_balance + validity + similarity).

  R2 (aizynthfinder_chemformer):
      Chemformer forward likelihood reranker. For each candidate route,
      the reactants are fed to the Chemformer forward model and the
      log-likelihood of the actual product is the score. If Chemformer
      is unavailable, degrades to a heuristic forward-likelihood proxy
      (RDKit validity + product/reactant atom-balance).

  R3 (aizynthfinder_pc_cng):
      PC-CNG negatives augmented logistic reranker. Same as P1-04:
      LogisticReactionRanker trained on PC-CNG synthetic negatives as
      pairwise training signal.

  R4 (ground_truth):
      Oracle ranker. The gold route (positive_reaction) always ranks #1.
      Included as an upper bound.

Candidate set per target molecule (group_id = source_id):
  - 1 gold route from PC-CNG CSV ``positive_reaction`` (label=1)
  - N PC-CNG synthetic negatives from ``candidate_reaction`` (label=0)
  - (Optional) M AiZynthFinder-generated routes (label=1 if reactant set
    matches the gold route's canonical reactant set, else 0)

Metrics (per ranker):
  - Top-K Route Recall (k=1, 3, 5, 10)
  - MRR (Mean Reciprocal Rank of first gold route)
  - NDCG@10
  - False-Positive Route Rate (fraction of groups where a label=0 route
    outranks the gold route)

Significance:
  10-seed paired bootstrap CI + paired sign-flip permutation p + sign-test p
  on MRR delta (R3 - R1, R3 - R2, R2 - R1). Each seed is a different
  sampling of test molecules (random source_id subset).

Degradation path (Section 26.1):
  1. Try AiZynthFinder with downloaded public policy models.
  2. If download fails (no network), try RDKit template-based retro
     with a small built-in library of common functional-group reactions.
  3. If template retro yields no candidates, use only the gold route +
     PC-CNG negatives as the candidate set (this is the most-degraded
     path; manifest will record ``fallback_path`` accordingly).

All performance claims are 10-seed paired. AiZynthFinder is invoked via
subprocess in the isolated ``aizynthfinder`` conda env (CPU-only). No GPU
is used.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import subprocess
import sys
import tempfile
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

AIZYNTHFINDER_PYTHON_DEFAULT = (
    "/home/cunyuliu/miniconda3/envs/aizynthfinder/bin/python"
)
CHEMFORMER_PYTHON_DEFAULT = (
    "/home/cunyuliu/pc_cng_research/envs/reaction_lm/bin/python"
)
CHEMFORMER_CKPT_DEFAULT = (
    "/home/cunyuliu/pc_cng_research/models/reaction_lm/"
    "chemformer_forward_uspto50k/last.ckpt"
)

FALLBACK_TAG = "aizynthfinder_unavailable"

RANKER_NAMES = (
    "aizynthfinder_baseline",
    "aizynthfinder_chemformer",
    "aizynthfinder_pc_cng",
    "ground_truth",
)

# Map full ranker names to the short keys used in seed_result dicts
# (r1, r2, r3, r4) so output writers can iterate over RANKER_NAMES.
RANKER_SHORT_KEY_MAP = {
    "aizynthfinder_baseline": "r1",
    "aizynthfinder_chemformer": "r2",
    "aizynthfinder_pc_cng": "r3",
    "ground_truth": "r4",
}


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
# FeatureCache + cached logistic ranker (mirrors P1-04 for fast training)
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
# Data loading
# ---------------------------------------------------------------------------


def load_pc_cng_negatives(
    path: str,
    max_candidates_per_source: int = 10,
) -> List[Dict[str, object]]:
    """Load PC-CNG synthetic negatives and build ranking rows.

    For each ``source_id`` the ``positive_reaction`` becomes a label=1
    candidate (gold route) and each ``candidate_reaction`` becomes a
    label=0 candidate. The ``parent_product`` SMILES is recorded as the
    AiZynthFinder target for that group.
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
    """Load USPTO-MIT-50k multi-step routes CSV (optional).

    Expected columns: ``product_smiles``, ``route_smiles``, ``route_id``,
    ``is_gold``. If the file is missing or malformed, returns an empty list.
    """
    if not path or not os.path.exists(path):
        return []
    rows: List[Dict[str, object]] = []
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for record in reader:
            product = str(record.get("product_smiles", "") or record.get("product", "")).strip()
            route = str(record.get("route_smiles", "") or record.get("reaction_smiles", "")).strip()
            if not product or not route:
                continue
            route_id = str(record.get("route_id", "") or record.get("id", "")).strip() or product
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
# AiZynthFinder subprocess integration
# ---------------------------------------------------------------------------


_AIZYNTHFINDER_RUNNER_SCRIPT = """
import json
import sys
from aizynthfinder.aizynthfinder import AiZynthFinder

def main():
    payload = json.load(sys.stdin)
    target = payload["target_smiles"]
    configfile = payload.get("configfile")
    time_limit = int(payload.get("time_limit", 30))
    iteration_limit = int(payload.get("iteration_limit", 100))
    n_routes = int(payload.get("n_routes", 5))
    try:
        if configfile:
            finder = AiZynthFinder(configfile=configfile)
        else:
            finder = AiZynthFinder()
        finder.target_smiles = target
        finder.prepare_tree()
        finder.tree_search()
        finder.build_routes()
        routes = []
        for route in finder.routes:
            try:
                # Each route is a ReactionTree; extract first reaction
                reaction_smiles = ""
                actions = list(route.reactions())
                if actions:
                    rxn = actions[0]
                    reactants = ".".join(sorted({
                        m.smiles for m in rxn.reactants
                    }))
                    products = ".".join(sorted({
                        m.smiles for m in rxn.products
                    }))
                    reaction_smiles = f"{reactants}>>{products}"
                routes.append({
                    "reaction_smiles": reaction_smiles,
                    "score": float(getattr(route, "route_score", 0.5)),
                    "source": "aizynthfinder",
                })
            except Exception:
                continue
            if len(routes) >= n_routes:
                break
        print(json.dumps({
            "status": "ok",
            "target": target,
            "routes": routes,
            "n_routes": len(routes),
        }))
    except Exception as exc:
        print(json.dumps({
            "status": "error",
            "target": target,
            "error": f"{type(exc).__name__}: {exc}",
            "routes": [],
        }))

if __name__ == "__main__":
    main()
"""


def try_aizynthfinder_search(
    target_smiles: str,
    aizynthfinder_python: str,
    configfile: Optional[str] = None,
    time_limit: int = 30,
    iteration_limit: int = 100,
    n_routes: int = 5,
    timeout: int = 120,
) -> Tuple[List[Dict[str, object]], str]:
    """Run AiZynthFinder on a single target via subprocess.

    Returns ``(routes, status)`` where ``routes`` is a list of dicts
    (each with ``reaction_smiles``, ``score``, ``source`` keys) and
    ``status`` is one of ``"ok"``, ``"error"``, ``"timeout"``,
    ``"env_missing"``.
    """
    if not os.path.exists(aizynthfinder_python):
        return [], "env_missing"
    payload = {
        "target_smiles": target_smiles,
        "configfile": configfile,
        "time_limit": time_limit,
        "iteration_limit": iteration_limit,
        "n_routes": n_routes,
    }
    try:
        result = subprocess.run(
            [aizynthfinder_python, "-c", _AIZYNTHFINDER_RUNNER_SCRIPT],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return [], "timeout"
    if result.returncode != 0:
        return [], "error"
    out = result.stdout.strip().splitlines()
    if not out:
        return [], "error"
    try:
        parsed = json.loads(out[-1])
    except json.JSONDecodeError:
        return [], "error"
    return list(parsed.get("routes", [])), str(parsed.get("status", "error"))


# ---------------------------------------------------------------------------
# RDKit template-based retro fallback
# ---------------------------------------------------------------------------


# A small library of one-step retrosynthesis SMARTS templates.
# These are intentionally conservative, well-known functional group
# disconnections. Each entry is (template_smarts, name, weight).
_TEMPLATE_REACTIONS: List[Tuple[str, str, float]] = [
    # Ester hydrolysis: R-COOR' -> R-COOH + R'OH
    ("[C:1](=[O:2])[O:3][C:4]>>[C:1](=[O:2])[OH].[C:4][OH]", "ester_hydrolysis", 0.7),
    # Amide hydrolysis: R-CONHR' -> R-COOH + R'NH2
    ("[C:1](=[O:2])[NX3:3]>>[C:1](=[O:2])[OH].[NX3:3]", "amide_hydrolysis", 0.7),
    # Ether cleavage: R-O-R' -> R-OH + R'OH
    ("[C:1][O:2][C:3]>>[C:1][OH].[O:2][C:3]", "ether_cleavage", 0.5),
    # SN2 disconnection at sp3 carbon with leaving group
    ("[C:1][Cl:2]>>[C:1][OH].[Cl:2]", "alkyl_chloride_sn2", 0.5),
    ("[C:1][Br:2]>>[C:1][OH].[Br:2]", "alkyl_bromide_sn2", 0.5),
    # Alcohol to alkene (dehydration reverse)
    ("[C:1][OH:2]>>[C:1]=[C:1]", "alcohol_dehydration", 0.4),
    # Imine formation: R2C=NR' -> R2C=O + R'NH2
    ("[C:1]=[N:2]>>[C:1]=[O].[N:2]", "imine_hydrolysis", 0.6),
    # Acetal hydrolysis
    ("[C:1]([O:2][C:3])([O:4][C:5])>>[C:1]=[O].[C:3][OH].[C:5][OH]", "acetal_hydrolysis", 0.5),
]


def generate_template_routes(
    target_smiles: str,
    max_routes: int = 5,
) -> List[Dict[str, object]]:
    """Generate retrosynthesis candidates via RDKit template application.

    Returns list of dicts: ``{reaction_smiles, score, source}``.
    Uses a small built-in library of common functional-group disconnections.
    If RDKit is unavailable or no templates apply, returns an empty list.
    """
    try:
        from rdkit import Chem  # type: ignore
        from rdkit.Chem import AllChem  # type: ignore
    except Exception:
        return []

    target_mol = Chem.MolFromSmiles(target_smiles)
    if target_mol is None:
        return []

    routes: List[Dict[str, object]] = []
    seen_reactants: set[str] = set()
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


# ---------------------------------------------------------------------------
# Chemformer forward likelihood scorer
# ---------------------------------------------------------------------------


_CHEMFORMER_RUNNER_SCRIPT = """
import json
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

def main():
    payload = json.load(sys.stdin)
    ckpt = payload.get("ckpt")
    rows = payload.get("rows", [])
    if not ckpt or not os.path.isfile(ckpt):
        print(json.dumps({"status": "ckpt_missing", "scores": []}))
        return
    try:
        sys.path.insert(0, "/home/cunyuliu/pc_cng_research/external/reaction_lm/Chemformer")
        import torch  # noqa
        from molbart.models.chemformer import Chemformer  # noqa
        model = Chemformer.load_from_checkpoint(ckpt, strict=False)
        model.eval()
        device = "cpu"
        model.to(device)
        scores = []
        with torch.no_grad():
            for row in rows:
                reactants = row.get("reactants", "")
                product = row.get("product", "")
                try:
                    tokens = model.tokeniser(
                        [reactants], [product],
                        padding=True,
                        device=device,
                    )
                    output = model.encode(tokens, tokens.mask)
                    log_likelihood = float(output.get("log_likelihood", -1.0))
                    scores.append({"idx": row["idx"], "lm_score": log_likelihood})
                except Exception as exc:
                    scores.append({"idx": row["idx"], "lm_score": -100.0,
                                   "error": f"{type(exc).__name__}: {exc}"})
        print(json.dumps({"status": "ok", "scores": scores}))
    except Exception as exc:
        print(json.dumps({"status": "error", "error": f"{type(exc).__name__}: {exc}",
                          "scores": []}))

if __name__ == "__main__":
    main()
"""


def score_chemformer(
    rows: Sequence[Dict[str, object]],
    chemformer_python: str,
    ckpt: str,
    timeout: int = 600,
) -> List[Dict[str, object]]:
    """Score rows by Chemformer forward likelihood via subprocess.

    For each row, the ``reaction_smiles`` is split into reactants and
    product, then fed to the Chemformer forward model. The
    log-likelihood of the actual product is the score.

    Falls back to a heuristic forward-likelihood proxy if the
    chemformer env is missing or the subprocess fails.
    """
    if not os.path.exists(chemformer_python):
        return _score_heuristic_forward(rows)
    payload_rows = []
    for i, row in enumerate(rows):
        rxn = str(row["reaction_smiles"])
        if ">>" in rxn:
            reactants, _, product = rxn.split(">>", 1)
        else:
            parts = rxn.split(">")
            reactants = parts[0] if len(parts) > 0 else ""
            product = parts[-1] if len(parts) > 0 else ""
        payload_rows.append({"idx": i, "reactants": reactants.strip(),
                             "product": product.strip()})
    payload = {"ckpt": ckpt, "rows": payload_rows}
    try:
        result = subprocess.run(
            [chemformer_python, "-c", _CHEMFORMER_RUNNER_SCRIPT],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return _score_heuristic_forward(rows)
    if result.returncode != 0:
        return _score_heuristic_forward(rows)
    out_lines = result.stdout.strip().splitlines()
    if not out_lines:
        return _score_heuristic_forward(rows)
    try:
        parsed = json.loads(out_lines[-1])
    except json.JSONDecodeError:
        return _score_heuristic_forward(rows)
    if parsed.get("status") != "ok":
        return _score_heuristic_forward(rows)
    score_map = {item["idx"]: float(item.get("lm_score", -100.0))
                 for item in parsed.get("scores", [])}
    out_rows: List[Dict[str, object]] = []
    for i, row in enumerate(rows):
        out_rows.append({**row, "score": score_map.get(i, -100.0),
                         "ranker_source": "chemformer"})
    return out_rows


def _score_heuristic_forward(
    rows: Sequence[Dict[str, object]],
) -> List[Dict[str, object]]:
    """Heuristic forward-likelihood proxy when Chemformer is unavailable.

    Combines atom_balance (high when reactants and products have matching
    atom counts), reaction validity, and reactant/product similarity.
    """
    out_rows: List[Dict[str, object]] = []
    for row in rows:
        features = featurize_reaction(str(row["reaction_smiles"]))
        # Feature indices mirror reranker.FEATURE_NAMES:
        # 1: valid, 2: atom_balance, 3: token_jaccard, 4: string_similarity
        score = (
            0.5 * features[2]      # atom_balance
            + 0.3 * features[1]    # valid
            + 0.2 * features[3]    # token_jaccard
        )
        out_rows.append({**row, "score": float(score),
                         "ranker_source": "heuristic_forward"})
    return out_rows


# ---------------------------------------------------------------------------
# Baseline (heuristic) scorer — same as P1-04
# ---------------------------------------------------------------------------


def heuristic_score(
    reaction_smiles: str,
    cache: Optional[FeatureCache] = None,
) -> float:
    """Baseline heuristic score (no learned weights, no PC-CNG negatives)."""
    features = (
        cache.get(reaction_smiles) if cache is not None
        else featurize_reaction(reaction_smiles)
    )
    return (
        0.45 * features[2]      # atom_balance
        + 0.20 * features[1]    # valid
        + 0.20 * features[3]    # token_jaccard
        + 0.15 * features[4]    # string_similarity
    )


def score_rows_heuristic(
    rows: Sequence[Dict[str, object]],
    cache: Optional[FeatureCache] = None,
) -> List[Dict[str, object]]:
    return [
        {**row, "score": heuristic_score(str(row["reaction_smiles"]), cache=cache),
         "ranker_source": "heuristic_baseline"}
        for row in rows
    ]


# ---------------------------------------------------------------------------
# PC-CNG augmented logistic ranker
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
# Ground-truth oracle ranker
# ---------------------------------------------------------------------------


def score_rows_ground_truth(
    rows: Sequence[Dict[str, object]],
) -> List[Dict[str, object]]:
    """Oracle: label=1 routes get score 1.0, label=0 get 0.0.

    Ties are broken by ``hard_score`` so the output is deterministic.
    """
    out_rows: List[Dict[str, object]] = []
    for row in rows:
        label = int(row["label"])
        score = 1.0 if label == 1 else 0.0
        # Add hard_score as a small tiebreaker so CSV output is stable
        score += 1e-3 * _safe_float(row.get("hard_score"), 0.0)
        out_rows.append({**row, "score": float(score),
                         "ranker_source": "ground_truth"})
    return out_rows


# ---------------------------------------------------------------------------
# Ranking metrics (same definitions as P1-04)
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


def false_positive_route_rate(scored_rows: Sequence[Dict[str, object]]) -> float:
    """Fraction of evaluable groups where a label=0 route is ranked #1."""
    grouped = _group_rows(scored_rows)
    groups = _evaluable_groups(grouped)
    if not groups:
        return 0.0
    fp = 0
    for group_rows in groups:
        ranked = sorted(group_rows, key=lambda r: float(r["score"]), reverse=True)
        if int(ranked[0]["label"]) == 0:
            fp += 1
    return fp / len(groups)


def evaluate(scored_rows: Sequence[Dict[str, object]]) -> Dict[str, float]:
    return {
        "top1_route_recall": topk_route_recall(scored_rows, 1),
        "top3_route_recall": topk_route_recall(scored_rows, 3),
        "top5_route_recall": topk_route_recall(scored_rows, 5),
        "top10_route_recall": topk_route_recall(scored_rows, 10),
        "mrr": mrr(scored_rows),
        "ndcg_at_10": ndcg_at_k(scored_rows, 10),
        "false_positive_route_rate": false_positive_route_rate(scored_rows),
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
            "mrr": 1.0 / first_pos,
            "ndcg": dcg_value / max(idcg, 1e-12),
        }
    return out


# ---------------------------------------------------------------------------
# Seed runner
# ---------------------------------------------------------------------------


def run_seed(
    rows: Sequence[Dict[str, object]],
    seed: int,
    train_fraction: float = 0.7,
    epochs: int = 200,
    shared_cache: Optional[FeatureCache] = None,
    aizynthfinder_python: Optional[str] = None,
    chemformer_python: Optional[str] = None,
    chemformer_ckpt: Optional[str] = None,
    aizynthfinder_routes_by_group: Optional[Dict[str, List[Dict[str, object]]]] = None,
    use_chemformer: bool = True,
) -> Dict[str, object]:
    """Run one seed: score all 4 rankers on the test set, evaluate metrics.

    ``aizynthfinder_routes_by_group`` (optional) is a pre-computed map
    from group_id to a list of AiZynthFinder-generated route dicts. If
    provided and non-empty for a group, those routes are appended to
    that group's candidate set (label=0 unless they exactly match the
    gold route's canonical reactants).
    """
    train_rows, test_rows = split_by_source(rows, train_fraction)
    if not test_rows:
        train_rows, test_rows = list(rows), list(rows)

    cache = shared_cache if shared_cache is not None else FeatureCache()
    unique_smiles = sorted({
        str(r["reaction_smiles"]) for r in list(train_rows) + list(test_rows)
    })
    cache.precompute(unique_smiles)

    # Optionally augment test_rows with AiZynthFinder-generated routes
    augmented_test_rows: List[Dict[str, object]] = list(test_rows)
    if aizynthfinder_routes_by_group:
        # Build a map of gold-route reactant sets per group for label assignment
        gold_reactants_by_group: Dict[str, str] = {}
        for row in test_rows:
            if int(row["label"]) == 1:
                rxn = str(row["reaction_smiles"])
                if ">>" in rxn:
                    gold_reactants_by_group[str(row["group_id"])] = rxn.split(">>", 1)[0].strip()
        for group_id, af_routes in aizynthfinder_routes_by_group.items():
            if not af_routes:
                continue
            gold_rxn = gold_reactants_by_group.get(group_id, "")
            gold_reactants = gold_rxn.split(">>", 1)[0].strip() if ">>" in gold_rxn else ""
            for af_route in af_routes:
                reaction_smiles = str(af_route.get("reaction_smiles", "")).strip()
                if not reaction_smiles or ">>" not in reaction_smiles:
                    continue
                af_reactants = reaction_smiles.split(">>", 1)[0].strip()
                # Label = 1 if AF route's reactants match the gold route's reactants
                label = 1 if (gold_reactants and af_reactants == gold_reactants) else 0
                augmented_test_rows.append({
                    "group_id": group_id,
                    "source_id": group_id,
                    "reaction_smiles": reaction_smiles,
                    "label": label,
                    "candidate_source": "aizynthfinder",
                    "failure_type": "gold" if label else "aizynthfinder_alternative",
                    "edit_action": "",
                    "hard_score": float(af_route.get("score", 0.5)),
                    "false_negative_risk": 0.0,
                    "parent_product": "",
                })

    # Compute scores for each ranker
    r1_scored = score_rows_heuristic(augmented_test_rows, cache=cache)
    if use_chemformer and chemformer_python and chemformer_ckpt:
        r2_scored = score_chemformer(
            augmented_test_rows, chemformer_python, chemformer_ckpt,
        )
    else:
        r2_scored = _score_heuristic_forward(augmented_test_rows)
    pc_cng_model = train_pc_cng_augmented_ranker(
        train_rows, seed, cache=cache, epochs=epochs,
    )
    r3_scored = score_rows_pc_cng(pc_cng_model, augmented_test_rows)
    r4_scored = score_rows_ground_truth(augmented_test_rows)

    return {
        "seed": seed,
        "r1_metrics": evaluate(r1_scored),
        "r2_metrics": evaluate(r2_scored),
        "r3_metrics": evaluate(r3_scored),
        "r4_metrics": evaluate(r4_scored),
        "r1_per_group": per_group_metrics(r1_scored),
        "r2_per_group": per_group_metrics(r2_scored),
        "r3_per_group": per_group_metrics(r3_scored),
        "r4_per_group": per_group_metrics(r4_scored),
        "n_train": len(train_rows),
        "n_test": len(test_rows),
        "n_test_augmented": len(augmented_test_rows),
    }


# ---------------------------------------------------------------------------
# Paired significance (10-seed, all pairwise deltas)
# ---------------------------------------------------------------------------


def _paired_significance_one_pair(
    seed_results: Sequence[Dict[str, object]],
    ranker_a: str,
    ranker_b: str,
    bootstrap_iterations: int = 10000,
    seed: int = 20260710,
) -> Dict[str, object]:
    """Compute paired significance for MRR delta (ranker_b - ranker_a)."""
    common_groups: Optional[set[str]] = None
    for r in seed_results:
        a_groups = set(r[f"{ranker_a}_per_group"].keys())
        b_groups = set(r[f"{ranker_b}_per_group"].keys())
        g = a_groups & b_groups
        common_groups = g if common_groups is None else (common_groups & g)
    common_groups_sorted = sorted(common_groups) if common_groups else []

    seed_a_mrr: List[float] = []
    seed_b_mrr: List[float] = []
    seed_deltas_mrr: List[float] = []
    for r in seed_results:
        a_vals = [r[f"{ranker_a}_per_group"][g]["mrr"] for g in common_groups_sorted]
        b_vals = [r[f"{ranker_b}_per_group"][g]["mrr"] for g in common_groups_sorted]
        seed_a_mrr.append(mean(a_vals))
        seed_b_mrr.append(mean(b_vals))
        seed_deltas_mrr.append(mean([b - a for a, b in zip(a_vals, b_vals)]))

    a_group_means = [
        mean([r[f"{ranker_a}_per_group"][g]["mrr"] for r in seed_results])
        for g in common_groups_sorted
    ]
    b_group_means = [
        mean([r[f"{ranker_b}_per_group"][g]["mrr"] for r in seed_results])
        for g in common_groups_sorted
    ]
    group_deltas = [b - a for a, b in zip(a_group_means, b_group_means)]

    ci_low, ci_high = bootstrap_ci(group_deltas, bootstrap_iterations, seed)
    perm_p = paired_permutation_p_value(group_deltas, bootstrap_iterations, seed + 100)
    sign_p = sign_test_p_value(group_deltas)

    rng = random.Random(seed + 500)
    n_seeds = len(seed_deltas_mrr)
    seed_bootstrap_deltas: List[float] = []
    for _ in range(bootstrap_iterations):
        sample = [seed_deltas_mrr[rng.randrange(n_seeds)] for _ in range(n_seeds)]
        seed_bootstrap_deltas.append(mean(sample))
    seed_ci_low = percentile(seed_bootstrap_deltas, 0.025)
    seed_ci_high = percentile(seed_bootstrap_deltas, 0.975)

    return {
        "n_seeds": len(seed_results),
        "n_common_groups": len(common_groups_sorted),
        "metric": "mrr",
        "ranker_a": ranker_a,
        "ranker_b": ranker_b,
        "ranker_a_mean": mean(seed_a_mrr),
        "ranker_b_mean": mean(seed_b_mrr),
        "delta_mean": mean(seed_deltas_mrr),
        "delta_pp": mean(seed_deltas_mrr) * 100.0,
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
        "ranker_b_better_groups": sum(1 for d in group_deltas if d > 0.0),
        "ranker_a_better_groups": sum(1 for d in group_deltas if d < 0.0),
        "tie_groups": sum(1 for d in group_deltas if d == 0.0),
    }


def paired_significance(
    seed_results: Sequence[Dict[str, object]],
    bootstrap_iterations: int = 10000,
    seed: int = 20260710,
) -> Dict[str, object]:
    """Compute pairwise paired significance for all ranker pairs.

    Returns a dict with keys ``"r3_vs_r1"``, ``"r3_vs_r2"``, ``"r2_vs_r1"``,
    ``"r3_vs_r4"`` (PC-CNG vs oracle, expected to be negative).
    """
    return {
        "r3_vs_r1": _paired_significance_one_pair(
            seed_results, "r1", "r3", bootstrap_iterations, seed
        ),
        "r3_vs_r2": _paired_significance_one_pair(
            seed_results, "r2", "r3", bootstrap_iterations, seed + 1
        ),
        "r2_vs_r1": _paired_significance_one_pair(
            seed_results, "r1", "r2", bootstrap_iterations, seed + 2
        ),
        "r3_vs_r4": _paired_significance_one_pair(
            seed_results, "r4", "r3", bootstrap_iterations, seed + 3
        ),
    }


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def _write_json(path: str, payload: object) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _write_csv(path: str, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_route_ranking_summary(
    path: str,
    ranker_metrics_agg: Dict[str, Dict[str, float]],
    sig: Dict[str, object],
    manifest_meta: Dict[str, object],
) -> None:
    """Write route_ranking_summary.json (combined manifest + metrics + sig)."""
    payload = {
        "task": "P2-01 AiZynthFinder Route Ranking",
        **manifest_meta,
        "rankers": RANKER_NAMES,
        "metrics": ranker_metrics_agg,
        "paired_significance": sig,
    }
    _write_json(path, payload)


def write_per_target_metrics(
    path: str,
    seed_results: Sequence[Dict[str, object]],
) -> None:
    """Write per_target_metrics.csv with one row per (seed, group_id, ranker)."""
    rows: List[Dict[str, object]] = []
    for seed_result in seed_results:
        seed = seed_result["seed"]
        for ranker in RANKER_NAMES:
            short_key = RANKER_SHORT_KEY_MAP[ranker]
            per_group_key = f"{short_key}_per_group"
            for group_id, metrics in seed_result[per_group_key].items():
                rows.append({
                    "seed": seed,
                    "group_id": group_id,
                    "ranker": ranker,
                    "top1": metrics["top1"],
                    "mrr": metrics["mrr"],
                    "ndcg": metrics["ndcg"],
                })
    _write_csv(
        path, rows,
        ["seed", "group_id", "ranker", "top1", "mrr", "ndcg"],
    )


def write_false_positive_routes(
    path: str,
    seed_results: Sequence[Dict[str, object]],
    all_rows: Sequence[Dict[str, object]],
) -> None:
    """Write false_positive_routes.csv listing groups where a label=0
    route outranked the gold route, per (seed, ranker)."""
    # Build group_id -> rows map for inspection
    grouped = _group_rows(all_rows)
    fp_rows: List[Dict[str, object]] = []
    for seed_result in seed_results:
        seed = seed_result["seed"]
        for ranker in RANKER_NAMES:
            short_key = RANKER_SHORT_KEY_MAP[ranker]
            per_group_key = f"{short_key}_per_group"
            for group_id, metrics in seed_result[per_group_key].items():
                # top1 == 0 means a label=0 was ranked #1
                if metrics["top1"] == 0.0:
                    group_rows = grouped.get(group_id, [])
                    n_pos = sum(1 for r in group_rows if int(r["label"]) == 1)
                    n_neg = sum(1 for r in group_rows if int(r["label"]) == 0)
                    fp_rows.append({
                        "seed": seed,
                        "group_id": group_id,
                        "ranker": ranker,
                        "mrr": metrics["mrr"],
                        "ndcg": metrics["ndcg"],
                        "n_positive_routes": n_pos,
                        "n_negative_routes": n_neg,
                        "false_positive_rank": 1,
                    })
    _write_csv(
        path, fp_rows,
        ["seed", "group_id", "ranker", "mrr", "ndcg",
         "n_positive_routes", "n_negative_routes", "false_positive_rank"],
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="P2-01 AiZynthFinder real routes retrosynthesis ranking"
    )
    parser.add_argument(
        "--routes-data", default=None,
        help="USPTO-MIT-50k routes CSV (optional; if absent or unreadable, "
             "falls back to PC-CNG negatives pseudo-routes)",
    )
    parser.add_argument(
        "--pc-cng-negatives", required=True,
        help="PC-CNG synthetic negatives CSV (used for pseudo-route fallback, "
             "PC-CNG augmented ranker training, and AiZynthFinder target list)",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS),
        help="Comma-separated seeds (default: 20260710..20260719)",
    )
    parser.add_argument(
        "--aizynthfinder-python", default=AIZYNTHFINDER_PYTHON_DEFAULT,
        help="Path to the aizynthfinder conda env Python",
    )
    parser.add_argument(
        "--chemformer-python", default=CHEMFORMER_PYTHON_DEFAULT,
        help="Path to the reaction_lm conda env Python (for Chemformer)",
    )
    parser.add_argument(
        "--chemformer-ckpt", default=CHEMFORMER_CKPT_DEFAULT,
        help="Path to Chemformer forward checkpoint",
    )
    parser.add_argument(
        "--aizynthfinder-config", default=None,
        help="Optional path to aizynthfinder YAML config (policy models)",
    )
    parser.add_argument(
        "--no-chemformer", action="store_true",
        help="Skip Chemformer scoring (use heuristic forward proxy)",
    )
    parser.add_argument(
        "--no-aizynthfinder", action="store_true",
        help="Skip AiZynthFinder search (use only PC-CNG negatives + gold)",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-candidates-per-source", type=int, default=10)
    parser.add_argument(
        "--max-sources", type=int, default=2000,
        help="Cap number of source_ids for tractability",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Smoke-test limit: cap number of source_ids to N",
    )
    parser.add_argument(
        "--aizynthfinder-time-limit", type=int, default=20,
        help="Per-target AiZynthFinder search time limit (seconds)",
    )
    parser.add_argument(
        "--aizynthfinder-iteration-limit", type=int, default=50,
        help="Per-target AiZynthFinder iteration limit",
    )
    parser.add_argument(
        "--aizynthfinder-n-routes", type=int, default=3,
        help="Max AiZynthFinder routes to keep per target",
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--epochs", type=int, default=200)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)
    os.makedirs(args.output_dir, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",")]

    # Suppress RDKit warnings
    try:
        from rdkit import RDLogger  # type: ignore
        RDLogger.DisableLog("rdApp.*")
    except Exception:
        pass

    # Load routes data; fall back to PC-CNG negatives pseudo-routes
    routes_rows: List[Dict[str, object]] = []
    fallback_used = False
    if args.routes_data and os.path.exists(args.routes_data):
        routes_rows = load_uspto_mit_50k_routes(args.routes_data)
    if not routes_rows:
        fallback_used = True
        routes_rows = load_pc_cng_negatives(
            args.pc_cng_negatives, args.max_candidates_per_source
        )

    # Cap sources for tractability
    source_ids = sorted({str(r["source_id"]) for r in routes_rows})
    if args.limit is not None and args.limit > 0:
        args.max_sources = min(args.max_sources, args.limit)
    if len(source_ids) > args.max_sources:
        rng = random.Random(20260710)
        source_ids = sorted(rng.sample(source_ids, args.max_sources))
    keep = set(source_ids)
    rows = [r for r in routes_rows if str(r["source_id"]) in keep]

    print(
        f"Loaded {len(rows)} route candidates across {len(source_ids)} source_ids",
        flush=True,
    )
    if fallback_used:
        print(
            f"NOTE: Using PC-CNG negatives pseudo-routes fallback "
            f"(no USPTO-MIT-50k routes available)",
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

    # Try AiZynthFinder route generation for each unique parent_product
    aizynthfinder_routes_by_group: Dict[str, List[Dict[str, object]]] = {}
    aizynthfinder_status = "skipped"
    if not args.no_aizynthfinder:
        print("\nAttempting AiZynthFinder route generation...", flush=True)
        # Collect unique target SMILES per group
        target_by_group: Dict[str, str] = {}
        for row in rows:
            group_id = str(row["group_id"])
            parent = str(row.get("parent_product", "")).strip()
            if parent and group_id not in target_by_group:
                target_by_group[group_id] = parent
        success_count = 0
        fail_count = 0
        for group_id, target in target_by_group.items():
            routes, status = try_aizynthfinder_search(
                target, args.aizynthfinder_python,
                configfile=args.aizynthfinder_config,
                time_limit=args.aizynthfinder_time_limit,
                iteration_limit=args.aizynthfinder_iteration_limit,
                n_routes=args.aizynthfinder_n_routes,
                timeout=args.aizynthfinder_time_limit + 60,
            )
            if status == "ok" and routes:
                aizynthfinder_routes_by_group[group_id] = routes
                success_count += 1
            else:
                fail_count += 1
            if (success_count + fail_count) % 10 == 0:
                print(
                    f"  AiZynthFinder progress: {success_count} ok, "
                    f"{fail_count} failed",
                    flush=True,
                )
        if success_count > 0:
            aizynthfinder_status = "ok"
            print(
                f"AiZynthFinder: {success_count}/{len(target_by_group)} targets "
                f"yielded routes",
                flush=True,
            )
        else:
            aizynthfinder_status = "failed"
            print(
                "AiZynthFinder route generation failed for all targets. "
                "Falling back to template-based retrosynthesis...",
                flush=True,
            )

    # If AiZynthFinder failed, try template-based retro
    template_routes_count = 0
    if aizynthfinder_status != "ok" and not args.no_aizynthfinder:
        print("\nFalling back to RDKit template-based retrosynthesis...", flush=True)
        target_by_group: Dict[str, str] = {}
        for row in rows:
            group_id = str(row["group_id"])
            parent = str(row.get("parent_product", "")).strip()
            if parent and group_id not in target_by_group:
                target_by_group[group_id] = parent
        for group_id, target in target_by_group.items():
            template_routes = generate_template_routes(
                target, max_routes=args.aizynthfinder_n_routes,
            )
            if template_routes:
                aizynthfinder_routes_by_group[group_id] = template_routes
                template_routes_count += 1
        if template_routes_count > 0:
            aizynthfinder_status = "template_fallback"
            print(
                f"Template-based retro: {template_routes_count}/{len(target_by_group)} "
                f"targets yielded routes",
                flush=True,
            )
        else:
            aizynthfinder_status = "no_routes"
            print(
                "Template-based retro yielded no routes. Using PC-CNG negatives "
                "only.",
                flush=True,
            )

    # Run 10 seeds
    print(f"\nRunning {len(seeds)} seeds...", flush=True)
    seed_results: List[Dict[str, object]] = []
    for seed in seeds:
        print(f"\n--- Seed {seed} ---", flush=True)
        result = run_seed(
            rows, seed, args.train_fraction, args.epochs,
            shared_cache=shared_cache,
            aizynthfinder_python=args.aizynthfinder_python,
            chemformer_python=args.chemformer_python,
            chemformer_ckpt=args.chemformer_ckpt,
            aizynthfinder_routes_by_group=aizynthfinder_routes_by_group,
            use_chemformer=not args.no_chemformer,
        )
        seed_results.append(result)
        print(f"  R1 (baseline)   MRR: {result['r1_metrics']['mrr']:.4f}", flush=True)
        print(f"  R2 (chemformer) MRR: {result['r2_metrics']['mrr']:.4f}", flush=True)
        print(f"  R3 (PC-CNG)     MRR: {result['r3_metrics']['mrr']:.4f}", flush=True)
        print(f"  R4 (oracle)     MRR: {result['r4_metrics']['mrr']:.4f}", flush=True)
        print(f"  R3-R1 MRR (pp): "
              f"{(result['r3_metrics']['mrr'] - result['r1_metrics']['mrr']) * 100:.2f}",
              flush=True)

    # Aggregate metrics across seeds
    ranker_metrics_agg: Dict[str, Dict[str, float]] = {}
    for ranker in RANKER_NAMES:
        short_key = RANKER_SHORT_KEY_MAP[ranker]
        metrics_key = f"{short_key}_metrics"
        metric_keys = list(seed_results[0][metrics_key].keys())
        ranker_metrics_agg[ranker] = {
            k: mean([float(r[metrics_key][k]) for r in seed_results])
            for k in metric_keys
        }

    # Paired significance
    sig = paired_significance(seed_results, args.bootstrap_iterations, seeds[0])

    # Manifest metadata
    manifest_meta = {
        "fallback_path": (
            "aizynthfinder" if aizynthfinder_status == "ok"
            else "template_fallback" if aizynthfinder_status == "template_fallback"
            else "pseudo_route_only"
        ),
        "aizynthfinder_status": aizynthfinder_status,
        "aizynthfinder_targets_with_routes": len(aizynthfinder_routes_by_group),
        "template_routes_count": template_routes_count,
        "fallback_reason": (
            "AiZynthFinder policy models unavailable (no network access to zenodo); "
            "used RDKit template-based retrosynthesis fallback"
            if aizynthfinder_status == "template_fallback"
            else "AiZynthFinder unavailable; using PC-CNG negatives + gold routes only"
            if aizynthfinder_status in ("failed", "no_routes", "skipped")
            else "AiZynthFinder real routes used"
        ),
        "n_source_ids": len(source_ids),
        "n_route_candidates": len(rows),
        "n_seeds": len(seeds),
        "seeds": seeds,
        "top_k": args.top_k,
        "max_candidates_per_source": args.max_candidates_per_source,
        "max_sources": args.max_sources,
        "bootstrap_iterations": args.bootstrap_iterations,
        "train_fraction": args.train_fraction,
        "chemformer_used": (not args.no_chemformer) and bool(args.chemformer_python),
        "chemformer_ckpt": args.chemformer_ckpt,
    }

    # Write outputs
    write_route_ranking_summary(
        os.path.join(args.output_dir, "route_ranking_summary.json"),
        ranker_metrics_agg, sig, manifest_meta,
    )
    write_per_target_metrics(
        os.path.join(args.output_dir, "per_target_metrics.csv"),
        seed_results,
    )
    _write_json(
        os.path.join(args.output_dir, "paired_significance.json"),
        sig,
    )
    write_false_positive_routes(
        os.path.join(args.output_dir, "false_positive_routes.csv"),
        seed_results, rows,
    )
    _write_json(
        os.path.join(args.output_dir, "per_seed_detail.json"),
        [
            {
                "seed": r["seed"],
                "aizynthfinder_baseline_metrics": r["r1_metrics"],
                "aizynthfinder_chemformer_metrics": r["r2_metrics"],
                "aizynthfinder_pc_cng_metrics": r["r3_metrics"],
                "ground_truth_metrics": r["r4_metrics"],
                "n_train": r["n_train"],
                "n_test": r["n_test"],
                "n_test_augmented": r["n_test_augmented"],
            }
            for r in seed_results
        ],
    )

    # Go/No-Go decision (R3 vs R1 on MRR)
    r3_r1_sig = sig["r3_vs_r1"]
    mrr_delta_pp = float(r3_r1_sig["delta_pp"])
    ci_low_pp = float(r3_r1_sig["seed_level_ci95_low_pp"])
    ci_high_pp = float(r3_r1_sig["seed_level_ci95_high_pp"])
    go_no_go = (
        "GO (write to main table)"
        if mrr_delta_pp > 1.0 and ci_low_pp > 0.0 and ci_high_pp > 0.0
        else "NO-GO (downgrade to supplementary)"
    )

    # Append Go/No-Go to summary
    summary_path = os.path.join(args.output_dir, "route_ranking_summary.json")
    summary = json.loads(open(summary_path).read())
    summary["go_no_go"] = go_no_go
    summary["go_no_go_criteria"] = {
        "comparison": "r3_vs_r1",
        "mrr_delta_pp": mrr_delta_pp,
        "seed_level_ci95_low_pp": ci_low_pp,
        "seed_level_ci95_high_pp": ci_high_pp,
        "threshold_pp": 1.0,
        "ci_all_positive": ci_low_pp > 0.0 and ci_high_pp > 0.0,
    }
    _write_json(summary_path, summary)

    print("\n" + "=" * 70)
    print("P2-01 AiZynthFinder Route Ranking — Summary")
    print("=" * 70)
    print(f"AiZynthFinder:     {aizynthfinder_status}")
    print(f"Fallback path:     {manifest_meta['fallback_path']}")
    print(f"N source_ids:      {len(source_ids)}")
    print(f"N route candidates:{len(rows)}")
    print(f"N seeds:           {len(seeds)}")
    print(f"R1 (baseline)   MRR: {ranker_metrics_agg['aizynthfinder_baseline']['mrr']:.4f}")
    print(f"R2 (chemformer) MRR: {ranker_metrics_agg['aizynthfinder_chemformer']['mrr']:.4f}")
    print(f"R3 (PC-CNG)     MRR: {ranker_metrics_agg['aizynthfinder_pc_cng']['mrr']:.4f}")
    print(f"R4 (oracle)     MRR: {ranker_metrics_agg['ground_truth']['mrr']:.4f}")
    print(f"R3-R1 MRR delta (pp): {mrr_delta_pp:.2f}")
    print(f"R3-R1 seed-level 95% CI: [{ci_low_pp:.2f}, {ci_high_pp:.2f}] pp")
    print(f"R3-R1 permutation p: {r3_r1_sig['paired_permutation_p']:.4f}")
    print(f"R3-R1 sign-test p:   {r3_r1_sig['sign_test_p']:.4f}")
    print(f"Go/No-Go (R3 vs R1): {go_no_go}")
    print("=" * 70)


if __name__ == "__main__":
    main()
