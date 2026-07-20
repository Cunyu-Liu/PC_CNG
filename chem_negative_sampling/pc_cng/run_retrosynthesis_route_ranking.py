"""Retrosynthesis Route Ranking with PC-CNG augmented reranker (P1-04).

Section 22.1 P1-04 task: rank candidate retrosynthesis routes per product,
comparing a baseline reranker (no PC-CNG negatives) vs a PC-CNG augmented
reranker that uses synthetic counterfactual negatives as pairwise training
signal.

FALLBACK NOTE
-------------
This implementation uses the "pseudo-route fallback" path because:
1. AiZynthFinder installation would downgrade rdkit from 2026.03.3 to
   2023.9.6, breaking the existing pc_cng environment.
2. USPTO-MIT-50k data is not available locally and TDC is not installed.

The pseudo-route dataset is derived from the PC-CNG synthetic negatives CSV
(which itself is derived from USPTO OpenMolecules): each ``source_id`` is
treated as a "product" with the ``positive_reaction`` as the gold route
(label=1) and each ``candidate_reaction`` as an alternative (incorrect)
route (label=0). This is explicitly tagged as "pseudo-route fallback" in
all output manifests.

Metrics
-------
- Top-K Route Recall (top-1/3/5/10): fraction of groups whose gold route
  appears in the top-K ranked candidates.
- MRR: Mean Reciprocal Rank of the first gold route.
- NDCG@10: Normalized Discounted Cumulative Gain at K=10.
- False-Positive Route Rate: fraction of groups where a label=0 route
  outranks the gold route.

Significance
------------
10-seed paired bootstrap CI + paired sign-flip permutation p + sign-test p
on MRR delta (PC-CNG augmented - baseline).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

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

FALLBACK_TAG = "pseudo-route"


# ---------------------------------------------------------------------------
# Cached featurizer (avoids repeated RDKit parsing across epochs)
# ---------------------------------------------------------------------------


class FeatureCache:
    """Cache featurize_reaction results to avoid repeated RDKit parsing.

    The base LogisticReactionRanker.fit calls featurize_reaction once per
    (epoch, row), which is O(epochs * N) RDKit calls. With 200 epochs and
    ~22k rows that is ~4.4M RDKit parses per seed — prohibitively slow.
    This cache computes each unique reaction's features once.
    """

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
    """LogisticReactionRanker backed by a FeatureCache.

    Same model class (logistic regression with SGD) but features are
    pre-cached so training is O(epochs * N) cheap arithmetic, not RDKit.
    """

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
        # Pre-compute features for all training rows (one RDKit pass)
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
    """Load PC-CNG synthetic negatives and build pseudo-route ranking rows.

    For each ``source_id`` the ``positive_reaction`` becomes a label=1
    candidate (gold route) and each ``candidate_reaction`` becomes a
    label=0 candidate (plausible but incorrect route).
    """
    rows: List[Dict[str, object]] = []
    seen_positives: Dict[str, str] = {}
    candidate_counts: Dict[str, int] = defaultdict(int)

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
            })
            candidate_counts[source_id] += 1

    return rows


def load_uspto_mit_50k_routes(path: str) -> List[Dict[str, object]]:
    """Load USPTO-MIT-50k multi-step routes CSV.

    Expected columns: ``product_smiles``, ``route_smiles``, ``route_id``,
    ``is_gold`` (1 for gold route, 0 for alternative). If the file is not
    present or malformed, returns an empty list (caller falls back to
    pseudo-routes from PC-CNG negatives).
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
            is_gold = int(_safe_float(record.get("is_gold", 0), 0.0))
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
            })
    return rows


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Baseline (heuristic, no PC-CNG negatives) and PC-CNG augmented ranker
# ---------------------------------------------------------------------------


def heuristic_score(
    reaction_smiles: str,
    cache: "FeatureCache | None" = None,
) -> float:
    """Baseline heuristic score with no learned weights and no PC-CNG negatives.

    Combines atom_balance, validity, and reactant-product similarity. This
    represents "route ranking without PC-CNG negatives" — only chemistry
    heuristics are used.
    """
    features = (
        cache.get(reaction_smiles) if cache is not None
        else featurize_reaction(reaction_smiles)
    )
    # Feature indices mirror reranker.FEATURE_NAMES:
    # 0: bias, 1: valid, 2: atom_balance, 3: token_jaccard,
    # 4: string_similarity, 5: one_reactant, 6: multi_reactant,
    # 7: product_reused_as_reactant, 8: has_leaving_group,
    # 9: normalized_atom_distance
    return (
        0.45 * features[2]      # atom_balance (most informative)
        + 0.20 * features[1]    # valid
        + 0.20 * features[3]    # token_jaccard
        + 0.15 * features[4]    # string_similarity
    )


def train_pc_cng_augmented_ranker(
    train_rows: Sequence[Dict[str, object]],
    seed: int,
    cache: FeatureCache,
    epochs: int = 200,
) -> CachedLogisticReactionRanker:
    """PC-CNG augmented ranker: LogisticReactionRanker trained on PC-CNG
    synthetic negatives as pairwise training signal (positives + candidates).

    Uses FeatureCache so RDKit parsing happens once per unique reaction,
    not once per (epoch, row).
    """
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


def score_rows_heuristic(
    rows: Sequence[Dict[str, object]],
    cache: "FeatureCache | None" = None,
) -> List[Dict[str, object]]:
    return [
        {**row, "score": heuristic_score(str(row["reaction_smiles"]), cache=cache)}
        for row in rows
    ]


def score_rows_model(
    model: CachedLogisticReactionRanker,
    rows: Sequence[Dict[str, object]],
) -> List[Dict[str, object]]:
    return [{**row, "score": float(model.predict_proba(str(row["reaction_smiles"])))} for row in rows]


# ---------------------------------------------------------------------------
# Ranking metrics
# ---------------------------------------------------------------------------


def _group_rows(
    scored_rows: Sequence[Dict[str, object]],
) -> Dict[str, List[Dict[str, object]]]:
    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in scored_rows:
        grouped[str(row["group_id"])].append(row)
    return grouped


def _evaluable_groups(grouped: Dict[str, List[Dict[str, object]]]) -> List[List[Dict[str, object]]]:
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
    """Mean Reciprocal Rank of the first gold route."""
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
    """NDCG@K with binary relevance (label=1 is relevant)."""
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
    """Fraction of evaluable groups where a label=0 route outranks the gold.

    A "false-positive route" is a label=0 candidate ranked at position 1
    (above the gold route).
    """
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
    shared_cache: "FeatureCache | None" = None,
) -> Dict[str, object]:
    """Run one seed: score baseline (heuristic) + PC-CNG augmented, evaluate.

    A shared FeatureCache can be passed across seeds so RDKit parsing of the
    same reaction happens only once across the entire 10-seed run.
    """
    train_rows, test_rows = split_by_source(rows, train_fraction)
    if not test_rows:
        train_rows, test_rows = list(rows), list(rows)

    cache = shared_cache if shared_cache is not None else FeatureCache()
    # Pre-compute features for all unique reactions in this seed's data
    unique_smiles = sorted({
        str(r["reaction_smiles"]) for r in list(train_rows) + list(test_rows)
    })
    cache.precompute(unique_smiles)

    # Baseline: heuristic scoring (no training, no PC-CNG negatives)
    baseline_scored = score_rows_heuristic(test_rows, cache=cache)
    baseline_metrics = evaluate(baseline_scored)

    # PC-CNG augmented: LogisticReactionRanker trained on PC-CNG negatives
    pc_cng_model = train_pc_cng_augmented_ranker(
        train_rows, seed, cache=cache, epochs=epochs,
    )
    pc_cng_scored = score_rows_model(pc_cng_model, test_rows)
    pc_cng_metrics = evaluate(pc_cng_scored)

    return {
        "seed": seed,
        "baseline_metrics": baseline_metrics,
        "pc_cng_metrics": pc_cng_metrics,
        "baseline_per_group": per_group_metrics(baseline_scored),
        "pc_cng_per_group": per_group_metrics(pc_cng_scored),
        "n_train": len(train_rows),
        "n_test": len(test_rows),
    }


# ---------------------------------------------------------------------------
# Paired significance (10-seed)
# ---------------------------------------------------------------------------


def paired_significance(
    seed_results: Sequence[Dict[str, object]],
    bootstrap_iterations: int = 10000,
    seed: int = 20260710,
) -> Dict[str, object]:
    """10-seed paired bootstrap CI + permutation p + sign-test p on MRR delta.

    Two levels of bootstrap:
    - Group-level: ensemble (mean across seeds) per-group MRR delta, then
      bootstrap over groups.
    - Seed-level: bootstrap over seeds (resample 10 seeds with replacement),
      each seed's MRR delta is the mean over common groups.
    """
    common_groups: set[str] | None = None
    for r in seed_results:
        g = set(r["baseline_per_group"].keys()) & set(r["pc_cng_per_group"].keys())
        common_groups = g if common_groups is None else (common_groups & g)
    common_groups_sorted = sorted(common_groups) if common_groups else []

    # Per-seed MRR delta (averaged over common groups)
    seed_baseline_mrr: List[float] = []
    seed_pc_cng_mrr: List[float] = []
    seed_deltas_mrr: List[float] = []
    for r in seed_results:
        b_vals = [r["baseline_per_group"][g]["mrr"] for g in common_groups_sorted]
        c_vals = [r["pc_cng_per_group"][g]["mrr"] for g in common_groups_sorted]
        seed_baseline_mrr.append(mean(b_vals))
        seed_pc_cng_mrr.append(mean(c_vals))
        seed_deltas_mrr.append(mean([c - b for c, b in zip(c_vals, b_vals)]))

    # Group-level deltas (ensemble mean across seeds per group)
    baseline_group_means = [
        mean([r["baseline_per_group"][g]["mrr"] for r in seed_results])
        for g in common_groups_sorted
    ]
    pc_cng_group_means = [
        mean([r["pc_cng_per_group"][g]["mrr"] for r in seed_results])
        for g in common_groups_sorted
    ]
    group_deltas = [c - b for c, b in zip(pc_cng_group_means, baseline_group_means)]

    # Group-level bootstrap CI + permutation p + sign-test p
    ci_low, ci_high = bootstrap_ci(group_deltas, bootstrap_iterations, seed)
    perm_p = paired_permutation_p_value(group_deltas, bootstrap_iterations, seed + 100)
    sign_p = sign_test_p_value(group_deltas)

    # Seed-level bootstrap CI (resample seeds with replacement)
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
        "baseline_mean": mean(seed_baseline_mrr),
        "pc_cng_mean": mean(seed_pc_cng_mrr),
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
        "candidate_better_groups": sum(1 for d in group_deltas if d > 0.0),
        "baseline_better_groups": sum(1 for d in group_deltas if d < 0.0),
        "tie_groups": sum(1 for d in group_deltas if d == 0.0),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _write_json(path: str, payload: object) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="P1-04 Retrosynthesis Route Ranking with PC-CNG augmented reranker"
    )
    parser.add_argument(
        "--routes-data", default=None,
        help="USPTO-MIT-50k routes CSV (optional; if absent or unreadable, "
             "falls back to pseudo-routes from --pc-cng-negatives)",
    )
    parser.add_argument(
        "--pc-cng-negatives", required=True,
        help="PC-CNG synthetic negatives CSV (used for both pseudo-route "
             "fallback and PC-CNG augmented ranker training)",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS),
        help="Comma-separated seeds (default: 20260710..20260719)",
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
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",")]

    # Suppress RDKit warnings (they flood the log during featurization)
    try:
        from rdkit import RDLogger  # type: ignore
        RDLogger.DisableLog("rdApp.*")
    except Exception:
        pass

    # Load routes data; fall back to pseudo-routes from PC-CNG negatives
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
            f"NOTE: Using {FALLBACK_TAG!r} fallback (no USPTO-MIT-50k routes available)",
            flush=True,
        )
        print(
            "      Pseudo-routes derived from PC-CNG synthetic negatives CSV",
            flush=True,
        )
    else:
        print("NOTE: Using USPTO-MIT-50k multi-step routes", flush=True)

    # Shared feature cache across all seeds (one RDKit pass per unique reaction)
    shared_cache = FeatureCache()
    # Pre-compute features for ALL unique reactions upfront
    unique_smiles = sorted({str(r["reaction_smiles"]) for r in rows})
    print(
        f"Pre-computing features for {len(unique_smiles)} unique reactions...",
        flush=True,
    )
    shared_cache.precompute(unique_smiles)
    print("Feature pre-computation complete.", flush=True)

    # Run 10 seeds
    seed_results: List[Dict[str, object]] = []
    for seed in seeds:
        print(f"\n--- Seed {seed} ---", flush=True)
        result = run_seed(
            rows, seed, args.train_fraction, args.epochs,
            shared_cache=shared_cache,
        )
        seed_results.append(result)
        b_mrr = result["baseline_metrics"]["mrr"]
        c_mrr = result["pc_cng_metrics"]["mrr"]
        print(f"  Baseline MRR:     {b_mrr:.4f}", flush=True)
        print(f"  PC-CNG MRR:       {c_mrr:.4f}", flush=True)
        print(f"  Delta MRR (pp):   {(c_mrr - b_mrr) * 100:.2f}", flush=True)

    # Aggregate metrics across seeds
    metric_keys = list(seed_results[0]["baseline_metrics"].keys())
    baseline_metrics_agg = {
        k: mean([float(r["baseline_metrics"][k]) for r in seed_results])
        for k in metric_keys
    }
    pc_cng_metrics_agg = {
        k: mean([float(r["pc_cng_metrics"][k]) for r in seed_results])
        for k in metric_keys
    }

    # Build output JSONs
    topk_data = {
        "fallback": FALLBACK_TAG if fallback_used else "uspto_mit_50k",
        "baseline": {
            "top1": baseline_metrics_agg["top1_route_recall"],
            "top3": baseline_metrics_agg["top3_route_recall"],
            "top5": baseline_metrics_agg["top5_route_recall"],
            "top10": baseline_metrics_agg["top10_route_recall"],
        },
        "pc_cng_augmented": {
            "top1": pc_cng_metrics_agg["top1_route_recall"],
            "top3": pc_cng_metrics_agg["top3_route_recall"],
            "top5": pc_cng_metrics_agg["top5_route_recall"],
            "top10": pc_cng_metrics_agg["top10_route_recall"],
        },
        "delta": {
            k: pc_cng_metrics_agg[k] - baseline_metrics_agg[k]
            for k in (
                "top1_route_recall", "top3_route_recall",
                "top5_route_recall", "top10_route_recall",
            )
        },
    }
    mrr_data = {
        "fallback": FALLBACK_TAG if fallback_used else "uspto_mit_50k",
        "baseline": baseline_metrics_agg["mrr"],
        "pc_cng_augmented": pc_cng_metrics_agg["mrr"],
        "delta": pc_cng_metrics_agg["mrr"] - baseline_metrics_agg["mrr"],
        "delta_pp": (pc_cng_metrics_agg["mrr"] - baseline_metrics_agg["mrr"]) * 100.0,
    }
    ndcg_data = {
        "fallback": FALLBACK_TAG if fallback_used else "uspto_mit_50k",
        "baseline": baseline_metrics_agg["ndcg_at_10"],
        "pc_cng_augmented": pc_cng_metrics_agg["ndcg_at_10"],
        "delta": pc_cng_metrics_agg["ndcg_at_10"] - baseline_metrics_agg["ndcg_at_10"],
        "delta_pp": (pc_cng_metrics_agg["ndcg_at_10"] - baseline_metrics_agg["ndcg_at_10"]) * 100.0,
    }
    fp_data = {
        "fallback": FALLBACK_TAG if fallback_used else "uspto_mit_50k",
        "baseline": baseline_metrics_agg["false_positive_route_rate"],
        "pc_cng_augmented": pc_cng_metrics_agg["false_positive_route_rate"],
        "delta": pc_cng_metrics_agg["false_positive_route_rate"] - baseline_metrics_agg["false_positive_route_rate"],
        "delta_pp": (pc_cng_metrics_agg["false_positive_route_rate"] - baseline_metrics_agg["false_positive_route_rate"]) * 100.0,
    }

    sig = paired_significance(seed_results, args.bootstrap_iterations, seeds[0])

    _write_json(os.path.join(args.output_dir, "topk_route_recall.json"), topk_data)
    _write_json(os.path.join(args.output_dir, "mrr.json"), mrr_data)
    _write_json(os.path.join(args.output_dir, "ndcg.json"), ndcg_data)
    _write_json(os.path.join(args.output_dir, "false_positive_route_rate.json"), fp_data)
    _write_json(os.path.join(args.output_dir, "paired_significance.json"), sig)

    # Go/No-Go decision
    mrr_delta_pp = (pc_cng_metrics_agg["mrr"] - baseline_metrics_agg["mrr"]) * 100.0
    ci_low_pp = sig["seed_level_ci95_low_pp"]
    ci_high_pp = sig["seed_level_ci95_high_pp"]
    go_no_go = (
        "GO (write to main table)"
        if mrr_delta_pp > 1.0 and ci_low_pp > 0.0 and ci_high_pp > 0.0
        else "NO-GO (downgrade to supplementary)"
    )

    manifest = {
        "task": "P1-04 Retrosynthesis Route Ranking",
        "fallback_path": FALLBACK_TAG if fallback_used else "uspto_mit_50k",
        "fallback_reason": (
            "AiZynthFinder install would downgrade rdkit 2026.03.3 -> 2023.9.6; "
            "USPTO-MIT-50k not available locally; TDC not installed"
            if fallback_used else "USPTO-MIT-50k routes loaded successfully"
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
        "baseline_metrics_mean": baseline_metrics_agg,
        "pc_cng_metrics_mean": pc_cng_metrics_agg,
        "paired_significance": sig,
        "go_no_go": go_no_go,
        "go_no_go_criteria": {
            "mrr_delta_pp": mrr_delta_pp,
            "seed_level_ci95_low_pp": ci_low_pp,
            "seed_level_ci95_high_pp": ci_high_pp,
            "threshold_pp": 1.0,
            "ci_all_positive": ci_low_pp > 0.0 and ci_high_pp > 0.0,
        },
    }
    _write_json(os.path.join(args.output_dir, "manifest.json"), manifest)

    # Per-seed detail
    _write_json(
        os.path.join(args.output_dir, "per_seed_detail.json"),
        [
            {
                "seed": r["seed"],
                "baseline_metrics": r["baseline_metrics"],
                "pc_cng_metrics": r["pc_cng_metrics"],
                "n_train": r["n_train"],
                "n_test": r["n_test"],
            }
            for r in seed_results
        ],
    )

    print("\n" + "=" * 60)
    print("P1-04 Retrosynthesis Route Ranking — Summary")
    print("=" * 60)
    print(f"Fallback:          {manifest['fallback_path']}")
    print(f"N source_ids:      {len(source_ids)}")
    print(f"N route candidates:{len(rows)}")
    print(f"N seeds:           {len(seeds)}")
    print(f"Baseline MRR:      {baseline_metrics_agg['mrr']:.4f}")
    print(f"PC-CNG MRR:        {pc_cng_metrics_agg['mrr']:.4f}")
    print(f"Delta MRR (pp):    {mrr_delta_pp:.2f}")
    print(f"Seed-level 95% CI: [{ci_low_pp:.2f}, {ci_high_pp:.2f}] pp")
    print(f"Permutation p:     {sig['paired_permutation_p']:.4f}")
    print(f"Sign-test p:       {sig['sign_test_p']:.4f}")
    print(f"Go/No-Go:          {go_no_go}")
    print("=" * 60)


if __name__ == "__main__":
    main()
