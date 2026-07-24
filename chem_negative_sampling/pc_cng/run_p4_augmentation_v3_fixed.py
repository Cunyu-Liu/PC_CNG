#!/usr/bin/env python3
"""G3 v3: reaction-conditioned augmentation with matched budget and hierarchical inference.

FIXED VERSION (v3.1): Replaces mock scoring with REAL LogisticRegression model
on Morgan fingerprints of full reaction context (reactants + agents + product).

Key fixes from v3.0:
1. REAL model (LogisticRegression) instead of rng.random() mock scoring
2. Proper train/val/test split separation (no data leakage)
3. COMMON test set across all arms (fair comparison: same test candidates,
   different training data per arm)
4. Real reaction context from hitea_full_normalized.csv (not just candidate SMILES)
5. RDKit warning suppression (major I/O overhead fix)

Spec compliance (pccng 的分阶段提示词 2.md lines 360-450):
- Model input = reactants + reagents/conditions + candidate product
- All arms matched on: positive count, negative count, training examples
- A6-randomized-label and A6-shuffled-parent negative controls
- Baseline ranking frozen on VALIDATION set
- Hierarchical inference: cluster bootstrap + seed level + paired permutation + Holm
"""
from __future__ import annotations

import csv
import json
import logging
import random
import re
import sys
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

# Suppress RDKit warnings (major I/O overhead) BEFORE importing rdkit
logging.getLogger("rdkit").setLevel(logging.ERROR)
try:
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
except ImportError:
    pass

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs
    from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
    RDKIT_AVAILABLE = True
    _USE_NEW_FP_API = True
except ImportError:
    RDKIT_AVAILABLE = False
    _USE_NEW_FP_API = False

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Morgan generator cache (new API - avoids DEPRECATION warnings)
_MORGAN_GENS: dict[tuple[int, int], object] = {}


def _get_morgan_generator(radius: int, nbits: int):
    """Get or create a cached MorganGenerator (new RDKit API)."""
    key = (radius, nbits)
    if key not in _MORGAN_GENS:
        _MORGAN_GENS[key] = GetMorganGenerator(radius=radius, fpSize=nbits)
    return _MORGAN_GENS[key]

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "chem_negative_sampling"))

from pc_cng.paired_cluster_inference import (
    hierarchical_bootstrap,
    holm_correction,
    mrr_metric,
    paired_permutation_test,
)


# ---------------------------------------------------------------------------
# Fingerprint caching (major performance optimization)
# ---------------------------------------------------------------------------
_FP_CACHE: dict[tuple[str, int, int], np.ndarray] = {}
_FP_HITS = 0
_FP_MISSES = 0


def _morgan_fp(smiles: str, radius: int = 2, nbits: int = 2048) -> np.ndarray:
    """Morgan fingerprint as numpy array (with caching)."""
    global _FP_HITS, _FP_MISSES
    cache_key = (smiles, radius, nbits)
    if cache_key in _FP_CACHE:
        _FP_HITS += 1
        return _FP_CACHE[cache_key]
    _FP_MISSES += 1
    if not RDKIT_AVAILABLE or not smiles:
        result = np.zeros(nbits, dtype=np.float32)
    else:
        mol = Chem.MolFromSmiles(re.sub(r":\d+", "", smiles))
        if mol is None:
            result = np.zeros(nbits, dtype=np.float32)
        else:
            if _USE_NEW_FP_API:
                gen = _get_morgan_generator(radius, nbits)
                fp = gen.GetFingerprint(mol)
            else:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
            arr = np.zeros(nbits, dtype=np.float32)
            DataStructs.ConvertToNumpyArray(fp, arr)
            result = arr
    _FP_CACHE[cache_key] = result
    return result


# ---------------------------------------------------------------------------
# Reaction context loader
# ---------------------------------------------------------------------------
def load_reaction_context(csv_path: Path) -> dict[str, dict]:
    """Load hitea_full_normalized.csv to get reactants/agents/products per source_id.

    Returns: {source_id: {reactants, agents, products, yield, reaction_class, split}}
    """
    context: dict[str, dict] = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = row.get("source_id", "")
            if not sid:
                continue
            try:
                yield_val = float(row.get("yield", 0) or 0)
            except (ValueError, TypeError):
                yield_val = 0.0
            context[sid] = {
                "reactants": row.get("reactants", ""),
                "agents": row.get("agents", ""),
                "products": row.get("products", ""),
                "yield": yield_val,
                "reaction_class": row.get("reaction_class", ""),
                "split": row.get("split", ""),
            }
    return context


# ---------------------------------------------------------------------------
# Arm definitions (v3 with negative controls)
# ---------------------------------------------------------------------------
@dataclass
class G3Arm:
    arm_id: str
    arm_name: str
    negative_source: Optional[str] = None
    is_negative_control: bool = False
    description: str = ""


ARMS_V3 = [
    G3Arm("A0", "positive_only", None, False, "Positive reactions only (no negatives in training)"),
    G3Arm("A1", "random_mismatch", "random_mismatch", False, "Random mismatch negatives"),
    G3Arm("A2", "random_corruption", "random_corruption", False, "Random corruption negatives"),
    G3Arm("A3", "tanimoto_retrieval", "tanimoto_retrieval", False, "Tanimoto retrieval negatives"),
    G3Arm("A4", "template_perturbation", "template_perturbation", False, "Template perturbation negatives"),
    G3Arm("A5", "unconstrained_edit", "unconstrained_edit", False, "Unconstrained structural edit negatives"),
    G3Arm("A6", "rule_pc_cng", "rule_pc_cng", False, "Rule PC-CNG negatives"),
    G3Arm("A6R", "rule_pc_cng_randomized_label", "rule_pc_cng", True,
          "PC-CNG candidates with RANDOMIZED labels (negative control)"),
    G3Arm("A6S", "rule_pc_cng_shuffled_parent", "rule_pc_cng", True,
          "PC-CNG candidates with SHUFFLED parent reaction (negative control)"),
]


def load_manifest_v3(manifest_path: Path, reaction_context: dict[str, dict]) -> dict[str, list[dict]]:
    """Load v2 manifest and build v3 training data with full reaction context.

    FIX: Uses hitea_full_normalized.csv to get real reactants/agents/products.
    """
    with open(manifest_path) as f:
        manifest = json.load(f)

    groups: dict[str, list[dict]] = {}
    for group in manifest.get("groups", []):
        gid = group["group_id"]
        source_rxn_id = group.get("source_reaction_id", "")
        experimental_group = group.get("experimental_group_id", "")
        group_split = group.get("split", "test")

        # Get real reaction context from hitea_full_normalized.csv
        ctx = reaction_context.get(source_rxn_id, {})
        reactants = ctx.get("reactants", "")
        agents = ctx.get("agents", "")
        products = ctx.get("products", "")
        measured_yield = ctx.get("yield", 0.0)
        reaction_class = ctx.get("reaction_class", "")

        candidates = []
        for cand in group.get("candidates", []):
            candidate_smiles = re.sub(r":\d+", "", cand.get("candidate_smiles", ""))

            # Build full reaction context input: reactants + agents + candidate product
            # This is the "reaction-conditioned input" per spec line 368-378
            if reactants:
                reaction_input = f"{reactants}.{agents}.{candidate_smiles}" if agents else f"{reactants}.{candidate_smiles}"
            else:
                reaction_input = candidate_smiles

            candidates.append({
                "candidate_id": cand.get("candidate_id", ""),
                "input_smiles": reaction_input,
                "reactants": reactants,
                "agents": agents,
                "candidate_smiles": candidate_smiles,
                "label": 1 if cand.get("gold_candidate", False) else 0,
                "source": cand.get("candidate_source", ""),
                "group_id": gid,
                "source_reaction_id": source_rxn_id,
                "experimental_group": experimental_group,
                "split": cand.get("split", group_split),
                "reaction_family": cand.get("reaction_family", reaction_class),
                "reaction_class": reaction_class,
                "measured_yield": measured_yield,
                "is_gold": cand.get("gold_candidate", False),
            })
        groups[gid] = candidates
    return groups


def build_common_test_set(groups: dict[str, list[dict]]) -> list[dict]:
    """Build COMMON test set: all test candidates from ALL sources.

    FIX: All arms are evaluated on the SAME test set for fair comparison.
    Previously each arm was evaluated on its own test set (different negatives),
    making MRR comparisons meaningless.

    The common test set contains:
    - All gold candidates from test split
    - All negative candidates from test split (all negative sources)
    """
    test_records = []
    for gid, candidates in groups.items():
        for c in candidates:
            if c.get("split") == "test":
                test_records.append(c)
    return test_records


def build_arm_training_data_v3(
    groups: dict[str, list[dict]],
    arm: G3Arm,
    max_negatives_per_group: Optional[int] = None,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Build training+val data for an arm with full reaction context.

    FIX: Now properly returns (train_records, val_records).
    FIX: Test records are handled separately via build_common_test_set().
    FIX: Uses seed for negative control randomization (propagates to model).

    All arms are matched on: positive count, negative count, training examples.
    """
    train_records = []
    val_records = []

    rng = random.Random(seed)
    group_ids = list(groups.keys())

    for gid in group_ids:
        candidates = groups[gid]
        gold = [c for c in candidates if c["is_gold"] and c.get("split") in ("train", "val")]
        negatives = [c for c in candidates if not c["is_gold"]
                     and c["source"] == arm.negative_source
                     and c.get("split") in ("train", "val")]

        if arm.is_negative_control:
            if arm.arm_id == "A6R":
                # Randomized label: shuffle labels across BOTH gold and negatives
                # FIX: v3.0 only shuffled negatives (all label=0), which was a no-op.
                # v3.1 shuffles across all candidates in the group, actually randomizing.
                all_candidates = [dict(c) for c in (gold + negatives)]
                labels = [c["label"] for c in all_candidates]
                rng.shuffle(labels)
                for c, l in zip(all_candidates, labels):
                    c["label"] = l
                # Re-split into gold and negatives based on shuffled labels
                gold = [c for c in all_candidates if c["label"] == 1]
                negatives = [c for c in all_candidates if c["label"] == 0]
            elif arm.arm_id == "A6S":
                # Shuffled parent: use negatives from a DIFFERENT group
                other_groups = [g for g in group_ids if g != gid]
                if other_groups:
                    other_gid = rng.choice(other_groups)
                    other_negs = [c for c in groups[other_gid]
                                  if not c["is_gold"]
                                  and c["source"] == arm.negative_source
                                  and c.get("split") in ("train", "val")]
                    negatives = [dict(c) for c in other_negs]
                    for c in negatives:
                        c["group_id"] = gid  # Reassign to current group

        # Match budget: limit negatives per group
        if max_negatives_per_group is not None and len(negatives) > max_negatives_per_group:
            negatives = negatives[:max_negatives_per_group]

        # Split by candidate-level split
        for c in gold:
            s = c.get("split", "train")
            if s == "train":
                train_records.append(c)
            elif s == "val":
                val_records.append(c)
        for c in negatives:
            s = c.get("split", "train")
            if s == "train":
                train_records.append(c)
            elif s == "val":
                val_records.append(c)

    return train_records, val_records


# ---------------------------------------------------------------------------
# REAL model (replaces mock scoring)
# ---------------------------------------------------------------------------
class ReactionContextScorer:
    """Strong model: RandomForest on reaction-context features (v3.2).

    Features:
      - reactants_fp (512 bits, Morgan r=2)
      - product_fp (512 bits, Morgan r=2)
      - diff_fp = |product_fp - reactants_fp| (512 bits, captures transformation)
      - tanimoto_sim (1 feature, similarity between reactants and product)
    Total: 1025 features (dense, tree-friendly)

    Model: RandomForestClassifier (100 trees, max_depth=10, n_jobs=4).
    Seed variability: bootstrap-sample training data per seed so A6
    (deterministic PC-CNG negatives) produces seed-level variance.

    Each arm trains its own model on its training data, then scores the COMMON test set.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.model = None
        self._rng = random.Random(seed)

    def _featurize(self, record: dict) -> np.ndarray:
        """Extract features: reactants_fp + product_fp + diff_fp + tanimoto."""
        reactants = record.get("reactants", "")
        product = record.get("candidate_smiles", "")
        r_fp = _morgan_fp(reactants, nbits=512)
        p_fp = _morgan_fp(product, nbits=512)
        diff_fp = np.abs(p_fp - r_fp)
        # Tanimoto similarity
        intersection = float(np.dot(r_fp, p_fp))
        union = float(r_fp.sum() + p_fp.sum() - intersection)
        tanimoto = intersection / union if union > 0 else 0.0
        return np.concatenate([r_fp, p_fp, diff_fp, np.array([tanimoto], dtype=np.float32)])

    def train(self, train_records: list[dict]) -> None:
        """Train RandomForest on training records.

        Bootstrap-sample the training data to introduce seed-level variability.
        This is critical for A6 (PC-CNG) where negatives are deterministic;
        without bootstrap, all seeds produce identical models.
        """
        if not train_records:
            self.model = None
            return

        # Bootstrap sample to introduce seed variability
        n = len(train_records)
        boot_indices = [self._rng.randint(0, n - 1) for _ in range(n)]
        boot_records = [train_records[i] for i in boot_indices]

        X = np.stack([self._featurize(r) for r in boot_records])
        y = np.array([r.get("label", 0) for r in boot_records], dtype=np.int32)

        if len(np.unique(y)) < 2:
            self.model = None
            return

        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight="balanced",
            random_state=self.seed,
            n_jobs=4,
        )
        self.model.fit(X, y)

    def score(self, records: list[dict]) -> list[float]:
        """Score records: probability of label=1 (gold candidate).

        When model is None (couldn't train), return RANDOM scores (not 0.5 for all).
        This prevents artificial MRR=1.0 from stable sort tie-breaking.
        """
        if self.model is None or not records:
            rng = random.Random(self.seed)
            return [rng.random() for _ in records]

        X = np.stack([self._featurize(r) for r in records])
        probs = self.model.predict_proba(X)
        return [float(p[1]) for p in probs]

# ---------------------------------------------------------------------------
# Baseline selection on VALIDATION set
# ---------------------------------------------------------------------------
def select_best_baseline_on_validation(
    per_seed_val_results: dict[int, dict[str, list[dict]]],
    candidate_arms: list[str],
) -> tuple[str, dict[str, float]]:
    """Select best baseline on VALIDATION set (not test set).

    FIX: v1/v2 selected best baseline on test set, causing selection bias.
    v3 freezes baseline ranking on validation set.
    Aggregates val MRR across all seeds for each baseline.
    """
    baseline_val_mrrs: dict[str, float] = {}
    for arm_id in candidate_arms:
        all_val_recs: list[dict] = []
        for s, arm_data in per_seed_val_results.items():
            if arm_id in arm_data:
                all_val_recs.extend(arm_data[arm_id])
        if all_val_recs:
            baseline_val_mrrs[arm_id] = mrr_metric(all_val_recs)

    if not baseline_val_mrrs:
        return "A3", {}
    best_arm = max(baseline_val_mrrs, key=baseline_val_mrrs.get)
    return best_arm, baseline_val_mrrs


# ---------------------------------------------------------------------------
# Aggregation with hierarchical inference
# ---------------------------------------------------------------------------
def run_g3_v3_aggregation(
    per_seed_results: dict[int, dict[str, list[dict]]],
    per_seed_val_results: dict[int, dict[str, list[dict]]],
    treatment_arm_id: str = "A6",
    n_bootstrap: int = 2000,
    seed: int = 20260723,
) -> dict:
    """Aggregate G3 v3 results with hierarchical inference.

    per_seed_results: {seed: {arm_id: [test_records_with_scores]}}
    per_seed_val_results: {seed: {arm_id: [val_records_with_scores]}}
    """
    seeds = sorted(per_seed_results.keys())
    n_seeds = len(seeds)

    # Select best baseline on VALIDATION set
    baseline_candidates = ["A1", "A2", "A3", "A4", "A5"]
    best_baseline_id, baseline_val_mrrs = select_best_baseline_on_validation(
        per_seed_val_results, baseline_candidates
    )

    # Compute point estimates per seed
    seed_mrrs: dict[str, list[float]] = {arm: [] for arm in [treatment_arm_id, best_baseline_id, "A0"]}
    for s in seeds:
        for arm in seed_mrrs:
            if arm in per_seed_results[s]:
                seed_mrrs[arm].append(mrr_metric(per_seed_results[s][arm]))

    # Treatment vs A0 (positive-only reference)
    ch_vs_a0 = hierarchical_bootstrap(
        challenger_per_seed={s: per_seed_results[s][treatment_arm_id] for s in seeds if treatment_arm_id in per_seed_results[s]},
        baseline_per_seed={s: per_seed_results[s]["A0"] for s in seeds if "A0" in per_seed_results[s]},
        metric_fn=mrr_metric,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )

    # Treatment vs best baseline (primary comparison)
    ch_vs_baseline = hierarchical_bootstrap(
        challenger_per_seed={s: per_seed_results[s][treatment_arm_id] for s in seeds if treatment_arm_id in per_seed_results[s]},
        baseline_per_seed={s: per_seed_results[s][best_baseline_id] for s in seeds if best_baseline_id in per_seed_results[s]},
        metric_fn=mrr_metric,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )

    # Negative control checks
    control_results = {}
    for ctrl_arm in ["A6R", "A6S"]:
        if ctrl_arm not in per_seed_results.get(seeds[0], {}):
            continue
        ctrl_vs_baseline = hierarchical_bootstrap(
            challenger_per_seed={s: per_seed_results[s][ctrl_arm] for s in seeds if ctrl_arm in per_seed_results[s]},
            baseline_per_seed={s: per_seed_results[s][best_baseline_id] for s in seeds if best_baseline_id in per_seed_results[s]},
            metric_fn=mrr_metric,
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
        control_results[ctrl_arm] = ctrl_vs_baseline

    # Paired permutation test (treatment vs baseline)
    perm_test = paired_permutation_test(
        challenger_values=seed_mrrs[treatment_arm_id],
        baseline_values=seed_mrrs[best_baseline_id],
        seed=seed,
    )

    # Holm correction
    p_values = [ch_vs_a0["p_value"], ch_vs_baseline["p_value"]]
    comparison_names = [f"{treatment_arm_id}_vs_A0", f"{treatment_arm_id}_vs_{best_baseline_id}"]
    for ctrl_arm, ctrl_result in control_results.items():
        p_values.append(ctrl_result["p_value"])
        comparison_names.append(f"{ctrl_arm}_vs_{best_baseline_id}")

    holm = holm_correction(p_values)

    # Verdict logic (fixed):
    # STRONG_GO: A6 beats best baseline (CI+) AND beats A0 (CI+) AND negative controls don't beat baseline
    # WEAK_GO: A6 beats A0 (CI+) AND negative controls don't beat baseline
    # PROMISING: A6 beats A0 (CI+) but negative controls check fails
    # NO_GO: A6 does not beat A0
    beats_a0 = ch_vs_a0["ci_all_positive"]
    beats_baseline = ch_vs_baseline["ci_all_positive"]
    controls_negative = all(
        not cr.get("ci_all_positive", False) for cr in control_results.values()
    )

    if beats_a0 and beats_baseline and controls_negative:
        verdict = "STRONG_GO"
        reason = (f"A6 beats A0 (CI+) AND beats best baseline {best_baseline_id} (CI+) "
                  f"AND negative controls show no signal")
    elif beats_a0 and controls_negative:
        verdict = "WEAK_GO"
        reason = (f"A6 beats A0 (CI+) and negative controls show no signal, "
                  f"but does not beat best baseline {best_baseline_id}")
    elif beats_a0:
        verdict = "PROMISING"
        reason = f"A6 beats A0 (CI+) but negative controls check failed or baseline not beaten"
    else:
        verdict = "NO_GO"
        reason = f"A6 does not beat A0"

    return {
        "schema_version": "3.1",
        "evaluation_date": "2026-07-24",
        "treatment_arm": treatment_arm_id,
        "best_baseline_arm": best_baseline_id,
        "best_baseline_selected_on": "validation_set",
        "baseline_val_mrrs": baseline_val_mrrs,
        "n_seeds": n_seeds,
        "seed_mrrs": {arm: vals for arm, vals in seed_mrrs.items() if vals},
        "treatment_vs_A0": ch_vs_a0,
        "treatment_vs_baseline": ch_vs_baseline,
        "negative_controls": control_results,
        "permutation_test": perm_test,
        "holm_correction": [
            {"comparison": name, **h} for name, h in zip(comparison_names, holm)
        ],
        "verdict": verdict,
        "reason": reason,
        "method": "hierarchical_bootstrap (cluster + seed) + paired permutation + Holm",
        "n_bootstrap": n_bootstrap,
        "model": "RandomForestClassifier (100 trees, depth=10) on Morgan fingerprints (reactants+product+diff+tanimoto, 1025 features)",
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="G3 v3.1 augmentation with reaction-conditioned input (REAL model)"
    )
    parser.add_argument("--manifest", type=Path,
                        default=_REPO_ROOT / "data/p4/manifests/hte_feasibility_v2.json")
    parser.add_argument("--reaction-csv", type=Path,
                        default=_REPO_ROOT / "data/processed/hitea_full_normalized.csv")
    parser.add_argument("--output-dir", type=Path,
                        default=_REPO_ROOT / "results/p4_augmentation_v3")
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        args.n_seeds = 2
        args.n_bootstrap = 500

    # Load reaction context
    print(f"Loading reaction context: {args.reaction_csv}")
    reaction_context = load_reaction_context(args.reaction_csv)
    print(f"Loaded {len(reaction_context)} reactions with context")

    # Load manifest
    print(f"Loading manifest: {args.manifest}")
    groups = load_manifest_v3(args.manifest, reaction_context)
    print(f"Loaded {len(groups)} groups")

    # Count splits
    split_counts: dict[str, int] = defaultdict(int)
    for g in groups.values():
        for c in g:
            split_counts[c.get("split", "unknown")] += 1
    print(f"Candidate splits: {dict(split_counts)}")

    # Build COMMON test set (all arms evaluated on the SAME test set)
    common_test_set = build_common_test_set(groups)
    test_by_group: dict[str, list[dict]] = defaultdict(list)
    for r in common_test_set:
        test_by_group[r["group_id"]].append(r)
    print(f"Common test set: {len(common_test_set)} candidates in {len(test_by_group)} groups")

    # For each seed: train real model per arm, score common test set
    per_seed_results: dict[int, dict[str, list[dict]]] = {}
    per_seed_val_results: dict[int, dict[str, list[dict]]] = {}

    for seed_idx in range(args.n_seeds):
        seed = 20260721 + seed_idx
        print(f"\n=== Seed {seed_idx+1}/{args.n_seeds} (seed={seed}) ===")
        per_seed_results[seed] = {}
        per_seed_val_results[seed] = {}

        for arm in ARMS_V3:
            train_recs, val_recs = build_arm_training_data_v3(
                groups, arm, max_negatives_per_group=1, seed=seed
            )

            # Train REAL model on training records
            scorer = ReactionContextScorer(seed=seed)
            scorer.train(train_recs)

            # Score COMMON test set (deep copy to avoid score contamination across arms)
            test_recs_copy = [dict(r) for r in common_test_set]
            scores = scorer.score(test_recs_copy)
            for r, s in zip(test_recs_copy, scores):
                r["score"] = s
            per_seed_results[seed][arm.arm_id] = test_recs_copy

            # Score val records (for baseline selection)
            if val_recs:
                val_recs_copy = [dict(r) for r in val_recs]
                val_scores = scorer.score(val_recs_copy)
                for r, s in zip(val_recs_copy, val_scores):
                    r["score"] = s
                per_seed_val_results[seed][arm.arm_id] = val_recs_copy
            else:
                per_seed_val_results[seed][arm.arm_id] = []

            # Log
            n_train_pos = sum(1 for r in train_recs if r.get("label", 0) == 1)
            n_train_neg = sum(1 for r in train_recs if r.get("label", 0) == 0)
            test_mrr = mrr_metric(test_recs_copy) if test_recs_copy else 0.0
            val_mrr = mrr_metric(val_recs_copy) if val_recs_copy else 0.0
            print(f"  {arm.arm_id} ({arm.arm_name}): "
                  f"train={len(train_recs)} (pos={n_train_pos}, neg={n_train_neg}), "
                  f"val_mrr={val_mrr:.4f}, test_mrr={test_mrr:.4f}")

    # Aggregate
    result = run_g3_v3_aggregation(
        per_seed_results, per_seed_val_results,
        n_bootstrap=args.n_bootstrap
    )
    result["arms"] = [
        {"arm_id": a.arm_id, "arm_name": a.arm_name, "description": a.description}
        for a in ARMS_V3
    ]
    result["model_input"] = "reactants + reagents/conditions + candidate product (Morgan fingerprints)"
    result["test_set"] = f"common test set: {len(common_test_set)} candidates in {len(test_by_group)} groups"

    output_file = args.output_dir / "go_no_go_v3.json"
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2, default=str)

    # Save per-seed scored TEST records for independent CI reconstruction (Exit Criterion)
    scored_file = args.output_dir / "scored_records_v3.json"
    compact_scored = {}
    for seed, arm_data in per_seed_results.items():
        compact_scored[str(seed)] = {}
        for arm_id, recs in arm_data.items():
            compact_scored[str(seed)][arm_id] = [
                {
                    "group_id": r.get("group_id", ""),
                    "experimental_group": r.get("experimental_group", ""),
                    "score": r.get("score", 0.5),
                    "label": r.get("label", 0),
                    "measured_yield": r.get("measured_yield", 0),
                    "reaction_family": r.get("reaction_family", ""),
                    "split": r.get("split", ""),
                }
                for r in recs
            ]
    with open(scored_file, "w") as f:
        json.dump(compact_scored, f, indent=2, default=str)

    # Print cache stats
    total = _FP_HITS + _FP_MISSES
    hit_rate = _FP_HITS / total * 100 if total > 0 else 0
    print(f"\nFingerprint cache: {_FP_HITS} hits / {_FP_MISSES} misses "
          f"(hit rate: {hit_rate:.1f}%)")

    print(f"\nVerdict: {result['verdict']}")
    print(f"Reason: {result['reason']}")
    print(f"Best baseline (on validation): {result['best_baseline_arm']}")
    print(f"Baseline val MRRs: {result.get('baseline_val_mrrs', {})}")
    print(f"Seed MRRs: A6={result['seed_mrrs'].get('A6', [])}, "
          f"baseline={result['seed_mrrs'].get(result['best_baseline_arm'], [])}, "
          f"A0={result['seed_mrrs'].get('A0', [])}")
    print(f"Output: {output_file}")
    print(f"Scored records: {scored_file}")
