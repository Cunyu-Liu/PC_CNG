"""Phase D: development and sealed-test evaluation of a training-time source gate.

The adaptive policy in this module chooses exactly one negative source for
each positive training reaction.  It is not an inference-time ensemble.

Development protocol
--------------------
1. Generate one valid candidate from each source for the same parent reaction.
2. Restrict every compared arm to the complete-case parent set.
3. Train two-fold out-of-fold downstream classifiers on a balanced uniform
   source allocation.
4. Score every held-out candidate and create a risk-adjusted hardness reward.
5. Fit a reaction-conditioned source gate to those out-of-fold rewards.
6. Freeze the gate, choose one source per parent, and train the downstream
   classifier from scratch.
7. Evaluate every arm on the already-frozen Phase-4 shared pool.

The current Phase-4 pools were used for method design and therefore may only
be used with ``--development-run``.  ``--formal-run`` requires a separate
evaluation directory whose manifest explicitly declares that labels were
sealed until after the model and analysis were frozen.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from pc_cng.chem_utils import atom_count_distance
from pc_cng.models.risk_aware_scorer import (
    FalseNegativeRiskModel,
    ObservedPool,
    RiskFeatureExtractor,
    canonical_smiles,
)
from pc_cng.paired_cluster_inference import paired_cluster_bootstrap
from pc_cng.phase3_enhanced import EnhancedMLP, morgan_fingerprint, reaction_fp_enhanced
from pc_cng.phase_e_sealed_contract import verify_sealed_manifest
from pc_cng.run_phase3_external_validation import (
    DEFAULT_NI_CSV,
    DEFAULT_OOD_DIR,
    DEFAULT_PARQUET,
    METHOD_LEARNED,
    METHOD_RANDOM,
    METHOD_RULE,
    NegativeGenerator,
    load_g8c_model,
    load_hitea_split,
    load_ni_coupling,
)
from pc_cng.run_phase4_fixed_testset import (
    cluster_bootstrap_metric,
    per_source_auprc,
    source_macro_auprc_metric,
)
from pc_cng.source_aware_policy import (
    GateTrainingConfig,
    SourceAwareSoftmaxGate,
    train_source_gate,
)

try:
    from pc_cng.reaction_gnn import ReactionAwareClassifier

    _HAS_GNN = True
except Exception:  # pragma: no cover
    ReactionAwareClassifier = None
    _HAS_GNN = False

try:
    from rdkit import Chem, DataStructs, RDLogger

    RDLogger.DisableLog("rdApp.*")
    _HAS_RDKIT = True
except Exception:  # pragma: no cover
    Chem = DataStructs = None
    _HAS_RDKIT = False


_REPO_ROOT = Path(__file__).resolve().parents[2]
_CNS_ROOT = _REPO_ROOT / "chem_negative_sampling"
DEFAULT_PHASE_C_CHECKPOINT = (
    _CNS_ROOT
    / "results"
    / "p4_g8c_formal_v2_20260729"
    / "model_checkpoint.pt"
)
DEFAULT_FIXED_POOL = _CNS_ROOT / "results" / "phase4_fixed_testset_v41"
DEFAULT_OUTPUT = _CNS_ROOT / "results" / "phase_d_source_policy_dev"

SOURCE_RANDOM = METHOD_RANDOM
SOURCE_SHUFFLED = "shuffled_real"
SOURCE_RETRIEVAL = "similarity_retrieval"
SOURCE_TEMPLATE = "template_perturbation"
SOURCE_RULE = METHOD_RULE
SOURCE_LEARNED = METHOD_LEARNED
SOURCE_NAMES = (
    SOURCE_RANDOM,
    SOURCE_SHUFFLED,
    SOURCE_RETRIEVAL,
    SOURCE_TEMPLATE,
    SOURCE_RULE,
    SOURCE_LEARNED,
)

SOURCE_FEATURE_NAMES = tuple(
    [f"source_onehot::{name}" for name in SOURCE_NAMES]
    + [
        "positive_similarity",
        "boundary_closeness",
        "false_negative_risk",
        "fnr_uncertainty",
        "chemical_validity",
        "atom_balance_quality",
        "reaction_family_support",
        "nearest_positive_similarity",
        "nearest_negative_similarity",
    ]
)
REACTION_FP_DIM = 8192
FAMILY_HASH_DIM = 16
REACTION_FEATURE_DIM = REACTION_FP_DIM + FAMILY_HASH_DIM
SOURCE_FEATURE_DIM = len(SOURCE_FEATURE_NAMES)

MAIN_POLICY_ARMS = (
    "positive_only",
    *SOURCE_NAMES,
    "uniform_union",
    "validation_selected_global_mixture",
    "learned_source_gate",
    "oracle_source_policy",
    "randomized_label_null",
)
GATE_ABLATIONS = (
    "gate_no_reaction_context",
    "gate_no_fnr",
    "gate_no_difficulty",
    "gate_no_family",
    "gate_no_learned_source",
    "gate_no_shuffled_real",
    "gate_no_source_dropout",
    "gate_no_entropy_regularization",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPO_ROOT,
            text=True,
        ).strip()
    except Exception:
        return "UNKNOWN"


def _stable_int(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def _product_of(reaction_smiles: str) -> Optional[str]:
    parts = str(reaction_smiles).split(">")
    if len(parts) == 3 and parts[2]:
        return parts[2]
    return None


def _negative_reaction(parent_reaction: str, product: str) -> Optional[str]:
    parts = str(parent_reaction).split(">")
    if len(parts) != 3 or not product:
        return None
    return f"{parts[0]}>{parts[1]}>{product}"


def _row_family(row: Any) -> str:
    for key in ("reaction_family", "reaction_class", "source"):
        value = row.get(key)
        if value is not None and str(value) not in {"", "nan", "None"}:
            return str(value)
    return "unknown"


def _row_group(row: Any) -> str:
    for key in ("experimental_group", "split_key", "record_id", "source_id"):
        value = row.get(key)
        if value is not None and str(value) not in {"", "nan", "None"}:
            return str(value)
    return str(row.get("reaction_smiles", "unknown"))


def _valid_product(product: str) -> bool:
    if not _HAS_RDKIT or not product:
        return False
    return Chem.MolFromSmiles(product) is not None


def _tanimoto_product(a: str, b: str) -> float:
    fp_a = morgan_fingerprint(a)
    fp_b = morgan_fingerprint(b)
    if fp_a is None or fp_b is None:
        return 0.0
    inter = float(np.minimum(fp_a, fp_b).sum())
    union = float(np.maximum(fp_a, fp_b).sum())
    return inter / union if union > 0 else 0.0


def _family_hash_features(family: str) -> np.ndarray:
    out = np.zeros(FAMILY_HASH_DIM, dtype=np.float32)
    out[_stable_int(f"phase_d_family|{family}") % FAMILY_HASH_DIM] = 1.0
    return out


def _normalise_reaction_fp(reaction_smiles: str) -> Optional[np.ndarray]:
    fp = reaction_fp_enhanced(reaction_smiles)
    if fp is None:
        return None
    fp = np.asarray(fp, dtype=np.float32)
    if fp.shape != (REACTION_FP_DIM,):
        raise RuntimeError(
            f"unexpected reaction fingerprint shape {fp.shape}; "
            f"expected {(REACTION_FP_DIM,)}"
        )
    return fp


def _source_feature_index(name: str) -> int:
    return SOURCE_FEATURE_NAMES.index(name)


def _build_observed_risk_components(
    train_rows: pd.DataFrame,
    *,
    seed: int,
    max_per_class: int = 1200,
) -> Tuple[RiskFeatureExtractor, FalseNegativeRiskModel, Dict[str, Any]]:
    """Fit an FNR model on observed outcomes from this scenario's train rows."""
    yield_column = "measured_yield" if "measured_yield" in train_rows else "yield"
    if yield_column not in train_rows:
        raise RuntimeError("observed FNR calibration requires a yield column")
    rows: List[Dict[str, Any]] = []
    for _, row in train_rows.iterrows():
        product = row.get("products") or _product_of(row.get("reaction_smiles", ""))
        canon = canonical_smiles(str(product or ""))
        if not canon:
            continue
        try:
            measured = float(row.get(yield_column))
        except (TypeError, ValueError):
            continue
        rows.append(
            {
                "product": canon,
                "family": _row_family(row),
                "group": _row_group(row),
                "positive": measured > 0.0,
            }
        )
    positive = [row for row in rows if row["positive"]]
    negative = [row for row in rows if not row["positive"]]
    if len(positive) < 20 or len(negative) < 20:
        raise RuntimeError(
            "scenario training rows do not contain enough observed outcomes "
            f"for FNR calibration: positive={len(positive)}, negative={len(negative)}"
        )
    rng = random.Random(seed)
    if len(positive) > max_per_class:
        positive = rng.sample(positive, max_per_class)
    if len(negative) > max_per_class:
        negative = rng.sample(negative, max_per_class)

    pool = ObservedPool()
    for row in positive:
        pool.pos_smiles.append(row["product"])
        pool.pos_family.append(row["family"])
    for row in negative:
        pool.neg_smiles.append(row["product"])
        pool.neg_family.append(row["family"])
    for row in rows:
        family = row["family"]
        product = row["product"]
        group = row["group"]
        pool.family_counts[family] = pool.family_counts.get(family, 0) + 1
        pool.group_sizes[group] = pool.group_sizes.get(group, 0) + 1
        pool.family_products.setdefault(family, set()).add(product)
        pool.all_products.add(product)

    extractor = RiskFeatureExtractor(
        pool,
        ensemble=None,
        max_ref=min(max_per_class, 1200),
        seed=seed,
        device="cpu",
    )

    def _observed_candidates(subset: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [
            {
                "candidate_smiles": row["product"],
                "gold_smiles": row["product"],
                "reaction_family": row["family"],
                "experimental_group_id": row["group"],
                "atom_mapping_status": "unmapped",
            }
            for row in subset
        ]

    positive_features = extractor.extract_batch(_observed_candidates(positive))
    negative_features = extractor.extract_batch(_observed_candidates(negative))
    model = FalseNegativeRiskModel()
    fit = model.fit(
        positive_features,
        negative_features,
        epochs=500,
        lr=0.1,
        seed=seed,
    )
    return extractor, model, {
        "source": "scenario_train_observed_outcomes_only",
        "yield_column": yield_column,
        "n_positive": len(positive),
        "n_negative": len(negative),
        "fit": fit,
        "model": model.to_dict(),
    }


def _choose_different_product(
    products: Sequence[str],
    true_product: str,
    start: int,
) -> Optional[str]:
    if not products:
        return None
    true_canon = canonical_smiles(true_product)
    for offset in range(len(products)):
        candidate = products[(start + offset) % len(products)]
        if canonical_smiles(candidate) != true_canon and _valid_product(candidate):
            return candidate
    return None


def _retrieval_product(
    products: Sequence[str],
    true_product: str,
    cached_fps: Dict[str, Any],
) -> Optional[str]:
    true_canon = canonical_smiles(true_product)
    true_fp = cached_fps.get(true_product)
    if true_fp is None:
        return None
    best: Tuple[float, Optional[str]] = (-1.0, None)
    for candidate in products:
        if canonical_smiles(candidate) == true_canon:
            continue
        cand_fp = cached_fps.get(candidate)
        if cand_fp is None:
            continue
        similarity = float(DataStructs.TanimotoSimilarity(true_fp, cand_fp))
        if similarity > best[0]:
            best = (similarity, candidate)
    return best[1]


def _generate_source_candidates(
    train_rows: pd.DataFrame,
    learned_model: Any,
    device: torch.device,
    extractor: RiskFeatureExtractor,
    fnr_model: FalseNegativeRiskModel,
    *,
    seed: int,
    min_complete_parents: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Generate a complete six-source candidate matrix for train parents."""
    parsed: List[Dict[str, Any]] = []
    for _, row in train_rows.iterrows():
        reaction = row.get("reaction_smiles")
        if not isinstance(reaction, str):
            continue
        product = _product_of(reaction)
        reaction_fp = _normalise_reaction_fp(reaction)
        if not product or reaction_fp is None or not _valid_product(product):
            continue
        parsed.append(
            {
                "reaction_smiles": reaction,
                "true_product": product,
                "family": _row_family(row),
                "group": _row_group(row),
                "reaction_fp": reaction_fp,
            }
        )
    products = [row["true_product"] for row in parsed]
    product_fps: Dict[str, Any] = {}
    for product in set(products):
        mol = Chem.MolFromSmiles(product)
        product_fps[product] = (
            Chem.RDKFingerprint(mol) if mol is not None else None
        )
    family_products: Dict[str, List[str]] = defaultdict(list)
    for row in parsed:
        family_products[row["family"]].append(row["true_product"])

    rule_generator = NegativeGenerator(SOURCE_RULE, seed=seed)
    learned_generator = NegativeGenerator(
        SOURCE_LEARNED,
        model=learned_model,
        top_k=1,
        device=device,
        seed=seed,
    )
    generation_counts: Dict[str, Counter] = {
        source: Counter() for source in SOURCE_NAMES
    }
    examples: List[Dict[str, Any]] = []
    incomplete: List[Dict[str, Any]] = []

    for index, row in enumerate(parsed):
        reaction = row["reaction_smiles"]
        true_product = row["true_product"]
        family = row["family"]
        source_products: Dict[str, Optional[str]] = {}
        source_products[SOURCE_RANDOM] = _choose_different_product(
            products,
            true_product,
            _stable_int(f"random|{seed}|{reaction}") % max(1, len(products)),
        )
        source_products[SOURCE_SHUFFLED] = _choose_different_product(
            products,
            true_product,
            index + 7,
        )
        source_products[SOURCE_RETRIEVAL] = _retrieval_product(
            products,
            true_product,
            product_fps,
        )
        same_family = family_products.get(family, [])
        source_products[SOURCE_TEMPLATE] = _choose_different_product(
            same_family if len(same_family) > 1 else products,
            true_product,
            _stable_int(f"template|{seed}|{family}|{reaction}")
            % max(1, len(same_family if len(same_family) > 1 else products)),
        )
        rule_reaction = rule_generator.generate(reaction)
        learned_reaction = learned_generator.generate(reaction)
        source_products[SOURCE_RULE] = (
            _product_of(rule_reaction) if rule_reaction else None
        )
        source_products[SOURCE_LEARNED] = (
            _product_of(learned_reaction) if learned_reaction else None
        )

        candidates: Dict[str, Dict[str, Any]] = {}
        for source in SOURCE_NAMES:
            product = source_products.get(source)
            if not product:
                generation_counts[source]["missing"] += 1
                continue
            if not _valid_product(product):
                generation_counts[source]["invalid"] += 1
                continue
            if canonical_smiles(product) == canonical_smiles(true_product):
                generation_counts[source]["collision_with_parent_positive"] += 1
                continue
            negative_reaction = _negative_reaction(reaction, product)
            negative_fp = (
                _normalise_reaction_fp(negative_reaction)
                if negative_reaction
                else None
            )
            if negative_reaction is None or negative_fp is None:
                generation_counts[source]["unfeaturizable"] += 1
                continue
            candidates[source] = {
                "source": source,
                "negative_product": product,
                "negative_reaction": negative_reaction,
                "negative_fp": negative_fp,
            }
            generation_counts[source]["valid"] += 1

        missing = [source for source in SOURCE_NAMES if source not in candidates]
        if missing:
            incomplete.append(
                {
                    "group": row["group"],
                    "missing_sources": missing,
                }
            )
            continue

        risk_inputs = [
            {
                "candidate_smiles": candidates[source]["negative_product"],
                "gold_smiles": true_product,
                "reaction_family": family,
                "experimental_group_id": row["group"],
                "atom_mapping_status": "unmapped",
            }
            for source in SOURCE_NAMES
        ]
        risk_features = extractor.extract_batch(risk_inputs)
        fnr_values = fnr_model.predict_fnr(risk_features)
        reactants = reaction.split(">")[0]
        try:
            true_distance = atom_count_distance(reactants, true_product)
        except Exception:
            true_distance = 0
        for source, risk_features_for_source, fnr in zip(
            SOURCE_NAMES,
            risk_features,
            fnr_values,
        ):
            candidate = candidates[source]
            similarity = _tanimoto_product(
                candidate["negative_product"],
                true_product,
            )
            boundary = float(
                np.clip(1.0 - abs(similarity - 0.575) / 0.575, 0.0, 1.0)
            )
            try:
                candidate_distance = atom_count_distance(
                    reactants,
                    candidate["negative_product"],
                )
                balance_excess = max(0, candidate_distance - true_distance)
                balance_quality = 1.0 / (1.0 + float(balance_excess))
            except Exception:
                balance_quality = 0.0
            onehot = [
                1.0 if source == source_name else 0.0
                for source_name in SOURCE_NAMES
            ]
            values = onehot + [
                similarity,
                boundary,
                float(fnr),
                1.0 - abs(2.0 * float(fnr) - 1.0),
                float(risk_features_for_source["chemical_validity"]),
                balance_quality,
                float(risk_features_for_source["reaction_family_support"]),
                float(risk_features_for_source["nearest_positive_similarity"]),
                float(risk_features_for_source["nearest_negative_similarity"]),
            ]
            candidate["positive_similarity"] = similarity
            candidate["boundary_closeness"] = boundary
            candidate["false_negative_risk"] = float(fnr)
            candidate["source_features"] = values
            candidate["risk_features"] = {
                key: float(value)
                for key, value in risk_features_for_source.items()
            }

        reaction_features = np.concatenate(
            [row["reaction_fp"], _family_hash_features(family)]
        ).astype(np.float32)
        examples.append(
            {
                "reaction_smiles": reaction,
                "true_product": true_product,
                "family": family,
                "group": row["group"],
                "reaction_features": reaction_features,
                "candidates": candidates,
            }
        )

    if len(examples) < min_complete_parents:
        count_summary = {
            source: dict(counts)
            for source, counts in generation_counts.items()
        }
        raise RuntimeError(
            f"only {len(examples)} complete six-source parents were generated; "
            f"required={min_complete_parents}; counts={count_summary}"
        )
    audit = {
        "n_input_rows": len(train_rows),
        "n_parsed_rows": len(parsed),
        "n_complete_parents": len(examples),
        "complete_case_fraction": len(examples) / max(1, len(parsed)),
        "generation_counts": {
            source: dict(counts)
            for source, counts in generation_counts.items()
        },
        "incomplete_examples": incomplete[:50],
        "source_names": list(SOURCE_NAMES),
        "candidate_budget_per_parent": 1,
        "comparison_parent_set": "complete cases shared by all source arms",
    }
    return examples, audit


def _candidate_cache_payload(examples: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    payload: List[Dict[str, Any]] = []
    for example in examples:
        payload.append(
            {
                "reaction_smiles": example["reaction_smiles"],
                "true_product": example["true_product"],
                "family": example["family"],
                "group": example["group"],
                "candidates": {
                    source: {
                        key: value
                        for key, value in candidate.items()
                        if key not in {"negative_fp"}
                    }
                    for source, candidate in example["candidates"].items()
                },
            }
        )
    return payload


def _build_training_dataset(
    examples: Sequence[Dict[str, Any]],
    selections: Sequence[str],
    *,
    arm: str,
    positive_only: bool = False,
    shuffle_labels: bool = False,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
    if len(examples) != len(selections):
        raise ValueError("selection count must equal example count")
    features: List[np.ndarray] = []
    labels: List[float] = []
    records: List[Dict[str, Any]] = []
    source_counts: Counter = Counter()
    for example, source in zip(examples, selections):
        positive_fp = _normalise_reaction_fp(example["reaction_smiles"])
        if positive_fp is None:
            raise RuntimeError("complete-case positive became unfeaturizable")
        features.append(positive_fp)
        labels.append(1.0)
        records.append(
            {
                "reaction_smiles": example["reaction_smiles"],
                "label": 1,
                "is_positive": True,
                "experimental_group": example["group"],
                "reaction_family": example["family"],
                "method": arm,
                "source": "positive",
            }
        )
        if positive_only:
            continue
        candidate = example["candidates"].get(source)
        if candidate is None:
            raise RuntimeError(f"selected unavailable source {source}")
        features.append(candidate["negative_fp"])
        labels.append(0.0)
        records.append(
            {
                "reaction_smiles": candidate["negative_reaction"],
                "negative_smiles": candidate["negative_product"],
                "label": 0,
                "is_positive": False,
                "experimental_group": example["group"],
                "reaction_family": example["family"],
                "method": arm,
                "source": source,
                "false_negative_risk": candidate["false_negative_risk"],
                "positive_similarity": candidate["positive_similarity"],
            }
        )
        source_counts[source] += 1
    y = np.asarray(labels, dtype=np.float32)
    if shuffle_labels:
        rng = np.random.default_rng(seed)
        y = y[rng.permutation(len(y))]
        for record, label in zip(records, y.tolist()):
            record["label"] = int(label)
            record["is_positive"] = bool(label)
    X = np.vstack(features).astype(np.float32)
    n_positive = int((y == 1).sum())
    n_negative = int((y == 0).sum())
    return X, y, records, {
        "n_records": len(y),
        "n_positive": n_positive,
        "n_negative": n_negative,
        "negative_to_positive_ratio": (
            n_negative / n_positive if n_positive else None
        ),
        "source_counts": dict(source_counts),
        "budget_exact": positive_only or n_positive == n_negative == len(examples),
        "positive_only_control": positive_only,
    }


def _train_downstream(
    X: np.ndarray,
    y: np.ndarray,
    records: List[Dict[str, Any]],
    *,
    backbone: str,
    seed: int,
    epochs: int,
) -> Any:
    if not torch.cuda.is_available():
        raise RuntimeError("Phase D training requires CUDA")
    if backbone == "gnn":
        if not _HAS_GNN:
            raise RuntimeError("ReactionAwareClassifier is unavailable")
        model = ReactionAwareClassifier(input_dim=X.shape[1], seed=seed)
        model.fit_reactions(
            [record["reaction_smiles"] for record in records],
            y,
            epochs=epochs,
            batch_size=64,
            lr=1e-3,
            verbose=False,
        )
        return model
    if backbone != "mlp":
        raise ValueError(f"unknown backbone {backbone}")
    model = EnhancedMLP(input_dim=X.shape[1], seed=seed)
    model.train(
        X,
        y,
        epochs=epochs,
        batch_size=64,
        lr=1e-3,
        verbose=False,
    )
    return model


def _predict_reactions(
    model: Any,
    reaction_smiles: Sequence[str],
    *,
    backbone: str,
) -> np.ndarray:
    if backbone == "gnn":
        return np.asarray(
            model.predict_proba_reactions(list(reaction_smiles)),
            dtype=np.float64,
        )
    features = []
    valid_indices = []
    for index, reaction in enumerate(reaction_smiles):
        fp = _normalise_reaction_fp(reaction)
        if fp is not None:
            features.append(fp)
            valid_indices.append(index)
    output = np.full(len(reaction_smiles), 0.5, dtype=np.float64)
    if features:
        output[valid_indices] = model.predict_proba(
            np.vstack(features).astype(np.float32)
        )
    return output


def _score_fixed_records(
    model: Any,
    records: Sequence[Dict[str, Any]],
    *,
    backbone: str,
) -> List[Dict[str, Any]]:
    scored = copy.deepcopy(list(records))
    values = _predict_reactions(
        model,
        [record["reaction_smiles"] for record in scored],
        backbone=backbone,
    )
    for record, value in zip(scored, values.tolist()):
        record["score"] = float(value)
    return scored


def _uniform_selections(
    examples: Sequence[Dict[str, Any]],
    sources: Sequence[str] = SOURCE_NAMES,
    *,
    salt: str,
) -> List[str]:
    names = tuple(sources)
    return [
        names[_stable_int(f"{salt}|{example['group']}") % len(names)]
        for example in examples
    ]


def _weighted_selections(
    examples: Sequence[Dict[str, Any]],
    weights: np.ndarray,
    *,
    seed: int,
) -> List[str]:
    if weights.shape != (len(SOURCE_NAMES),):
        raise ValueError("global source weights have wrong shape")
    cumulative = np.cumsum(weights / weights.sum())
    output = []
    for example in examples:
        draw = (
            _stable_int(f"global_mix|{seed}|{example['group']}") % 10_000_000
        ) / 10_000_000.0
        index = int(np.searchsorted(cumulative, draw, side="right"))
        output.append(SOURCE_NAMES[min(index, len(SOURCE_NAMES) - 1)])
    return output


def _cross_fitted_hardness(
    examples: Sequence[Dict[str, Any]],
    *,
    backbone: str,
    folds: int,
    seed: int,
    epochs: int,
    min_fold_parents: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    if folds < 2:
        raise ValueError("cross-fitting requires at least two folds")
    fold_ids = np.asarray(
        [
            _stable_int(f"phase_d_fold|{seed}|{example['group']}") % folds
            for example in examples
        ],
        dtype=np.int64,
    )
    hardness = np.full(
        (len(examples), len(SOURCE_NAMES)),
        np.nan,
        dtype=np.float32,
    )
    fold_audit: List[Dict[str, Any]] = []
    for fold in range(folds):
        train_indices = np.flatnonzero(fold_ids != fold)
        held_indices = np.flatnonzero(fold_ids == fold)
        if (
            len(train_indices) < min_fold_parents
            or len(held_indices) < min_fold_parents
        ):
            raise RuntimeError(
                f"cross-fit fold {fold} too small: train={len(train_indices)}, "
                f"held={len(held_indices)}, required={min_fold_parents}"
            )
        fold_train = [examples[index] for index in train_indices]
        selections = _uniform_selections(
            fold_train,
            salt=f"oof|{backbone}|{seed}|{fold}",
        )
        X, y, records, budget = _build_training_dataset(
            fold_train,
            selections,
            arm=f"oof_uniform_fold_{fold}",
            seed=seed + fold,
        )
        model = _train_downstream(
            X,
            y,
            records,
            backbone=backbone,
            seed=seed + fold,
            epochs=epochs,
        )
        candidate_reactions: List[str] = []
        for index in held_indices:
            for source in SOURCE_NAMES:
                candidate_reactions.append(
                    examples[index]["candidates"][source]["negative_reaction"]
                )
        scores = _predict_reactions(
            model,
            candidate_reactions,
            backbone=backbone,
        ).reshape(len(held_indices), len(SOURCE_NAMES))
        hardness[held_indices] = scores.astype(np.float32)
        fold_audit.append(
            {
                "fold": fold,
                "n_train": len(train_indices),
                "n_heldout": len(held_indices),
                "train_budget": budget,
                "heldout_mean_hardness_by_source": {
                    source: float(scores[:, index].mean())
                    for index, source in enumerate(SOURCE_NAMES)
                },
            }
        )
        del model
        torch.cuda.empty_cache()
    if not np.isfinite(hardness).all():
        raise RuntimeError("cross-fitted hardness contains missing values")
    return hardness, {
        "fold_assignment": (
            "sha256(experimental_group) modulo folds; each reward scored by "
            "a downstream model that did not train on that parent"
        ),
        "folds": folds,
        "fold_results": fold_audit,
    }


def _example_tensors(
    examples: Sequence[Dict[str, Any]],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    reaction = np.vstack(
        [example["reaction_features"] for example in examples]
    ).astype(np.float32)
    source = np.asarray(
        [
            [
                example["candidates"][source_name]["source_features"]
                for source_name in SOURCE_NAMES
            ]
            for example in examples
        ],
        dtype=np.float32,
    )
    available = np.ones(
        (len(examples), len(SOURCE_NAMES)),
        dtype=bool,
    )
    return (
        torch.from_numpy(reaction).to(device),
        torch.from_numpy(source).to(device),
        torch.from_numpy(available).to(device),
    )


def _policy_variant_inputs(
    reaction: torch.Tensor,
    source: torch.Tensor,
    available: torch.Tensor,
    hardness: np.ndarray,
    variant: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    reaction_variant = reaction.clone()
    source_variant = source.clone()
    available_variant = available.clone()
    use_fnr = variant != "gate_no_fnr"
    use_difficulty = variant != "gate_no_difficulty"

    if variant == "gate_no_reaction_context":
        reaction_variant[:, :REACTION_FP_DIM] = 0.0
    if variant == "gate_no_family":
        reaction_variant[:, REACTION_FP_DIM:] = 0.0
        source_variant[:, :, _source_feature_index("reaction_family_support")] = 0.0
    if not use_fnr:
        for name in ("false_negative_risk", "fnr_uncertainty"):
            source_variant[:, :, _source_feature_index(name)] = 0.0
    if not use_difficulty:
        for name in ("positive_similarity", "boundary_closeness"):
            source_variant[:, :, _source_feature_index(name)] = 0.0
    if variant == "gate_no_learned_source":
        available_variant[:, SOURCE_NAMES.index(SOURCE_LEARNED)] = False
    if variant == "gate_no_shuffled_real":
        available_variant[:, SOURCE_NAMES.index(SOURCE_SHUFFLED)] = False

    rewards = torch.from_numpy(hardness.astype(np.float32)).to(reaction.device)
    if use_fnr:
        fnr = source[:, :, _source_feature_index("false_negative_risk")]
        rewards = rewards * (1.0 - fnr)
    if use_difficulty:
        boundary = source[:, :, _source_feature_index("boundary_closeness")]
        rewards = rewards * (0.5 + 0.5 * boundary)
    rewards = rewards.masked_fill(~available_variant, -1e6)
    return reaction_variant, source_variant, available_variant, rewards


def _fit_gate_variant(
    examples: Sequence[Dict[str, Any]],
    reaction: torch.Tensor,
    source: torch.Tensor,
    available: torch.Tensor,
    hardness: np.ndarray,
    *,
    variant: str,
    seed: int,
    gate_epochs: int,
    gate_dir: Path,
) -> Tuple[List[str], Dict[str, Any], np.ndarray]:
    reaction_v, source_v, available_v, rewards = _policy_variant_inputs(
        reaction,
        source,
        available,
        hardness,
        variant,
    )
    config = GateTrainingConfig(
        epochs=gate_epochs,
        learning_rate=2e-3,
        target_temperature=0.20,
        entropy_weight=(
            0.0 if variant == "gate_no_entropy_regularization" else 0.02
        ),
        source_dropout=(
            0.0 if variant == "gate_no_source_dropout" else 0.10
        ),
        seed=seed,
    )
    gate = SourceAwareSoftmaxGate(
        reaction_dim=REACTION_FEATURE_DIM,
        source_feature_dim=SOURCE_FEATURE_DIM,
        n_sources=len(SOURCE_NAMES),
        hidden_dim=64,
        temperature=1.0,
    ).to(reaction.device)
    audit = train_source_gate(
        gate,
        reaction_v,
        source_v,
        rewards,
        available_v,
        config,
    )
    with torch.no_grad():
        probabilities = (
            gate(reaction_v, source_v, available_v).detach().cpu().numpy()
        )
        indices = probabilities.argmax(axis=1)
    selections = [SOURCE_NAMES[int(index)] for index in indices]
    gate_dir.mkdir(parents=True, exist_ok=True)
    model_path = gate_dir / f"{variant}.pt"
    torch.save(
        {
            "state_dict": gate.state_dict(),
            "variant": variant,
            "source_names": SOURCE_NAMES,
            "source_feature_names": SOURCE_FEATURE_NAMES,
            "reaction_feature_dim": REACTION_FEATURE_DIM,
            "training_audit": audit,
        },
        model_path,
    )
    audit["checkpoint"] = str(model_path)
    audit["checkpoint_sha256"] = _sha256(model_path)
    audit["selected_source_counts"] = dict(Counter(selections))
    return selections, audit, probabilities


def _load_fixed_records(pool_dir: Path, scenario: str) -> List[Dict[str, Any]]:
    path = (
        pool_dir
        / "per_scenario_records"
        / f"{scenario}__rule_pc_cng__semi_hard.csv"
    )
    if not path.exists():
        raise FileNotFoundError(f"fixed evaluation records not found: {path}")
    records: List[Dict[str, Any]] = []
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            row["label"] = int(float(row["label"]))
            row["score"] = 0.0
            row["is_positive"] = str(row.get("is_positive", "")).lower() == "true"
            row["sim"] = float(row.get("sim") or 0.0)
            records.append(row)
    if not records:
        raise RuntimeError(f"empty fixed evaluation pool: {path}")
    return records


def _write_scored_csv(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for record in records for key in record})
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def _source_macro_chance(records: Sequence[Dict[str, Any]]) -> float:
    positives = [record for record in records if record.get("is_positive")]
    negative_counts: Counter = Counter(
        str(record.get("source", "unknown"))
        for record in records
        if not record.get("is_positive")
    )
    if not negative_counts:
        return len(positives) / max(1, len(records))
    return float(
        np.mean(
            [
                len(positives) / (len(positives) + count)
                for count in negative_counts.values()
            ]
        )
    )


def _scenario_loader(
    scenario: str,
    *,
    max_train: int,
    max_test: int,
) -> Dict[str, Any]:
    if scenario == "ni_coupling":
        return load_ni_coupling(DEFAULT_NI_CSV, max_train, max_test)
    return load_hitea_split(
        DEFAULT_PARQUET,
        DEFAULT_OOD_DIR,
        scenario,
        max_train,
        max_test,
    )


def _formal_pool_contract(pool_dir: Path) -> Dict[str, Any]:
    manifest_path = pool_dir / "sealed_test_manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(
            "--formal-run requires sealed_test_manifest.json in the evaluation pool"
        )
    verification = verify_sealed_manifest(
        manifest_path,
        expected_pool_dir=pool_dir,
    )
    if not verification["verified"]:
        raise RuntimeError(
            "Phase-E sealed-test contract verification failed: "
            + "; ".join(verification["failures"])
        )
    manifest = json.loads(manifest_path.read_text())
    if pool_dir.resolve() == DEFAULT_FIXED_POOL.resolve():
        raise RuntimeError(
            "the Phase-4 fixed pool is development-only and cannot be formal"
        )
    manifest["pre_run_contract_verification"] = verification
    return manifest


def run_one(
    scenario: str,
    backbone: str,
    train_rows: pd.DataFrame,
    fixed_records: List[Dict[str, Any]],
    examples: List[Dict[str, Any]],
    *,
    output_dir: Path,
    seed: int,
    epochs: int,
    gate_epochs: int,
    folds: int,
    min_fold_parents: int,
    n_bootstrap: int,
    run_ablations: bool,
) -> Dict[str, Any]:
    print(f"[phase-d] {scenario}/{backbone}: cross-fitted source rewards")
    hardness, cross_fit_audit = _cross_fitted_hardness(
        examples,
        backbone=backbone,
        folds=folds,
        seed=seed,
        epochs=epochs,
        min_fold_parents=min_fold_parents,
    )
    reaction, source, available = _example_tensors(
        examples,
        torch.device("cuda"),
    )
    gate_root = output_dir / "gate_checkpoints" / scenario / backbone
    gate_selections, gate_audit, gate_probabilities = _fit_gate_variant(
        examples,
        reaction,
        source,
        available,
        hardness,
        variant="learned_source_gate",
        seed=seed,
        gate_epochs=gate_epochs,
        gate_dir=gate_root,
    )
    _, _, _, full_rewards_t = _policy_variant_inputs(
        reaction,
        source,
        available,
        hardness,
        "learned_source_gate",
    )
    full_rewards = full_rewards_t.detach().cpu().numpy()
    mean_reward = full_rewards.mean(axis=0)
    global_weights = np.exp(
        (mean_reward - mean_reward.max()) / 0.20
    )
    global_weights = global_weights / global_weights.sum()
    global_selections = _weighted_selections(
        examples,
        global_weights,
        seed=seed,
    )
    oracle_selections = [
        SOURCE_NAMES[int(index)] for index in full_rewards.argmax(axis=1)
    ]
    uniform_selections = _uniform_selections(
        examples,
        salt=f"uniform_final|{scenario}|{backbone}|{seed}",
    )
    best_source_index = int(mean_reward.argmax())
    validation_selected_best_source = SOURCE_NAMES[best_source_index]

    selections_by_arm: Dict[str, List[str]] = {
        "positive_only": [SOURCE_RANDOM] * len(examples),
        **{
            source_name: [source_name] * len(examples)
            for source_name in SOURCE_NAMES
        },
        "uniform_union": uniform_selections,
        "validation_selected_global_mixture": global_selections,
        "learned_source_gate": gate_selections,
        "oracle_source_policy": oracle_selections,
        "randomized_label_null": gate_selections,
    }
    gate_audits: Dict[str, Any] = {"learned_source_gate": gate_audit}
    gate_probabilities_by_variant: Dict[str, np.ndarray] = {
        "learned_source_gate": gate_probabilities
    }
    if run_ablations:
        for offset, variant in enumerate(GATE_ABLATIONS, start=1):
            selections, audit, probabilities = _fit_gate_variant(
                examples,
                reaction,
                source,
                available,
                hardness,
                variant=variant,
                seed=seed + offset,
                gate_epochs=gate_epochs,
                gate_dir=gate_root,
            )
            selections_by_arm[variant] = selections
            gate_audits[variant] = audit
            gate_probabilities_by_variant[variant] = probabilities

    arm_results: Dict[str, Any] = {}
    scored_by_arm: Dict[str, List[Dict[str, Any]]] = {}
    for offset, (arm, selections) in enumerate(selections_by_arm.items()):
        print(f"[phase-d] {scenario}/{backbone}: train {arm}")
        positive_only = arm == "positive_only"
        shuffled_labels = arm == "randomized_label_null"
        X, y, train_records, budget = _build_training_dataset(
            examples,
            selections,
            arm=arm,
            positive_only=positive_only,
            shuffle_labels=shuffled_labels,
            seed=seed + 1000 + offset,
        )
        model = _train_downstream(
            X,
            y,
            train_records,
            backbone=backbone,
            seed=seed + 1000 + offset,
            epochs=epochs,
        )
        scored = _score_fixed_records(
            model,
            fixed_records,
            backbone=backbone,
        )
        scored_by_arm[arm] = scored
        score_path = (
            output_dir
            / "scored_records"
            / f"{scenario}__{backbone}__{arm}.csv"
        )
        _write_scored_csv(score_path, scored)
        metric = source_macro_auprc_metric(scored)
        arm_results[arm] = {
            "source_macro_auprc": metric,
            "per_source_auprc": per_source_auprc(scored),
            "train_budget": budget,
            "scored_records": str(score_path),
            "selected_source_counts": dict(Counter(selections)),
        }
        del model
        torch.cuda.empty_cache()

    inference = {
        "gate_vs_validation_selected_best_single": paired_cluster_bootstrap(
            scored_by_arm["learned_source_gate"],
            scored_by_arm[validation_selected_best_source],
            source_macro_auprc_metric,
            n_bootstrap=n_bootstrap,
            seed=seed + 2001,
        ),
        "gate_vs_uniform_union": paired_cluster_bootstrap(
            scored_by_arm["learned_source_gate"],
            scored_by_arm["uniform_union"],
            source_macro_auprc_metric,
            n_bootstrap=n_bootstrap,
            seed=seed + 2002,
        ),
        "gate_vs_global_mixture": paired_cluster_bootstrap(
            scored_by_arm["learned_source_gate"],
            scored_by_arm["validation_selected_global_mixture"],
            source_macro_auprc_metric,
            n_bootstrap=n_bootstrap,
            seed=seed + 2003,
        ),
    }
    if "gate_no_learned_source" in scored_by_arm:
        inference["gate_vs_no_learned_source"] = paired_cluster_bootstrap(
            scored_by_arm["learned_source_gate"],
            scored_by_arm["gate_no_learned_source"],
            source_macro_auprc_metric,
            n_bootstrap=n_bootstrap,
            seed=seed + 2004,
        )
    null_interval = cluster_bootstrap_metric(
        scored_by_arm["randomized_label_null"],
        metric_fn=source_macro_auprc_metric,
        n_bootstrap=n_bootstrap,
        seed=seed + 2005,
    )
    null_chance = _source_macro_chance(fixed_records)
    null_interval["source_macro_chance"] = null_chance
    null_interval["chance_inside_ci"] = (
        null_interval["ci_low"] <= null_chance <= null_interval["ci_high"]
    )
    inference["randomized_label_null"] = null_interval

    policy_rows = []
    for index, example in enumerate(examples):
        row = {
            "group": example["group"],
            "reaction_family": example["family"],
            "selected_source": gate_selections[index],
        }
        for source_index, source_name in enumerate(SOURCE_NAMES):
            row[f"prob::{source_name}"] = float(
                gate_probabilities[index, source_index]
            )
            row[f"oof_hardness::{source_name}"] = float(
                hardness[index, source_index]
            )
            row[f"reward::{source_name}"] = float(
                full_rewards[index, source_index]
            )
        policy_rows.append(row)
    policy_path = (
        output_dir / "policy_maps" / f"{scenario}__{backbone}.csv"
    )
    _write_scored_csv(policy_path, policy_rows)
    del reaction, source, available
    torch.cuda.empty_cache()
    return {
        "scenario": scenario,
        "backbone": backbone,
        "seed": seed,
        "n_bootstrap": n_bootstrap,
        "n_complete_parents": len(examples),
        "cross_fit": cross_fit_audit,
        "oof_mean_hardness_by_source": {
            source_name: float(hardness[:, index].mean())
            for index, source_name in enumerate(SOURCE_NAMES)
        },
        "validation_selected_best_single": validation_selected_best_source,
        "global_mixture_weights": {
            source_name: float(global_weights[index])
            for index, source_name in enumerate(SOURCE_NAMES)
        },
        "gate_training": gate_audits,
        "arms": arm_results,
        "inference": inference,
        "policy_map": str(policy_path),
        "evaluation_status": "DEVELOPMENT_ONLY_USED_FOR_METHOD_DESIGN",
    }


def _aggregate_exit(
    results: Sequence[Dict[str, Any]],
    *,
    formal_run: bool,
) -> Dict[str, Any]:
    primary_cells = list(results)
    gate_vs_single = [
        result["inference"]["gate_vs_validation_selected_best_single"][
            "ci_all_positive"
        ]
        for result in primary_cells
    ]
    gate_vs_uniform = [
        result["inference"]["gate_vs_uniform_union"]["ci_all_positive"]
        for result in primary_cells
    ]
    learned_increment = [
        result["inference"]["gate_vs_no_learned_source"]["ci_all_positive"]
        for result in primary_cells
        if "gate_vs_no_learned_source" in result["inference"]
    ]
    null_ok = [
        result["inference"]["randomized_label_null"]["chance_inside_ci"]
        for result in primary_cells
    ]
    exact_budget = all(
        arm_result["train_budget"]["budget_exact"]
        for result in primary_cells
        for arm, arm_result in result["arms"].items()
        if arm != "positive_only"
    )
    datasets = sorted({result["scenario"] for result in primary_cells})
    backbones = sorted({result["backbone"] for result in primary_cells})
    statistical_exit = (
        len(datasets) >= 2
        and len(backbones) >= 2
        and bool(gate_vs_single)
        and all(gate_vs_single)
        and bool(gate_vs_uniform)
        and all(gate_vs_uniform)
        and len(learned_increment) == len(primary_cells)
        and all(learned_increment)
        and bool(null_ok)
        and all(null_ok)
        and exact_budget
    )
    return {
        "datasets": datasets,
        "backbones": backbones,
        "n_primary_cells": len(primary_cells),
        "gate_vs_best_single_ci_all_positive": gate_vs_single,
        "gate_vs_uniform_ci_all_positive": gate_vs_uniform,
        "leave_one_learned_ci_all_positive": learned_increment,
        "randomized_label_null_at_chance": null_ok,
        "candidate_budget_exact": exact_budget,
        "development_statistical_exit_met": statistical_exit,
        "confirmatory_exit_met": bool(formal_run and statistical_exit),
        "status": (
            "FORMAL_EXIT_MET"
            if formal_run and statistical_exit
            else (
                "FORMAL_NO_GO"
                if formal_run
                else "DEVELOPMENT_ONLY_CONFIRMATORY_TEST_REQUIRED"
            )
        ),
        "interpretation": (
            "Development results may guide method design but cannot establish "
            "the Phase-D primary claim. A new sealed test is required."
            if not formal_run
            else "Formal result follows the sealed-test manifest."
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    run_mode = parser.add_mutually_exclusive_group(required=True)
    run_mode.add_argument("--development-run", action="store_true")
    run_mode.add_argument("--formal-run", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=["random", "ni_coupling"],
    )
    parser.add_argument(
        "--backbones",
        nargs="+",
        choices=["mlp", "gnn"],
        default=["mlp", "gnn"],
    )
    parser.add_argument(
        "--ablation-backbones",
        nargs="+",
        choices=["mlp", "gnn"],
        default=["mlp", "gnn"],
    )
    parser.add_argument("--max-train", type=int, default=240)
    parser.add_argument("--max-test", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--gate-epochs", type=int, default=300)
    parser.add_argument("--folds", type=int, default=2)
    parser.add_argument("--min-fold-parents", type=int, default=20)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--min-complete-parents", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_PHASE_C_CHECKPOINT,
    )
    parser.add_argument(
        "--evaluation-pool",
        type=Path,
        default=DEFAULT_FIXED_POOL,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if not _HAS_RDKIT:
        raise RuntimeError("Phase D requires RDKit")
    if not torch.cuda.is_available():
        raise RuntimeError("Phase D training requires a visible CUDA GPU")
    if args.gpu < 0 or args.gpu >= torch.cuda.device_count():
        raise RuntimeError(
            f"requested GPU {args.gpu}, visible count={torch.cuda.device_count()}"
        )
    torch.cuda.set_device(args.gpu)
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    if not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)
    formal_manifest = (
        _formal_pool_contract(args.evaluation_pool)
        if args.formal_run
        else None
    )
    if args.development_run and args.evaluation_pool.resolve() != DEFAULT_FIXED_POOL.resolve():
        print(
            "[phase-d] warning: development run uses a non-default evaluation pool; "
            "it will still be marked development-only"
        )
    model, loaded = load_g8c_model(args.checkpoint, device=device)
    if model is None:
        raise RuntimeError(f"failed to load Phase-C source expert: {loaded}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    run_manifest = {
        "timestamp_epoch": started,
        "git_commit_sha": _git_sha(),
        "exact_command": [
            sys.executable,
            "-m",
            "pc_cng.run_phase_d_source_policy",
            *sys.argv[1:],
        ],
        "script_sha256": _sha256(Path(__file__)),
        "mode": "formal" if args.formal_run else "development",
        "evaluation_status": (
            "SEALED_CONFIRMATORY"
            if args.formal_run
            else "DEVELOPMENT_SET_USED_FOR_METHOD_DESIGN"
        ),
        "formal_manifest": formal_manifest,
        "scenarios": args.scenarios,
        "backbones": args.backbones,
        "ablation_backbones": args.ablation_backbones,
        "max_train": args.max_train,
        "max_test": args.max_test,
        "epochs": args.epochs,
        "gate_epochs": args.gate_epochs,
        "folds": args.folds,
        "min_fold_parents": args.min_fold_parents,
        "n_bootstrap": args.n_bootstrap,
        "min_complete_parents": args.min_complete_parents,
        "seed": args.seed,
        "source_names": SOURCE_NAMES,
        "source_feature_names": SOURCE_FEATURE_NAMES,
        "candidate_budget_per_parent": 1,
        "phase_c_checkpoint": str(args.checkpoint.resolve()),
        "phase_c_checkpoint_sha256": _sha256(args.checkpoint),
        "evaluation_pool": str(args.evaluation_pool.resolve()),
        "cuda_device": torch.cuda.get_device_name(args.gpu),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "primary_hypothesis": (
            "reaction-conditioned source gate > validation-selected best "
            "single source and uniform union on source-macro AUPRC"
        ),
        "development_warning": (
            None
            if args.formal_run
            else (
                "The Phase-4 fixed pool has been inspected and used for "
                "method design. No confirmatory or SOTA claim is permitted."
            )
        ),
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2)
    )

    results: List[Dict[str, Any]] = []
    scenario_audits: Dict[str, Any] = {}
    for scenario_offset, scenario in enumerate(args.scenarios):
        print(f"[phase-d] prepare scenario={scenario}")
        split = _scenario_loader(
            scenario,
            max_train=args.max_train,
            max_test=args.max_test,
        )
        train_rows = split["train"].reset_index(drop=True)
        fixed_records = _load_fixed_records(args.evaluation_pool, scenario)
        extractor, fnr_model, risk_audit = _build_observed_risk_components(
            train_rows,
            seed=args.seed + scenario_offset * 100,
        )
        examples, generation_audit = _generate_source_candidates(
            train_rows,
            model,
            device,
            extractor,
            fnr_model,
            seed=args.seed + scenario_offset * 100,
            min_complete_parents=args.min_complete_parents,
        )
        cache_path = output_dir / "candidate_cache" / f"{scenario}.json"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(_candidate_cache_payload(examples), indent=2)
        )
        scenario_audits[scenario] = {
            "risk_model": risk_audit,
            "candidate_generation": generation_audit,
            "candidate_cache": str(cache_path),
            "candidate_cache_sha256": _sha256(cache_path),
            "fixed_pool_n_records": len(fixed_records),
            "fixed_pool_chance": _source_macro_chance(fixed_records),
        }
        for backbone_offset, backbone in enumerate(args.backbones):
            result = run_one(
                scenario,
                backbone,
                train_rows,
                fixed_records,
                examples,
                output_dir=output_dir,
                seed=args.seed + scenario_offset * 100 + backbone_offset * 10,
                epochs=args.epochs,
                gate_epochs=args.gate_epochs,
                folds=args.folds,
                min_fold_parents=args.min_fold_parents,
                n_bootstrap=args.n_bootstrap,
                run_ablations=backbone in args.ablation_backbones,
            )
            results.append(result)
            partial = {
                "results": results,
                "scenario_audits": scenario_audits,
                "status": "RUNNING_PARTIAL_NOT_A_CLAIM",
            }
            (output_dir / "partial_results.json").write_text(
                json.dumps(partial, indent=2)
            )

    exit_status = _aggregate_exit(results, formal_run=args.formal_run)
    final = {
        "run_manifest": run_manifest,
        "scenario_audits": scenario_audits,
        "results": results,
        "exit_status": exit_status,
        "elapsed_sec": time.time() - started,
    }
    final_path = output_dir / "phase_d_results.json"
    final_path.write_text(json.dumps(final, indent=2))
    verdict_path = output_dir / "verdict.json"
    verdict_path.write_text(json.dumps(exit_status, indent=2))
    run_manifest["elapsed_sec"] = final["elapsed_sec"]
    run_manifest["result_sha256"] = _sha256(final_path)
    run_manifest["system_metrics"] = {
        "cuda_peak_memory_allocated_mib": (
            torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        ),
        "cuda_peak_memory_reserved_mib": (
            torch.cuda.max_memory_reserved(device) / (1024 * 1024)
        ),
        "cuda_memory_allocated_at_end_mib": (
            torch.cuda.memory_allocated(device) / (1024 * 1024)
        ),
        "cuda_memory_reserved_at_end_mib": (
            torch.cuda.memory_reserved(device) / (1024 * 1024)
        ),
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2)
    )
    print(json.dumps(exit_status, indent=2))
    print(f"[phase-d] results: {final_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
