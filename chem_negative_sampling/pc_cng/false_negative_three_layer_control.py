"""False-Negative three-layer control for PC-CNG v3 (P1-08, Section 9 E3).

This module implements the three-layer fusion pipeline that filters the
reviewed synthetic negatives down to a high-confidence subset:

- Layer 1 (ensemble agreement): multiple reranker seeds vote on each parent
  reaction; high cross-seed score variance flags potential false negatives
  (model uncertainty => the synthetic negative is unreliable).
- Layer 2 (database retrieval): canonical reaction / reactants lookup against
  USPTO normalized positives; an exact (or Tanimoto >= 0.95) match means the
  "negative" is actually an observed positive => false positive => excluded.
- Layer 3 (expert review): if reviewer verdicts exist they are used directly;
  otherwise the layer degrades to a conservative rule-based plausibility check
  over the reviewed-negatives columns (valid / atom_balance /
  false_negative_risk / review_status).

A synthetic negative is retained as "high-confidence" iff it is NOT excluded
by any of the three layers.

The module also ships the expert-review sampling helper and the
inter-annotator agreement metrics (Cohen's kappa for 2 raters, Fleiss' kappa
for 3+ raters) required by the double-blind protocol in
``docs/expert_review_protocol_20260719.md``.
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
import statistics
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .chem_utils import (
    canonicalize_reaction,
    canonicalize_smiles,
    is_valid_reaction,
    split_reaction,
)


# --------------------------------------------------------------------------- #
# Schema constants
# --------------------------------------------------------------------------- #

REVIEWED_FIELDS = [
    "source_id", "positive_reaction", "candidate_reaction", "task", "failure_type",
    "edit_action", "parent_reactants", "parent_product", "candidate_reactants",
    "candidate_product", "valid", "atom_balance", "locality", "closeness",
    "hard_score", "false_negative_risk", "passes_filter", "label", "provenance",
    "review_status", "review_reasons", "product_overlap",
]

SAMPLED_FOR_REVIEW_FIELDS = [
    "sample_id", "reaction_smiles", "parent_reaction_smiles", "failure_type",
    "task", "source_origin", "true_label",
    "chemical_validity", "mechanistic_plausibility", "side_product_likelihood",
    "feasibility_score", "overall_verdict", "comment", "reviewer_id",
    "review_timestamp",
]

DEFAULT_ENSEMBLE_STD_THRESHOLD = 0.15   # exclude when cross-seed score std > 0.15
DEFAULT_TANIMOTO_THRESHOLD = 0.95       # exclude when reactants Tanimoto >= 0.95
DEFAULT_TANIMOTO_SAMPLE_SIZE = 10000    # cap DB sample for tractable pairwise Tanimoto
DEFAULT_SAMPLE_SIZE = 400               # stratified sample size (200-500 range midpoint)
DEFAULT_CONTROL_SIZE = 100              # real-negative controls injected for blinding
DEFAULT_SEED = 20260719

KEEP_STATUS = "keep_synthetic_negative"


# --------------------------------------------------------------------------- #
# CSV helpers
# --------------------------------------------------------------------------- #

def read_csv(path: str) -> Tuple[List[Dict[str, str]], List[str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
        return rows, list(reader.fieldnames or [])


def write_csv(path: str, rows: Sequence[Dict[str, object]], fields: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _to_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None or value == "" or value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: object, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


# --------------------------------------------------------------------------- #
# Layer 1: ensemble agreement
# --------------------------------------------------------------------------- #

def load_ensemble_scores(ensemble_dir: str) -> Dict[str, List[float]]:
    """Load per-source_id score vectors across all seed subdirectories.

    Each seed subdir must contain ``test_predictions.csv`` with at least the
    columns ``source_id`` and ``score``.  Returns a mapping
    ``source_id -> [score_seed_0, score_seed_1, ...]``.
    """
    if not ensemble_dir or not os.path.isdir(ensemble_dir):
        return {}

    seed_dirs = sorted(
        entry for entry in os.listdir(ensemble_dir)
        if os.path.isdir(os.path.join(ensemble_dir, entry))
        and "seed" in entry.lower()
    )

    per_source: Dict[str, List[float]] = defaultdict(list)
    for seed_dir in seed_dirs:
        pred_path = os.path.join(ensemble_dir, seed_dir, "test_predictions.csv")
        if not os.path.isfile(pred_path):
            continue
        rows, _ = read_csv(pred_path)
        for row in rows:
            sid = row.get("source_id", "")
            if not sid:
                continue
            per_source[sid].append(_to_float(row.get("score")))

    return dict(per_source)


def ensemble_agreement_layer(
    rows: Sequence[Dict[str, str]],
    ensemble_dir: str,
    std_threshold: float = DEFAULT_ENSEMBLE_STD_THRESHOLD,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], Dict[str, object]]:
    """Layer 1: exclude synthetic negatives whose parent reaction has high
    cross-seed reranker score variance (ensemble disagreement => unreliable
    negative).  Synthetic negatives whose ``source_id`` has no ensemble
    coverage are KEPT (conservative) and flagged ``no_ensemble_coverage``.
    """
    per_source = load_ensemble_scores(ensemble_dir)
    kept: List[Dict[str, str]] = []
    excluded: List[Dict[str, str]] = []
    coverage = 0
    no_coverage = 0
    for row in rows:
        sid = row.get("source_id", "")
        scores = per_source.get(sid)
        if not scores or len(scores) < 2:
            no_coverage += 1
            row = dict(row)
            row["layer1_ensemble_std"] = ""
            row["layer1_verdict"] = "keep_no_coverage"
            kept.append(row)
            continue
        coverage += 1
        std = statistics.pstdev(scores) if len(scores) > 1 else 0.0
        row = dict(row)
        row["layer1_ensemble_std"] = f"{std:.6f}"
        if std > std_threshold:
            row["layer1_verdict"] = "exclude_disagreement"
            excluded.append(row)
        else:
            row["layer1_verdict"] = "keep_agreement"
            kept.append(row)

    stats: Dict[str, object] = {
        "layer": 1,
        "name": "ensemble_agreement",
        "input_rows": len(rows),
        "kept": len(kept),
        "excluded": len(excluded),
        "exclusion_rate": (len(excluded) / len(rows)) if rows else 0.0,
        "ensemble_seed_dirs": len(per_source),
        "parent_coverage": coverage,
        "parent_no_coverage": no_coverage,
        "std_threshold": std_threshold,
    }
    return kept, excluded, stats


# --------------------------------------------------------------------------- #
# Layer 2: database retrieval
# --------------------------------------------------------------------------- #

def _tanimoto(fps_a, fps_b) -> float:
    """Mean Tanimoto over fingerprint lists (used as a coarse similarity)."""
    if not fps_a or not fps_b:
        return 0.0
    best = 0.0
    for fa in fps_a:
        for fb in fps_b:
            if fa is None or fb is None:
                continue
            inter = (fa & fb).GetNumOnBits()
            union = (fa | fb).GetNumOnBits()
            if union == 0:
                continue
            sim = inter / union
            if sim > best:
                best = sim
            if best >= 1.0:
                break
        if best >= 1.0:
            break
    return best


def _build_database_index(database_csv: str, sample_size: int) -> Tuple[
    set, set, set, List[Tuple[str, str]], List
]:
    """Build canonical reaction / reactants / products sets and a fingerprint
    sample for Tanimoto near-duplicate detection.

    Returns ``(canon_reactions, canon_reactants, canon_products,
    reactants_sample, fps_sample)``.
    """
    canon_reactions: set = set()
    canon_reactants: set = set()
    canon_products: set = set()
    reactants_sample: List[Tuple[str, str]] = []
    fps_sample: List = []

    if not database_csv or not os.path.isfile(database_csv):
        return canon_reactions, canon_reactants, canon_products, reactants_sample, fps_sample

    rows, _ = read_csv(database_csv)
    rows_positive = [r for r in rows if r.get("label_type", "") == "positive"]

    try:
        from rdkit import Chem  # type: ignore
        from rdkit.Chem import AllChem  # type: ignore
        have_rdkit = True
    except Exception:
        have_rdkit = False

    rng = random.Random(DEFAULT_SEED)
    if len(rows_positive) > sample_size:
        sampled_idx = rng.sample(range(len(rows_positive)), sample_size)
    else:
        sampled_idx = list(range(len(rows_positive)))

    for idx in sampled_idx:
        row = rows_positive[idx]
        rxn = row.get("reaction_smiles", "")
        reactants = row.get("reactants", "") or ""
        products = row.get("products", "") or ""
        canon_rxn = canonicalize_reaction(rxn)
        if canon_rxn:
            canon_reactions.add(canon_rxn)
        canon_r = canonicalize_smiles(reactants)
        if canon_r:
            canon_reactants.add(canon_r)
        canon_p = canonicalize_smiles(products)
        if canon_p:
            canon_products.add(canon_p)
        if reactants and have_rdkit:
            reactants_sample.append((canon_r or reactants, reactants))
            mol_fps = []
            for part in reactants.split("."):
                part = part.strip()
                if not part:
                    continue
                mol = Chem.MolFromSmiles(part)
                if mol is None:
                    continue
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
                mol_fps.append(fp)
            fps_sample.append(mol_fps)

    return canon_reactions, canon_reactants, canon_products, reactants_sample, fps_sample


def database_retrieval_layer(
    rows: Sequence[Dict[str, str]],
    database_csv: str,
    tanimoto_threshold: float = DEFAULT_TANIMOTO_THRESHOLD,
    tanimoto_sample_size: int = DEFAULT_TANIMOTO_SAMPLE_SIZE,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], Dict[str, object]]:
    """Layer 2: exclude synthetic negatives whose candidate reaction (or
    reactants with Tanimoto >= 0.95) appears in the USPTO normalized positives
    database (observed positive => the "negative" is a false positive).
    """
    canon_reactions, canon_reactants, canon_products, reactants_sample, fps_sample = (
        _build_database_index(database_csv, tanimoto_sample_size)
    )

    try:
        from rdkit import Chem  # type: ignore
        from rdkit.Chem import AllChem  # type: ignore
        have_rdkit = True
    except Exception:
        have_rdkit = False

    kept: List[Dict[str, str]] = []
    excluded: List[Dict[str, str]] = []
    exact_hits = 0
    tanimoto_hits = 0

    for row in rows:
        candidate_rxn = row.get("candidate_reaction", "") or ""
        candidate_reactants = row.get("candidate_reactants", "") or row.get("parent_reactants", "") or ""
        canon_candidate = canonicalize_reaction(candidate_rxn)
        canon_candidate_reactants = canonicalize_smiles(candidate_reactants)

        verdict = "keep"
        hit_reason = ""
        if canon_candidate and canon_candidate in canon_reactions:
            verdict = "exclude"
            hit_reason = "exact_reaction_match"
            exact_hits += 1
        elif canon_candidate_reactants and canon_candidate_reactants in canon_reactants:
            verdict = "exclude"
            hit_reason = "exact_reactants_match"
            exact_hits += 1
        elif have_rdkit and canon_candidate_reactants and fps_sample:
            cand_fps = []
            for part in candidate_reactants.split("."):
                part = part.strip()
                if not part:
                    continue
                mol = Chem.MolFromSmiles(part)
                if mol is None:
                    continue
                cand_fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024))
            if cand_fps:
                best_sim = 0.0
                for db_fps in fps_sample:
                    sim = _tanimoto(cand_fps, db_fps)
                    if sim > best_sim:
                        best_sim = sim
                    if best_sim >= tanimoto_threshold:
                        break
                if best_sim >= tanimoto_threshold:
                    verdict = "exclude"
                    hit_reason = f"tanimoto_{best_sim:.4f}"
                    tanimoto_hits += 1

        row = dict(row)
        row["layer2_verdict"] = verdict
        row["layer2_hit_reason"] = hit_reason
        (excluded if verdict == "exclude" else kept).append(row)

    stats: Dict[str, object] = {
        "layer": 2,
        "name": "database_retrieval",
        "input_rows": len(rows),
        "kept": len(kept),
        "excluded": len(excluded),
        "exclusion_rate": (len(excluded) / len(rows)) if rows else 0.0,
        "database_rows_loaded": len(reactants_sample),
        "database_canonical_reactions": len(canon_reactions),
        "database_canonical_reactants": len(canon_reactants),
        "exact_match_hits": exact_hits,
        "tanimoto_hits": tanimoto_hits,
        "tanimoto_threshold": tanimoto_threshold,
    }
    return kept, excluded, stats


# --------------------------------------------------------------------------- #
# Layer 3: expert review (with rule-based fallback)
# --------------------------------------------------------------------------- #

def rule_based_plausibility_check(row: Dict[str, str]) -> str:
    """Conservative rule-based surrogate for expert review.

    Returns one of ``keep`` / ``exclude`` / ``uncertain``.
    """
    valid = _to_int(row.get("valid"), 1)
    atom_balance = _to_float(row.get("atom_balance"), 0.0)
    fnr = _to_float(row.get("false_negative_risk"), 0.0)
    review_status = row.get("review_status", "")

    if valid == 0:
        return "exclude"
    if atom_balance < 0.5:
        return "exclude"
    if review_status == "needs_review_or_downweight" and fnr > 0.7:
        return "exclude"
    if review_status == KEEP_STATUS:
        return "keep"
    if fnr < 0.5 and atom_balance >= 0.7 and valid == 1:
        return "keep"
    return "uncertain"


def _load_expert_verdicts(expert_review_dir: str) -> Dict[str, str]:
    """Load reviewer verdicts if available.  Returns mapping
    ``sample_id -> verdict`` (verdict in {keep, exclude, uncertain}).

    The verdict is derived from the ``overall_verdict`` column (1-5):
    1-2 => exclude, 3 => uncertain, 4-5 => keep.  Returns empty dict if the
    reviewer ratings file does not exist (=> rule-based fallback).
    """
    if not expert_review_dir:
        return {}
    ratings_path = os.path.join(expert_review_dir, "reviewer_ratings_raw.csv")
    if not os.path.isfile(ratings_path):
        return {}
    rows, _ = read_csv(ratings_path)
    verdicts: Dict[str, str] = {}
    for row in rows:
        sample_id = row.get("sample_id", "")
        overall = _to_int(row.get("overall_verdict"), 3)
        if overall <= 2:
            verdicts[sample_id] = "exclude"
        elif overall == 3:
            verdicts[sample_id] = "uncertain"
        else:
            verdicts[sample_id] = "keep"
    return verdicts


def expert_review_layer(
    rows: Sequence[Dict[str, str]],
    expert_review_dir: Optional[str],
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], Dict[str, object]]:
    """Layer 3: expert review with rule-based fallback.

    Since expert review is deferred to the revision stage (see protocol doc
    Section 5), this layer runs the rule-based plausibility check by default.
    The ``uncertain`` pool is excluded from high-confidence negatives but
    counted separately in the summary.
    """
    verdicts = _load_expert_verdicts(expert_review_dir or "")
    expert_executed = bool(verdicts)

    kept: List[Dict[str, str]] = []
    excluded: List[Dict[str, str]] = []
    uncertain: List[Dict[str, str]] = []
    for row in rows:
        sample_id = row.get("sample_id", row.get("source_id", ""))
        if expert_executed and sample_id in verdicts:
            verdict = verdicts[sample_id]
            source = "expert_verdict"
        else:
            verdict = rule_based_plausibility_check(row)
            source = "rule_based_fallback"
        row = dict(row)
        row["layer3_verdict"] = verdict
        row["layer3_source"] = source
        if verdict == "keep":
            kept.append(row)
        elif verdict == "exclude":
            excluded.append(row)
        else:
            uncertain.append(row)

    stats: Dict[str, object] = {
        "layer": 3,
        "name": "expert_review",
        "input_rows": len(rows),
        "kept": len(kept),
        "excluded": len(excluded),
        "uncertain": len(uncertain),
        "exclusion_rate": (len(excluded) / len(rows)) if rows else 0.0,
        "expert_executed": expert_executed,
        "fallback": "rule_based_plausibility_check" if not expert_executed else None,
    }
    # uncertain rows are not "kept" for high-confidence purposes but tracked
    return kept, excluded + uncertain, stats


# --------------------------------------------------------------------------- #
# Three-layer fusion
# --------------------------------------------------------------------------- #

def run_three_layer_control(
    rows: Sequence[Dict[str, str]],
    ensemble_dir: str,
    database_csv: str,
    expert_review_dir: Optional[str],
    ensemble_std_threshold: float = DEFAULT_ENSEMBLE_STD_THRESHOLD,
    tanimoto_threshold: float = DEFAULT_TANIMOTO_THRESHOLD,
    tanimoto_sample_size: int = DEFAULT_TANIMOTO_SAMPLE_SIZE,
) -> Tuple[List[Dict[str, str]], Dict[str, object]]:
    """Run all three layers sequentially and return the high-confidence subset
    plus a summary dict.
    """
    layer1_kept, layer1_excluded, layer1_stats = ensemble_agreement_layer(
        rows, ensemble_dir, std_threshold=ensemble_std_threshold
    )
    layer2_kept, layer2_excluded, layer2_stats = database_retrieval_layer(
        layer1_kept, database_csv,
        tanimoto_threshold=tanimoto_threshold,
        tanimoto_sample_size=tanimoto_sample_size,
    )
    layer3_kept, layer3_excluded, layer3_stats = expert_review_layer(
        layer2_kept, expert_review_dir
    )

    total_excluded = len(rows) - len(layer3_kept)
    summary: Dict[str, object] = {
        "input_rows": len(rows),
        "high_confidence_rows": len(layer3_kept),
        "total_excluded": total_excluded,
        "high_confidence_rate": (len(layer3_kept) / len(rows)) if rows else 0.0,
        "layer1": layer1_stats,
        "layer2": layer2_stats,
        "layer3": layer3_stats,
        "go_no_go_threshold": 0.30,
        "go_no_go_verdict": (
            "GO" if (len(layer3_kept) / len(rows) if rows else 0.0) >= 0.30 else "NO_GO_high_false_negative_risk"
        ),
    }
    return layer3_kept, summary


# --------------------------------------------------------------------------- #
# Stratified sampling for expert review
# --------------------------------------------------------------------------- #

def stratified_sample_for_review(
    rows: Sequence[Dict[str, str]],
    n_samples: int = DEFAULT_SAMPLE_SIZE,
    seed: int = DEFAULT_SEED,
    control_rows: Optional[Sequence[Dict[str, str]]] = None,
    n_controls: int = DEFAULT_CONTROL_SIZE,
) -> List[Dict[str, str]]:
    """Stratified random sample by ``failure_type`` (fallback ``task``).

    Injects ``n_controls`` real-negative controls (from ``control_rows``) for
    double-blind calibration.  Returns a shuffled list with the
    ``SAMPLED_FOR_REVIEW_FIELDS`` schema (verdict fields left blank).
    """
    rng = random.Random(seed)

    strata: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = row.get("failure_type", "") or row.get("task", "") or "unknown"
        strata[key].append(row)

    total = sum(len(v) for v in strata.values())
    if total == 0:
        return []

    # proportional allocation with minimum 1 per stratum
    sampled: List[Dict[str, str]] = []
    for key, members in strata.items():
        quota = max(1, round(n_samples * len(members) / total))
        quota = min(quota, len(members))
        sampled.extend(rng.sample(members, quota))

    # if proportional allocation overshot/undershot, trim/pad deterministically
    if len(sampled) > n_samples:
        rng.shuffle(sampled)
        sampled = sampled[:n_samples]
    elif len(sampled) < n_samples and len(rows) >= n_samples:
        already = {id(r) for r in sampled}
        pool = [r for r in rows if id(r) not in already]
        rng.shuffle(pool)
        sampled.extend(pool[: n_samples - len(sampled)])

    # build anonymized records
    output: List[Dict[str, str]] = []
    syn_records = []
    for row in sampled:
        syn_records.append({
            "reaction_smiles": row.get("candidate_reaction", ""),
            "parent_reaction_smiles": row.get("positive_reaction", ""),
            "failure_type": row.get("failure_type", ""),
            "task": row.get("task", ""),
            "source_origin": "pc_cng_synthetic",
            "true_label": "0",
        })

    control_records = []
    if control_rows:
        ctrls = list(control_rows)
        if len(ctrls) > n_controls:
            ctrls = rng.sample(ctrls, n_controls)
        for row in ctrls:
            control_records.append({
                "reaction_smiles": row.get("reaction_smiles", row.get("candidate_reaction", "")),
                "parent_reaction_smiles": "",
                "failure_type": "",
                "task": "",
                "source_origin": "real_negative_control",
                "true_label": "1",
            })

    combined = syn_records + control_records
    rng.shuffle(combined)

    for idx, rec in enumerate(combined, start=1):
        rec["sample_id"] = f"S{idx:04d}"
        for field in ["chemical_validity", "mechanistic_plausibility",
                      "side_product_likelihood", "feasibility_score",
                      "overall_verdict", "comment", "reviewer_id",
                      "review_timestamp"]:
            rec[field] = ""
        output.append(rec)

    return output


# --------------------------------------------------------------------------- #
# Inter-annotator agreement
# --------------------------------------------------------------------------- #

def cohens_kappa(rater_a: Sequence[int], rater_b: Sequence[int]) -> Tuple[float, Dict[str, object]]:
    """Cohen's kappa for two raters on a shared ordinal/categorical scale.

    Returns ``(kappa, info_dict)`` where ``info_dict`` contains the
    observed/expected agreement rates and the confusion matrix.
    """
    if len(rater_a) != len(rater_b):
        raise ValueError("rater vectors must have equal length")
    n = len(rater_a)
    if n == 0:
        return 0.0, {"observed_agreement": 0.0, "expected_agreement": 0.0,
                     "confusion_matrix": {}, "n": 0}

    categories = sorted(set(rater_a) | set(rater_b))
    cat_index = {c: i for i, c in enumerate(categories)}
    k = len(categories)
    matrix = [[0] * k for _ in range(k)]
    for a, b in zip(rater_a, rater_b):
        matrix[cat_index[a]][cat_index[b]] += 1

    po = sum(matrix[i][i] for i in range(k)) / n
    row_marginal = [sum(matrix[i]) / n for i in range(k)]
    col_marginal = [sum(matrix[i][j] for i in range(k)) / n for j in range(k)]
    pe = sum(row_marginal[i] * col_marginal[i] for i in range(k))
    kappa = 0.0 if pe >= 1.0 else (po - pe) / (1.0 - pe)

    confusion = {f"{categories[i]}|{categories[j]}": matrix[i][j]
                 for i in range(k) for j in range(k)}
    return kappa, {
        "observed_agreement": po,
        "expected_agreement": pe,
        "confusion_matrix": confusion,
        "categories": categories,
        "n": n,
    }


def fleiss_kappa(ratings: Sequence[Sequence[int]], n_categories: int) -> Tuple[float, List[float]]:
    """Fleiss' kappa for >= 2 raters.

    ``ratings`` is an N x K matrix (N subjects, K raters); each entry is an
    integer category in ``[0, n_categories)``.  Returns
    ``(kappa, per_category_proportions)``.
    """
    n_subjects = len(ratings)
    if n_subjects == 0:
        return 0.0, [0.0] * n_categories
    k_raters = len(ratings[0])
    if k_raters < 2:
        return 0.0, [0.0] * n_categories

    counts = [[0] * n_categories for _ in range(n_subjects)]
    for i, subject in enumerate(ratings):
        for vote in subject:
            if 0 <= vote < n_categories:
                counts[i][vote] += 1

    p_j = [0.0] * n_categories
    for i in range(n_subjects):
        for j in range(n_categories):
            p_j[j] += counts[i][j]
    total_votes = n_subjects * k_raters
    p_j = [p / total_votes for p in p_j]

    p_i = []
    for i in range(n_subjects):
        s = sum(c * c for c in counts[i])
        p_i.append((s - k_raters) / (k_raters * (k_raters - 1)))
    p_bar = sum(p_i) / n_subjects
    p_e = sum(p * p for p in p_j)
    kappa = 0.0 if p_e >= 1.0 else (p_bar - p_e) / (1.0 - p_e)
    return kappa, p_j


def verdict_to_binary(verdict: int) -> int:
    """Map 1-5 overall_verdict to binary category for kappa:
    1-2 => 0 (false_neg_risk), 3 => 1 (uncertain), 4-5 => 2 (true_neg).
    """
    if verdict <= 2:
        return 0
    if verdict == 3:
        return 1
    return 2
