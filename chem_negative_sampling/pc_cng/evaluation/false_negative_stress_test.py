"""P4-G5: False-negative stress tests.

Three pre-declared stress families:

1. ``known_positive``   — held-out observed positives (HTEa yield>0,
   split != train) disguised as ranking candidates among synthetic
   negatives. A model with runaway false rejection ranks them last.
2. ``near_positive``    — synthetic manifest candidates that are
   near-duplicates of real training positives
   (nearest_train_similarity >= threshold, no exact overlap). A
   risk-aware model should NOT score them near zero.
3. ``ood_family``       — observed HTEa val/test rows whose reaction
   class never appears among manifest train families. Calibration
   (ECE) must hold out-of-family.

Plus generic selective-prediction analysis (coverage-risk curve) and
collision sensitivity on test predictions.

All construction/metric functions are pure (no torch) for testability.
"""

from __future__ import annotations

import csv
import json
import math
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

STRESS_SET_NAMES = ["known_positive", "near_positive", "ood_family"]

DEFAULT_N_KNOWN_POSITIVE = 200
DEFAULT_K_NEG = 7
DEFAULT_NEAR_POSITIVE_MIN_SIM = 0.7
DEFAULT_MAX_OOD_PER_FAMILY = 50
COVERAGE_GRID = [round(0.1 * i, 1) for i in range(5, 11)]  # 0.5..1.0


# ---------------------------------------------------------------------------
# Set construction
# ---------------------------------------------------------------------------

def build_known_positive_set(
    heldout_positives: Sequence[Dict[str, Any]],
    train_negatives_by_family: Dict[str, List[Dict[str, Any]]],
    fallback_negatives: Sequence[Dict[str, Any]],
    n: int = DEFAULT_N_KNOWN_POSITIVE,
    k_neg: int = DEFAULT_K_NEG,
    seed: int = 20260723,
) -> List[Dict[str, Any]]:
    """Disguised known-positive ranking groups.

    Args:
        heldout_positives: dicts with at least ``smiles`` and
            ``reaction_family`` (observed, yield>0, split != train).
        train_negatives_by_family: family -> synthetic train negatives
            (dicts with ``smiles``).
        fallback_negatives: flat negative pool when a family has none.
        n: number of groups to sample.
        k_neg: synthetic negatives per group.
        seed: sampling seed.

    Returns:
        List of groups: {"group_id", "candidates": [...]} where exactly
        one candidate has ``is_disguised_positive`` True and label 1.
    """
    rng = random.Random(seed)
    pool = list(heldout_positives)
    rng.shuffle(pool)
    pool = pool[:n]
    groups: List[Dict[str, Any]] = []
    for i, pos in enumerate(pool):
        fam = pos.get("reaction_family", "") or "unknown"
        negs = train_negatives_by_family.get(fam) or list(fallback_negatives)
        if len(negs) < k_neg:
            negs = list(negs) + list(fallback_negatives)
        chosen = rng.sample(negs, k_neg) if len(negs) >= k_neg else negs[:k_neg]
        cands = [{
            "smiles": pos["smiles"],
            "label": 1,
            "is_disguised_positive": True,
            "reaction_family": fam,
        }]
        for j, neg in enumerate(chosen):
            cands.append({
                "smiles": neg["smiles"],
                "label": 0,
                "is_disguised_positive": False,
                "reaction_family": fam,
            })
        groups.append({"group_id": f"knownpos_{i}", "candidates": cands})
    return groups


def build_near_positive_set(
    candidates: Sequence[Dict[str, Any]],
    min_sim: float = DEFAULT_NEAR_POSITIVE_MIN_SIM,
) -> List[Dict[str, Any]]:
    """Synthetic candidates that are near-duplicates of train positives.

    Keeps non-gold candidates with nearest_train_similarity >= min_sim
    and no exact train overlap. Each returned dict carries smiles,
    nearest_train_similarity, candidate_id and (if present) fnr.
    """
    out: List[Dict[str, Any]] = []
    for c in candidates:
        if c.get("gold_candidate") or c.get("label") == 1:
            continue
        sim = c.get("nearest_train_similarity")
        if sim is None:
            continue
        try:
            sim = float(sim)
        except (TypeError, ValueError):
            continue
        if sim < min_sim:
            continue
        if c.get("train_overlap"):
            continue
        out.append({
            "smiles": c.get("smiles") or c.get("candidate_smiles") or "",
            "candidate_id": c.get("candidate_id", ""),
            "nearest_train_similarity": sim,
            "reaction_family": c.get("reaction_family", ""),
        })
    return out


def build_ood_family_set(
    heldout_rows: Sequence[Dict[str, Any]],
    train_families: Sequence[str],
    max_per_family: int = DEFAULT_MAX_OOD_PER_FAMILY,
    seed: int = 20260723,
) -> List[Dict[str, Any]]:
    """Observed rows whose reaction family is absent from train families.

    Args:
        heldout_rows: dicts with ``smiles``, ``label`` (0/1) and
            ``reaction_family``.
        train_families: families present in the manifest train split.
    """
    fam_set = set(train_families)
    rng = random.Random(seed)
    by_family: Dict[str, List[Dict[str, Any]]] = {}
    for row in heldout_rows:
        fam = row.get("reaction_family", "") or "unknown"
        if fam in fam_set:
            continue
        by_family.setdefault(fam, []).append({
            "smiles": row["smiles"],
            "label": int(row["label"]),
            "reaction_family": fam,
        })
    out: List[Dict[str, Any]] = []
    for fam, rows in sorted(by_family.items()):
        rng.shuffle(rows)
        out.extend(rows[:max_per_family])
    return out


def build_ood_scaffold_set(
    eval_candidates: Sequence[Dict[str, Any]],
    train_scaffolds: Sequence[str],
    max_n: int = 100,
    seed: int = 20260723,
) -> List[Dict[str, Any]]:
    """Scaffold-level OOD fallback: manifest eval candidates whose
    product_scaffold never appears in the train split.

    Used only when family-level OOD is empty (on v2 all HTEa val/test
    reaction families are covered by the 394 train groups). Each returned
    dict carries smiles, label (gold=1), reaction_family and
    product_scaffold.
    """
    sc_set = set(train_scaffolds)
    pool = []
    for c in eval_candidates:
        sc = c.get("product_scaffold", "") or ""
        if sc and sc not in sc_set:
            pool.append({
                "smiles": c.get("smiles") or "",
                "label": 1 if (c.get("gold_candidate") or c.get("label") == 1) else 0,
                "reaction_family": c.get("reaction_family", "") or "unknown",
                "product_scaffold": sc,
                "candidate_id": c.get("candidate_id", ""),
            })
    rng = random.Random(seed)
    rng.shuffle(pool)
    return pool[:max_n]


# ---------------------------------------------------------------------------
# Metrics (pure; probabilities passed in or derived from logits via sigmoid)
# ---------------------------------------------------------------------------

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, x))))


def known_positive_metrics(
    scored_groups: Sequence[Sequence[Dict[str, Any]]],
) -> Dict[str, float]:
    """Recovery + mean probability for disguised known positives.

    Args:
        scored_groups: one list of scored candidate dicts per group;
            each dict needs ``score`` and ``is_disguised_positive``.
    """
    n = 0
    top1 = 0
    probs: List[float] = []
    for group in scored_groups:
        if not group:
            continue
        best = max(group, key=lambda c: c["score"])
        n += 1
        if best.get("is_disguised_positive"):
            top1 += 1
        for c in group:
            if c.get("is_disguised_positive"):
                probs.append(_sigmoid(float(c["score"])))
    return {
        "n": n,
        "recovery_top1": (top1 / n) if n else 0.0,
        "mean_prob": (sum(probs) / len(probs)) if probs else 0.0,
    }


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return 0.0
    return cov / math.sqrt(vx * vy)


def near_positive_metrics(
    scored: Sequence[Dict[str, Any]],
    hard_reject_threshold: float = 0.1,
) -> Dict[str, float]:
    """Over-rejection + fnr-correlation for near-positive candidates.

    Each dict needs ``score``; ``fnr`` optional (enables correlation).
    """
    n = len(scored)
    if n == 0:
        return {"n": 0, "mean_prob": 0.0, "hard_reject_rate": 0.0, "fnr_corr": 0.0}
    probs = [_sigmoid(float(c["score"])) for c in scored]
    hard = sum(1 for p in probs if p < hard_reject_threshold)
    fnrs = [float(c["fnr"]) for c in scored if c.get("fnr") is not None]
    ps = [_sigmoid(float(c["score"])) for c in scored if c.get("fnr") is not None]
    return {
        "n": n,
        "mean_prob": sum(probs) / n,
        "hard_reject_rate": hard / n,
        "fnr_corr": _pearson(ps, fnrs) if len(fnrs) >= 2 else 0.0,
    }


def ece_brier_nll(
    probs: Sequence[float],
    labels: Sequence[int],
    n_bins: int = 10,
) -> Dict[str, float]:
    """ECE / Brier / NLL from probabilities."""
    n = len(probs)
    if n == 0:
        return {"ece": 0.0, "brier": 0.0, "nll": 0.0, "n": 0}
    brier = sum((p - y) ** 2 for p, y in zip(probs, labels)) / n
    nll = -sum(
        y * math.log(max(p, 1e-12)) + (1 - y) * math.log(max(1 - p, 1e-12))
        for p, y in zip(probs, labels)
    ) / n
    ece = 0.0
    for i in range(n_bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        idx = [j for j, p in enumerate(probs) if lo <= p < hi or (i == n_bins - 1 and p == 1.0)]
        if not idx:
            continue
        acc = sum(labels[j] for j in idx) / len(idx)
        conf = sum(probs[j] for j in idx) / len(idx)
        ece += (len(idx) / n) * abs(acc - conf)
    return {"ece": ece, "brier": brier, "nll": nll, "n": n}


def ood_metrics(scored: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    """Calibration on OOD-family observed rows (needs score + label)."""
    if not scored:
        return {"ece": 0.0, "brier": 0.0, "nll": 0.0, "n": 0}
    probs = [_sigmoid(float(c["score"])) for c in scored]
    labels = [int(c["label"]) for c in scored]
    return ece_brier_nll(probs, labels)


def collision_sensitivity(
    test_predictions: Sequence[Dict[str, Any]],
    collision_candidate_ids: Sequence[str],
    hard_reject_threshold: float = 0.1,
) -> Dict[str, float]:
    """How hard the model rejects known-positive collisions.

    Args:
        test_predictions: scored test candidates (score + candidate_id).
        collision_candidate_ids: candidate_ids flagged
            known_positive_collision=True in the manifest.
    """
    ids = set(collision_candidate_ids)
    probs = [
        _sigmoid(float(p["score"])) for p in test_predictions
        if p.get("candidate_id") in ids
    ]
    n = len(probs)
    if n == 0:
        return {"n": 0, "hard_reject_rate": 0.0, "mean_prob": 0.0}
    hard = sum(1 for p in probs if p < hard_reject_threshold)
    return {"n": n, "hard_reject_rate": hard / n, "mean_prob": sum(probs) / n}


def coverage_risk_curve(
    probs: Sequence[float],
    labels: Sequence[int],
    uncertainty: Sequence[float],
    coverages: Sequence[float] = COVERAGE_GRID,
) -> Dict[str, Any]:
    """Selective prediction: risk (Brier) vs coverage, abstaining on the
    highest-uncertainty tail.

    Returns {"coverage": [...], "risk": [...], "risk_at_0p8", "auc"}.
    """
    n = len(probs)
    if n == 0:
        return {"coverage": [], "risk": [], "risk_at_0p8": 0.0, "auc": 0.0}
    order = sorted(range(n), key=lambda i: uncertainty[i])  # keep low-unc first
    risks: List[float] = []
    used_coverages: List[float] = []
    for cov in coverages:
        k = max(1, int(round(cov * n)))
        keep = order[:k]
        risk = sum((probs[i] - labels[i]) ** 2 for i in keep) / k
        risks.append(risk)
        used_coverages.append(float(cov))
    # Trapezoidal AUC over the coverage grid
    auc = 0.0
    for i in range(1, len(risks)):
        auc += 0.5 * (risks[i] + risks[i - 1]) * (used_coverages[i] - used_coverages[i - 1])
    risk_at_0p8 = risks[used_coverages.index(0.8)] if 0.8 in used_coverages else risks[-1]
    return {
        "coverage": used_coverages,
        "risk": risks,
        "risk_at_0p8": risk_at_0p8,
        "auc": auc,
    }


# ---------------------------------------------------------------------------
# File-based wrapper (used by the orchestrator)
# ---------------------------------------------------------------------------

def load_htea_rows(
    htea_csv_path: Path,
    splits: Sequence[str],
    positives_only: bool = False,
) -> List[Dict[str, Any]]:
    """Load HTEa rows for given splits as {smiles, label, reaction_family}."""
    out: List[Dict[str, Any]] = []
    with open(htea_csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("split", "") not in splits:
                continue
            prod = (row.get("products") or "").strip()
            if not prod:
                continue
            try:
                y = float(row.get("yield", "") or 0.0)
            except ValueError:
                continue
            label = 1 if y > 0 else 0
            if positives_only and label != 1:
                continue
            out.append({
                "smiles": prod,
                "label": label,
                "reaction_family": row.get("reaction_class", "") or "unknown",
            })
    return out


def build_all_stress_sets(
    manifest_path: Path,
    htea_csv_path: Path,
    fnr_by_candidate: Optional[Dict[str, float]] = None,
    seed: int = 20260723,
) -> Dict[str, Any]:
    """Build the three stress sets from the v2 manifest + HTEa.

    Returns a JSON-serializable dict with keys in STRESS_SET_NAMES plus
    ``train_families`` and ``collision_candidate_ids`` (test split).
    """
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    train_families: set = set()
    train_scaffolds: set = set()
    train_negs_by_family: Dict[str, List[Dict[str, Any]]] = {}
    fallback_negs: List[Dict[str, Any]] = []
    eval_candidates: List[Dict[str, Any]] = []
    collision_ids: List[str] = []

    for group in manifest.get("groups", []):
        gsplit = group.get("split", "train")
        for cand in group.get("candidates", []):
            fam = cand.get("reaction_family", "") or "unknown"
            smi = re.sub(r":\d+", "", cand.get("candidate_smiles") or "")
            if gsplit == "train":
                train_families.add(fam)
                sc = cand.get("product_scaffold", "") or ""
                if sc:
                    train_scaffolds.add(sc)
                if not cand.get("gold_candidate", False):
                    entry = {"smiles": smi, "reaction_family": fam}
                    train_negs_by_family.setdefault(fam, []).append(entry)
                    fallback_negs.append(entry)
            else:
                c2 = dict(cand)
                c2["smiles"] = smi
                eval_candidates.append(c2)
                if gsplit == "test" and cand.get("known_positive_collision"):
                    collision_ids.append(cand.get("candidate_id", ""))

    heldout_pos = load_htea_rows(htea_csv_path, ["val", "test"], positives_only=True)
    heldout_all = load_htea_rows(htea_csv_path, ["val", "test"], positives_only=False)

    known_positive = build_known_positive_set(
        heldout_pos, train_negs_by_family, fallback_negs, seed=seed,
    )
    near_positive = build_near_positive_set(eval_candidates)
    if fnr_by_candidate:
        for c in near_positive:
            c["fnr"] = fnr_by_candidate.get(c.get("candidate_id", ""))
    ood_family = build_ood_family_set(heldout_all, sorted(train_families), seed=seed)
    ood_axis = "family"
    if not ood_family:
        # v2 data property: all HTEa val/test reaction families are covered
        # by the train split -> pre-declared scaffold-level fallback.
        ood_family = build_ood_scaffold_set(
            eval_candidates, sorted(train_scaffolds), seed=seed,
        )
        ood_axis = "scaffold"

    return {
        "known_positive": known_positive,
        "near_positive": near_positive,
        "ood_family": ood_family,
        "ood_axis": ood_axis,
        "train_families": sorted(train_families),
        "collision_candidate_ids": collision_ids,
    }
