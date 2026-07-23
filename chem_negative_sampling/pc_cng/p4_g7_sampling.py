"""P4-G7: Human Expert Calibration — stratified sampling and blinding.

Produces:
- ``results/p4_human_review/sampling_manifest.json`` — full mapping (blinded_id -> source, for post-review unblinding)
- ``results/p4_human_review/samples.csv`` — master sample list (blinded)
- ``results/p4_human_review/blinded_forms/{reviewer_id}.csv`` — per-reviewer blinded forms

Spec: pccng 的分阶段提示词.md#L1397-1569 (P4-G7)
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEED = 20260723

# 12 required strata (spec: 分层抽样)
STRATA = [
    "random",
    "tanimoto",
    "template",
    "pc_cng",
    "known_real_negative",
    "known_positive_control",
    "pc_cng_low_risk",
    "pc_cng_medium_risk",
    "pc_cng_high_risk",
    "llm_disagreement",
    "hte_false_positive",
    "hte_false_negative",
]

# Scoring dimensions (1-5 Likert)
SCORING_DIMENSIONS = [
    "structural_validity",
    "mechanistic_plausibility",
    "plausible_competing_outcome",
    "likely_low_yield_failure",
    "likely_feasible_positive",
    "confidence",
]

# Reason codes
REASON_CODES = [
    "wrong_reaction_center",
    "unlikely_bond_change",
    "condition_mismatch",
    "chemoselectivity_issue",
    "regioselectivity_issue",
    "stereochemistry_issue",
    "plausible_side_product",
    "likely_feasible_alternative",
    "insufficient_information",
    "other",
]

# FNR risk tiers
FNR_LOW_THRESHOLD = 0.2
FNR_HIGH_THRESHOLD = 0.5

# HTE false positive/negative thresholds
FP_SCORE_THRESHOLD = 0.6   # model predicted high
FP_YIELD_THRESHOLD = 5.0   # but actual yield is low
FN_SCORE_THRESHOLD = 0.4   # model predicted low
FN_YIELD_THRESHOLD = 50.0  # but actual yield is high

# Default sample counts per stratum for pilot (80 total)
PILOT_PER_STRATUM = {
    "random": 8,
    "tanimoto": 8,
    "template": 8,
    "pc_cng": 8,
    "known_real_negative": 8,
    "known_positive_control": 8,
    "pc_cng_low_risk": 5,
    "pc_cng_medium_risk": 5,
    "pc_cng_high_risk": 5,
    "llm_disagreement": 4,
    "hte_false_positive": 7,
    "hte_false_negative": 6,
}

# Default sample counts per stratum for main review (250 total)
MAIN_PER_STRATUM = {
    "random": 25,
    "tanimoto": 25,
    "template": 25,
    "pc_cng": 25,
    "known_real_negative": 20,
    "known_positive_control": 20,
    "pc_cng_low_risk": 18,
    "pc_cng_medium_risk": 18,
    "pc_cng_high_risk": 18,
    "llm_disagreement": 15,
    "hte_false_positive": 21,
    "hte_false_negative": 20,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_atom_mapping(smiles: str) -> str:
    """Remove atom mapping from SMILES for blinding."""
    if not smiles:
        return ""
    # Remove :NN patterns (atom mapping)
    s = re.sub(r":\d+", "", smiles)
    # Remove bracketed atom numbers like [C:1] -> [C]
    s = re.sub(r"\[([A-Za-z]+)(?:H\d*)?:\d+\]", r"[\1]", s)
    # Clean up any remaining colons
    s = s.replace(":", "")
    return s.strip()


def _blinded_id(seed: int, idx: int, source: str) -> str:
    """Generate a non-reversible blinded ID."""
    raw = f"{seed}:{idx}:{source}"
    return "BLD-" + hashlib.sha256(raw.encode()).hexdigest()[:12].upper()


def _load_manifest(manifest_path: Path) -> List[Dict[str, Any]]:
    """Load candidate manifest and flatten to list of candidates."""
    with open(manifest_path) as f:
        manifest = json.load(f)
    candidates = []
    for group in manifest.get("groups", []):
        for cand in group.get("candidates", []):
            cand["_group_id"] = group.get("group_id", "")
            cand["_source_reaction_id"] = group.get("source_reaction_id", "")
            cand["_experimental_group_id"] = group.get("experimental_group_id", "")
            candidates.append(cand)
    return candidates


def _load_risk_artifacts(risk_path: Path) -> Dict[str, Dict[str, Any]]:
    """Load risk artifacts keyed by candidate_id."""
    with open(risk_path) as f:
        artifacts = json.load(f)
    return artifacts.get("candidates", {})


def _load_g6_predictions(pred_path: Path) -> List[Dict[str, Any]]:
    """Load G6 raw predictions for a method."""
    with open(pred_path, newline="") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# Stratified sampling
# ---------------------------------------------------------------------------

def stratified_sample(
    manifest_path: Path,
    risk_artifacts_path: Path,
    g6_predictions_path: Path,
    n_per_stratum: Dict[str, int],
    seed: int = SEED,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Perform stratified sampling across 12 categories.

    Returns (samples, summary) where samples is a list of dicts with
    both blinded and unblinded fields.
    """
    rng = random.Random(seed)

    candidates = _load_manifest(manifest_path)
    risk_data = _load_risk_artifacts(risk_artifacts_path)
    g6_preds = _load_g6_predictions(g6_predictions_path)

    # Index candidates by source
    by_source: Dict[str, List[Dict]] = defaultdict(list)
    for c in candidates:
        by_source[c.get("candidate_source", "unknown")].append(c)

    # Index risk data by candidate_id
    risk_by_cid = risk_data

    # Index G6 predictions by record_id
    g6_by_rid = {p["record_id"]: p for p in g6_preds}

    samples: List[Dict[str, Any]] = []
    stratum_counts: Dict[str, int] = {}

    # Helper to pick n from a pool, avoiding duplicates
    used_ids: set = set()

    def _pick(pool: List[Dict], n: int) -> List[Dict]:
        available = [c for c in pool if c.get("candidate_id", "") not in used_ids]
        rng.shuffle(available)
        picked = available[:n]
        for c in picked:
            used_ids.add(c.get("candidate_id", ""))
        return picked

    # 1. random
    random_pool = by_source.get("random_corruption", []) + by_source.get("random_mismatch", [])
    for c in _pick(random_pool, n_per_stratum.get("random", 0)):
        samples.append(_make_sample(c, "random", rng))
    stratum_counts["random"] = len([s for s in samples if s["stratum"] == "random"])

    # 2. tanimoto
    for c in _pick(by_source.get("tanimoto_retrieval", []), n_per_stratum.get("tanimoto", 0)):
        samples.append(_make_sample(c, "tanimoto", rng))
    stratum_counts["tanimoto"] = len([s for s in samples if s["stratum"] == "tanimoto"])

    # 3. template
    for c in _pick(by_source.get("template_perturbation", []), n_per_stratum.get("template", 0)):
        samples.append(_make_sample(c, "template", rng))
    stratum_counts["template"] = len([s for s in samples if s["stratum"] == "template"])

    # 4. pc_cng (general, not risk-stratified)
    pc_cng_pool = by_source.get("rule_pc_cng", [])
    for c in _pick(pc_cng_pool, n_per_stratum.get("pc_cng", 0)):
        samples.append(_make_sample(c, "pc_cng", rng))
    stratum_counts["pc_cng"] = len([s for s in samples if s["stratum"] == "pc_cng"])

    # 5. known_real_negative (from HTE: below_detection or measured_zero)
    # Use gold candidates that have known zero yield in HTE
    # We'll use external_beam candidates as proxy for "known real negatives"
    # if they come from HTE zero-yield reactions. Otherwise use random_corruption
    # that are NOT in PC-CNG as "observed negative" proxy.
    neg_pool = [c for c in by_source.get("external_beam", [])
                if c.get("candidate_source") != "rule_pc_cng"]
    if len(neg_pool) < n_per_stratum.get("known_real_negative", 0):
        # Fallback: use random_corruption as negative-like
        neg_pool = by_source.get("random_corruption", [])
    for c in _pick(neg_pool, n_per_stratum.get("known_real_negative", 0)):
        samples.append(_make_sample(c, "known_real_negative", rng))
    stratum_counts["known_real_negative"] = len([s for s in samples if s["stratum"] == "known_real_negative"])

    # 6. known_positive_control (gold)
    for c in _pick(by_source.get("gold", []), n_per_stratum.get("known_positive_control", 0)):
        samples.append(_make_sample(c, "known_positive_control", rng))
    stratum_counts["known_positive_control"] = len([s for s in samples if s["stratum"] == "known_positive_control"])

    # 7-9. PC-CNG risk-stratified
    pc_cng_with_fnr = []
    for c in pc_cng_pool:
        cid = c.get("candidate_id", "")
        risk = risk_by_cid.get(cid, {})
        fnr = risk.get("false_negative_risk", 0.5)
        c["_fnr"] = fnr
        pc_cng_with_fnr.append(c)

    low_pool = [c for c in pc_cng_with_fnr if c["_fnr"] < FNR_LOW_THRESHOLD]
    med_pool = [c for c in pc_cng_with_fnr if FNR_LOW_THRESHOLD <= c["_fnr"] < FNR_HIGH_THRESHOLD]
    high_pool = [c for c in pc_cng_with_fnr if c["_fnr"] >= FNR_HIGH_THRESHOLD]

    for c in _pick(low_pool, n_per_stratum.get("pc_cng_low_risk", 0)):
        samples.append(_make_sample(c, "pc_cng_low_risk", rng, fnr=c["_fnr"]))
    stratum_counts["pc_cng_low_risk"] = len([s for s in samples if s["stratum"] == "pc_cng_low_risk"])

    for c in _pick(med_pool, n_per_stratum.get("pc_cng_medium_risk", 0)):
        samples.append(_make_sample(c, "pc_cng_medium_risk", rng, fnr=c["_fnr"]))
    stratum_counts["pc_cng_medium_risk"] = len([s for s in samples if s["stratum"] == "pc_cng_medium_risk"])

    for c in _pick(high_pool, n_per_stratum.get("pc_cng_high_risk", 0)):
        samples.append(_make_sample(c, "pc_cng_high_risk", rng, fnr=c["_fnr"]))
    stratum_counts["pc_cng_high_risk"] = len([s for s in samples if s["stratum"] == "pc_cng_high_risk"])

    # 10. llm_disagreement (proxy: high model uncertainty = score near 0.5)
    # Use G6 predictions where risk_aware score is near 0.5 (uncertain)
    uncertain_preds = [p for p in g6_preds
                       if 0.4 < float(p.get("score", 0.5)) < 0.6]
    rng.shuffle(uncertain_preds)
    for p in uncertain_preds[:n_per_stratum.get("llm_disagreement", 0)]:
        rid = p["record_id"]
        # Find matching candidate or create a synthetic sample from HTE
        # Use the HTE product SMILES
        # We don't have product SMILES in G6 preds, so we'll use a candidate
        # from the manifest that matches this record_id
        matched = None
        for c in candidates:
            if c.get("candidate_id", "") == rid or c.get("source_reaction_id", "") == rid:
                matched = c
                break
        if matched:
            samples.append(_make_sample(matched, "llm_disagreement", rng,
                                        model_score=float(p.get("score", 0.5))))
        else:
            # Create a minimal sample from the prediction
            samples.append({
                "blinded_id": _blinded_id(seed, len(samples), "llm_disagreement"),
                "stratum": "llm_disagreement",
                "reaction_smiles": "",  # Will be filled from HTE parquet
                "candidate_source": "hte_record",
                "candidate_id": rid,
                "fnr": None,
                "model_score": float(p.get("score", 0.5)),
                "actual_yield": float(p.get("yield", -1)),
            })
    stratum_counts["llm_disagreement"] = len([s for s in samples if s["stratum"] == "llm_disagreement"])

    # 11. hte_false_positive (model high score, actual low yield)
    fp_preds = [p for p in g6_preds
                if float(p.get("score", 0)) > FP_SCORE_THRESHOLD
                and float(p.get("yield", 100)) < FP_YIELD_THRESHOLD]
    rng.shuffle(fp_preds)
    for p in fp_preds[:n_per_stratum.get("hte_false_positive", 0)]:
        rid = p["record_id"]
        matched = None
        for c in candidates:
            if c.get("candidate_id", "") == rid or c.get("source_reaction_id", "") == rid:
                matched = c
                break
        if matched:
            samples.append(_make_sample(matched, "hte_false_positive", rng,
                                        model_score=float(p.get("score", 0)),
                                        actual_yield=float(p.get("yield", -1))))
        else:
            samples.append({
                "blinded_id": _blinded_id(seed, len(samples), "hte_false_positive"),
                "stratum": "hte_false_positive",
                "reaction_smiles": "",
                "candidate_source": "hte_record",
                "candidate_id": rid,
                "fnr": None,
                "model_score": float(p.get("score", 0)),
                "actual_yield": float(p.get("yield", -1)),
            })
    stratum_counts["hte_false_positive"] = len([s for s in samples if s["stratum"] == "hte_false_positive"])

    # 12. hte_false_negative (model low score, actual high yield)
    fn_preds = [p for p in g6_preds
                if float(p.get("score", 0)) < FN_SCORE_THRESHOLD
                and float(p.get("yield", 0)) > FN_YIELD_THRESHOLD]
    rng.shuffle(fn_preds)
    for p in fn_preds[:n_per_stratum.get("hte_false_negative", 0)]:
        rid = p["record_id"]
        matched = None
        for c in candidates:
            if c.get("candidate_id", "") == rid or c.get("source_reaction_id", "") == rid:
                matched = c
                break
        if matched:
            samples.append(_make_sample(matched, "hte_false_negative", rng,
                                        model_score=float(p.get("score", 0)),
                                        actual_yield=float(p.get("yield", -1))))
        else:
            samples.append({
                "blinded_id": _blinded_id(seed, len(samples), "hte_false_negative"),
                "stratum": "hte_false_negative",
                "reaction_smiles": "",
                "candidate_source": "hte_record",
                "candidate_id": rid,
                "fnr": None,
                "model_score": float(p.get("score", 0)),
                "actual_yield": float(p.get("yield", -1)),
            })
    stratum_counts["hte_false_negative"] = len([s for s in samples if s["stratum"] == "hte_false_negative"])

    # Assign sequential blinded IDs (re-number for cleaner appearance)
    rng.shuffle(samples)  # Final shuffle so strata are interleaved
    for i, s in enumerate(samples):
        s["blinded_id"] = f"S{i+1:04d}"

    summary = {
        "n_total": len(samples),
        "stratum_counts": stratum_counts,
        "seed": seed,
        "n_per_stratum_requested": n_per_stratum,
        "all_strata_covered": all(stratum_counts.get(s, 0) > 0 for s in STRATA),
    }

    return samples, summary


def _make_sample(cand: Dict, stratum: str, rng: random.Random,
                 fnr: Optional[float] = None,
                 model_score: Optional[float] = None,
                 actual_yield: Optional[float] = None) -> Dict[str, Any]:
    """Create a sample record from a candidate."""
    smiles = cand.get("candidate_smiles", "")
    return {
        "blinded_id": "",  # will be assigned after shuffle
        "stratum": stratum,
        "reaction_smiles": _strip_atom_mapping(smiles),
        "candidate_source": cand.get("candidate_source", ""),
        "candidate_id": cand.get("candidate_id", ""),
        "group_id": cand.get("_group_id", ""),
        "reaction_family": cand.get("reaction_family", ""),
        "fnr": fnr,
        "model_score": model_score,
        "actual_yield": actual_yield,
    }


# ---------------------------------------------------------------------------
# Blinded forms
# ---------------------------------------------------------------------------

def create_blinded_forms(
    samples: List[Dict[str, Any]],
    output_dir: Path,
    n_reviewers: int = 2,
    seed: int = SEED,
) -> List[Path]:
    """Create per-reviewer blinded form CSVs.

    Each form contains ONLY:
    - blinded_id
    - reaction_smiles (atom mapping stripped)
    - Empty columns for each scoring dimension
    - Empty column for reason_codes

    The form does NOT contain: candidate_source, fnr, model_score, stratum.
    Each reviewer gets a different randomization order.
    """
    rng = random.Random(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    forms_dir = output_dir / "blinded_forms"
    forms_dir.mkdir(exist_ok=True)

    form_paths = []
    for reviewer_idx in range(n_reviewers):
        reviewer_id = f"reviewer_{reviewer_idx + 1}"
        # Create a shuffled copy for this reviewer
        indices = list(range(len(samples)))
        rng.shuffle(indices)

        form_path = forms_dir / f"{reviewer_id}.csv"
        with open(form_path, "w", newline="") as f:
            w = csv.writer(f)
            # Header
            header = ["blinded_id", "reaction_smiles"]
            header.extend(SCORING_DIMENSIONS)
            header.append("reason_codes")
            header.append("notes")
            w.writerow(header)

            for idx in indices:
                s = samples[idx]
                row = [s["blinded_id"], s["reaction_smiles"]]
                # Empty scoring columns (experts fill these in)
                row.extend([""] * len(SCORING_DIMENSIONS))
                row.append("")  # reason_codes
                row.append("")  # notes
                w.writerow(row)

        form_paths.append(form_path)

    return form_paths


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_sampling_manifest(
    samples: List[Dict[str, Any]],
    summary: Dict[str, Any],
    output_path: Path,
    phase: str = "pilot",
) -> Path:
    """Write the sampling manifest JSON (for post-review unblinding)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": "p4_human_review_sampling_v1",
        "phase": phase,
        "seed": summary["seed"],
        "n_total": summary["n_total"],
        "stratum_counts": summary["stratum_counts"],
        "all_strata_covered": summary["all_strata_covered"],
        "scoring_dimensions": SCORING_DIMENSIONS,
        "reason_codes": REASON_CODES,
        "samples": samples,  # includes unblinded info (stratum, source, fnr, etc.)
    }
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return output_path


def write_samples_csv(
    samples: List[Dict[str, Any]],
    output_path: Path,
) -> Path:
    """Write the master samples CSV (blinded — no source, no fnr, no score)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["blinded_id", "reaction_smiles"])
        for s in samples:
            w.writerow([s["blinded_id"], s["reaction_smiles"]])
    return output_path


def run_pilot(
    manifest_path: Path,
    risk_artifacts_path: Path,
    g6_predictions_path: Path,
    output_dir: Path,
    seed: int = SEED,
) -> Dict[str, Any]:
    """Generate pilot review materials (80 samples, 2 reviewers)."""
    samples, summary = stratified_sample(
        manifest_path, risk_artifacts_path, g6_predictions_path,
        n_per_stratum=PILOT_PER_STRATUM, seed=seed,
    )

    write_sampling_manifest(samples, summary, output_dir / "sampling_manifest.json", phase="pilot")
    write_samples_csv(samples, output_dir / "samples.csv")
    create_blinded_forms(samples, output_dir, n_reviewers=2, seed=seed)

    return {
        "phase": "pilot",
        "n_samples": len(samples),
        "stratum_counts": summary["stratum_counts"],
        "all_strata_covered": summary["all_strata_covered"],
        "n_reviewers": 2,
        "output_dir": str(output_dir),
    }


def run_main_review(
    manifest_path: Path,
    risk_artifacts_path: Path,
    g6_predictions_path: Path,
    output_dir: Path,
    n_reviewers: int = 3,
    seed: int = SEED,
) -> Dict[str, Any]:
    """Generate main review materials (250 samples, 3 reviewers)."""
    samples, summary = stratified_sample(
        manifest_path, risk_artifacts_path, g6_predictions_path,
        n_per_stratum=MAIN_PER_STRATUM, seed=seed,
    )

    write_sampling_manifest(samples, summary, output_dir / "sampling_manifest.json", phase="main")
    write_samples_csv(samples, output_dir / "samples.csv")
    create_blinded_forms(samples, output_dir, n_reviewers=n_reviewers, seed=seed)

    return {
        "phase": "main",
        "n_samples": len(samples),
        "stratum_counts": summary["stratum_counts"],
        "all_strata_covered": summary["all_strata_covered"],
        "n_reviewers": n_reviewers,
        "output_dir": str(output_dir),
    }
