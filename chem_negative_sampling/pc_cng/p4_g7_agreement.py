"""P4-G7: Human Expert Calibration — agreement and statistical analysis.

Computes:
- Human-human weighted Cohen's kappa
- Krippendorff's alpha
- Control discrimination (positive vs negative controls)
- Source-level effect size
- Reviewer effect
- Confidence sensitivity
- LLM-human agreement (supplementary only)

Spec: pccng 的分阶段提示词.md#L1397-1569 (P4-G7)
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Weighted Cohen's Kappa
# ---------------------------------------------------------------------------

def weighted_kappa(rater1: List[int], rater2: List[int],
                   weights: Optional[np.ndarray] = None) -> float:
    """Quadratic weighted kappa for ordinal ratings (1-5 Likert).

    Uses quadratic weights by default: w_ij = (i-j)^2 / (k-1)^2
    """
    if len(rater1) != len(rater2) or len(rater1) == 0:
        return 0.0

    r1 = np.array(rater1, dtype=float)
    r2 = np.array(rater2, dtype=float)

    # Rating scale (assumed 1-5)
    categories = sorted(set(r1.tolist() + r2.tolist()))
    k = len(categories)
    if k < 2:
        return 1.0 if len(r1) > 0 else 0.0

    cat_to_idx = {c: i for i, c in enumerate(categories)}

    # Weight matrix (quadratic)
    if weights is None:
        weights = np.zeros((k, k))
        for i in range(k):
            for j in range(k):
                weights[i, j] = ((i - j) ** 2) / ((k - 1) ** 2) if k > 1 else 0

    # Observed confusion matrix
    obs = np.zeros((k, k))
    for a, b in zip(r1, r2):
        obs[cat_to_idx[a], cat_to_idx[b]] += 1
    obs /= obs.sum()

    # Expected matrix (outer product of marginals)
    marg1 = obs.sum(axis=1, keepdims=True)
    marg2 = obs.sum(axis=0, keepdims=True)
    expected = marg1 @ marg2

    # Weighted kappa
    num = np.sum(weights * obs)
    den = np.sum(weights * expected)
    if den == 0:
        return 0.0
    return float(1 - num / den)


# ---------------------------------------------------------------------------
# Krippendorff's Alpha
# ---------------------------------------------------------------------------

def krippendorff_alpha(
    data: List[List[Optional[int]]],
    level: str = "ordinal",
) -> float:
    """Krippendorff's alpha for multiple raters.

    Args:
        data: reliability data matrix, rows = items, cols = raters.
              None/NaN for missing values.
        level: "ordinal", "nominal", "interval", or "ratio".

    Returns:
        Alpha value (-1 to 1).
    """
    if not data or len(data) == 0:
        return 0.0

    n_raters = max(len(row) for row in data)
    if n_raters < 2:
        return 0.0

    # Collect all unique values
    all_vals = set()
    for row in data:
        for v in row:
            if v is not None and not (isinstance(v, float) and math.isnan(v)):
                all_vals.add(v)
    categories = sorted(all_vals)
    k = len(categories)
    if k < 2:
        return 1.0

    cat_to_idx = {c: i for i, c in enumerate(categories)}

    # Count pairable values per item
    u = sum(1 for row in data
            for v in row
            if v is not None and not (isinstance(v, float) and math.isnan(v)))

    # Observed disagreement Do
    Do = 0.0
    n_pairs = 0
    for row in data:
        vals = [v for v in row
                if v is not None and not (isinstance(v, float) and math.isnan(v))]
        m = len(vals)
        if m < 2:
            continue
        n_pairs += m * (m - 1)
        for i in range(len(vals)):
            for j in range(i + 1, len(vals)):
                vi = cat_to_idx[vals[i]]
                vj = cat_to_idx[vals[j]]
                if level == "ordinal":
                    diff = sum(range(min(vi, vj), max(vi, vj))) - (vi + vj) / 2 * (max(vi, vj) - min(vi, vj))
                    # Use squared distance for ordinal
                    Do += (vi - vj) ** 2
                elif level == "nominal":
                    Do += 0 if vi == vj else 1
                elif level == "interval":
                    Do += (vi - vj) ** 2
                else:  # ratio
                    Do += (vals[i] - vals[j]) ** 2
    if n_pairs == 0:
        return 0.0
    Do /= n_pairs

    # Expected disagreement De
    cat_counts = defaultdict(int)
    for row in data:
        for v in row:
            if v is not None and not (isinstance(v, float) and math.isnan(v)):
                cat_counts[cat_to_idx[v]] += 1

    De = 0.0
    total = sum(cat_counts.values())
    for i in range(k):
        for j in range(k):
            if i == j:
                continue
            pi = cat_counts[i] / total
            pj = cat_counts[j] / total
            if level == "ordinal":
                De += pi * pj * 2 * ((sum(range(i, j)) if i < j else sum(range(j, i))) ** 2) / (k - 1) ** 2 if k > 1 else 0
                # Simplified: use squared distance
                De = 0.0  # Recalculate below
    # Simplified De for ordinal (using squared distance)
    De = 0.0
    for i in range(k):
        for j in range(k):
            if i == j:
                continue
            pi = cat_counts[i] / total
            pj = cat_counts[j] / total
            De += pi * pj * (i - j) ** 2

    if De == 0:
        return 1.0

    return float(1 - Do / De)


# ---------------------------------------------------------------------------
# Control discrimination
# ---------------------------------------------------------------------------

def control_discrimination(
    positive_scores: List[float],
    negative_scores: List[float],
) -> Dict[str, Any]:
    """Test if experts can discriminate positive from negative controls.

    Uses Mann-Whitney U test (non-parametric) and effect size.
    """
    from scipy.stats import mannwhitneyu

    if len(positive_scores) < 2 or len(negative_scores) < 2:
        return {
            "discrimination_significant": False,
            "reason": "Insufficient samples",
            "n_positive": len(positive_scores),
            "n_negative": len(negative_scores),
        }

    pos = np.array(positive_scores, dtype=float)
    neg = np.array(negative_scores, dtype=float)

    # Mann-Whitney U
    try:
        stat, pval = mannwhitneyu(pos, neg, alternative="greater")
    except ValueError:
        stat, pval = 0.0, 1.0

    # Effect size (rank-biserial correlation)
    n1, n2 = len(pos), len(neg)
    r_effect = 1 - (2 * stat) / (n1 * n2) if n1 * n2 > 0 else 0.0

    return {
        "discrimination_significant": pval < 0.05,
        "p_value": round(float(pval), 6),
        "effect_size_r": round(float(r_effect), 4),
        "mean_positive": round(float(np.mean(pos)), 4),
        "mean_negative": round(float(np.mean(neg)), 4),
        "mean_diff": round(float(np.mean(pos) - np.mean(neg)), 4),
        "n_positive": len(positive_scores),
        "n_negative": len(negative_scores),
    }


# ---------------------------------------------------------------------------
# Source-level effect
# ---------------------------------------------------------------------------

def source_level_effect(
    scores_by_source: Dict[str, List[float]],
) -> Dict[str, Any]:
    """Compute effect sizes between PC-CNG sources and baselines.

    Uses Kruskal-Wallis test for overall differences, then
    pairwise Mann-Whitney for PC-CNG vs each baseline.
    """
    from scipy.stats import kruskal, mannwhitneyu

    sources = list(scores_by_source.keys())
    if len(sources) < 2:
        return {"overall_p": 1.0, "pairwise": {}}

    groups = [np.array(scores_by_source[s], dtype=float)
              for s in sources if len(scores_by_source[s]) > 0]
    if len(groups) < 2:
        return {"overall_p": 1.0, "pairwise": {}}

    # Kruskal-Wallis
    try:
        kw_stat, kw_p = kruskal(*groups)
    except ValueError:
        kw_stat, kw_p = 0.0, 1.0

    # Pairwise: PC-CNG sources vs baselines
    pc_cng_sources = [s for s in sources if "pc_cng" in s]
    baseline_sources = [s for s in sources if "pc_cng" not in s
                        and s not in ("known_positive_control", "known_real_negative")]

    pairwise = {}
    for pc in pc_cng_sources:
        for bl in baseline_sources:
            if len(scores_by_source[pc]) < 2 or len(scores_by_source[bl]) < 2:
                continue
            try:
                stat, pval = mannwhitneyu(
                    scores_by_source[pc], scores_by_source[bl],
                    alternative="greater",
                )
                # Cliff's delta
                delta = _cliffs_delta(scores_by_source[pc], scores_by_source[bl])
                pairwise[f"{pc}_vs_{bl}"] = {
                    "p_value": round(float(pval), 6),
                    "cliffs_delta": round(float(delta), 4),
                    "pc_cng_mean": round(float(np.mean(scores_by_source[pc])), 4),
                    "baseline_mean": round(float(np.mean(scores_by_source[bl])), 4),
                }
            except (ValueError, IndexError):
                continue

    return {
        "kruskal_wallis_p": round(float(kw_p), 6),
        "pairwise": pairwise,
    }


def _cliffs_delta(a: List[float], b: List[float]) -> float:
    """Cliff's delta effect size."""
    a_arr = np.array(a)
    b_arr = np.array(b)
    n = len(a_arr) * len(b_arr)
    if n == 0:
        return 0.0
    more = sum(1 for ai in a_arr for bi in b_arr if ai > bi)
    less = sum(1 for ai in a_arr for bi in b_arr if ai < bi)
    return (more - less) / n


# ---------------------------------------------------------------------------
# Reviewer effect
# ---------------------------------------------------------------------------

def reviewer_effect(
    scores_by_reviewer: Dict[str, List[float]],
) -> Dict[str, Any]:
    """Test for systematic differences between reviewers."""
    reviewers = list(scores_by_reviewer.keys())
    if len(reviewers) < 2:
        return {"reviewer_effect_significant": False, "reason": "Need >= 2 reviewers"}

    from scipy.stats import kruskal

    groups = [np.array(scores_by_reviewer[r], dtype=float)
              for r in reviewers if len(scores_by_reviewer[r]) > 0]
    if len(groups) < 2:
        return {"reviewer_effect_significant": False, "reason": "Insufficient data"}

    try:
        stat, pval = kruskal(*groups)
    except ValueError:
        stat, pval = 0.0, 1.0

    means = {r: round(float(np.mean(scores_by_reviewer[r])), 4)
             for r in reviewers if len(scores_by_reviewer[r]) > 0}

    return {
        "reviewer_effect_significant": pval < 0.05,
        "p_value": round(float(pval), 6),
        "reviewer_means": means,
    }


# ---------------------------------------------------------------------------
# Confidence sensitivity
# ---------------------------------------------------------------------------

def confidence_sensitivity(
    scores: List[float],
    confidence_ratings: List[float],
) -> Dict[str, Any]:
    """Correlation between expert confidence and score extremity.

    Higher confidence should correlate with more extreme scores
    (further from neutral 3.0 on a 1-5 scale).
    """
    if len(scores) != len(confidence_ratings) or len(scores) < 3:
        return {"correlation": 0.0, "significant": False}

    from scipy.stats import spearmanr

    scores_arr = np.array(scores, dtype=float)
    conf_arr = np.array(confidence_ratings, dtype=float)

    # Score extremity = distance from neutral
    extremity = np.abs(scores_arr - 3.0)

    r, p = spearmanr(extremity, conf_arr)
    return {
        "correlation": round(float(r) if not math.isnan(r) else 0.0, 4),
        "p_value": round(float(p) if not math.isnan(p) else 1.0, 6),
        "significant": p < 0.05,
        "interpretation": "Higher confidence correlates with more extreme scores" if r > 0.1 else "No clear confidence-score relationship",
    }


# ---------------------------------------------------------------------------
# LLM-human agreement (supplementary)
# ---------------------------------------------------------------------------

def llm_human_agreement(
    llm_scores: List[float],
    human_scores: List[float],
) -> Dict[str, Any]:
    """Compute LLM-human agreement (supplementary only).

    This CANNOT replace human-human agreement.
    """
    if len(llm_scores) != len(human_scores) or len(llm_scores) < 3:
        return {"agreement": 0.0, "note": "Insufficient data"}

    from scipy.stats import spearmanr

    r, p = spearmanr(llm_scores, human_scores)
    mae = float(np.mean(np.abs(np.array(llm_scores) - np.array(human_scores))))

    return {
        "spearman_rho": round(float(r) if not math.isnan(r) else 0.0, 4),
        "p_value": round(float(p) if not math.isnan(p) else 1.0, 6),
        "mae": round(mae, 4),
        "note": "Supplementary analysis only; cannot replace human-human agreement",
    }


# ---------------------------------------------------------------------------
# Full analysis pipeline
# ---------------------------------------------------------------------------

def analyze_responses(
    responses_path: Path,
    manifest_path: Path,
    output_path: Path,
) -> Dict[str, Any]:
    """Full analysis of expert review responses.

    Args:
        responses_path: Path to CSV with expert responses.
            Columns: blinded_id, reviewer_id, structural_validity, ...,
                     confidence, reason_codes.
        manifest_path: Path to sampling_manifest.json (for unblinding).
        output_path: Path to write agreement_report.json.
    """
    # Load manifest for unblinding
    with open(manifest_path) as f:
        manifest = json.load(f)
    sample_info = {s["blinded_id"]: s for s in manifest["samples"]}

    # Load responses
    with open(responses_path, newline="") as f:
        responses = list(csv.DictReader(f))

    if not responses:
        return {"error": "No responses found", "n_responses": 0}

    # Group by reviewer
    by_reviewer: Dict[str, Dict[str, Dict]] = defaultdict(lambda: defaultdict(dict))
    for r in responses:
        rid = r.get("reviewer_id", "")
        bid = r.get("blinded_id", "")
        by_reviewer[rid][bid] = r

    reviewers = list(by_reviewer.keys())

    # 1. Human-human weighted kappa (pairwise)
    kappa_results = {}
    for i in range(len(reviewers)):
        for j in range(i + 1, len(reviewers)):
            r1, r2 = reviewers[i], reviewers[j]
            shared_ids = set(by_reviewer[r1].keys()) & set(by_reviewer[r2].keys())
            if len(shared_ids) < 5:
                continue
            # Use structural_validity as primary dimension
            r1_scores = [int(by_reviewer[r1][bid].get("structural_validity", 3))
                         for bid in sorted(shared_ids)
                         if by_reviewer[r1][bid].get("structural_validity", "").isdigit()]
            r2_scores = [int(by_reviewer[r2][bid].get("structural_validity", 3))
                         for bid in sorted(shared_ids)
                         if by_reviewer[r2][bid].get("structural_validity", "").isdigit()]
            if len(r1_scores) >= 5:
                kappa_results[f"{r1}_vs_{r2}"] = round(weighted_kappa(r1_scores, r2_scores), 4)

    # 2. Krippendorff's alpha
    all_items = set()
    for r in by_reviewer.values():
        all_items.update(r.keys())
    alpha_data = []
    for bid in sorted(all_items):
        row = []
        for rid in reviewers:
            val = by_reviewer[rid].get(bid, {}).get("structural_validity", "")
            row.append(int(val) if val.isdigit() else None)
        alpha_data.append(row)
    alpha = round(krippendorff_alpha(alpha_data, level="ordinal"), 4) if alpha_data else 0.0

    # 3. Control discrimination
    pos_scores = []
    neg_scores = []
    for r in responses:
        bid = r.get("blinded_id", "")
        info = sample_info.get(bid, {})
        sv = r.get("structural_validity", "")
        if not sv.isdigit():
            continue
        sv = int(sv)
        if info.get("stratum") == "known_positive_control":
            pos_scores.append(float(sv))
        elif info.get("stratum") == "known_real_negative":
            neg_scores.append(float(sv))
    control_disc = control_discrimination(pos_scores, neg_scores)

    # 4. Source-level effect
    scores_by_source: Dict[str, List[float]] = defaultdict(list)
    for r in responses:
        bid = r.get("blinded_id", "")
        info = sample_info.get(bid, {})
        sv = r.get("structural_validity", "")
        if sv.isdigit():
            source = info.get("stratum", "unknown")
            scores_by_source[source].append(float(sv))
    source_effect = source_level_effect(scores_by_source)

    # 5. Reviewer effect
    scores_by_reviewer: Dict[str, List[float]] = defaultdict(list)
    for r in responses:
        sv = r.get("structural_validity", "")
        if sv.isdigit():
            scores_by_reviewer[r.get("reviewer_id", "")].append(float(sv))
    rev_effect = reviewer_effect(scores_by_reviewer)

    # 6. Confidence sensitivity
    all_scores = []
    all_conf = []
    for r in responses:
        sv = r.get("structural_validity", "")
        conf = r.get("confidence", "")
        if sv.isdigit() and conf.isdigit():
            all_scores.append(float(sv))
            all_conf.append(float(conf))
    conf_sens = confidence_sensitivity(all_scores, all_conf)

    # Compile report
    report = {
        "schema": "p4_human_review_agreement_v1",
        "n_responses": len(responses),
        "n_reviewers": len(reviewers),
        "reviewers": reviewers,
        "weighted_kappa": kappa_results,
        "krippendorff_alpha": alpha,
        "control_discrimination": control_disc,
        "source_level_effect": source_effect,
        "reviewer_effect": rev_effect,
        "confidence_sensitivity": conf_sens,
        "verdict": _compute_verdict(kappa_results, alpha, control_disc, source_effect),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    return report


def _compute_verdict(
    kappa_results: Dict[str, float],
    alpha: float,
    control_disc: Dict[str, Any],
    source_effect: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute go/no-go verdict based on spec criteria."""
    max_kappa = max(kappa_results.values()) if kappa_results else 0.0

    # GO: kappa or alpha >= 0.5, controls discriminated, PC-CNG > random/template
    pc_cng_better = False
    pairwise = source_effect.get("pairwise", {})
    for key, val in pairwise.items():
        if "pc_cng" in key and "random" in key:
            if val.get("p_value", 1.0) < 0.05 and val.get("cliffs_delta", 0) > 0:
                pc_cng_better = True
        if "pc_cng" in key and "template" in key:
            if val.get("p_value", 1.0) < 0.05 and val.get("cliffs_delta", 0) > 0:
                pc_cng_better = True

    controls_ok = control_disc.get("discrimination_significant", False)
    agreement_ok = max(max_kappa, alpha) >= 0.5

    if agreement_ok and controls_ok and pc_cng_better:
        verdict = "GO"
        reason = (f"Agreement {'kappa' if max_kappa >= alpha else 'alpha'} >= 0.5, "
                  f"controls discriminated, PC-CNG > baseline")
    elif agreement_ok or controls_ok:
        verdict = "PARTIAL_GO"
        reason = "Moderate agreement or control discrimination; qualitative evidence only"
    else:
        verdict = "NO_GO"
        reason = "Insufficient agreement or control discrimination"

    return {
        "verdict": verdict,
        "reason": reason,
        "max_weighted_kappa": round(max_kappa, 4),
        "krippendorff_alpha": alpha,
        "controls_discriminated": controls_ok,
        "pc_cng_superior": pc_cng_better,
        "next_phase_allowed": verdict in ("GO", "PARTIAL_GO"),
    }
