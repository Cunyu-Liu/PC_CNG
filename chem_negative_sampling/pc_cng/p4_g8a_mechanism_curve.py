"""P4-G8A: Difficulty-benefit-risk mechanism curve analysis.

Determines the real relationship between candidate difficulty and
utility/risk. Does NOT force an inverted-U; reports the true shape
(monotonic, inverted-U, threshold, dataset-specific, scorer-specific).

Spec: 提示词/pccng 的分阶段提示词.md#L1572-1833 (P4-G8A)

Outputs (results/p4_mechanism_curve/):
    per_candidate_metrics.csv   — all candidates with difficulty + utility + risk
    curve_specs.json            — frozen binning + fitted curve specs
    risk_curve.json             — risk (FNR) vs difficulty
    utility_curve.json          — utility (MRR) vs difficulty
    shape_comparison.csv        — R² for each candidate shape
    go_no_go.json               — verdict
    run_manifest.json, environment.json, input_hashes.json, commands.log
"""

from __future__ import annotations

import json
import os
import platform
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Suppress RDKit warnings
os.environ["RDKitRDLogger"] = "0"
try:
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
except Exception:
    pass

PHASE = "P4-G8A"
SEED = 20260723
N_BINS = 10  # deciles for binning

# Difficulty metrics to analyze
DIFFICULTY_METRICS = [
    "positive_similarity",         # Tanimoto to gold candidate
    "nearest_train_similarity",    # from manifest
    "scoring_margin",              # top-1 - top-2 score
    "ensemble_uncertainty",        # std across seeds
    "false_negative_risk",         # from G5 risk artifacts
    "edit_distance",               # from manifest
    "known_positive_collision",    # database collision (binary)
]

# Curve shapes to test
CURVE_SHAPES = ["monotonic_decreasing", "monotonic_increasing",
                "inverted_u", "threshold", "flat"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_manifest(manifest_path: Path) -> Dict[str, dict]:
    """Load v2 manifest and return candidate_id -> candidate dict."""
    with open(manifest_path) as f:
        manifest = json.load(f)
    candidates = {}
    for group in manifest.get("groups", []):
        for cand in group.get("candidates", []):
            candidates[cand["candidate_id"]] = cand
    return candidates


def load_predictions(pred_dir: Path) -> Dict[str, List[dict]]:
    """Load all prediction files from a paired_predictions directory.

    Returns dict: {scorer_arm_seed -> [predictions]}
    """
    results = {}
    if not pred_dir.exists():
        return results
    for d in sorted(pred_dir.iterdir()):
        if not d.is_dir():
            continue
        test_file = d / "test_predictions.json"
        if test_file.exists():
            with open(test_file) as f:
                results[d.name] = json.load(f)
    return results


def load_risk_artifacts(risk_path: Path) -> Dict[str, dict]:
    """Load G5 risk artifacts."""
    if not risk_path.exists():
        return {}
    with open(risk_path) as f:
        data = json.load(f)
    return data.get("candidates", {})


# ---------------------------------------------------------------------------
# Difficulty metrics computation
# ---------------------------------------------------------------------------

def _tanimoto(smiles1: str, smiles2: str) -> float:
    """Tanimoto similarity between two SMILES."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.DataStructs import TanimotoSimilarity
    m1, m2 = Chem.MolFromSmiles(smiles1), Chem.MolFromSmiles(smiles2)
    if m1 is None or m2 is None:
        return 0.0
    fp1 = AllChem.GetMorganFingerprintAsBitVect(m1, 2, nBits=2048)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(m2, 2, nBits=2048)
    return TanimotoSimilarity(fp1, fp2)


def compute_positive_similarity(candidates: Dict[str, dict]) -> Dict[str, float]:
    """Tanimoto similarity of each candidate to its group's gold candidate."""
    # Group candidates by group_id
    gold_by_group = {}
    for cid, cand in candidates.items():
        if cand.get("gold_candidate"):
            gold_by_group[cand["group_id"]] = cand["candidate_smiles"]

    result = {}
    for cid, cand in candidates.items():
        gold_smi = gold_by_group.get(cand["group_id"])
        if gold_smi and cand["candidate_smiles"] != gold_smi:
            result[cid] = _tanimoto(cand["candidate_smiles"], gold_smi)
        else:
            result[cid] = 1.0  # gold candidate has similarity 1.0
    return result


def compute_scoring_margin(predictions: List[dict]) -> Dict[str, float]:
    """Score margin: top-1 - top-2 score per candidate (within group)."""
    by_group = defaultdict(list)
    for pred in predictions:
        by_group[pred["group_id"]].append(pred)

    result = {}
    for group_id, group_preds in by_group.items():
        sorted_preds = sorted(group_preds, key=lambda p: p["score"], reverse=True)
        if len(sorted_preds) >= 2:
            margin = sorted_preds[0]["score"] - sorted_preds[1]["score"]
        else:
            margin = 0.0
        for pred in sorted_preds:
            result[pred["candidate_id"]] = margin
    return result


def compute_ensemble_uncertainty(
    all_preds: Dict[str, List[dict]]
) -> Dict[str, float]:
    """Std of scores across seeds for each candidate."""
    # Group by scorer prefix (e.g., "chemformer_A0_" -> all seeds)
    scorer_seed_scores = defaultdict(lambda: defaultdict(list))
    for key, preds in all_preds.items():
        # Parse key: "scorer_arm_seedXXXXX"
        parts = key.rsplit("_seed", 1)
        if len(parts) == 2:
            prefix = parts[0]  # "chemformer_A0"
        else:
            prefix = key
        for pred in preds:
            scorer_seed_scores[pred["candidate_id"]][prefix].append(pred["score"])

    result = {}
    for cid, scorer_scores in scorer_seed_scores.items():
        # Average std across scorers
        stds = []
        for prefix, scores in scorer_scores.items():
            if len(scores) > 1:
                stds.append(statistics.stdev(scores))
        result[cid] = statistics.mean(stds) if stds else 0.0
    return result


def compute_downstream_loss(
    predictions: List[dict], candidates: Dict[str, dict]
) -> Dict[str, float]:
    """1 - (rank-based MRR contribution) per candidate."""
    by_group = defaultdict(list)
    for pred in predictions:
        by_group[pred["group_id"]].append(pred)

    result = {}
    for group_id, group_preds in by_group.items():
        sorted_preds = sorted(group_preds, key=lambda p: p["score"], reverse=True)
        for rank, pred in enumerate(sorted_preds, 1):
            # MRR contribution = 1/rank if label==1, else 0
            mrr_contrib = (1.0 / rank) if pred.get("label") == 1 else 0.0
            result[pred["candidate_id"]] = 1.0 - mrr_contrib
    return result


# ---------------------------------------------------------------------------
# Curve fitting
# ---------------------------------------------------------------------------

def bin_values(values: List[float], n_bins: int = N_BINS) -> List[int]:
    """Assign each value to a bin (0 to n_bins-1)."""
    if not values:
        return []
    arr = np.array(values)
    bins = np.linspace(arr.min(), arr.max() + 1e-10, n_bins + 1)
    return np.digitize(arr, bins[1:-1]).tolist()


def fit_curve(x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """Fit different curve shapes and return R² for each.

    Shapes:
    - monotonic_decreasing: y = a - b*x (b > 0)
    - monotonic_increasing: y = a + b*x (b > 0)
    - inverted_u: y = a + b*x - c*x² (c > 0)
    - threshold: piecewise linear with breakpoint
    - flat: y = mean(y)
    """
    if len(x) < 3:
        return {"best_shape": "insufficient_data", "r2": {}}

    x_mean, y_mean = x.mean(), y.mean()
    ss_tot = np.sum((y - y_mean) ** 2)
    if ss_tot == 0:
        return {"best_shape": "flat", "r2": {"flat": 1.0}}

    results = {}

    # Linear fit (monotonic)
    coeffs_linear = np.polyfit(x, y, 1)
    y_pred_linear = np.polyval(coeffs_linear, x)
    ss_res_linear = np.sum((y - y_pred_linear) ** 2)
    r2_linear = 1 - ss_res_linear / ss_tot

    if coeffs_linear[0] > 0:
        results["monotonic_increasing"] = r2_linear
        results["monotonic_decreasing"] = -1.0
    else:
        results["monotonic_decreasing"] = r2_linear
        results["monotonic_increasing"] = -1.0

    # Quadratic fit (inverted-U)
    if len(x) >= 4:
        coeffs_quad = np.polyfit(x, y, 2)
        y_pred_quad = np.polyval(coeffs_quad, x)
        ss_res_quad = np.sum((y - y_pred_quad) ** 2)
        r2_quad = 1 - ss_res_quad / ss_tot
        # Inverted-U requires negative leading coefficient
        if coeffs_quad[0] < 0:
            results["inverted_u"] = r2_quad
        else:
            results["inverted_u"] = -1.0
    else:
        results["inverted_u"] = -1.0

    # Threshold: try each unique x as breakpoint, pick best
    best_r2_threshold = -1.0
    for bp in np.percentile(x, [25, 50, 75]):
        mask_left = x <= bp
        mask_right = x > bp
        if mask_left.sum() < 2 or mask_right.sum() < 2:
            continue
        y_left_mean = y[mask_left].mean()
        y_right_mean = y[mask_right].mean()
        y_pred_thresh = np.where(mask_left, y_left_mean, y_right_mean)
        ss_res_thresh = np.sum((y - y_pred_thresh) ** 2)
        r2_thresh = 1 - ss_res_thresh / ss_tot
        if r2_thresh > best_r2_threshold:
            best_r2_threshold = r2_thresh
    results["threshold"] = best_r2_threshold

    # Flat
    results["flat"] = 0.0  # R² = 0 by definition

    # Determine best shape
    best_shape = max(results, key=results.get)
    return {
        "best_shape": best_shape,
        "r2": {k: round(v, 4) for k, v in results.items()},
        "n_points": len(x),
    }


def analyze_curve(
    difficulty_values: List[float],
    utility_values: List[float],
    risk_values: List[float],
    metric_name: str,
) -> Dict[str, Any]:
    """Analyze one difficulty metric's relationship to utility and risk."""
    # Bin by difficulty
    bins = bin_values(difficulty_values)

    # Compute per-bin means
    bin_utility = defaultdict(list)
    bin_risk = defaultdict(list)
    for b, u, r in zip(bins, utility_values, risk_values):
        bin_utility[b].append(u)
        bin_risk[b].append(r)

    bin_centers = []
    bin_util_means = []
    bin_risk_means = []
    for b in sorted(bin_utility.keys()):
        bin_centers.append(b)
        bin_util_means.append(statistics.mean(bin_utility[b]))
        bin_risk_means.append(statistics.mean(bin_risk[b]))

    # Fit curves
    x = np.array(bin_centers, dtype=float)
    util_y = np.array(bin_util_means, dtype=float)
    risk_y = np.array(bin_risk_means, dtype=float)

    util_fit = fit_curve(x, util_y)
    risk_fit = fit_curve(x, risk_y)

    return {
        "metric": metric_name,
        "n_bins": len(bin_centers),
        "bin_centers": bin_centers,
        "utility_means": [round(v, 6) for v in bin_util_means],
        "risk_means": [round(v, 6) for v in bin_risk_means],
        "utility_curve_shape": util_fit["best_shape"],
        "utility_r2": util_fit["r2"],
        "risk_curve_shape": risk_fit["best_shape"],
        "risk_r2": risk_fit["r2"],
        "n_candidates": len(difficulty_values),
    }


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyze_mechanism_curve(
    manifest_path: Path,
    g3_dir: Path,
    g4_dir: Optional[Path],
    risk_path: Path,
    g6_dir: Optional[Path],
    output_dir: Path,
) -> Dict[str, Any]:
    """Full mechanism curve analysis.

    Analyzes 2 datasets (G3 test, G6 HTE) × 2+ scorers.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # Load data
    print(f"[{PHASE}] Loading manifest: {manifest_path}")
    candidates = load_manifest(manifest_path)
    print(f"[{PHASE}] Loaded {len(candidates)} candidates")

    print(f"[{PHASE}] Loading G5 risk artifacts: {risk_path}")
    risk_artifacts = load_risk_artifacts(risk_path)
    print(f"[{PHASE}] Loaded {len(risk_artifacts)} risk entries")

    # Compute positive similarity (Tanimoto to gold)
    print(f"[{PHASE}] Computing positive similarity...")
    positive_sim = compute_positive_similarity(candidates)

    # Datasets and scorers to analyze
    datasets = {}
    scorers = {}

    # Dataset 1: G3 v2 test predictions
    # G3 v2 has separate subdirectories: g3_dir_chemformer/ and g3_dir_gnn/
    # Also check g3_dir/paired_predictions/ for combined format
    g3_pred_dirs = []
    for suffix in ("_chemformer", "_gnn"):
        d = Path(str(g3_dir) + suffix) / "paired_predictions"
        if d.exists():
            g3_pred_dirs.append(d)
    if not g3_pred_dirs:
        d = g3_dir / "paired_predictions"
        if d.exists():
            g3_pred_dirs.append(d)

    if g3_pred_dirs:
        datasets["g3_v2_test"] = candidates
        total_g3_preds = 0
        for g3_pred_dir in g3_pred_dirs:
            g3_preds = load_predictions(g3_pred_dir)
            total_g3_preds += len(g3_preds)
            for key in g3_preds:
                if key.startswith("chemformer"):
                    scorers.setdefault("chemformer", {})["g3_v2_test"] = g3_preds[key]
                elif key.startswith("gnn"):
                    scorers.setdefault("gnn", {})["g3_v2_test"] = g3_preds[key]
        print(f"[{PHASE}] G3 v2: {total_g3_preds} prediction files, "
              f"scorers={list(scorers.keys())}")

    # Dataset 2: G6 HTE (if available)
    if g6_dir and g6_dir.exists():
        g6_pred_dir = g6_dir / "raw_predictions"
        if g6_pred_dir.exists():
            g6_preds = load_predictions(g6_pred_dir)
            if g6_preds:
                datasets["g6_hte"] = candidates
                for key in g6_preds:
                    if "risk_aware" in key:
                        scorers.setdefault("risk_aware_pc_cng", {})["g6_hte"] = g6_preds[key]
                print(f"[{PHASE}] G6 HTE: {len(g6_preds)} prediction files")

    # Dataset 3: G4 v2 MLP predictions (if available)
    if g4_dir and g4_dir.exists():
        g4_pred_dir = g4_dir / "raw_predictions"
        if g4_pred_dir.exists():
            g4_preds = load_predictions(g4_pred_dir)
            if g4_preds:
                datasets["g4_v2_test"] = candidates
                for key in g4_preds:
                    if key.startswith("morgan_mlp"):
                        scorers.setdefault("morgan_mlp", {})["g4_v2_test"] = g4_preds[key]
                print(f"[{PHASE}] G4 v2: {len(g4_preds)} prediction files")

    # Run analysis for each dataset × scorer
    all_curves = []
    per_candidate_rows = []

    for scorer_name, dataset_preds in scorers.items():
        for dataset_name, preds_list in dataset_preds.items():
            print(f"\n[{PHASE}] Analyzing {scorer_name} × {dataset_name}...")

            # Compute scoring metrics
            scoring_margin = compute_scoring_margin(preds_list)

            # Get all seed predictions for ensemble uncertainty
            all_scorer_preds = {}
            if dataset_name.startswith("g3"):
                for d in g3_pred_dirs:
                    all_scorer_preds.update(load_predictions(d))
            elif dataset_name.startswith("g4") and g4_dir:
                g4_pd = g4_dir / "raw_predictions"
                if g4_pd.exists():
                    all_scorer_preds = load_predictions(g4_pd)
            elif g6_dir:
                g6_pd = g6_dir / "raw_predictions"
                if g6_pd.exists():
                    all_scorer_preds = load_predictions(g6_pd)

            ensemble_unc = compute_ensemble_uncertainty(all_scorer_preds)
            downstream_loss = compute_downstream_loss(preds_list, candidates)

            # Build per-candidate metrics
            for pred in preds_list:
                cid = pred["candidate_id"]
                cand = candidates.get(cid, {})
                risk = risk_artifacts.get(cid, {})

                row = {
                    "candidate_id": cid,
                    "dataset": dataset_name,
                    "scorer": scorer_name,
                    "candidate_source": cand.get("candidate_source", ""),
                    "reaction_family": cand.get("reaction_family", ""),
                    "edit_distance": cand.get("edit_distance", 0),
                    "known_positive_collision": int(cand.get("known_positive_collision", False)),
                    "nearest_train_similarity": cand.get("nearest_train_similarity", 0.0),
                    "positive_similarity": positive_sim.get(cid, 0.0),
                    "score": pred["score"],
                    "label": pred["label"],
                    "scoring_margin": scoring_margin.get(cid, 0.0),
                    "ensemble_uncertainty": ensemble_unc.get(cid, 0.0),
                    "false_negative_risk": risk.get("features", {}).get("false_negative_risk", 0.5),
                    "downstream_loss": downstream_loss.get(cid, 1.0),
                }
                per_candidate_rows.append(row)

            # Analyze each difficulty metric
            for metric in DIFFICULTY_METRICS:
                values = [r[metric] for r in per_candidate_rows
                          if r["dataset"] == dataset_name and r["scorer"] == scorer_name]
                utilities = [1.0 - r["downstream_loss"] for r in per_candidate_rows
                             if r["dataset"] == dataset_name and r["scorer"] == scorer_name]
                risks = [r["false_negative_risk"] for r in per_candidate_rows
                         if r["dataset"] == dataset_name and r["scorer"] == scorer_name]

                if len(values) < 10:
                    continue

                curve = analyze_curve(values, utilities, risks, metric)
                curve["dataset"] = dataset_name
                curve["scorer"] = scorer_name
                all_curves.append(curve)

    # Write per-candidate metrics CSV
    if per_candidate_rows:
        import csv
        csv_path = output_dir / "per_candidate_metrics.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=per_candidate_rows[0].keys())
            writer.writeheader()
            writer.writerows(per_candidate_rows)
        print(f"[{PHASE}] Wrote {len(per_candidate_rows)} rows to {csv_path}")

    # Write curve specs
    curve_specs = {
        "phase": PHASE,
        "n_bins": N_BINS,
        "difficulty_metrics": DIFFICULTY_METRICS,
        "curve_shapes_tested": CURVE_SHAPES,
        "curves": all_curves,
    }
    with open(output_dir / "curve_specs.json", "w") as f:
        json.dump(curve_specs, f, indent=2)

    # Write utility and risk curves separately
    utility_curves = [c for c in all_curves if c["metric"] != "false_negative_risk"]
    risk_curves = [c for c in all_curves if c["metric"] == "false_negative_risk"
                   or c["metric"] == "database_collision"]
    with open(output_dir / "utility_curve.json", "w") as f:
        json.dump(utility_curves, f, indent=2)
    with open(output_dir / "risk_curve.json", "w") as f:
        json.dump(risk_curves, f, indent=2)

    # Shape comparison
    import csv as csv_mod
    with open(output_dir / "shape_comparison.csv", "w", newline="") as f:
        writer = csv_mod.DictWriter(f, fieldnames=[
            "dataset", "scorer", "metric", "utility_shape", "utility_best_r2",
            "risk_shape", "risk_best_r2", "n_candidates"])
        writer.writeheader()
        for c in all_curves:
            util_r2 = max(c["utility_r2"].values()) if c["utility_r2"] else 0
            risk_r2 = max(c["risk_r2"].values()) if c["risk_r2"] else 0
            writer.writerow({
                "dataset": c.get("dataset", ""),
                "scorer": c.get("scorer", ""),
                "metric": c["metric"],
                "utility_shape": c["utility_curve_shape"],
                "utility_best_r2": round(util_r2, 4),
                "risk_shape": c["risk_curve_shape"],
                "risk_best_r2": round(risk_r2, 4),
                "n_candidates": c["n_candidates"],
            })

    # Compute verdict
    verdict = compute_verdict(all_curves)

    go_no_go = {
        "phase": PHASE,
        "status": verdict["verdict"],
        "primary_metric": {"name": "curve_shape_reproducibility"},
        "predeclared_threshold": {
            "go": "Relationship reproduced in >=2 datasets AND >=2 scorers consistent",
            "partial_go": "Relationship found but not fully reproduced",
            "no_go": "No consistent relationship found",
        },
        "verdict_details": verdict,
        "evidence_paths": [
            str(output_dir / "per_candidate_metrics.csv"),
            str(output_dir / "curve_specs.json"),
            str(output_dir / "utility_curve.json"),
            str(output_dir / "risk_curve.json"),
            str(output_dir / "shape_comparison.csv"),
        ],
        "next_phase_allowed": verdict["next_phase_allowed"],
    }
    with open(output_dir / "go_no_go.json", "w") as f:
        json.dump(go_no_go, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n[{PHASE}] Complete ({elapsed:.1f}s)")
    print(f"[{PHASE}] Verdict: {verdict['verdict']}")
    print(f"[{PHASE}] next_phase_allowed: {verdict['next_phase_allowed']}")
    return go_no_go


def compute_verdict(curves: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Determine GO/PARTIAL_GO/NO_GO based on curve reproducibility."""
    if not curves:
        return {"verdict": "NO_GO", "reason": "No curves computed",
                "next_phase_allowed": False}

    # Group by metric
    by_metric = defaultdict(list)
    for c in curves:
        by_metric[c["metric"]].append(c)

    # Check reproducibility: same shape in >=2 datasets AND >=2 scorers
    reproduced_metrics = []
    for metric, metric_curves in by_metric.items():
        datasets = set(c.get("dataset", "") for c in metric_curves)
        scorers = set(c.get("scorer", "") for c in metric_curves)
        shapes = [c["utility_curve_shape"] for c in metric_curves]

        # Check if shape is consistent across datasets and scorers
        if len(datasets) >= 2 and len(scorers) >= 2:
            shape_counts = defaultdict(int)
            for s in shapes:
                if s != "insufficient_data" and s != "flat":
                    shape_counts[s] += 1
            if shape_counts:
                best_shape = max(shape_counts, key=shape_counts.get)
                if shape_counts[best_shape] >= 2:
                    reproduced_metrics.append({
                        "metric": metric,
                        "shape": best_shape,
                        "n_datasets": len(datasets),
                        "n_scorers": len(scorers),
                        "n_consistent": shape_counts[best_shape],
                    })

    n_reproduced = len(reproduced_metrics)

    if n_reproduced >= 2:
        verdict = "GO"
        reason = (f"{n_reproduced} difficulty metrics reproduced across "
                  f">=2 datasets and >=2 scorers with consistent curve shapes")
    elif n_reproduced >= 1:
        verdict = "PARTIAL_GO"
        reason = (f"{n_reproduced} difficulty metric reproduced across "
                  f">=2 datasets and >=2 scorers; "
                  f"some metrics are dataset/scorer-specific")
    else:
        # Check if any relationship exists at all (even if not reproduced)
        any_non_flat = any(
            c["utility_curve_shape"] not in ("flat", "insufficient_data")
            for c in curves
        )
        if any_non_flat:
            verdict = "PARTIAL_GO"
            reason = ("Relationships found but not reproduced across "
                      "datasets/scorers; results are dataset/scorer-specific")
        else:
            verdict = "NO_GO"
            reason = "No consistent difficulty-utility relationship found"

    return {
        "verdict": verdict,
        "reason": reason,
        "n_reproduced_metrics": n_reproduced,
        "reproduced_metrics": reproduced_metrics,
        "next_phase_allowed": verdict in ("GO", "PARTIAL_GO"),
    }


# ---------------------------------------------------------------------------
# Contract files
# ---------------------------------------------------------------------------

def write_contract_files(output_dir: Path, manifest_path: Path,
                         g3_dir: Path, risk_path: Path):
    """Write run_manifest, environment, input_hashes, commands.log."""
    import hashlib

    def sha256(p):
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    with open(output_dir / "run_manifest.json", "w") as f:
        json.dump({
            "phase": PHASE,
            "analysis": "difficulty_benefit_risk_mechanism_curve",
            "n_bins": N_BINS,
            "difficulty_metrics": DIFFICULTY_METRICS,
            "curve_shapes_tested": CURVE_SHAPES,
            "seed": SEED,
        }, f, indent=2)

    with open(output_dir / "environment.json", "w") as f:
        env = {"python": sys.version.split()[0], "platform": platform.platform()}
        try:
            import numpy as np
            env["numpy"] = np.__version__
        except ImportError:
            pass
        try:
            import rdkit
            env["rdkit"] = rdkit.__version__
        except ImportError:
            pass
        json.dump(env, f, indent=2)

    with open(output_dir / "input_hashes.json", "w") as f:
        hashes = {}
        for p in [manifest_path, risk_path]:
            if p.exists():
                hashes[str(p)] = sha256(p)
        json.dump(hashes, f, indent=2)

    with open(output_dir / "commands.log", "w") as f:
        f.write(f"python3 -m pc_cng.p4_g8a_mechanism_curve "
                f"--manifest {manifest_path} "
                f"--g3-dir {g3_dir} "
                f"--risk-path {risk_path} "
                f"--output-dir {output_dir}\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=f"{PHASE} mechanism curve analysis")
    parser.add_argument("--manifest", type=Path,
                        default=Path("data/p4/manifests/hte_feasibility_v2.json"))
    parser.add_argument("--g3-dir", type=Path,
                        default=Path("results/p4_augmentation_v2"))
    parser.add_argument("--g4-dir", type=Path,
                        default=Path("results/p4_generator_scorer_matrix_v2"))
    parser.add_argument("--risk-path", type=Path,
                        default=Path("results/p4_risk_aware/risk_artifacts.json"))
    parser.add_argument("--g6-dir", type=Path,
                        default=Path("results/p4_hte_external_validation"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/p4_mechanism_curve"))
    args = parser.parse_args()

    result = analyze_mechanism_curve(
        args.manifest, args.g3_dir, args.g4_dir, args.risk_path,
        args.g6_dir, args.output_dir)
    write_contract_files(args.output_dir, args.manifest, args.g3_dir, args.risk_path)
