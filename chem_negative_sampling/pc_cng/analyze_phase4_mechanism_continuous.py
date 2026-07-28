"""Continuous, uncertainty-aware reanalysis of Phase 4 mechanism evidence.

This module replaces the arbitrary-bin curve fitting in the historical G8-A
analysis.  It deliberately does *not* reinterpret candidate-level ranking
scores as downstream utility.  Instead, it estimates an observational
``negative hardness`` curve: the within-reaction percentile rank of a
negative candidate's score, where larger values mean that the scorer ranked
the negative more like the positive product.

The curve is a cubic regression spline with reaction-group bootstrap
confidence bands.  Controlled downstream utility remains the responsibility
of the frozen easy/semi-hard/hard intervention.  This separation prevents an
observational score curve from being promoted into a causal utility claim.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


SEED = 20260729
CONTINUOUS_FEATURES = (
    "positive_similarity",
    "nearest_train_similarity",
    "scoring_margin",
    "ensemble_uncertainty",
    "false_negative_risk",
    "edit_distance",
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _as_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def load_candidate_groups(manifest_path: Path) -> Dict[str, str]:
    data = json.loads(manifest_path.read_text())
    out: Dict[str, str] = {}
    for group in data.get("groups", []):
        group_id = str(group.get("group_id", ""))
        for candidate in group.get("candidates", []):
            candidate_id = str(candidate.get("candidate_id", ""))
            if candidate_id:
                out[candidate_id] = group_id
    return out


def load_candidate_rows(
    metrics_path: Path,
    candidate_groups: Mapping[str, str],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with metrics_path.open(newline="") as fh:
        for raw in csv.DictReader(fh):
            candidate_id = str(raw.get("candidate_id", ""))
            group_id = candidate_groups.get(candidate_id)
            score = _as_float(raw.get("score"))
            label = _as_float(raw.get("label"))
            if not group_id or score is None or label is None:
                continue
            row: Dict[str, Any] = dict(raw)
            row["group_id"] = group_id
            row["score"] = score
            row["label"] = int(label)
            for feature in CONTINUOUS_FEATURES:
                row[feature] = _as_float(raw.get(feature))
            row["known_positive_collision"] = int(
                _as_float(raw.get("known_positive_collision")) or 0
            )
            rows.append(row)
    return rows


def attach_within_group_hardness(rows: Sequence[Dict[str, Any]]) -> None:
    """Attach a tie-aware score percentile within each reaction group."""
    grouped: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row["dataset"]),
                str(row["scorer"]),
                str(row["group_id"]),
            )
        ].append(row)

    for group_rows in grouped.values():
        scores = np.asarray([float(r["score"]) for r in group_rows])
        n = len(scores)
        for row in group_rows:
            score = float(row["score"])
            lower = float(np.sum(scores < score))
            equal = float(np.sum(scores == score))
            row["negative_hardness"] = (lower + 0.5 * equal) / n


def _spline_basis(
    x: np.ndarray,
    x_min: float,
    x_max: float,
    knots: np.ndarray,
) -> np.ndarray:
    scale = max(x_max - x_min, 1e-12)
    z = np.clip((x - x_min) / scale, 0.0, 1.0)
    columns = [np.ones_like(z), z, z**2, z**3]
    columns.extend(np.maximum(z - knot, 0.0) ** 3 for knot in knots)
    return np.column_stack(columns)


def _fit_spline(
    x: np.ndarray,
    y: np.ndarray,
    grid: np.ndarray,
    x_min: float,
    x_max: float,
    knots: np.ndarray,
    ridge: float = 1e-5,
) -> np.ndarray:
    design = _spline_basis(x, x_min, x_max, knots)
    grid_design = _spline_basis(grid, x_min, x_max, knots)
    penalty = np.eye(design.shape[1]) * ridge
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    return np.clip(grid_design @ beta, 0.0, 1.0)


def fit_cluster_bootstrap_spline(
    rows: Sequence[Dict[str, Any]],
    feature: str,
    n_bootstrap: int,
    seed: int,
) -> Optional[Dict[str, Any]]:
    usable = [
        row
        for row in rows
        if row["label"] == 0
        and row.get(feature) is not None
        and not (feature == "edit_distance" and float(row[feature]) < 0)
    ]
    if len(usable) < 40:
        return None
    x = np.asarray([float(row[feature]) for row in usable])
    y = np.asarray([float(row["negative_hardness"]) for row in usable])
    unique = np.unique(x)
    if len(unique) < 8:
        return {
            "feature": feature,
            "status": "UNAVAILABLE_INSUFFICIENT_VARIATION",
            "n": len(usable),
            "n_unique": int(len(unique)),
        }

    x_min, x_max = (float(v) for v in np.quantile(x, [0.05, 0.95]))
    if x_max <= x_min:
        return None
    grid = np.linspace(x_min, x_max, 51)
    raw_knots = np.quantile(x, [0.2, 0.4, 0.6, 0.8])
    knots = np.unique((raw_knots - x_min) / (x_max - x_min))
    knots = knots[(knots > 0.0) & (knots < 1.0)]
    point = _fit_spline(x, y, grid, x_min, x_max, knots)

    by_cluster: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in usable:
        by_cluster[str(row["group_id"])].append(row)
    clusters = sorted(by_cluster)
    if len(clusters) < 8:
        return None

    rng = np.random.default_rng(seed)
    bootstrap_predictions: List[np.ndarray] = []
    for _ in range(n_bootstrap):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        sample_rows = [r for cluster in sampled for r in by_cluster[cluster]]
        bx = np.asarray([float(row[feature]) for row in sample_rows])
        by = np.asarray([float(row["negative_hardness"]) for row in sample_rows])
        try:
            bootstrap_predictions.append(
                _fit_spline(bx, by, grid, x_min, x_max, knots)
            )
        except np.linalg.LinAlgError:
            continue

    if len(bootstrap_predictions) < max(50, n_bootstrap // 2):
        return None
    boot = np.vstack(bootstrap_predictions)
    low = np.quantile(boot, 0.025, axis=0)
    high = np.quantile(boot, 0.975, axis=0)
    peak_index = int(np.argmax(point))
    peak_fraction = peak_index / (len(grid) - 1)
    strong_inverted_u = bool(
        0.2 <= peak_fraction <= 0.8
        and low[peak_index] > max(high[0], high[-1])
    )
    end_delta = float(point[-1] - point[0])
    return {
        "feature": feature,
        "status": "ESTIMATED",
        "outcome": "within_reaction_negative_score_percentile",
        "interpretation": "higher means the scorer ranks a negative more like a positive",
        "n": len(usable),
        "n_clusters": len(clusters),
        "n_unique": int(len(unique)),
        "grid": grid.tolist(),
        "estimate": point.tolist(),
        "ci_low": low.tolist(),
        "ci_high": high.tolist(),
        "knots_normalized": knots.tolist(),
        "end_delta": end_delta,
        "peak_x": float(grid[peak_index]),
        "strong_inverted_u": strong_inverted_u,
        "n_bootstrap_success": int(len(bootstrap_predictions)),
    }


def _bootstrap_mean(
    rows: Sequence[Dict[str, Any]],
    seed: int,
    n_bootstrap: int,
) -> Tuple[float, float, float]:
    by_cluster: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        by_cluster[str(row["group_id"])].append(
            float(row["negative_hardness"])
        )
    clusters = sorted(by_cluster)
    point = float(np.mean([v for values in by_cluster.values() for v in values]))
    if len(clusters) < 5:
        return point, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_bootstrap):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        values = [v for cluster in sampled for v in by_cluster[cluster]]
        draws.append(float(np.mean(values)))
    return point, float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def categorical_driver_summary(
    rows: Sequence[Dict[str, Any]],
    field: str,
    n_bootstrap: int,
    seed: int,
) -> List[Dict[str, Any]]:
    negatives = [row for row in rows if row["label"] == 0]
    by_value: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in negatives:
        by_value[str(row.get(field, "unknown"))].append(row)
    out = []
    for index, (value, subset) in enumerate(sorted(by_value.items())):
        if len(subset) < 10:
            continue
        mean, low, high = _bootstrap_mean(
            subset, seed=seed + index, n_bootstrap=n_bootstrap
        )
        out.append(
            {
                "field": field,
                "value": value,
                "n": len(subset),
                "n_clusters": len({str(row["group_id"]) for row in subset}),
                "mean_negative_hardness": mean,
                "ci_low": low,
                "ci_high": high,
            }
        )
    return out


def _read_json(path: Optional[Path]) -> Dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text())


def _phase4_matching_audit(per_scenario: Mapping[str, Any]) -> Dict[str, Any]:
    audits = [v.get("audit", {}) for v in per_scenario.values()]
    validity = []
    candidate_budget = []
    family_equal = []
    source_equal = []
    for audit in audits:
        pools = audit.get("pools", {})
        if not all(name in pools for name in ("easy", "semi_hard", "hard")):
            continue
        validity.extend(
            float(pools[name].get("validity_rate", 0.0))
            for name in ("easy", "semi_hard", "hard")
        )
        candidate_budget.append(
            len(
                {
                    int(pools[name].get("candidate_count_per_positive_per_source", -1))
                    for name in ("easy", "semi_hard", "hard")
                }
            )
            == 1
        )
        family_equal.append(
            len(
                {
                    json.dumps(
                        pools[name].get("family_distribution", {}),
                        sort_keys=True,
                    )
                    for name in ("easy", "semi_hard", "hard")
                }
            )
            == 1
        )
        source_equal.append(
            len(
                {
                    json.dumps(
                        pools[name].get("source_composition", {}),
                        sort_keys=True,
                    )
                    for name in ("easy", "semi_hard", "hard")
                }
            )
            == 1
        )
    return {
        "validity_matched": bool(validity) and min(validity) == 1.0,
        "candidate_count_per_positive_per_source_matched": bool(candidate_budget)
        and all(candidate_budget),
        "family_distribution_exactly_matched": bool(family_equal)
        and all(family_equal),
        "source_distribution_exactly_matched": bool(source_equal)
        and all(source_equal),
        "similarity": "frozen stratification variable, intentionally different",
        "edit_count": "NOT_MEASURED; historical code used 1-sim as a proxy",
        "scorer_margin": "NOT_FROZEN_OR_MATCHED_BEFORE_POOL_CONSTRUCTION",
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidate_groups = load_candidate_groups(args.manifest)
    rows = load_candidate_rows(args.per_candidate_metrics, candidate_groups)
    attach_within_group_hardness(rows)

    cells: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        cells[(str(row["dataset"]), str(row["scorer"]))].append(row)

    curves: List[Dict[str, Any]] = []
    drivers: Dict[str, Any] = {}
    for cell_index, ((dataset, scorer), cell_rows) in enumerate(sorted(cells.items())):
        cell_key = f"{dataset}::{scorer}"
        drivers[cell_key] = {
            "candidate_source": categorical_driver_summary(
                cell_rows,
                "candidate_source",
                args.n_bootstrap,
                args.seed + 1000 + cell_index,
            ),
            "reaction_family": categorical_driver_summary(
                cell_rows,
                "reaction_family",
                args.n_bootstrap,
                args.seed + 2000 + cell_index,
            ),
            "known_positive_collision": categorical_driver_summary(
                cell_rows,
                "known_positive_collision",
                args.n_bootstrap,
                args.seed + 3000 + cell_index,
            ),
        }
        for feature_index, feature in enumerate(CONTINUOUS_FEATURES):
            result = fit_cluster_bootstrap_spline(
                cell_rows,
                feature,
                n_bootstrap=args.n_bootstrap,
                seed=args.seed + cell_index * 100 + feature_index,
            )
            if result is None:
                result = {
                    "feature": feature,
                    "status": "UNAVAILABLE_INSUFFICIENT_DATA",
                }
            result["dataset"] = dataset
            result["scorer"] = scorer
            curves.append(result)

    phase4_agg = _read_json(args.phase4_aggregation)
    second_agg = _read_json(args.second_scorer_aggregation)
    g8b = _read_json(args.g8b_transfer)
    per_scenario = _read_json(args.phase4_per_scenario)
    matching = _phase4_matching_audit(per_scenario)

    h3_primary = (
        phase4_agg.get("verdict", {}).get("H3_inverted_u", {})
    )
    h3_second = (
        second_agg.get("verdict", {}).get("H3_inverted_u", {})
    )
    estimated = [c for c in curves if c.get("status") == "ESTIMATED"]
    unavailable = [c for c in curves if c.get("status", "").startswith("UNAVAILABLE")]
    strong_by_feature: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for curve in estimated:
        if curve.get("strong_inverted_u"):
            strong_by_feature[str(curve["feature"])].append(curve)
    replicated_strong_features = []
    for feature, feature_curves in strong_by_feature.items():
        datasets = {str(c["dataset"]) for c in feature_curves}
        scorers = {str(c["scorer"]) for c in feature_curves}
        if len(datasets) >= 2 and len(scorers) >= 2:
            replicated_strong_features.append(feature)

    status = {
        "schema_version": "phase4_mechanism_continuous_v1",
        "status": "EXPLORATORY_ONLY_EXIT_NOT_MET",
        "reason": (
            "Continuous observational associations and uncertainty bands were "
            "estimated, but the controlled pools were not exactly matched on "
            "family/source/edit-count/scorer-margin and no sealed held-out "
            "confirmatory replication exists. The semi-hard inverted-U is not "
            "general, and G8-B reports no transfer direction with CI > 0."
        ),
        "historical_g8a_status": "DEPRECATED_EXPLORATORY_BINNED_ANALYSIS",
        "observational_association": {
            "status": "AVAILABLE_EXPLORATORY",
            "outcome": "within-reaction negative score percentile, not downstream utility",
            "n_curve_cells_estimated": len(estimated),
            "n_curve_cells_unavailable": len(unavailable),
            "strong_inverted_u_cells": sum(
                bool(c.get("strong_inverted_u")) for c in estimated
            ),
            "replicated_strong_inverted_u_features": replicated_strong_features,
        },
        "controlled_intervention": {
            "status": "UNSUPPORTED_GENERAL_MECHANISM",
            "primary_scorer_h3_scenarios": h3_primary.get("n_sota"),
            "primary_scorer_n_scenarios": h3_primary.get("n_scenarios"),
            "second_scorer_h3_scenarios": h3_second.get("n_sota"),
            "second_scorer_n_scenarios": h3_second.get("n_scenarios"),
            "matching_audit": matching,
        },
        "held_out_replication": {
            "status": "NOT_MET",
            "note": (
                "The second scorer used the same development pools. It is an "
                "exploratory robustness check, not sealed confirmation."
            ),
        },
        "uncertainty_interval": {
            "status": "MET_FOR_OBSERVATIONAL_CURVES",
            "method": "reaction-group cluster bootstrap 95% bands",
            "n_bootstrap": args.n_bootstrap,
        },
        "competing_explanation_ablation": {
            "status": "PARTIAL",
            "covered": [
                "candidate source",
                "reaction family",
                "randomized-label null",
                "formula-preserved versus formula-changed shuffled transfer",
            ],
            "missing_or_unusable": [
                "candidate-level FNR (constant 0.5 in historical table)",
                "known-positive collision (too sparse for a two-level estimate)",
                "reaction-centre locality",
                "true edit count",
                "edit type in the analysis table",
                "family support density",
            ],
        },
        "g8b_transfer": {
            "status": "NO_GO_NEGATIVE_TRANSFER_RETAINED",
            "directions_with_any_positive_ci": g8b.get(
                "directions_with_any_positive_ci"
            ),
            "total_directions": g8b.get("total_directions"),
        },
        "claim_policy": (
            "Do not claim a causal semi-hard boundary mechanism. Report the "
            "continuous curves, null checks, source/family heterogeneity and "
            "negative transfer as exploratory evidence."
        ),
    }

    (args.output_dir / "continuous_curves.json").write_text(
        json.dumps(curves, indent=2)
    )
    (args.output_dir / "driver_effects.json").write_text(
        json.dumps(drivers, indent=2)
    )
    (args.output_dir / "mechanism_status.json").write_text(
        json.dumps(status, indent=2)
    )
    summary_rows = []
    for curve in curves:
        summary_rows.append(
            {
                "dataset": curve.get("dataset"),
                "scorer": curve.get("scorer"),
                "feature": curve.get("feature"),
                "status": curve.get("status"),
                "n": curve.get("n"),
                "n_clusters": curve.get("n_clusters"),
                "end_delta": curve.get("end_delta"),
                "peak_x": curve.get("peak_x"),
                "strong_inverted_u": curve.get("strong_inverted_u"),
            }
        )
    with (args.output_dir / "curve_summary.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    inputs = {
        str(path): _sha256(path)
        for path in (
            args.per_candidate_metrics,
            args.manifest,
            args.phase4_aggregation,
            args.phase4_per_scenario,
            args.second_scorer_aggregation,
            args.g8b_transfer,
        )
        if path is not None and path.exists()
    }
    (args.output_dir / "input_hashes.json").write_text(
        json.dumps(inputs, indent=2)
    )
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "analysis": "phase4_continuous_mechanism_reanalysis",
                "seed": args.seed,
                "n_bootstrap": args.n_bootstrap,
                "n_input_rows": len(rows),
                "n_candidate_groups": len(candidate_groups),
                "curve_model": "cubic regression spline with ridge stabilization",
                "ci": "reaction-group cluster bootstrap percentile 95%",
                "outcome": "within-reaction negative score percentile",
            },
            indent=2,
        )
    )
    return status


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-candidate-metrics", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--phase4-aggregation", type=Path, required=True)
    parser.add_argument("--phase4-per-scenario", type=Path, required=True)
    parser.add_argument("--second-scorer-aggregation", type=Path)
    parser.add_argument("--g8b-transfer", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-bootstrap", type=int, default=500)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    status = run(args)
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
