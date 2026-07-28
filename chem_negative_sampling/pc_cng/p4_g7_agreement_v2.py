"""Strict analysis for completed Phase-F / G7 expert pilot v2 forms."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from pc_cng.p4_g7_agreement import krippendorff_alpha, weighted_kappa
from pc_cng.p4_g7_sampling_v2 import SCHEMA, SCORING_DIMENSIONS


PRIMARY_RELIABILITY_DIMENSION = "mechanistic_plausibility"
PRIMARY_SOURCE_DIMENSION = "plausible_competing_outcome"
AGREEMENT_MIN = 0.5
AGREEMENT_TARGET = 0.6
REVIEWER_DRIFT_MAX = 0.75


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _parse_score(value: str, *, field: str, item: str) -> int:
    try:
        score = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"missing/invalid {field} for {item}") from exc
    if score < 1 or score > 5:
        raise ValueError(f"{field} must be an integer from 1 to 5 for {item}")
    return score


def _bootstrap_mean_difference(
    treatment: Sequence[float],
    baseline: Sequence[float],
    *,
    seed: int,
    n_bootstrap: int,
) -> Dict[str, Any]:
    if not treatment or not baseline:
        raise ValueError("source comparison requires non-empty samples")
    treatment_array = np.asarray(treatment, dtype=float)
    baseline_array = np.asarray(baseline, dtype=float)
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_bootstrap, dtype=float)
    for index in range(n_bootstrap):
        t_sample = rng.choice(treatment_array, size=len(treatment_array), replace=True)
        b_sample = rng.choice(baseline_array, size=len(baseline_array), replace=True)
        deltas[index] = float(t_sample.mean() - b_sample.mean())
    point = float(treatment_array.mean() - baseline_array.mean())
    low, high = np.quantile(deltas, [0.025, 0.975])
    return {
        "delta": point,
        "ci_low": float(low),
        "ci_high": float(high),
        "ci_all_positive": bool(low > 0),
        "n_treatment": len(treatment),
        "n_baseline": len(baseline),
    }


def _load_complete_forms(
    form_paths: Sequence[Path],
    *,
    expected_ids: set,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    if len(form_paths) < 3:
        raise ValueError("G7 v2 analysis requires at least three completed forms")
    by_reviewer: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for path in form_paths:
        reviewer = path.stem
        rows = _read_csv(path)
        if len(rows) != len(expected_ids):
            raise ValueError(f"{reviewer} form has {len(rows)} rows, expected {len(expected_ids)}")
        ids = [row.get("blinded_id", "") for row in rows]
        if len(ids) != len(set(ids)) or set(ids) != expected_ids:
            raise ValueError(f"{reviewer} form does not match the frozen item set")
        parsed: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            blinded_id = row["blinded_id"]
            parsed[blinded_id] = {
                dimension: _parse_score(
                    row.get(dimension, ""),
                    field=dimension,
                    item=f"{reviewer}/{blinded_id}",
                )
                for dimension in SCORING_DIMENSIONS
            }
            parsed[blinded_id]["reason_code"] = row.get("reason_code", "")
            parsed[blinded_id]["notes"] = row.get("notes", "")
        by_reviewer[reviewer] = parsed
    return by_reviewer


def analyze_completed_forms(
    pilot_dir: Path,
    form_paths: Sequence[Path],
    *,
    n_bootstrap: int = 5000,
    seed: int = 20260729,
) -> Dict[str, Any]:
    pilot_dir = pilot_dir.resolve()
    manifest = json.loads(
        (pilot_dir / "pilot_manifest.json").read_text(encoding="utf-8")
    )
    key = json.loads(
        (pilot_dir / "sampling_key_unblinded.json").read_text(encoding="utf-8")
    )
    if manifest.get("schema") != SCHEMA or key.get("schema") != SCHEMA:
        raise ValueError("pilot artifacts do not use the G7 v2 schema")
    items = {item["blinded_id"]: item for item in key["items"]}
    if len(items) != manifest.get("n_items"):
        raise ValueError("sampling key count does not match pilot manifest")
    forms = _load_complete_forms(form_paths, expected_ids=set(items))
    reviewers = sorted(forms)
    item_ids = sorted(items)

    reliability: Dict[str, Any] = {}
    for dimension in SCORING_DIMENSIONS:
        pairwise = {}
        for left, right in combinations(reviewers, 2):
            left_scores = [forms[left][item][dimension] for item in item_ids]
            right_scores = [forms[right][item][dimension] for item in item_ids]
            pairwise[f"{left}__{right}"] = weighted_kappa(
                left_scores,
                right_scores,
            )
        alpha_matrix = [
            [forms[reviewer][item][dimension] for reviewer in reviewers]
            for item in item_ids
        ]
        reliability[dimension] = {
            "pairwise_weighted_kappa": pairwise,
            "minimum_pairwise_weighted_kappa": min(pairwise.values()),
            "mean_pairwise_weighted_kappa": float(np.mean(list(pairwise.values()))),
            "krippendorff_alpha_ordinal": krippendorff_alpha(
                alpha_matrix,
                level="ordinal",
            ),
        }

    item_mean: Dict[str, Dict[str, float]] = {}
    for item in item_ids:
        item_mean[item] = {
            dimension: float(
                np.mean([forms[reviewer][item][dimension] for reviewer in reviewers])
            )
            for dimension in SCORING_DIMENSIONS
        }

    positive_ids = [
        item for item, info in items.items() if info["stratum"] == "positive_control"
    ]
    negative_ids = [
        item
        for item, info in items.items()
        if info["stratum"] == "obvious_negative_control"
    ]
    controls = {}
    for dimension in ("structural_validity", "mechanistic_plausibility"):
        controls[dimension] = _bootstrap_mean_difference(
            [item_mean[item][dimension] for item in positive_ids],
            [item_mean[item][dimension] for item in negative_ids],
            seed=seed + len(controls),
            n_bootstrap=n_bootstrap,
        )

    by_stratum: Dict[str, List[float]] = defaultdict(list)
    for item, info in items.items():
        by_stratum[info["stratum"]].append(
            item_mean[item][PRIMARY_SOURCE_DIMENSION]
        )
    single_strata = (
        "random_mismatch",
        "rule_pc_cng",
        "learned_structured",
        "shuffled_real",
    )
    single_means = {
        stratum: float(np.mean(by_stratum[stratum]))
        for stratum in single_strata
    }
    best_single = max(single_means, key=single_means.get)
    source_comparisons = {
        "gate_vs_uniform_union": _bootstrap_mean_difference(
            by_stratum["learned_source_gate"],
            by_stratum["uniform_union"],
            seed=seed + 101,
            n_bootstrap=n_bootstrap,
        ),
        "gate_vs_expert_best_single": {
            "baseline": best_single,
            **_bootstrap_mean_difference(
                by_stratum["learned_source_gate"],
                by_stratum[best_single],
                seed=seed + 102,
                n_bootstrap=n_bootstrap,
            ),
        },
    }

    reviewer_means = {
        reviewer: float(
            np.mean(
                [
                    forms[reviewer][item][PRIMARY_RELIABILITY_DIMENSION]
                    for item in item_ids
                ]
            )
        )
        for reviewer in reviewers
    }
    reviewer_spread = max(reviewer_means.values()) - min(reviewer_means.values())
    primary_reliability = reliability[PRIMARY_RELIABILITY_DIMENSION]
    agreement_value = max(
        primary_reliability["krippendorff_alpha_ordinal"],
        primary_reliability["minimum_pairwise_weighted_kappa"],
    )
    controls_pass = all(
        result["ci_all_positive"] for result in controls.values()
    )
    pilot_reliability_pass = agreement_value >= AGREEMENT_MIN
    reviewer_drift_pass = reviewer_spread <= REVIEWER_DRIFT_MAX
    source_quality_pass = all(
        comparison["ci_all_positive"]
        for comparison in source_comparisons.values()
    )
    pilot_exit = (
        pilot_reliability_pass
        and controls_pass
        and reviewer_drift_pass
        and source_quality_pass
    )

    return {
        "schema": "p4_g7_expert_agreement_v2",
        "n_items": len(item_ids),
        "n_reviewers": len(reviewers),
        "reviewers": reviewers,
        "reliability": reliability,
        "control_discrimination": controls,
        "source_stratum_means": {
            stratum: float(np.mean(scores))
            for stratum, scores in by_stratum.items()
        },
        "source_comparisons": source_comparisons,
        "reviewer_effect": {
            "means": reviewer_means,
            "max_mean_spread": reviewer_spread,
            "threshold": REVIEWER_DRIFT_MAX,
            "pass": reviewer_drift_pass,
        },
        "verdict": {
            "status": "PILOT_GO_MAIN_REVIEW_ALLOWED" if pilot_exit else "PILOT_NO_GO",
            "pilot_exit_met": pilot_exit,
            "agreement_minimum": AGREEMENT_MIN,
            "agreement_target": AGREEMENT_TARGET,
            "primary_agreement_value": agreement_value,
            "reliability_pass": pilot_reliability_pass,
            "controls_pass": controls_pass,
            "reviewer_drift_pass": reviewer_drift_pass,
            "source_quality_pass": source_quality_pass,
            "main_review_allowed": pilot_exit,
        },
        "limitations": [
            "Expert source comparisons use different matched parents by design and are not paired within parent.",
            "Pilot evidence does not replace a prospective chemistry experiment.",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-dir", type=Path, required=True)
    parser.add_argument("--completed-form", action="append", type=Path, required=True)
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    report = analyze_completed_forms(
        args.pilot_dir,
        args.completed_form,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
    output = args.output or args.pilot_dir / "agreement_report_v2.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["verdict"], indent=2))
    return 0 if report["verdict"]["pilot_exit_met"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
