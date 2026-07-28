"""Phase G analysis of source-policy heterogeneity and complementarity.

This analysis is descriptive on development runs.  It does not upgrade
source selection to a causal mechanism claim.  Confirmatory status requires a
sealed external result whose manifest passed the Phase-E contract.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd


EPS = 1e-12


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _entropy(probabilities: Sequence[float]) -> float:
    values = np.asarray(probabilities, dtype=float)
    values = values[values > 0]
    if values.size == 0:
        return 0.0
    return float(-np.sum(values * np.log(values)))


def _normalised_counts(counts: Mapping[str, int], names: Sequence[str]) -> np.ndarray:
    values = np.asarray([counts.get(name, 0) for name in names], dtype=float)
    total = float(values.sum())
    if total <= 0:
        return np.zeros(len(names), dtype=float)
    return values / total


def _js_divergence(a: np.ndarray, b: np.ndarray) -> float:
    midpoint = 0.5 * (a + b)

    def _kl(x, y):
        mask = x > 0
        return float(np.sum(x[mask] * np.log((x[mask] + EPS) / (y[mask] + EPS))))

    return 0.5 * _kl(a, midpoint) + 0.5 * _kl(b, midpoint)


def _mutual_information(
    rows: Sequence[Mapping[str, str]],
    *,
    source_names: Sequence[str],
) -> Dict[str, float]:
    if not rows:
        return {"mutual_information": 0.0, "normalised_mutual_information": 0.0}
    families = sorted({str(row["reaction_family"]) for row in rows})
    family_index = {name: index for index, name in enumerate(families)}
    source_index = {name: index for index, name in enumerate(source_names)}
    joint = np.zeros((len(families), len(source_names)), dtype=float)
    for row in rows:
        source = str(row["selected_source"])
        if source in source_index:
            joint[family_index[str(row["reaction_family"])], source_index[source]] += 1
    joint /= max(float(joint.sum()), 1.0)
    p_family = joint.sum(axis=1, keepdims=True)
    p_source = joint.sum(axis=0, keepdims=True)
    expected = p_family @ p_source
    mask = joint > 0
    mi = float(np.sum(joint[mask] * np.log((joint[mask] + EPS) / (expected[mask] + EPS))))
    h_family = _entropy(p_family.ravel())
    h_source = _entropy(p_source.ravel())
    denominator = max(math.sqrt(max(h_family * h_source, 0.0)), EPS)
    return {
        "mutual_information": mi,
        "normalised_mutual_information": float(mi / denominator),
    }


def _candidate_feature_map(cache_path: Path) -> Dict[tuple, Dict[str, float]]:
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    mapped: Dict[tuple, Dict[str, float]] = {}
    for entry in payload:
        group = str(entry["group"])
        for source, candidate in entry["candidates"].items():
            mapped[(group, source)] = {
                "false_negative_risk": float(candidate.get("false_negative_risk", np.nan)),
                "positive_similarity": float(candidate.get("positive_similarity", np.nan)),
                "boundary_closeness": float(candidate.get("boundary_closeness", np.nan)),
            }
    return mapped


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _rank_correlations(
    frame: pd.DataFrame,
    target: str,
) -> Dict[str, Optional[float]]:
    result: Dict[str, Optional[float]] = {}
    for column in (
        "false_negative_risk",
        "positive_similarity",
        "boundary_closeness",
        "oof_hardness",
        "reward",
    ):
        subset = frame[[target, column]].replace([np.inf, -np.inf], np.nan).dropna()
        if (
            len(subset) < 5
            or subset[column].nunique() < 2
            or subset[target].nunique() < 2
        ):
            result[column] = None
        else:
            correlation = float(
                subset[target].corr(subset[column], method="spearman")
            )
            result[column] = correlation if math.isfinite(correlation) else None
    return result


def _arm_metric(cell: Mapping[str, Any], arm: str) -> Optional[float]:
    record = cell.get("arms", {}).get(arm)
    if not isinstance(record, dict):
        return None
    value = record.get("source_macro_auprc")
    return None if value is None else float(value)


def analyze(result_dir: Path) -> Dict[str, Any]:
    result_dir = result_dir.resolve()
    result_path = result_dir / "phase_d_results.json"
    manifest_path = result_dir / "run_manifest.json"
    if not result_path.exists() or not manifest_path.exists():
        raise FileNotFoundError("Phase D result_dir lacks required result/manifest")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_names = list(manifest["source_names"])

    cache_by_scenario = {
        scenario: Path(audit["candidate_cache"])
        for scenario, audit in result["scenario_audits"].items()
    }
    cache_features = {
        scenario: _candidate_feature_map(path)
        for scenario, path in cache_by_scenario.items()
    }

    cell_reports: List[Dict[str, Any]] = []
    distributions: Dict[str, Dict[str, np.ndarray]] = defaultdict(dict)
    for cell in result["results"]:
        scenario = str(cell["scenario"])
        backbone = str(cell["backbone"])
        policy_path = Path(cell["policy_map"])
        rows = _read_csv(policy_path)
        counts = Counter(row["selected_source"] for row in rows)
        probabilities = _normalised_counts(counts, source_names)
        distributions[backbone][scenario] = probabilities

        long_rows: List[Dict[str, Any]] = []
        for row in rows:
            group = str(row["group"])
            for source in source_names:
                feature = cache_features[scenario].get((group, source), {})
                long_rows.append(
                    {
                        "group": group,
                        "reaction_family": row["reaction_family"],
                        "source": source,
                        "selected": int(row["selected_source"] == source),
                        "probability": _safe_float(row.get(f"prob::{source}")),
                        "oof_hardness": _safe_float(row.get(f"oof_hardness::{source}")),
                        "reward": _safe_float(row.get(f"reward::{source}")),
                        **feature,
                    }
                )
        frame = pd.DataFrame(long_rows)
        selected_frame = frame[frame["selected"] == 1]
        by_family: Dict[str, Dict[str, int]] = {}
        for family, family_rows in pd.DataFrame(rows).groupby("reaction_family"):
            by_family[str(family)] = dict(Counter(family_rows["selected_source"]))

        gate_metric = _arm_metric(cell, "learned_source_gate")
        leave_one_out = {}
        for source in source_names:
            arm_name = {
                "learned_structured": "gate_no_learned_source",
                "shuffled_real": "gate_no_shuffled_real",
            }.get(source)
            if arm_name is None:
                continue
            metric = _arm_metric(cell, arm_name)
            leave_one_out[source] = {
                "ablation_arm": arm_name,
                "full_gate_minus_ablation": (
                    None
                    if gate_metric is None or metric is None
                    else float(gate_metric - metric)
                ),
            }

        cell_reports.append(
            {
                "scenario": scenario,
                "backbone": backbone,
                "n_parents": len(rows),
                "selection_counts": dict(counts),
                "selection_proportions": {
                    source: float(probabilities[index])
                    for index, source in enumerate(source_names)
                },
                "selection_entropy": _entropy(probabilities),
                "normalised_selection_entropy": (
                    _entropy(probabilities) / math.log(len(source_names))
                ),
                "family_dependence": _mutual_information(
                    rows,
                    source_names=source_names,
                ),
                "selection_by_family": by_family,
                "selected_source_driver_spearman": _rank_correlations(
                    selected_frame,
                    "probability",
                ),
                "all_candidate_probability_driver_spearman": _rank_correlations(
                    frame,
                    "probability",
                ),
                "leave_one_source_out": leave_one_out,
                "policy_map": str(policy_path),
            }
        )

    policy_shifts: List[Dict[str, Any]] = []
    for backbone, scenario_map in distributions.items():
        scenarios = sorted(scenario_map)
        for left_index, left in enumerate(scenarios):
            for right in scenarios[left_index + 1 :]:
                policy_shifts.append(
                    {
                        "backbone": backbone,
                        "scenario_a": left,
                        "scenario_b": right,
                        "jensen_shannon_divergence": _js_divergence(
                            scenario_map[left],
                            scenario_map[right],
                        ),
                    }
                )

    mode = str(manifest.get("mode", "development"))
    formal_contract = manifest.get("formal_manifest")
    sealed_verified = bool(
        mode == "formal"
        and isinstance(formal_contract, dict)
        and formal_contract.get("pre_run_contract_verification", {}).get("verified")
    )
    return {
        "schema": "phase_g_source_complementarity_v1",
        "source_phase_d_result": str(result_path),
        "phase_d_mode": mode,
        "sealed_contract_verified": sealed_verified,
        "cells": cell_reports,
        "policy_shift": policy_shifts,
        "mechanism_exit_met": False,
        "status": (
            "CONFIRMATORY_ASSOCIATION_ONLY_CONTROLLED_INTERVENTION_REQUIRED"
            if sealed_verified
            else "EXPLORATORY_SOURCE_COMPLEMENTARITY_ONLY"
        ),
        "interpretation": (
            "Selection heterogeneity, feature associations and leave-one-source "
            "ablations can motivate a complementarity mechanism. They do not "
            "establish causality without a matched source intervention and "
            "held-out replication."
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    report = analyze(args.result_dir)
    output = args.output or args.result_dir / "phase_g_source_complementarity.json"
    output.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": report["status"], "output": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
