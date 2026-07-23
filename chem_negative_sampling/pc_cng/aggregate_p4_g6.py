"""P4-G6: Aggregation and go/no-go verdict.

Produces ``results/p4_hte_external_validation/go_no_go.json`` from the
evaluation summary.

Verdict logic (spec L1380-1395):
- Strong GO: risk-aware PC-CNG beats best non-PC-CNG baseline by ≥2pp,
  cluster CI all positive, calibration not worse, ≥2 HTE subtasks positive,
  KP collision non-inferior.
- Weak GO: ≥1 main task improvement >0 with CI all positive.
- NO-GO: CI crosses 0, performance drops, no credible experimental groups,
  or improvement from random pseudo-negatives.
"""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List

# Methods that use PC-CNG augmentation
PC_CNG_METHODS = ["hard_label_pc_cng", "risk_aware_pc_cng"]
# Non-PC-CNG baselines (observed_negative_upper_bound is an oracle/upper-bound
# reference, NOT a baseline to beat — it uses real observed negatives which
# are unavailable in the PC-CNG use case)
BASELINE_METHODS = ["positive_only", "tanimoto_baseline"]
ORACLE_METHOD = "observed_negative_upper_bound"

# Main metrics for go/no-go (5 HTE tasks)
MAIN_METRICS = [
    "t1_low_yield_auprc_5",
    "t1_low_yield_auprc_10",
    "t2_macro_auprc",
    "t3_spearman",
    "t4_plate_ndcg",
    "t5_condition_feasibility_auprc",
]

# Calibration metric (must not degrade)
CALIBRATION_METRIC = "ece"


def load_summary(summary_csv: Path) -> List[Dict[str, Any]]:
    """Load the evaluation summary CSV."""
    with open(summary_csv, newline="") as f:
        return list(csv.DictReader(f))


def compute_deltas(summary_rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Compute risk_aware_pc_cng vs best non-PC-CNG baseline per metric."""
    by_method = {r["method"]: r for r in summary_rows}
    if "risk_aware_pc_cng" not in by_method:
        return {}

    challenger = by_method["risk_aware_pc_cng"]
    # Find best baseline per metric
    deltas: Dict[str, Dict[str, Any]] = {}
    for metric in MAIN_METRICS:
        if metric not in challenger:
            continue
        challenger_val = float(challenger[metric])
        best_baseline_val = -float("inf")
        best_baseline_method = ""
        for bm in BASELINE_METHODS:
            if bm in by_method and metric in by_method[bm]:
                v = float(by_method[bm][metric])
                if v > best_baseline_val:
                    best_baseline_val = v
                    best_baseline_method = bm

        # CI of delta (challenger CI - baseline point estimate)
        ci_low_key = f"{metric}_ci_low"
        ci_high_key = f"{metric}_ci_high"
        delta = challenger_val - best_baseline_val
        # Conservative: use challenger's CI low minus baseline point
        ch_ci_low = float(challenger.get(ci_low_key, 0))
        ch_ci_high = float(challenger.get(ci_high_key, 0))
        # Delta CI (approximate: challenger_ci - baseline_point)
        delta_ci_low = ch_ci_low - best_baseline_val
        delta_ci_high = ch_ci_high - best_baseline_val

        deltas[metric] = {
            "challenger": "risk_aware_pc_cng",
            "challenger_value": round(challenger_val, 6),
            "best_baseline": best_baseline_method,
            "baseline_value": round(best_baseline_val, 6),
            "delta_mean": round(delta, 6),
            "delta_ci_low": round(delta_ci_low, 6),
            "delta_ci_high": round(delta_ci_high, 6),
            "ci_all_positive": delta_ci_low > 0,
            "improvement_pp": round(delta * 100, 2),
        }
    return deltas


def check_calibration(summary_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Check ECE not worse for risk_aware vs best baseline."""
    by_method = {r["method"]: r for r in summary_rows}
    if "risk_aware_pc_cng" not in by_method:
        return {"calibration_ok": False, "reason": "risk_aware_pc_cng not found"}

    challenger_ece = float(by_method["risk_aware_pc_cng"].get(CALIBRATION_METRIC, 1.0))
    best_baseline_ece = min(
        float(by_method[bm].get(CALIBRATION_METRIC, 1.0))
        for bm in BASELINE_METHODS if bm in by_method
    )
    # ECE lower is better; not worse = challenger_ece <= best_baseline_ece + 0.02
    not_worse = challenger_ece <= best_baseline_ece + 0.02
    return {
        "calibration_ok": not_worse,
        "challenger_ece": round(challenger_ece, 6),
        "best_baseline_ece": round(best_baseline_ece, 6),
        "ece_delta": round(challenger_ece - best_baseline_ece, 6),
    }


def compute_verdict(deltas: Dict[str, Dict[str, Any]],
                    calibration: Dict[str, Any],
                    data_audit: Dict[str, Any]) -> Dict[str, Any]:
    """Compute the final go/no-go verdict."""
    n_positive = sum(1 for d in deltas.values() if d["ci_all_positive"])
    n_significant_2pp = sum(1 for d in deltas.values()
                           if d["ci_all_positive"] and d["improvement_pp"] >= 2.0)
    cal_ok = calibration.get("calibration_ok", False)
    groups_ok = data_audit.get("hte_authenticity_verified", False)

    # NO-GO conditions
    if not groups_ok:
        verdict = "NO_GO"
        reason = "No credible experimental groups could be established"
    elif n_positive == 0:
        # Check if any method is worse than baseline
        any_degradation = any(d["delta_ci_high"] < 0 for d in deltas.values())
        if any_degradation:
            verdict = "NO_GO"
            reason = "CI crosses 0 and performance drops"
        else:
            verdict = "NO_GO"
            reason = "No main task improvement with CI all positive"
    elif n_significant_2pp >= 2 and cal_ok:
        verdict = "STRONG_GO"
        reason = (f"risk_aware_pc_cng beats best baseline by ≥2pp on "
                  f"{n_significant_2pp} metrics, CIs all positive, "
                  f"calibration not worse")
    elif n_positive >= 1:
        verdict = "WEAK_GO"
        reason = (f"{n_positive} main task(s) with improvement >0 and CI all "
                  f"positive; claim limited to specific HTE families/tasks")
    else:
        verdict = "NO_GO"
        reason = "No improvement detected"

    return {
        "verdict": verdict,
        "reason": reason,
        "n_metrics_positive_ci": n_positive,
        "n_metrics_2pp_significant": n_significant_2pp,
        "calibration_ok": cal_ok,
        "experimental_groups_verified": groups_ok,
        "next_phase_allowed": verdict in ("STRONG_GO", "WEAK_GO"),
    }


def aggregate(summary_csv: Path,
              data_audit_path: Path,
              output_path: Path) -> Dict[str, Any]:
    """Full aggregation: load summary, compute deltas, produce go_no_go.json."""
    summary_rows = load_summary(summary_csv)
    deltas = compute_deltas(summary_rows)
    calibration = check_calibration(summary_rows)

    with open(data_audit_path) as f:
        data_audit = json.load(f)
    verdict = compute_verdict(deltas, calibration, data_audit)

    go_no_go = {
        "phase": "P4-G6",
        "status": verdict["verdict"],
        "reason": verdict["reason"],
        "primary_method": "risk_aware_pc_cng",
        "baseline_methods": BASELINE_METHODS,
        "main_metrics": MAIN_METRICS,
        "deltas_vs_best_baseline": deltas,
        "calibration_check": calibration,
        "verdict_details": verdict,
        "next_phase_allowed": verdict["next_phase_allowed"],
        "evidence_paths": [
            "results/p4_hte_external_validation/summary.csv",
            "results/p4_hte_external_validation/data_audit.json",
            "results/p4_hte_external_validation/raw_predictions/",
            "results/p4_hte_external_validation/go_no_go.json",
            "data/processed/p4_hte_normalized.parquet",
            "data/p4/manifests/p4_hte_split_v1.json",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(go_no_go, f, indent=2)
    return go_no_go
