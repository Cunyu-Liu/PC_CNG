"""P4-G5 aggregation: summary.csv, ablation.csv, go_no_go.json.

Usage::

    python3 -m pc_cng.aggregate_p4_g5 --output-dir results/p4_risk_aware

GO/NO-GO contract (pre-declared, from the P4-G5 spec):

GO — at least TWO criteria satisfied, at least ONE of which is an
external task metric (criterion 1 or 2):
  1. HTE AUPRC improvement vs hard_binary has all-positive CI
  2. fixed-candidate MRR improvement vs hard_binary has all-positive CI
  3. ECE relative reduction >= 20% vs hard_binary
  4. collision sensitivity (hard-reject rate) significantly reduced
  5. selective risk (risk@0.8 coverage) significantly reduced
  6. training instability (mean max val-MRR drop) significantly reduced

A criterion is satisfied if ANY pre-declared non-baseline method meets
it (paired seed-level bootstrap vs hard_binary).

PARTIAL GO — not GO, external-metric CIs contain zero, but at least one
calibration/risk criterion (3-5) is satisfied: claim is risk control.

NO-GO — any of:
  * risk-aware methods simultaneously degrade external performance AND
    calibration (no method satisfies any of 1-5 AND at least one method
    is significantly worse on both 1 and 3)
  * the FNR risk score is dominated by generator self-scores
    (|coef| share of ensemble-derivative features > 0.8)
  * known-positive stress test still severely fails
    (best-method recovery_top1 <= 0.2)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CNS_ROOT = _REPO_ROOT / "chem_negative_sampling"
if str(_CNS_ROOT) not in sys.path:
    sys.path.insert(0, str(_CNS_ROOT))

from pc_cng.run_p4_augmentation import paired_bootstrap_ci  # noqa: E402
from pc_cng.training.train_risk_aware import METHODS  # noqa: E402

PHASE = "P4-G5"
BASELINE_METHOD = "hard_binary"
ENSEMBLE_DERIVED_FEATURES = [
    "ensemble_mean",
    "ensemble_variance",
    "epistemic_uncertainty",
    "aleatoric_uncertainty",
]
SELF_SCORE_DOMINANCE_THRESHOLD = 0.8
KNOWN_POSITIVE_SEVERE_FAILURE = 0.2
ECE_RELATIVE_REDUCTION = 0.20


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_runs(output_dir: Path) -> Dict[str, List[dict]]:
    """Load main runs: {method: [run jsons sorted by seed]}."""
    runs: Dict[str, List[dict]] = {}
    runs_dir = output_dir / "runs"
    if not runs_dir.exists():
        return runs
    for method_dir in sorted(runs_dir.iterdir()):
        if not method_dir.is_dir():
            continue
        method_runs = []
        for f in sorted(method_dir.glob("seed_*.json")):
            with open(f) as fh:
                method_runs.append(json.load(fh))
        if method_runs:
            runs[method_dir.name] = method_runs
    return runs


def load_ablation_runs(output_dir: Path) -> Dict[str, List[dict]]:
    """Load ablation runs: {component: [run jsons]}."""
    out: Dict[str, List[dict]] = {}
    abl_dir = output_dir / "ablation"
    if not abl_dir.exists():
        return out
    for comp_dir in sorted(abl_dir.iterdir()):
        if not comp_dir.is_dir():
            continue
        runs = []
        for f in sorted(comp_dir.glob("seed_*.json")):
            with open(f) as fh:
                runs.append(json.load(fh))
        if runs:
            out[comp_dir.name] = runs
    return out


# ---------------------------------------------------------------------------
# Summary CSV
# ---------------------------------------------------------------------------

def write_summary_csv(runs: Dict[str, List[dict]], path: Path) -> None:
    rows = []
    for method, method_runs in runs.items():
        for r in method_runs:
            rows.append({
                "method": method,
                "seed": r["seed"],
                "n_train": r["n_train"],
                "pu_prior": r["pu_prior"],
                "best_epoch": r["best_epoch"],
                "wall_clock_seconds": r["wall_clock_seconds"],
                "val_mrr": r["val_metrics"]["mrr"],
                "val_auprc": r["val_metrics"]["auprc"],
                "val_ece": r["val_metrics"]["ece"],
                "val_nll": r["val_metrics"]["nll"],
                "test_mrr": r["test_metrics"]["mrr"],
                "test_top1": r["test_metrics"]["top1"],
                "test_auprc": r["test_metrics"]["auprc"],
                "test_ece": r["test_metrics"]["ece"],
                "test_brier": r["test_metrics"]["brier"],
                "test_nll": r["test_metrics"]["nll"],
                "fixed_forward_test_mrr": r.get("fixed_forward_test_mrr"),
                "kp_recovery_top1": r["stress"]["known_positive"]["recovery_top1"],
                "kp_mean_prob": r["stress"]["known_positive"]["mean_prob"],
                "np_hard_reject_rate": r["stress"]["near_positive"]["hard_reject_rate"],
                "np_fnr_corr": r["stress"]["near_positive"]["fnr_corr"],
                "ood_ece": r["stress"]["ood_family"]["ece"],
                "collision_hard_reject_rate": r["stress"]["collision_sensitivity"]["hard_reject_rate"],
                "selective_risk_at_0p8": r["selective"]["risk_at_0p8"],
                "selective_auc": r["selective"]["auc"],
                "max_val_mrr_drop": r["training_stability"]["max_val_mrr_drop"],
            })
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_ablation_csv(
    ablation: Dict[str, List[dict]],
    full_runs: Optional[List[dict]],
    path: Path,
) -> None:
    """Per-component ablation vs the full risk_weighted_pairwise model."""
    rows = []
    full_mrr = full_auprc = full_ece = None
    if full_runs:
        full_mrr = statistics.mean(r["test_metrics"]["mrr"] for r in full_runs)
        full_auprc = statistics.mean(r["test_metrics"]["auprc"] for r in full_runs)
        full_ece = statistics.mean(r["test_metrics"]["ece"] for r in full_runs)
    for comp, runs in ablation.items():
        mrr = statistics.mean(r["test_metrics"]["mrr"] for r in runs)
        auprc = statistics.mean(r["test_metrics"]["auprc"] for r in runs)
        ece = statistics.mean(r["test_metrics"]["ece"] for r in runs)
        coll = statistics.mean(
            r["stress"]["collision_sensitivity"]["hard_reject_rate"] for r in runs
        )
        row = {
            "ablated_component": comp,
            "n_seeds": len(runs),
            "test_mrr": round(mrr, 6),
            "test_auprc": round(auprc, 6),
            "test_ece": round(ece, 6),
            "collision_hard_reject_rate": round(coll, 6),
        }
        if full_mrr is not None:
            row["delta_mrr_vs_full"] = round(mrr - full_mrr, 6)
            row["delta_auprc_vs_full"] = round(auprc - full_auprc, 6)
            row["delta_ece_vs_full"] = round(ece - full_ece, 6)
        rows.append(row)
    if not rows:
        return
    fields = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# GO / NO-GO
# ---------------------------------------------------------------------------

def _series(runs: List[dict], getter) -> List[float]:
    vals = []
    for r in runs:
        v = getter(r)
        if v is not None:
            vals.append(float(v))
    return vals


def _paired_by_seed(
    treatment: List[dict], control: List[dict], getter,
) -> Optional[Dict[str, float]]:
    """Paired bootstrap on seed-aligned values. None if unalignable."""
    t_by_seed = {r["seed"]: getter(r) for r in treatment}
    c_by_seed = {r["seed"]: getter(r) for r in control}
    common = sorted(set(t_by_seed) & set(c_by_seed))
    common = [s for s in common if t_by_seed[s] is not None and c_by_seed[s] is not None]
    if len(common) < 3:
        return None
    return paired_bootstrap_ci(
        [t_by_seed[s] for s in common],
        [c_by_seed[s] for s in common],
    )


def compute_go_no_go(
    runs: Dict[str, List[dict]],
    risk_model_manifest: Optional[dict],
) -> Dict[str, Any]:
    """Evaluate the pre-declared P4-G5 gate."""
    if BASELINE_METHOD not in runs:
        return {"phase": PHASE, "status": "NO_GO", "reason": "missing hard_binary baseline runs"}
    baseline = runs[BASELINE_METHOD]
    challengers = {m: rs for m, rs in runs.items() if m != BASELINE_METHOD}
    if not challengers:
        return {"phase": PHASE, "status": "NO_GO", "reason": "no challenger method runs"}

    metrics = {
        "hte_auprc": lambda r: r["test_metrics"]["auprc"],
        "fixed_mrr": lambda r: r.get("fixed_forward_test_mrr"),
        "ece": lambda r: r["test_metrics"]["ece"],
        "collision": lambda r: r["stress"]["collision_sensitivity"]["hard_reject_rate"],
        "selective": lambda r: r["selective"]["risk_at_0p8"],
        "instability": lambda r: r["training_stability"]["max_val_mrr_drop"],
    }

    # --- per-method paired comparisons vs baseline
    comparisons: Dict[str, Dict[str, Any]] = {}
    for method, mruns in challengers.items():
        comp: Dict[str, Any] = {}
        for name, getter in metrics.items():
            ci = _paired_by_seed(mruns, baseline, getter)
            if ci is None:
                continue
            entry: Dict[str, Any] = dict(ci)
            if name == "ece":
                b_mean = statistics.mean(_series(baseline, getter))
                m_mean = statistics.mean(_series(mruns, getter))
                entry["relative_reduction"] = (
                    round((b_mean - m_mean) / b_mean, 6) if b_mean > 0 else 0.0
                )
            comp[name] = entry
        comparisons[method] = comp

    # --- criteria (ANY challenger satisfies)
    def any_challenger(pred) -> List[str]:
        return [m for m, c in comparisons.items() if c and pred(c)]

    crit = {}
    crit["hte_auprc_ci_positive"] = any_challenger(
        lambda c: c.get("hte_auprc", {}).get("ci_low", 0) > 0
    )
    crit["fixed_mrr_ci_positive"] = any_challenger(
        lambda c: c.get("fixed_mrr", {}).get("ci_low", 0) > 0
    )
    crit["ece_relative_reduction_20"] = any_challenger(
        lambda c: (c.get("ece", {}).get("relative_reduction", 0) >= ECE_RELATIVE_REDUCTION
                   and c.get("ece", {}).get("ci_high", 0) < 0)
    )
    crit["collision_sensitivity_reduced"] = any_challenger(
        lambda c: c.get("collision", {}).get("ci_high", 0) < 0
    )
    crit["selective_risk_reduced"] = any_challenger(
        lambda c: c.get("selective", {}).get("ci_high", 0) < 0
    )
    crit["training_instability_reduced"] = any_challenger(
        lambda c: c.get("instability", {}).get("ci_high", 0) < 0
    )

    satisfied = {k: bool(v) for k, v in crit.items()}
    n_satisfied = sum(satisfied.values())
    external_satisfied = satisfied["hte_auprc_ci_positive"] or satisfied["fixed_mrr_ci_positive"]
    calibration_satisfied = (
        satisfied["ece_relative_reduction_20"]
        or satisfied["collision_sensitivity_reduced"]
        or satisfied["selective_risk_reduced"]
    )

    # --- NO-GO forensic checks
    nogo_reasons: List[str] = []
    fnr_dominance: Optional[float] = None
    if risk_model_manifest:
        rm = risk_model_manifest.get("risk_model", {})
        names = rm.get("feature_names", [])
        coefs = [abs(float(c)) for c in rm.get("coef", [])]
        total = sum(coefs)
        if total > 0 and names:
            share = sum(
                c for n, c in zip(names, coefs) if n in ENSEMBLE_DERIVED_FEATURES
            ) / total
            fnr_dominance = round(share, 6)
            if share > SELF_SCORE_DOMINANCE_THRESHOLD:
                nogo_reasons.append(
                    f"fnr_self_score_dominance={share:.3f}>{SELF_SCORE_DOMINANCE_THRESHOLD}"
                )
    best_kp = max(
        (statistics.mean(r["stress"]["known_positive"]["recovery_top1"] for r in rs)
         for rs in challengers.values()),
        default=0.0,
    )
    if best_kp <= KNOWN_POSITIVE_SEVERE_FAILURE:
        nogo_reasons.append(
            f"known_positive_severe_failure: best recovery_top1={best_kp:.3f}"
            f" <= {KNOWN_POSITIVE_SEVERE_FAILURE}"
        )
    simultaneous_degradation = any(
        c.get("hte_auprc", {}).get("ci_high", 0) < 0 and c.get("ece", {}).get("ci_low", 0) > 0
        for c in comparisons.values()
    ) and n_satisfied == 0
    if simultaneous_degradation:
        nogo_reasons.append("simultaneous external+calibration degradation")

    # --- verdict
    limitations: List[str] = []
    if nogo_reasons:
        status = "NO_GO"
        limitations = list(nogo_reasons)
    elif n_satisfied >= 2 and external_satisfied:
        status = "GO"
    elif calibration_satisfied and not external_satisfied:
        status = "PARTIAL_GO"
        limitations.append("external task metrics unchanged; claim is risk control only")
    else:
        status = "NO_GO"
        if not nogo_reasons:
            nogo_reasons.append("insufficient criteria satisfied")
            limitations.append("insufficient criteria satisfied")

    baseline_auprc = _series(baseline, metrics["hte_auprc"])
    primary_metric = {
        "name": "hte_auprc",
        "baseline_method": BASELINE_METHOD,
        "baseline_mean": round(statistics.mean(baseline_auprc), 6) if baseline_auprc else None,
        "challenger_deltas_vs_baseline": {
            m: {
                "ci_low": c.get("hte_auprc", {}).get("ci_low"),
                "ci_high": c.get("hte_auprc", {}).get("ci_high"),
                "delta_mean": c.get("hte_auprc", {}).get("delta_mean"),
            }
            for m, c in comparisons.items()
        },
    }
    predeclared_threshold = {
        "min_criteria": 2,
        "min_external_criteria": 1,
        "ece_relative_reduction": ECE_RELATIVE_REDUCTION,
        "self_score_dominance": SELF_SCORE_DOMINANCE_THRESHOLD,
        "known_positive_severe_failure": KNOWN_POSITIVE_SEVERE_FAILURE,
    }

    return {
        "phase": PHASE,
        "status": status,
        "primary_metric": primary_metric,
        "predeclared_threshold": predeclared_threshold,
        "evidence_paths": [],  # filled by caller (contract writer)
        "limitations": limitations,
        "criteria": {k: {"satisfied": bool(v), "methods": crit[k]} for k, v in satisfied.items()},
        "n_criteria_satisfied": n_satisfied,
        "external_task_metric_satisfied": external_satisfied,
        "comparisons_vs_hard_binary": comparisons,
        "fnr_self_score_coef_share": fnr_dominance,
        "best_known_positive_recovery_top1": round(best_kp, 6),
        "nogo_reasons": nogo_reasons,
        "predeclared_thresholds": predeclared_threshold,
        "baseline_method": BASELINE_METHOD,
        "challenger_methods": sorted(challengers.keys()),
        "next_phase_allowed": status in ("GO", "PARTIAL_GO"),
    }


# ---------------------------------------------------------------------------
# General P4 contract files (run_manifest / environment / input_hashes /
# commands.log) — required for every P4 phase per the global contract.
# ---------------------------------------------------------------------------

def _sha256(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit(repo: Path) -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def collect_environment() -> Dict[str, Any]:
    env: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    try:
        import torch
        env["torch"] = torch.__version__
        env["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            env["cuda_version"] = torch.version.cuda
            env["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except Exception:
        pass
    try:
        import rdkit
        env["rdkit"] = rdkit.__version__
    except Exception:
        pass
    try:
        import numpy
        env["numpy"] = numpy.__version__
    except Exception:
        pass
    return env


COMMANDS_LOG = """# P4-G5 executed commands (chronological)

# 1. risk model + artifacts (deterministic seed 20260723)
python -m pc_cng.run_p4_risk_aware risk-model \\
    --manifest data/p4/manifests/hte_feasibility_v2.json \\
    --htea-csv data/processed/hitea_full_normalized.csv \\
    --output-dir results/p4_risk_aware --device cuda:6

# 2. smoke: 1 method x 1 seed
python -m pc_cng.run_p4_risk_aware train --method hard_binary --seed 20260721 \\
    --device cuda:6 --manifest data/p4/manifests/hte_feasibility_v2.json \\
    --output-dir results/p4_risk_aware \\
    --fixed-forward-manifest data/p4/manifests/fixed_forward_candidates_v1.json

# 3a. main matrix GPU 6 (3 methods x 10 seeds) + ablation seeds 20260721-25
python -m pc_cng.run_p4_risk_aware train-all \\
    --methods hard_binary,label_smoothing,pu_nnpu \\
    --with-ablation --ablation-seeds 20260721,20260722,20260723,20260724,20260725 \\
    --device cuda:6 --manifest data/p4/manifests/hte_feasibility_v2.json \\
    --output-dir results/p4_risk_aware \\
    --fixed-forward-manifest data/p4/manifests/fixed_forward_candidates_v1.json

# 3b. main matrix GPU 7 (2 methods x 10 seeds) + ablation seeds 20260726-30
python -m pc_cng.run_p4_risk_aware train-all \\
    --methods risk_weighted_pairwise,risk_weighted_infonce \\
    --with-ablation --ablation-seeds 20260726,20260727,20260728,20260729,20260730 \\
    --device cuda:7 --manifest data/p4/manifests/hte_feasibility_v2.json \\
    --output-dir results/p4_risk_aware \\
    --fixed-forward-manifest data/p4/manifests/fixed_forward_candidates_v1.json

# 4. aggregation + go/no-go
python -m pc_cng.aggregate_p4_g5 --output-dir results/p4_risk_aware
"""


def write_phase_contract_files(
    output_dir: Path,
    runs: Dict[str, List[dict]],
    ablation: Dict[str, List[dict]],
    go: Dict[str, Any],
) -> None:
    """Write run_manifest.json, environment.json, input_hashes.json,
    commands.log next to go_no_go.json (P4 global contract section 5)."""
    now = datetime.now(timezone.utc).isoformat()

    run_entries = []
    for method, mruns in sorted(runs.items()):
        for r in mruns:
            run_entries.append({
                "method": method,
                "seed": r["seed"],
                "ablate": r.get("ablate", []),
                "best_epoch": r["best_epoch"],
                "wall_clock_seconds": r["wall_clock_seconds"],
                "result_path": f"runs/{method}/seed_{r['seed']}.json",
            })
    for comp, cruns in sorted(ablation.items()):
        for r in cruns:
            run_entries.append({
                "method": r.get("method", "risk_weighted_pairwise"),
                "seed": r["seed"],
                "ablate": r.get("ablate", [comp]),
                "best_epoch": r["best_epoch"],
                "wall_clock_seconds": r["wall_clock_seconds"],
                "result_path": f"ablation/{comp}/seed_{r['seed']}.json",
            })
    run_manifest = {
        "schema": "p4_run_manifest/v1",
        "phase": PHASE,
        "generated_at_utc": now,
        "n_main_runs": sum(len(v) for v in runs.values()),
        "n_ablation_runs": sum(len(v) for v in ablation.values()),
        "methods": sorted(runs.keys()),
        "ablated_components": sorted(ablation.keys()),
        "seeds": sorted({r["seed"] for v in runs.values() for r in v}),
        "git_commit": _git_commit(_CNS_ROOT),
        "runs": run_entries,
    }
    with open(output_dir / "run_manifest.json", "w") as f:
        json.dump(run_manifest, f, indent=2)

    with open(output_dir / "environment.json", "w") as f:
        json.dump(collect_environment(), f, indent=2)

    repo = _REPO_ROOT  # staging root (server: pc_cng_research)
    hash_targets = {
        "manifest_v2": repo / "data/p4/manifests/hte_feasibility_v2.json",
        "fixed_forward_manifest": repo / "data/p4/manifests/fixed_forward_candidates_v1.json",
        "htea_csv": repo / "data/processed/hitea_full_normalized.csv",
        "risk_artifacts": output_dir / "risk_artifacts.json",
        "stress_sets": output_dir / "stress_sets.json",
        "risk_model_manifest": output_dir / "risk_model_manifest.json",
        "code_run_p4_risk_aware": _CNS_ROOT / "pc_cng/run_p4_risk_aware.py",
        "code_risk_aware_scorer": _CNS_ROOT / "pc_cng/models/risk_aware_scorer.py",
        "code_train_risk_aware": _CNS_ROOT / "pc_cng/training/train_risk_aware.py",
        "code_false_negative_stress_test": _CNS_ROOT / "pc_cng/evaluation/false_negative_stress_test.py",
        "code_aggregate_p4_g5": _CNS_ROOT / "pc_cng/aggregate_p4_g5.py",
    }
    hashes = {name: _sha256(p) for name, p in hash_targets.items()}
    with open(output_dir / "input_hashes.json", "w") as f:
        json.dump({
            "schema": "p4_input_hashes/v1",
            "phase": PHASE,
            "generated_at_utc": now,
            "sha256": hashes,
        }, f, indent=2)

    with open(output_dir / "commands.log", "w") as f:
        f.write(COMMANDS_LOG)

    evidence = [
        "results/p4_risk_aware/summary.csv",
        "results/p4_risk_aware/ablation.csv",
        "results/p4_risk_aware/risk_model_manifest.json",
        "results/p4_risk_aware/risk_artifacts.json",
        "results/p4_risk_aware/stress_sets.json",
        "results/p4_risk_aware/raw_predictions/",
        "results/p4_risk_aware/run_manifest.json",
        "results/p4_risk_aware/environment.json",
        "results/p4_risk_aware/input_hashes.json",
        "results/p4_risk_aware/commands.log",
        "docs/p4_05_risk_aware_learning.md",
    ]
    go["evidence_paths"] = evidence
    with open(output_dir / "go_no_go.json", "w") as f:
        json.dump(go, f, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="P4-G5 aggregation")
    parser.add_argument("--output-dir", type=Path, default=Path("results/p4_risk_aware"))
    args = parser.parse_args()

    runs = load_runs(args.output_dir)
    if not runs:
        raise SystemExit(f"no runs found under {args.output_dir}/runs")
    ablation = load_ablation_runs(args.output_dir)

    write_summary_csv(runs, args.output_dir / "summary.csv")
    print(f"[aggregate] summary.csv: {sum(len(v) for v in runs.values())} runs, "
          f"{len(runs)} methods")

    full_pairwise = runs.get("risk_weighted_pairwise")
    write_ablation_csv(ablation, full_pairwise, args.output_dir / "ablation.csv")
    print(f"[aggregate] ablation.csv: {len(ablation)} components")

    rmm_path = args.output_dir / "risk_model_manifest.json"
    rmm = None
    if rmm_path.exists():
        with open(rmm_path) as f:
            rmm = json.load(f)

    go = compute_go_no_go(runs, rmm)
    with open(args.output_dir / "go_no_go.json", "w") as f:
        json.dump(go, f, indent=2)
    print(f"[aggregate] go_no_go: {go['status']} "
          f"({go['n_criteria_satisfied']} criteria, reasons={go['nogo_reasons']})")

    write_phase_contract_files(args.output_dir, runs, ablation, go)
    print("[aggregate] contract files: run_manifest.json, environment.json, "
          "input_hashes.json, commands.log, go_no_go.json(+evidence_paths)")


if __name__ == "__main__":
    main()
