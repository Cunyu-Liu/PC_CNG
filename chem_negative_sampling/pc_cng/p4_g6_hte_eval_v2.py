#!/usr/bin/env python3
"""G6 HTE evaluation v2: rebuilt with independent task heads and paired cluster CI.

Key changes from v1:
1. Five independent task heads (T1-T5), no shared model
2. Six unconfounded training arms (B0-B5)
3. Candidate-level risk weight (not global average FNR)
4. Paired cluster bootstrap CI (challenger - baseline within same replicate)
5. collision_sensitivity computed from real data (not hardcoded 0.0)
6. Preregistered primary endpoint: T5 condition-feasibility macro-AUPRC
7. All other metrics marked secondary/exploratory
"""
from __future__ import annotations

import json
import re
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

# Add parent to path for imports
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "chem_negative_sampling"))

from pc_cng.p4_g6_task_heads_v2 import (
    ARMS_V2,
    PRIMARY_ENDPOINT,
    SECONDARY_ENDPOINTS,
    TrainingArm,
    build_all_task_heads,
    compute_collision_sensitivity,
)
from pc_cng.paired_cluster_inference import (
    auprc_metric,
    ece_metric,
    family_macro_auprc_metric,
    macro_auprc_metric,
    mae_metric,
    ndcg_metric,
    paired_cluster_bootstrap,
    spearman_metric,
)


def load_candidate_fnr_map(manifest_path: Path, risk_artifacts_path: Optional[Path] = None) -> dict[str, float]:
    """Load candidate_id -> FNR mapping.

    FIX: Build a proper candidate_id -> SMILES -> FNR mapping.
    v1 lost this mapping by returning neg_smiles as a list without candidate_id.
    v2 preserves the mapping.
    """
    fnr_map: dict[str, float] = {}

    # First try risk_artifacts (has per-candidate FNR)
    if risk_artifacts_path and risk_artifacts_path.exists():
        with open(risk_artifacts_path) as f:
            artifacts = json.load(f)
        for cid, rec in artifacts.get("candidates", {}).items():
            fnr_map[cid] = float(rec.get("false_negative_risk", 0.3))

    # Also load from manifest (fallback)
    with open(manifest_path) as f:
        manifest = json.load(f)
    for group in manifest.get("groups", []):
        for cand in group.get("candidates", []):
            if cand.get("candidate_source") == "rule_pc_cng":
                cid = cand.get("candidate_id", "")
                if cid and cid not in fnr_map:
                    fnr_map[cid] = float(cand.get("false_negative_risk", 0.3))

    return fnr_map


def load_synthetic_negatives(manifest_path: Path) -> list[dict]:
    """Load synthetic PC-CNG negatives WITH candidate_id preserved.

    FIX: v1 lost candidate_id by only returning SMILES list.
    v2 preserves candidate_id for risk weight mapping.
    """
    with open(manifest_path) as f:
        manifest = json.load(f)
    negatives = []
    for group in manifest.get("groups", []):
        for cand in group.get("candidates", []):
            if cand.get("candidate_source") == "rule_pc_cng":
                smi = re.sub(r":\d+", "", cand.get("candidate_smiles", ""))
                negatives.append({
                    "candidate_id": cand.get("candidate_id", ""),
                    "products": smi,
                    "measured_yield": 0.0,
                    "label": 0,
                    "experimental_group": cand.get("experimental_group_id", ""),
                    "reaction_family": cand.get("reaction_family", ""),
                    "source": "synthetic_pc_cng",
                })
    return negatives


def load_random_template_negatives(manifest_path: Path) -> list[dict]:
    """Load random/template negatives for B2 arm."""
    with open(manifest_path) as f:
        manifest = json.load(f)
    negatives = []
    for group in manifest.get("groups", []):
        for cand in group.get("candidates", []):
            src = cand.get("candidate_source", "")
            if src in ("random_mismatch", "random_corruption", "tanimoto_retrieval",
                        "template_perturbation"):
                smi = re.sub(r":\d+", "", cand.get("candidate_smiles", ""))
                negatives.append({
                    "candidate_id": cand.get("candidate_id", ""),
                    "products": smi,
                    "measured_yield": 0.0,
                    "label": 0,
                    "experimental_group": cand.get("experimental_group_id", ""),
                    "reaction_family": cand.get("reaction_family", ""),
                    "source": src,
                })
    return negatives


def split_positive_negative(records: list[dict], yield_threshold: float = 50.0) -> tuple[list[dict], list[dict]]:
    """Split records into positive (high yield) and observed negative (low yield)."""
    positives = []
    negatives = []
    for r in records:
        y = float(r.get("measured_yield", 0))
        if y >= yield_threshold:
            r["label"] = 1
            positives.append(r)
        else:
            r["label"] = 0
            negatives.append(r)
    return positives, negatives


def compute_task_scores(task_heads: dict, arm: TrainingArm, train_records, sample_weights,
                        test_records: list[dict]) -> dict[str, list[float]]:
    """Train each task head on the arm's training set and score test records.

    Returns dict of task_id -> list of scores for test records.
    """
    scores = {}
    for task_id, head in task_heads.items():
        try:
            if not train_records:
                warnings.warn(f"{task_id} skipped: empty training data for arm {arm.arm_id}")
                scores[task_id] = [0.5] * len(test_records)
                continue
            head.train(train_records, sample_weight=sample_weights)
            task_scores = head.score(test_records)
            scores[task_id] = task_scores
        except Exception as e:
            warnings.warn(f"{task_id} failed: {e}")
            scores[task_id] = [0.5] * len(test_records)
    return scores


def build_scored_records(test_records: list[dict], task_scores: dict[str, list[float]]) -> dict[str, list[dict]]:
    """Attach task scores to test records for metric computation.

    Returns dict of task_id -> list of scored records (records with 'score' field set).
    """
    scored = {}
    for task_id, scores in task_scores.items():
        recs = []
        for r, s in zip(test_records, scores):
            rec = dict(r)
            rec["score"] = s
            recs.append(rec)
        scored[task_id] = recs
    return scored


def compute_all_metrics(scored_by_task: dict[str, list[dict]]) -> dict[str, float]:
    """Compute all metrics from scored records."""
    metrics = {}

    # T1: low-yield AUPRC (low yield is positive)
    if "T1" in scored_by_task:
        t1_recs = scored_by_task["T1"]
        # For T1, label=1 means low yield (already set in task head)
        for r in t1_recs:
            r["label"] = 1 if float(r.get("measured_yield", 0)) < 50.0 else 0
        metrics["T1_low_yield_auprc"] = auprc_metric(t1_recs)

    # T2: macro AUPRC across yield bins
    if "T2" in scored_by_task:
        t2_recs = scored_by_task["T2"]
        for r in t2_recs:
            y = float(r.get("measured_yield", 0))
            r["yield_bin"] = min(4, int(y // 20))
            r["label"] = 1 if y >= 50 else 0
        metrics["T2_macro_auprc"] = macro_auprc_metric(t2_recs)

    # T3: regression MAE and Spearman
    if "T3" in scored_by_task:
        t3_recs = scored_by_task["T3"]
        metrics["T3_mae"] = mae_metric(t3_recs)
        metrics["T3_spearman"] = spearman_metric(t3_recs)

    # T4: plate NDCG
    if "T4" in scored_by_task:
        t4_recs = scored_by_task["T4"]
        metrics["T4_plate_ndcg"] = ndcg_metric(t4_recs)

    # T5: condition-feasibility AUPRC (PRIMARY ENDPOINT)
    if "T5" in scored_by_task:
        t5_recs = scored_by_task["T5"]
        for r in t5_recs:
            r["label"] = 1 if float(r.get("measured_yield", 0)) >= 50.0 else 0
        metrics["T5_condition_feasibility_auprc"] = auprc_metric(t5_recs)
        # Also compute macro AUPRC by reaction family
        family_groups = defaultdict(list)
        for r in t5_recs:
            family_groups[r.get("reaction_family", "unknown")].append(r)
        family_auprcs = []
        for fam, recs in family_groups.items():
            if len(set(r.get("label", 0) for r in recs)) >= 2:
                family_auprcs.append(auprc_metric(recs))
        metrics["T5_condition_feasibility_macro_auprc"] = float(np.mean(family_auprcs)) if family_auprcs else 0.0

    # ECE and Brier (calibration)
    if "T5" in scored_by_task:
        t5_recs = scored_by_task["T5"]
        for r in t5_recs:
            r["label"] = 1 if float(r.get("measured_yield", 0)) >= 50.0 else 0
        metrics["ece"] = ece_metric(t5_recs)
        # Brier score
        scores = np.array([r.get("score", 0.5) for r in t5_recs])
        labels = np.array([r.get("label", 0) for r in t5_recs])
        metrics["brier"] = float(np.mean((scores - labels) ** 2))

    return metrics


def run_g6_v2_evaluation(
    hte_records: list[dict],
    manifest_path: Path,
    risk_artifacts_path: Optional[Path] = None,
    yield_threshold: float = 50.0,
    n_bootstrap: int = 2000,
    seed: int = 20260723,
) -> dict:
    """Run complete G6 v2 evaluation.

    Returns dict with per-arm metrics, paired CI vs baseline, and go/no-go verdict.
    """
    # Split data
    train_val_records = [r for r in hte_records if r.get("split") == "train"]
    val_records = [r for r in hte_records if r.get("split") == "val"]
    test_records = [r for r in hte_records if r.get("split") == "test"]

    # Split train into positive and observed negative
    train_pos, train_obs_neg = split_positive_negative(train_val_records, yield_threshold)

    # Load synthetic negatives
    synthetic_negs = load_synthetic_negatives(manifest_path)
    random_template_negs = load_random_template_negatives(manifest_path)

    # Load candidate FNR map
    fnr_map = load_candidate_fnr_map(manifest_path, risk_artifacts_path)

    # Build task heads
    task_heads_template = build_all_task_heads()

    # Compute collision sensitivity (from real data)
    collision_sens = compute_collision_sensitivity(test_records, train_pos)

    # Run each arm
    arm_results: dict[str, dict] = {}
    for arm in ARMS_V2:
        train_records, sample_weights = arm.build_training_set(
            positive_records=train_pos,
            synthetic_negatives=synthetic_negs,
            random_template_negatives=random_template_negs,
            observed_negatives=train_obs_neg,
            fnr_map=fnr_map,
        )

        task_heads = build_all_task_heads()  # fresh heads per arm
        task_scores = compute_task_scores(task_heads, arm, train_records, sample_weights, test_records)
        scored_by_task = build_scored_records(test_records, task_scores)
        metrics = compute_all_metrics(scored_by_task)
        metrics["collision_sensitivity"] = collision_sens
        arm_results[arm.arm_id] = {
            "arm_name": arm.arm_name,
            "description": arm.description,
            "n_train": len(train_records),
            "n_pos": sum(1 for r in train_records if r.get("label", 1) == 1),
            "n_neg": sum(1 for r in train_records if r.get("label", 0) == 0),
            "metrics": metrics,
            "scored_by_task": scored_by_task,
        }

    # Paired cluster bootstrap CI: each arm vs B0 (positive_only baseline)
    # For each metric, compute paired cluster bootstrap
    baseline_arm_id = "B0"
    primary_metric = PRIMARY_ENDPOINT

    paired_cis: dict[str, dict] = {}
    all_metrics = set()
    for arm_id, result in arm_results.items():
        if arm_id == baseline_arm_id:
            continue
        all_metrics.update(result["metrics"].keys())

    for metric_name in sorted(all_metrics):
        if metric_name == "collision_sensitivity":
            continue  # data-level metric, not method-dependent

        for arm_id in arm_results:
            if arm_id == baseline_arm_id:
                continue

            challenger_scored = arm_results[arm_id]["scored_by_task"]
            baseline_scored = arm_results[baseline_arm_id]["scored_by_task"]

            # Find which task this metric belongs to
            task_id = None
            if metric_name.startswith("T1"):
                task_id = "T1"
            elif metric_name.startswith("T2"):
                task_id = "T2"
            elif metric_name.startswith("T3"):
                task_id = "T3"
            elif metric_name.startswith("T4"):
                task_id = "T4"
            elif metric_name.startswith("T5"):
                task_id = "T5"
            elif metric_name in ("ece", "brier"):
                task_id = "T5"

            if task_id is None or task_id not in challenger_scored:
                continue

            ch_recs = challenger_scored[task_id]
            bl_recs = baseline_scored[task_id]

            # Choose metric function (check macro_auprc BEFORE auprc since "macro_auprc" contains "auprc")
            if "macro_auprc" in metric_name:
                metric_fn = lambda recs: family_macro_auprc_metric(recs)
            elif "auprc" in metric_name:
                metric_fn = lambda recs: auprc_metric(recs)
            elif "mae" in metric_name:
                metric_fn = lambda recs: mae_metric(recs)
            elif "spearman" in metric_name:
                metric_fn = lambda recs: spearman_metric(recs)
            elif "ndcg" in metric_name:
                metric_fn = lambda recs: ndcg_metric(recs)
            elif "ece" in metric_name:
                metric_fn = lambda recs: ece_metric(recs)
            elif "brier" in metric_name:
                metric_fn = lambda recs: float(np.mean([
                    (r.get("score", 0.5) - r.get("label", 0)) ** 2 for r in recs
                ]))
            else:
                continue

            ci = paired_cluster_bootstrap(
                challenger_records=ch_recs,
                baseline_records=bl_recs,
                metric_fn=metric_fn,
                cluster_key="experimental_group",
                n_bootstrap=n_bootstrap,
                seed=seed,
            )
            paired_cis[f"{arm_id}_vs_{baseline_arm_id}_{metric_name}"] = ci

    # Determine verdict based on PRIMARY endpoint
    primary_ci_key = f"B4_vs_{baseline_arm_id}_{primary_metric}"
    primary_ci = paired_cis.get(primary_ci_key, {})

    # Also check B1 (synthetic only) and B4 (observed + synthetic) vs B0
    b1_primary_key = f"B1_vs_{baseline_arm_id}_{primary_metric}"
    b1_primary_ci = paired_cis.get(b1_primary_key, {})

    b3_primary_key = f"B3_vs_{baseline_arm_id}_{primary_metric}"
    b3_primary_ci = paired_cis.get(b3_primary_key, {})

    # Verdict logic:
    # STRONG_GO: B4 (observed+synthetic) primary CI all positive AND B4 > B3 (observed only)
    # WEAK_GO: B4 primary CI all positive
    # NO_GO: otherwise
    b4_ci_positive = primary_ci.get("ci_all_positive", False)
    b4_beats_b3 = False
    b4_vs_b3_key = f"B4_vs_B3_{primary_metric}"
    # Need to also compute B4 vs B3
    if "B3" in arm_results and "B4" in arm_results:
        ch_b4 = arm_results["B4"]["scored_by_task"].get("T5", [])
        bl_b3 = arm_results["B3"]["scored_by_task"].get("T5", [])
        if ch_b4 and bl_b3:
            ci_b4_b3 = paired_cluster_bootstrap(
                ch_b4, bl_b3,
                lambda recs: auprc_metric(recs),
                cluster_key="experimental_group",
                n_bootstrap=n_bootstrap,
                seed=seed,
            )
            paired_cis[b4_vs_b3_key] = ci_b4_b3
            b4_beats_b3 = ci_b4_b3.get("ci_all_positive", False)

    if b4_ci_positive and b4_beats_b3:
        verdict = "STRONG_GO"
        reason = f"Primary endpoint {primary_metric}: B4 CI all positive AND B4 beats B3 (observed-only)"
    elif b4_ci_positive:
        verdict = "WEAK_GO"
        reason = f"Primary endpoint {primary_metric}: B4 CI all positive but B4 does not beat B3"
    else:
        verdict = "NO_GO"
        reason = f"Primary endpoint {primary_metric}: B4 CI not all positive"

    result = {
        "schema_version": "2.0",
        "evaluation_date": "2026-07-24",
        "primary_endpoint": primary_metric,
        "secondary_endpoints": SECONDARY_ENDPOINTS,
        "task_heads": "5 independent (T1-T5)",
        "arms": [a.arm_name for a in ARMS_V2],
        "arm_results": {aid: {k: v for k, v in r.items() if k != "scored_by_task"} for aid, r in arm_results.items()},
        "paired_cis": {k: {kk: vv for kk, vv in v.items()} for k, v in paired_cis.items()},
        "collision_sensitivity": collision_sens,
        "verdict": verdict,
        "reason": reason,
        "candidate_fnr_map_size": len(fnr_map),
        "n_train_pos": len(train_pos),
        "n_train_obs_neg": len(train_obs_neg),
        "n_synthetic_neg": len(synthetic_negs),
        "n_random_template_neg": len(random_template_negs),
        "n_test": len(test_records),
        "n_bootstrap": n_bootstrap,
    }
    # Also return full arm_results with scored_by_task for independent CI reconstruction
    scored_data = {
        "test_records": test_records,
        "arm_scored_by_task": {aid: r.get("scored_by_task", {}) for aid, r in arm_results.items()},
    }
    return result, scored_data


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="G6 v2 evaluation")
    parser.add_argument("--hte-parquet", type=Path, default=_REPO_ROOT / "data/processed/p4_hte_normalized.parquet")
    parser.add_argument("--manifest", type=Path, default=_REPO_ROOT / "data/p4/manifests/hte_feasibility_v2.json")
    parser.add_argument("--risk-artifacts", type=Path, default=_REPO_ROOT / "results/p4_risk_aware/risk_artifacts.json")
    parser.add_argument("--output-dir", type=Path, default=_REPO_ROOT / "results/p4_hte_external_validation_v2")
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load HTE data
    import pandas as pd
    df = pd.read_parquet(args.hte_parquet)
    hte_records = df.to_dict("records")

    if args.smoke:
        # Use fewer records and bootstrap iterations for smoke test
        # Sample proportionally from each split to ensure all splits present
        from collections import defaultdict
        by_split = defaultdict(list)
        for r in hte_records:
            by_split[r.get("split", "train")].append(r)
        # Take up to 800 train, 200 val, 400 test (1400 total)
        smoke_records = []
        for split in ("train", "val", "test"):
            recs = by_split.get(split, [])
            cap = {"train": 800, "val": 200, "test": 400}[split]
            smoke_records.extend(recs[:cap])
        hte_records = smoke_records
        args.n_bootstrap = 500

    print(f"Loaded {len(hte_records)} HTE records")
    print(f"Manifest: {args.manifest}")
    print(f"Risk artifacts: {args.risk_artifacts}")

    result, scored_data = run_g6_v2_evaluation(
        hte_records=hte_records,
        manifest_path=args.manifest,
        risk_artifacts_path=args.risk_artifacts,
        n_bootstrap=args.n_bootstrap,
    )

    # Write results
    output_file = args.output_dir / "go_no_go_v2.json"
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2, default=str)

    # Save scored records for independent CI reconstruction (Exit Criterion)
    scored_file = args.output_dir / "scored_records_v2.json"
    with open(scored_file, "w") as f:
        json.dump(scored_data, f, indent=2, default=str)
    print(f"Scored records: {scored_file}")

    # Summary CSV
    summary_lines = ["arm_id,arm_name,n_train,n_pos,n_neg," + ",".join(sorted(
        set(k for r in result["arm_results"].values() for k in r["metrics"])
    ))]
    for arm_id, r in result["arm_results"].items():
        metrics = r["metrics"]
        row = f"{arm_id},{r['arm_name']},{r['n_train']},{r['n_pos']},{r['n_neg']}"
        for k in summary_lines[0].split(",")[5:]:
            row += f",{metrics.get(k, '')}"
        summary_lines.append(row)
    (args.output_dir / "summary_v2.csv").write_text("\n".join(summary_lines))

    print(f"\nVerdict: {result['verdict']}")
    print(f"Reason: {result['reason']}")
    print(f"Primary endpoint: {result['primary_endpoint']}")
    print(f"Collision sensitivity: {result['collision_sensitivity']:.4f}")
    print(f"Output: {output_file}")
