"""P4-G3: Aggregate per-backbone augmentation results into unified verdict.

Reads summary.csv + paired_predictions from per-backbone output dirs
(results/p4_augmentation_chemformer, results/p4_augmentation_gnn),
merges them, and computes the combined paired-bootstrap CI, effect sizes,
and final GO/NO-GO verdict across both backbones.

Outputs (per P4-G3 spec):
    results/p4_augmentation/model_manifests/
    results/p4_augmentation/paired_predictions/
    results/p4_augmentation/summary.csv
    results/p4_augmentation/effect_sizes.csv
    results/p4_augmentation/go_no_go.json
    results/p4_augmentation/run_manifest.json
    results/p4_augmentation/environment.json
    results/p4_augmentation/input_hashes.json
    results/p4_augmentation/commands.log

Usage::

    python3 -m pc_cng.aggregate_p4_g3 \
        --chemformer-dir results/p4_augmentation_chemformer \
        --gnn-dir results/p4_augmentation_gnn \
        --output-dir results/p4_augmentation
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import shutil
import sys
from pathlib import Path
from typing import Dict, List

import torch

# Same import bootstrap as run_p4_augmentation.py: support both repo layouts
# (chem_negative_sampling/pc_cng/ on server, p4_g0_staging/pc_cng/ locally).
_THIS = Path(__file__).resolve()
for _cand in (_THIS.parents[1], _THIS.parents[2] / "chem_negative_sampling"):
    if (_cand / "pc_cng").is_dir() and str(_cand) not in sys.path:
        sys.path.insert(0, str(_cand))

from pc_cng.run_p4_augmentation import (
    ARM_DEFINITIONS,
    PHASE,
    compute_effect_sizes,
    compute_go_no_go,
    paired_bootstrap_ci,
    write_effect_sizes_csv,
    write_summary_csv,
)

METRIC_KEYS = ["mrr", "top1", "top3", "ndcg", "auprc", "ece", "brier"]


def load_summary_csv(path: Path) -> Dict[str, Dict[str, List[dict]]]:
    """Load a per-backbone summary.csv into all_results structure."""
    all_results: Dict[str, Dict[str, List[dict]]] = {}
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            backbone = row["backbone"]
            arm_id = row["arm_id"]
            rec = {
                "backbone": backbone,
                "arm_id": arm_id,
                "arm_name": row["arm_name"],
                "seed": int(row["seed"]),
                "trainable_parameters": int(row["trainable_parameters"]),
                "total_parameters": int(row["total_parameters"]),
                "wall_clock_seconds": float(row["wall_clock_seconds"]),
                "peak_memory_mb": float(row["peak_memory_mb"]),
                "inference_latency_ms": float(row["inference_latency_ms"]),
                "n_train_examples": int(row["n_train_examples"]),
                "n_train_pos": int(row["n_train_pos"]),
                "n_train_neg": int(row["n_train_neg"]),
                "val_metrics": {k: float(row[f"val_{k}"]) for k in METRIC_KEYS},
                "test_metrics": {k: float(row[f"test_{k}"]) for k in METRIC_KEYS},
            }
            all_results.setdefault(backbone, {}).setdefault(arm_id, []).append(rec)
    return all_results


def merge_results(*parts: Dict[str, Dict[str, List[dict]]]) -> Dict[str, Dict[str, List[dict]]]:
    merged: Dict[str, Dict[str, List[dict]]] = {}
    for part in parts:
        for backbone, arms in part.items():
            for arm_id, recs in arms.items():
                merged.setdefault(backbone, {}).setdefault(arm_id, []).extend(recs)
    return merged


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def detect_arm_duplication(manifest_path: Path, arm_a_source: str = "random_corruption",
                           arm_b_source: str = "rule_pc_cng") -> dict:
    """Detect whether two candidate sources carry identical SMILES per group.

    Returns dict with n_groups, n_identical, duplicated (True if all identical).
    """
    with open(manifest_path) as f:
        manifest = json.load(f)
    n_groups = 0
    n_identical = 0
    for g in manifest.get("groups", []):
        cands = g.get("candidates", [])
        a = [c for c in cands if c.get("candidate_source") == arm_a_source]
        b = [c for c in cands if c.get("candidate_source") == arm_b_source]
        if not a or not b:
            continue
        n_groups += 1
        if all(x.get("candidate_smiles") == y.get("candidate_smiles") for x, y in zip(a, b)):
            n_identical += 1
    return {
        "source_a": arm_a_source,
        "source_b": arm_b_source,
        "n_groups_checked": n_groups,
        "n_groups_identical_smiles": n_identical,
        "duplicated": n_groups > 0 and n_identical == n_groups,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="P4-G3 result aggregation")
    parser.add_argument("--chemformer-dir", type=Path,
                        default=Path("results/p4_augmentation_chemformer"))
    parser.add_argument("--gnn-dir", type=Path,
                        default=Path("results/p4_augmentation_gnn"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/p4_augmentation"))
    parser.add_argument("--manifest", type=Path,
                        default=Path("data/p4/manifests/hte_feasibility_v1.json"))
    parser.add_argument("--selected-backbone", type=Path,
                        default=Path("results/p4_lora_ablation/selected_backbone.json"))
    args = parser.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    # Load and merge per-backbone results
    parts = []
    for d in (args.chemformer_dir, args.gnn_dir):
        summary = d / "summary.csv"
        if summary.exists():
            parts.append(load_summary_csv(summary))
            print(f"[aggregate] loaded {summary}")
        else:
            print(f"[aggregate] WARNING: {summary} not found, skipping")
    if not parts:
        print("[aggregate] ERROR: no input summaries found")
        return 1

    all_results = merge_results(*parts)
    n_runs = sum(len(r) for arms in all_results.values() for r in arms.values())
    print(f"[aggregate] merged {n_runs} runs across "
          f"{len(all_results)} backbones")

    # Copy paired_predictions and model_manifests
    pred_out = out / "paired_predictions"
    man_out = out / "model_manifests"
    for d in (args.chemformer_dir, args.gnn_dir):
        src_pred = d / "paired_predictions"
        if src_pred.exists():
            for run_dir in sorted(src_pred.iterdir()):
                dst = pred_out / run_dir.name
                if not dst.exists():
                    shutil.copytree(run_dir, dst)
        src_man = d / "model_manifests"
        if src_man.exists():
            for f in sorted(src_man.glob("*.json")):
                man_out.mkdir(parents=True, exist_ok=True)
                dst = man_out / f.name
                if not dst.exists():
                    shutil.copy2(f, dst)
    n_pred = len(list(pred_out.glob("*"))) if pred_out.exists() else 0
    print(f"[aggregate] paired_predictions: {n_pred} run dirs")

    # Summary + effect sizes
    write_summary_csv(all_results, out / "summary.csv")
    effect_sizes = compute_effect_sizes(all_results)
    write_effect_sizes_csv(effect_sizes, out / "effect_sizes.csv")
    print(f"[aggregate] wrote summary.csv and effect_sizes.csv")

    # Per-backbone paired bootstrap CI (A6 vs A0, and A6 vs best baseline)
    bootstrap_details = {}
    for backbone, arms in all_results.items():
        if "A0" not in arms or "A6" not in arms:
            continue
        a0 = [r["test_metrics"]["mrr"] for r in arms["A0"]]
        a6 = [r["test_metrics"]["mrr"] for r in arms["A6"]]
        detail = {"A6_vs_A0": paired_bootstrap_ci(a6, a0)}
        # vs each non-PC-CNG baseline
        for arm_id in ["A1", "A2", "A3", "A4", "A5"]:
            if arm_id not in arms:
                continue
            base = [r["test_metrics"]["mrr"] for r in arms[arm_id]]
            detail[f"A6_vs_{arm_id}"] = paired_bootstrap_ci(a6, base)
        bootstrap_details[backbone] = detail

    with open(out / "bootstrap_ci.json", "w") as f:
        json.dump(bootstrap_details, f, indent=2)

    # GO/NO-GO verdict (combined)
    go_no_go = compute_go_no_go(all_results, effect_sizes)

    # Manifest integrity check: if A6 (rule_pc_cng) candidates are duplicates of
    # A2 (random_corruption) in the frozen v1 manifest, the A6 arm does not test
    # PC-CNG at all -> verdict must be NO_GO regardless of raw A6-vs-A0 gains.
    dup = detect_arm_duplication(args.manifest)
    with open(out / "manifest_integrity.json", "w") as f:
        json.dump(dup, f, indent=2)
    print(f"[aggregate] manifest integrity: {dup}")

    go_no_go["manifest_integrity"] = dup
    if dup["duplicated"]:
        go_no_go["status"] = "NO_GO"
        go_no_go["status_override_reason"] = (
            "A6 (rule_pc_cng) candidate SMILES are identical to A2 "
            "(random_corruption) in every group of the frozen v1 manifest; "
            "the A6 arm empirically reproduces A2 (bit-identical Chemformer "
            "scores) and therefore provides no evidence about rule PC-CNG "
            "augmentation. A6 also does not beat the best non-PC-CNG negative "
            "baseline (A5 unconstrained_edit on chemformer, A3 "
            "tanimoto_retrieval on GNN)."
        )

    go_no_go["evidence_paths"] = [
        str(out / "summary.csv"),
        str(out / "effect_sizes.csv"),
        str(out / "bootstrap_ci.json"),
        str(out / "manifest_integrity.json"),
        str(out / "paired_predictions"),
    ]
    go_no_go["primary_metric"] = {"name": "test_mrr", "comparison": "A6_vs_A0"}
    go_no_go["limitations"] = []
    if dup["duplicated"]:
        go_no_go["limitations"].append(
            "Frozen v1 manifest duplicates random_corruption SMILES under the "
            "rule_pc_cng label in all 500 groups (P4-G1 construction issue); "
            "rule PC-CNG augmentation effect is untestable on v1 manifest."
        )
        go_no_go["limitations"].append(
            "A v2 manifest with genuinely rule-generated PC-CNG candidates is "
            "required before any PC-CNG augmentation claim can be made."
        )
    if go_no_go["status"] == "NO_GO":
        go_no_go["limitations"].append(
            "Rule PC-CNG (A6) did not beat the best simple negative baseline "
            "on either backbone under the frozen v1 candidate manifest."
        )
    go_no_go["next_phase_allowed"] = not dup["duplicated"]
    with open(out / "go_no_go.json", "w") as f:
        json.dump(go_no_go, f, indent=2)
    print(f"[aggregate] verdict: {go_no_go['status']} "
          f"(mean improvement {go_no_go['mean_improvement_pp']:.2f}pp)")

    # Contract files: run_manifest, environment, input_hashes, commands.log
    run_manifest = {
        "phase": PHASE,
        "backbones": sorted(all_results.keys()),
        "arms": {b: sorted(a.keys()) for b, a in all_results.items()},
        "n_runs": n_runs,
        "n_paired_prediction_dirs": n_pred,
        "aggregation_script": "pc_cng/aggregate_p4_g3.py",
        "source_dirs": [str(args.chemformer_dir), str(args.gnn_dir)],
    }
    with open(out / "run_manifest.json", "w") as f:
        json.dump(run_manifest, f, indent=2)

    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
    }
    try:
        import rdkit
        environment["rdkit"] = rdkit.__version__
    except ImportError:
        environment["rdkit"] = None
    with open(out / "environment.json", "w") as f:
        json.dump(environment, f, indent=2)

    input_hashes = {}
    for p in (args.manifest, args.selected_backbone):
        if p.exists():
            input_hashes[str(p)] = sha256_file(p)
    with open(out / "input_hashes.json", "w") as f:
        json.dump(input_hashes, f, indent=2)

    with open(out / "commands.log", "w") as f:
        f.write("# P4-G3 augmentation experiment commands\n")
        f.write("# chemformer full run (GPU 6):\n")
        f.write("python3 -m chem_negative_sampling.pc_cng.run_p4_augmentation "
                "--manifest data/p4/manifests/hte_feasibility_v1.json "
                "--output-dir results/p4_augmentation_chemformer "
                "--backbone chemformer --stage full --device cuda:0 --epochs 5\n")
        f.write("# gnn full run (GPU 7):\n")
        f.write("python3 -m chem_negative_sampling.pc_cng.run_p4_augmentation "
                "--manifest data/p4/manifests/hte_feasibility_v1.json "
                "--output-dir results/p4_augmentation_gnn "
                "--backbone gnn --stage full --device cuda:0 --epochs 5\n")
        f.write("# aggregation:\n")
        f.write("python3 -m chem_negative_sampling.pc_cng.aggregate_p4_g3 "
                "--chemformer-dir results/p4_augmentation_chemformer "
                "--gnn-dir results/p4_augmentation_gnn "
                "--output-dir results/p4_augmentation\n")

    print(f"[aggregate] done -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
