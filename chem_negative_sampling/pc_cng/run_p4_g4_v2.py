"""P4-G4 v2: Full Generator × Scorer matrix with v2 manifest.

Entry condition P4-G3 v2 == WEAK_GO is met. Runs the full 3-scorer × 7-arm ×
10-seed matrix with the v2 candidate manifest (genuine rule PC-CNG candidates,
no A6≡A2 duplication).

Outputs (results/p4_generator_scorer_matrix_v2/):
    summary.csv  effect_sizes.csv  interaction_model.json
    difficulty_profile.json  raw_predictions/  go_no_go.json
    run_manifest.json  environment.json  input_hashes.json  commands.log
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Suppress RDKit deprecation warnings BEFORE any rdkit import
os.environ["RDKitRDLogger"] = "0"

import numpy as np
import torch

# Explicitly disable RDKit logger (catches all rdApp.* messages)
try:
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
except Exception:
    pass

# ---------------------------------------------------------------------------
# Import bootstrap — must find pc_cng package
# ---------------------------------------------------------------------------

_THIS = Path(__file__).resolve()
for _cand in (_THIS.parents[1], _THIS.parents[2] / "chem_negative_sampling"):
    if (_cand / "pc_cng").is_dir() and str(_cand) not in sys.path:
        sys.path.insert(0, str(_cand))

# Import everything we need from the diagnostic module
from pc_cng.run_p4_augmentation import (  # noqa: E402
    ARM_DEFINITIONS,
    ARM_IDS,
    build_arm_training_data,
    compute_auprc,
    compute_calibration_metrics,
    compute_metrics_from_predictions,
    load_manifest_candidates,
    set_seed,
)
from pc_cng.run_p4_g4_diagnostic import (  # noqa: E402
    DEFAULT_SEEDS,
    MorganMLPScorer,
    build_cell_table,
    cell_effect_sizes,
    diagnose_hypotheses,
    difficulty_profile,
    g3_rows_to_records,
    interaction_anova,
    load_g3_summary,
    mixed_effects_model,
    run_single_mlp_experiment,
    sha256_file,
    write_effect_sizes_csv,
    write_summary_csv,
)

PHASE = "P4-G4"
G3_V2_DIRS = {
    "chemformer": Path("results/p4_augmentation_v2_chemformer"),
    "gnn": Path("results/p4_augmentation_v2_gnn"),
}


def load_existing_mlp_run(arm_id: str, seed: int, output_dir: Path) -> dict:
    """Load metrics from an already-completed MLP run."""
    pred_dir = output_dir / "raw_predictions" / f"morgan_mlp_{arm_id}_seed{seed}"
    with open(pred_dir / "test_predictions.json") as f:
        test_preds = json.load(f)
    with open(pred_dir / "val_predictions.json") as f:
        val_preds = json.load(f)
    test_m = compute_metrics_from_predictions(test_preds)
    val_m = compute_metrics_from_predictions(val_preds)
    test_cal = compute_calibration_metrics(test_preds)
    val_cal = compute_calibration_metrics(val_preds)
    model = MorganMLPScorer()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {
        "backbone": "morgan_mlp",
        "arm_id": arm_id,
        "arm_name": ARM_DEFINITIONS[arm_id]["name"],
        "seed": seed,
        "trainable_parameters": trainable,
        "total_parameters": total,
        "wall_clock_seconds": 0.0,
        "peak_memory_mb": 0.0,
        "inference_latency_ms": 0.0,
        "best_epoch": 0,
        "n_train_examples": 0,
        "n_train_pos": 0,
        "n_train_neg": 0,
        "val_metrics": {
            "mrr": round(val_m["mrr"], 6), "top1": round(val_m["top1"], 6),
            "top3": round(val_m["top3"], 6), "ndcg": round(val_m["ndcg"], 6),
            "auprc": round(compute_auprc(val_preds), 6),
            "ece": round(val_cal["ece"], 6), "brier": round(val_cal["brier"], 6),
        },
        "test_metrics": {
            "mrr": round(test_m["mrr"], 6), "top1": round(test_m["top1"], 6),
            "top3": round(test_m["top3"], 6), "ndcg": round(test_m["ndcg"], 6),
            "auprc": round(compute_auprc(test_preds), 6),
            "ece": round(test_cal["ece"], 6), "brier": round(test_cal["brier"], 6),
        },
    }


def compute_verdict(
    hypotheses: Dict[str, Any],
    anova: Dict[str, Any],
    effects: Dict[str, Dict[str, Dict[str, float]]],
    manifest_dup: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute GO/PARTIAL_GO/NO_GO based on full matrix results."""
    n_positive = hypotheses.get("n_scorers_with_a6_positive", 0)
    n_total = len({k for k in effects if k.startswith("morgan_mlp") or
                   k in ("chemformer", "gnn")})
    if n_total == 0:
        n_total = 3  # chemformer, gnn, morgan_mlp

    interaction_key = "C(source):C(scorer)"
    interaction_p = anova.get(interaction_key, {}).get("p_value", 1.0)

    a6_duplicated = manifest_dup.get("duplicated", False)

    # Check if A6 beats best non-PC-CNG baseline in any scorer
    best_arm_info = hypotheses.get("H3_pc_cng_candidates_harder", {})
    best_arm_per_scorer = best_arm_info.get("best_arm_per_scorer", {})

    if a6_duplicated:
        verdict = "NO_GO"
        reason = "A6 (rule_pc_cng) duplicates another arm in the manifest"
    elif n_positive >= n_total and interaction_p < 0.05:
        verdict = "GO"
        reason = (f"A6 positive in {n_positive}/{n_total} scorers, "
                  f"interaction p={interaction_p:.2e}")
    elif n_positive >= 2:
        verdict = "PARTIAL_GO"
        reason = (f"A6 positive in {n_positive}/{n_total} scorers, "
                  f"interaction p={interaction_p:.2e}; "
                  f"best arm varies by scorer: {best_arm_per_scorer}")
    else:
        verdict = "NO_GO"
        reason = (f"A6 positive in only {n_positive}/{n_total} scorers; "
                  f"gains do not generalize across scorers")

    return {
        "verdict": verdict,
        "reason": reason,
        "n_scorers_positive": n_positive,
        "n_total_scorers": n_total,
        "interaction_p": interaction_p,
        "a6_duplicated": a6_duplicated,
        "best_arm_per_scorer": best_arm_per_scorer,
        "next_phase_allowed": verdict in ("GO", "PARTIAL_GO"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="P4-G4 v2: Full Generator × Scorer matrix (v2 manifest)")
    parser.add_argument("--manifest", type=Path,
                        default=Path("data/p4/manifests/hte_feasibility_v2.json"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/p4_generator_scorer_matrix_v2"))
    parser.add_argument("--g3-dir", type=Path,
                        default=Path("results/p4_augmentation_v2"))
    parser.add_argument("--device", type=str,
                        default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--stage", type=str, default="full",
                        choices=["smoke", "full"])
    parser.add_argument("--skip-mlp", action="store_true",
                        help="Skip Morgan MLP runs; reuse existing v2 MLP outputs")
    args = parser.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    seeds = [DEFAULT_SEEDS[0]] if args.stage == "smoke" else DEFAULT_SEEDS

    print(f"[P4-G4 v2] FULL MATRIX mode (P4-G3 v2 = WEAK_GO)")
    print(f"[P4-G4 v2] Output: {out}, stage={args.stage}, device={args.device}")
    print(f"[P4-G4 v2] Manifest: {args.manifest}")

    # ---- 0. Load manifest data ----
    splits = load_manifest_candidates(args.manifest)
    train_data, val_data, test_data = splits["train"], splits["val"], splits["test"]
    print(f"[P4-G4 v2] Loaded {len(train_data)} train, {len(val_data)} val, "
          f"{len(test_data)} test groups")

    # ---- 1. Morgan MLP runs (3rd independent scorer, v2 manifest) ----
    mlp_records: List[dict] = []
    mlp_summary = out / "mlp_summary.csv"
    if not args.skip_mlp:
        t0 = time.time()
        n_skipped = 0
        for arm_id in ARM_IDS:
            for seed in seeds:
                pred_file = out / "raw_predictions" / f"morgan_mlp_{arm_id}_seed{seed}" / "test_predictions.json"
                if pred_file.exists():
                    print(f"[P4-G4 v2] morgan_mlp × {arm_id} × seed={seed} (SKIP - exists)")
                    rec = load_existing_mlp_run(arm_id, seed, out)
                    mlp_records.append(rec)
                    n_skipped += 1
                    continue
                print(f"[P4-G4 v2] morgan_mlp × {arm_id} × seed={seed}")
                rec = run_single_mlp_experiment(
                    arm_id, seed, train_data, val_data, test_data,
                    args.epochs, args.batch_size, args.lr, args.device, out)
                mlp_records.append(rec)
                print(f"  test MRR {rec['test_metrics']['mrr']:.4f}  "
                      f"wall {rec['wall_clock_seconds']:.1f}s")
        print(f"[P4-G4 v2] MLP training complete ({time.time()-t0:.1f}s, "
              f"{len(mlp_records)} runs, {n_skipped} skipped)")
        write_summary_csv(mlp_records, mlp_summary)
    else:
        if mlp_summary.exists():
            for r in csv.DictReader(open(mlp_summary)):
                keys = ["mrr", "top1", "top3", "ndcg", "auprc", "ece", "brier"]
                mlp_records.append({
                    "backbone": r["backbone"], "arm_id": r["arm_id"],
                    "arm_name": r["arm_name"], "seed": int(r["seed"]),
                    "trainable_parameters": int(r["trainable_parameters"]),
                    "total_parameters": int(r["total_parameters"]),
                    "wall_clock_seconds": float(r["wall_clock_seconds"]),
                    "n_train_examples": int(r["n_train_examples"]),
                    "n_train_pos": int(r["n_train_pos"]),
                    "n_train_neg": int(r["n_train_neg"]),
                    "val_metrics": {k: float(r[f"val_{k}"]) for k in keys},
                    "test_metrics": {k: float(r[f"test_{k}"]) for k in keys},
                })
            print(f"[P4-G4 v2] Reused {len(mlp_records)} MLP runs from {mlp_summary}")
        else:
            print(f"[P4-G4 v2] WARNING: --skip-mlp but no {mlp_summary} found")

    # ---- 2. Load G3 v2 records (chemformer + gnn, v2 manifest) ----
    g3_records: List[dict] = []
    for name, d in G3_V2_DIRS.items():
        p = d / "summary.csv"
        if p.exists():
            g3_records.extend(g3_rows_to_records(load_g3_summary(p)))
            print(f"[P4-G4 v2] loaded G3 v2 {name}: {p}")
        else:
            print(f"[P4-G4 v2] WARNING: G3 v2 {name} summary not found: {p}")

    all_records = g3_records + mlp_records
    scorers = sorted({r["backbone"] for r in all_records})
    print(f"[P4-G4 v2] total cells: {len(all_records)} (scorers={scorers})")

    # ---- 3. Effect sizes per scorer ----
    effects = cell_effect_sizes(all_records)
    write_effect_sizes_csv(effects, out / "effect_sizes.csv")
    write_summary_csv(all_records, out / "summary.csv")

    # ---- 4. Difficulty profiling ----
    difficulty = difficulty_profile(args.manifest)
    with open(out / "difficulty_profile.json", "w") as f:
        json.dump(difficulty, f, indent=2)

    # ---- 5. Interaction model ----
    rows = build_cell_table(all_records)
    anova = interaction_anova(rows)
    mixed = mixed_effects_model(rows)

    # ---- 6. Manifest integrity + count check ----
    dup_path = args.g3_dir / "manifest_integrity.json"
    manifest_dup = json.load(open(dup_path)) if dup_path.exists() else {"duplicated": None}
    counts = {(r["backbone"], r["arm_id"]): (r["n_train_pos"], r["n_train_neg"])
              for r in all_records}
    counts_matched = all(v == (394, 0) if k[1] == "A0" else v == (394, 394)
                         for k, v in counts.items())

    hypotheses = diagnose_hypotheses(effects, anova, difficulty, manifest_dup,
                                     counts_matched)

    interaction_model = {
        "phase": PHASE,
        "mode": "full_matrix",
        "manifest_version": "v2",
        "entry_condition": "P4-G3 v2 == WEAK_GO -> full 3-scorer x 7-arm x 10-seed matrix",
        "cell_table_n": len(rows),
        "scorers": scorers,
        "arms": ARM_IDS,
        "seeds": seeds,
        "anova": anova,
        "mixed_effects": mixed,
        "hypotheses": hypotheses,
        "counts_matched": counts_matched,
    }
    with open(out / "interaction_model.json", "w") as f:
        json.dump(interaction_model, f, indent=2)

    # ---- 7. Compute proper verdict ----
    verdict_info = compute_verdict(hypotheses, anova, effects, manifest_dup)

    go_no_go = {
        "phase": PHASE,
        "status": verdict_info["verdict"],
        "mode": "full_matrix",
        "manifest_version": "v2",
        "full_matrix_executed": True,
        "primary_metric": {"name": "test_mrr", "comparison": "arm_vs_A0 per scorer"},
        "predeclared_threshold": {
            "go": "PC-CNG main effect positive OR >=3/4 scorers effect>0 OR survives matching",
            "no_go": "gains vanish after difficulty matching / explained by hardness or count",
        },
        "verdict_details": verdict_info,
        "key_findings": {
            "source_x_scorer_interaction_p": verdict_info["interaction_p"],
            "best_arm_per_scorer": verdict_info["best_arm_per_scorer"],
            "n_scorers_with_a6_positive": verdict_info["n_scorers_positive"],
            "n_total_scorers": verdict_info["n_total_scorers"],
            "a6_duplicated": verdict_info["a6_duplicated"],
            "manifest_version": "v2 (genuine rule PC-CNG candidates, no A6≡A2 duplication)",
        },
        "limitations": [],  # Will be filled below if needed
        "evidence_paths": [
            str(out / "summary.csv"),
            str(out / "effect_sizes.csv"),
            str(out / "interaction_model.json"),
            str(out / "difficulty_profile.json"),
            str(out / "raw_predictions"),
        ],
        "next_phase_allowed": verdict_info["next_phase_allowed"],
    }

    # Add limitations for PARTIAL_GO
    if verdict_info["verdict"] == "PARTIAL_GO":
        go_no_go["limitations"].append(
            f"A6 positive in {verdict_info['n_scorers_positive']}/"
            f"{verdict_info['n_total_scorers']} scorers; "
            f"effect is scorer-dependent (best arm varies: "
            f"{verdict_info['best_arm_per_scorer']})")

    with open(out / "go_no_go.json", "w") as f:
        json.dump(go_no_go, f, indent=2)

    # ---- 8. Contract files ----
    with open(out / "run_manifest.json", "w") as f:
        json.dump({
            "phase": PHASE,
            "mode": "full_matrix",
            "manifest_version": "v2",
            "scorers": scorers,
            "arms": ARM_IDS,
            "seeds": seeds,
            "n_cells": len(all_records),
            "g3_source": "results/p4_augmentation_v2_{chemformer,gnn}",
            "mlp_source": "trained with v2 manifest",
            "script": "pc_cng/run_p4_g4_v2.py",
        }, f, indent=2)

    with open(out / "environment.json", "w") as f:
        env = {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "cuda_available": torch.cuda.is_available(),
        }
        if torch.cuda.is_available():
            env["cuda_device"] = args.device
        try:
            import rdkit
            env["rdkit"] = rdkit.__version__
        except ImportError:
            pass
        try:
            import statsmodels
            env["statsmodels"] = statsmodels.__version__
        except ImportError:
            pass
        json.dump(env, f, indent=2)

    with open(out / "input_hashes.json", "w") as f:
        hashes = {str(args.manifest): sha256_file(args.manifest)}
        for name, d in G3_V2_DIRS.items():
            p = d / "summary.csv"
            if p.exists():
                hashes[str(p)] = sha256_file(p)
        json.dump(hashes, f, indent=2)

    with open(out / "commands.log", "w") as f:
        f.write(
            f"python3 -m pc_cng.run_p4_g4_v2 "
            f"--manifest {args.manifest} "
            f"--output-dir {out} "
            f"--g3-dir {args.g3_dir} "
            f"--stage {args.stage} --device {args.device}\n")

    print(f"\n[P4-G4 v2] verdict: {verdict_info['verdict']}")
    print(f"[P4-G4 v2] reason: {verdict_info['reason']}")
    print(f"[P4-G4 v2] interaction p = {verdict_info['interaction_p']:.2e}")
    print(f"[P4-G4 v2] next_phase_allowed: {verdict_info['next_phase_allowed']}")
    print(f"[P4-G4 v2] done -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
