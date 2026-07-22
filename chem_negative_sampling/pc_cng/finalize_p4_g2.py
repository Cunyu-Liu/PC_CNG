"""P4-G2 finalize: copy screening results to spec-required path and generate
selected_backbone.json + go_no_go.json with PARTIAL_GO verdict.

Usage (on remote, from repo root)::

    python3 -m pc_cng.finalize_p4_g2 \
        --screening-dir results/p4_lora_ablation_screening \
        --output-dir results/p4_lora_ablation
"""

from __future__ import annotations

import csv
import json
import shutil
import statistics
import sys
from pathlib import Path
from typing import Dict, List

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CNS_ROOT = _REPO_ROOT / "chem_negative_sampling"
if str(_CNS_ROOT) not in sys.path:
    sys.path.insert(0, str(_CNS_ROOT))

from pc_cng.run_p4_lora_ablation import (  # noqa: E402
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS_FULL,
    DEFAULT_LR,
    NONINFERIORITY_MARGIN,
    compute_checkpoint_hash,
    compute_go_no_go,
    select_best_backbone,
)
from models.pretrained_backbone import DEFAULT_CHECKPOINT_PATH  # noqa: E402


def load_summary_csv(csv_path: Path) -> Dict[str, List[dict]]:
    """Load summary.csv back into the all_results dict structure."""
    all_results: Dict[str, List[dict]] = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = row["config_id"]
            if cid not in all_results:
                all_results[cid] = []
            all_results[cid].append({
                "config_id": cid,
                "config_name": row["config_name"],
                "seed": int(row["seed"]),
                "trainable_parameters": int(row["trainable_parameters"]),
                "total_parameters": int(row["total_parameters"]),
                "param_ratio": float(row["param_ratio"]),
                "wall_clock_seconds": float(row["wall_clock_seconds"]),
                "peak_memory_mb": float(row["peak_memory_mb"]),
                "inference_latency_ms": float(row["inference_latency_ms"]),
                "val_metrics": {
                    "mrr": float(row["val_mrr"]),
                    "top1": float(row["val_top1"]),
                    "top3": float(row["val_top3"]),
                    "ndcg": float(row["val_ndcg"]),
                    "ece": float(row["val_ece"]),
                    "brier": float(row["val_brier"]),
                },
                "test_metrics": {
                    "mrr": float(row["test_mrr"]),
                    "top1": float(row["test_top1"]),
                    "top3": float(row["test_top3"]),
                    "ndcg": float(row["test_ndcg"]),
                    "ece": float(row["test_ece"]),
                    "brier": float(row["test_brier"]),
                },
            })
    return all_results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="P4-G2 finalize")
    parser.add_argument("--screening-dir", type=Path,
                        default=Path("results/p4_lora_ablation_screening"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/p4_lora_ablation"))
    args = parser.parse_args()

    screening = args.screening_dir
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    # 1. Copy static artifacts
    for fname in ["config_registry.json", "summary.csv", "noninferiority.json"]:
        src = screening / fname
        dst = output / fname
        if src.exists():
            shutil.copy2(src, dst)
            print(f"  Copied {fname}")

    # Copy raw_predictions/ and per_seed/ directories
    for subdir in ["raw_predictions", "per_seed"]:
        src = screening / subdir
        dst = output / subdir
        if src.exists() and not dst.exists():
            shutil.copytree(src, dst)
            print(f"  Copied {subdir}/")

    # 2. Load data
    with open(output / "config_registry.json") as f:
        config_registry = json.load(f)
    with open(output / "noninferiority.json") as f:
        noninferiority = json.load(f)
    all_results = load_summary_csv(output / "summary.csv")

    print(f"\nLoaded {len(all_results)} configs:")
    for cid, results in all_results.items():
        if results:
            mean_mrr = statistics.mean([r["test_metrics"]["mrr"] for r in results])
            print(f"  {cid}: n_seeds={len(results)}, mean_test_mrr={mean_mrr:.4f}, "
                  f"params={results[0]['trainable_parameters']}")

    # 3. Select best backbone with PARTIAL_GO fallback
    selected = select_best_backbone(
        config_registry, all_results, noninferiority,
        allow_partial_go=True,
    )
    if selected:
        sb_path = output / "selected_backbone.json"
        with open(sb_path, "w") as f:
            json.dump(selected, f, indent=2)
        print(f"\nselected_backbone.json: {sb_path}")
        print(f"  Config: {selected['config_id']}")
        print(f"  MRR: {selected['mean_mrr']}")
        print(f"  Trainable params: {selected['trainable_parameters']}")
        print(f"  Non-inferior: {selected.get('is_noninferior', False)}")
        print(f"  Selection rule: {selected['selection_rule']}")

        # Verify all 11 required fields from spec
        required_fields = [
            "checkpoint", "checkpoint_hash", "architecture", "target_modules",
            "rank", "alpha", "dropout", "trainable_parameters", "training_budget",
            "selection_metric", "selection_rule",
        ]
        missing = [f for f in required_fields if f not in selected]
        if missing:
            print(f"  WARNING: Missing required fields: {missing}")
        else:
            print(f"  All 11 required fields present")
    else:
        print("\nERROR: No backbone selected (NO-GO)")
        return 1

    # 4. Compute GO/NO-GO
    full_ft_results = all_results.get("C6", [])
    go_no_go = compute_go_no_go(all_results, noninferiority, selected, full_ft_results)

    if go_no_go["status"] == "PARTIAL_GO":
        go_no_go["partial_go_conditions"] = {
            "reason": "LoRA slightly below full fine-tuning but efficiency advantage clear (>=10x fewer params)",
            "requirement": "Formal augmentation main results MUST simultaneously report full fine-tuning sensitivity",
            "selected_config": selected["config_id"],
            "param_efficiency_ratio": (
                full_ft_results[0]["trainable_parameters"] / selected["trainable_parameters"]
                if full_ft_results else None
            ),
        }

    go_path = output / "go_no_go.json"
    with open(go_path, "w") as f:
        json.dump(go_no_go, f, indent=2)
    print(f"\ngo_no_go.json: {go_path}")
    print(f"  Status: {go_no_go['status']}")
    print(f"  Next phase allowed: {go_no_go['next_phase_allowed']}")
    if "partial_go_conditions" in go_no_go:
        pgc = go_no_go["partial_go_conditions"]
        print(f"  Param efficiency ratio: {pgc['param_efficiency_ratio']:.1f}x")

    return 0


if __name__ == "__main__":
    sys.exit(main())
