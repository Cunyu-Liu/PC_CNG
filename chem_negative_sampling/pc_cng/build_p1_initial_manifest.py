"""Generate P1 initial state manifest for PC-CNG project.

Captures the project state at P1 startup (2026-07-19) for later comparison.
Records: timestamp, result subdirs, key summary SHA-256, best models,
manuscript tables, test status, active processes, GPU state.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


ROOT = Path("/home/cunyuliu/pc_cng_research")
PY = "/home/cunyuliu/miniconda3/envs/pc_cng_gpu/bin/python"
OUTPUT = ROOT / "docs" / "P1_initial_state_manifest_20260719.json"


def sha256_file(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def run(cmd: List[str], cwd: Optional[Path] = None, timeout: int = 60) -> str:
    try:
        result = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.stdout.strip() or result.stderr.strip()
    except Exception as exc:
        return f"ERROR: {exc}"


def count_result_subdirs() -> Dict[str, object]:
    results_dir = ROOT / "results"
    subdirs = [p for p in results_dir.iterdir() if p.is_dir()]
    return {
        "total_subdirs": len(subdirs),
        "subdir_names": sorted([p.name for p in subdirs]),
    }


def key_summary_hashes() -> List[Dict[str, str]]:
    """SHA-256 of key summary.json files for reproducibility tracking."""
    key_paths = [
        "results/type1_diverse_anchor_ablation_full/paper_summary/paper_table.md",
        "results/type1_diverse_anchor_dpo_reference/paper_summary/paper_table.md",
        "results/type1_diverse_anchor_dpo_beta_sweep_full/paper_summary/paper_table.md",
        "results/type2_low_yield_branch_full/paper_summary/paper_table.md",
        "results/science_advances_dpo_reward_benchmark_full/paper_summary_10seed/paper_table.md",
        "results/science_advances_regiosqm_benchmark_full/paper_summary_10seed/paper_table.md",
        "results/type1_unreacted_substrate_supplement_v2_20260711/v2_unreacted_multiseed_summary/summary.json",
        "results/type1_combined_feature_v2_20260712/combined_feature_multiseed_summary/summary.json",
        "results/type1_curated_weak_class_contexts_20260711/curated_weak_class_contexts_multiseed_summary/summary.json",
        "results/type1_v2_hidden4096_20260712/hidden4096_multiseed_summary/summary.json",
        "results/type1_v2_dropout04_20260712/dropout04_multiseed_summary/summary.json",
        "results/type1_v2_coslr_warm5_20260712/coslr_warm5_multiseed_summary/summary.json",
        "results/type1_v2_filtered_baseline_20260712/filtered_baseline_multiseed_summary/summary.json",
        "results/type1_v2_nbits8192_10seed_20260714/nbits8192_10seed_summary/summary.json",
        "results/type1_v2_pairwise_margin_10seed_20260714/pw20_m000_10seed_summary/summary.json",
        "results/type1_v2_pairwise_margin_10seed_20260714/pw20_m005_10seed_summary/summary.json",
        "results/external_product_prediction_benchmark_25k_repaired_20260713/benchmark/benchmark_summary.json",
        "results/external_product_prediction_benchmark_25k_repaired_20260713/benchmark_validity_aware/benchmark_summary.json",
        "results/external_score_calibration_25k_repaired_strict_20260715/external_score_calibration_summary.json",
        "results/external_score_calibration_25k_repaired_validity_aware_20260715/external_score_calibration_summary.json",
        "results/external_calibration_heldout_base_scored_5k_20260716/benchmark_summary.json",
        "results/external_calibration_heldout_chemformer_ll_full_5k_20260717/benchmark_summary.json",
        "results/manuscript_tables_pc_cng_v3/manifest.json",
        "results/manuscript_tables_p1_baseline_20260719/manifest.json",
        "docs/00_当前有效文档/顶刊论文核心思想与从0到1落地方案.md",
        "docs/file_reorganization_manifest_20260719.md",
        "docs/PC-CNG-v3-reproducibility-manifest-20260712.json",
        "docs/PC-CNG-v3-benchmark-manifest-20260712.json",
    ]
    out = []
    for rel in key_paths:
        p = ROOT / rel
        sha = sha256_file(p)
        out.append({
            "path": rel,
            "exists": p.exists(),
            "sha256": sha,
            "size_bytes": p.stat().st_size if p.exists() and p.is_file() else 0,
        })
    return out


def best_models() -> List[Dict[str, str]]:
    return [
        {
            "name": "type1_diverse_anchor_pairwise_default",
            "description": "Main Type-1 same-context reranking model",
            "checkpoint_dir": "results/type1_diverse_anchor_full",
            "overall_top1": "97.49 +/- 0.06",
            "test_top1": "85.07 +/- 0.94",
            "seeds": "20260710-20260719 (10 seeds)",
        },
        {
            "name": "type1_v2_unreacted_expanded",
            "description": "v2/unreacted substrate supplement, main baseline for ablations",
            "checkpoint_dir": "results/type1_unreacted_substrate_supplement_v2_20260711",
            "test_top1": "87.16 +/- 1.58",
            "seeds": "20260710-20260719 (10 seeds)",
        },
        {
            "name": "type1_combined_feature",
            "description": "Combined Morgan+GraphStats, expanded curated scope winner",
            "checkpoint_dir": "results/type1_combined_feature_v2_20260712",
            "expanded_curated_test_top1": "76.61 +/- 1.94",
            "paired_significance_p": "< 0.0001",
            "seeds": "20260710-20260719 (10 seeds)",
        },
        {
            "name": "type1_classw050_rc",
            "description": "Curated weak-class contexts (Amide/Cu fix)",
            "checkpoint_dir": "results/type1_curated_weak_class_contexts_20260711",
            "expanded_curated_test_top1": "97.16 +/- 0.30",
            "seeds": "20260710-20260719 (10 seeds)",
        },
        {
            "name": "external_mlp_calibrator_v1",
            "description": "Frozen MLP recipe for held-out 5k (pending full-beam eval)",
            "checkpoint_dir": "results/external_score_mlp_calibrator_v1_repaired25k_train_20260715",
            "val_top1": "93.14",
            "heldout_base_top1": "94.51 (base-only diagnostic, not full-beam)",
        },
    ]


def test_status() -> Dict[str, object]:
    out = run(
        [PY, "-m", "pytest", "tests/", "-q", "--tb=no", "--disable-warnings"],
        cwd=ROOT / "chem_negative_sampling",
        timeout=120,
    )
    return {
        "command": "pytest tests/ -q --tb=no",
        "output_tail": out[-500:],
        "passed": "passed" in out and "failed" not in out.lower(),
    }


def active_processes() -> Dict[str, object]:
    checks = {
        "calibrate_pid_2544995": 2544995,
        "rf_cf5_scheduler_pid_1437378": 1437378,
        "rf_cf5_child_pid_2042374": 2042374,
    }
    results = {}
    for name, pid in checks.items():
        out = run(["ps", "-p", str(pid), "-o", "pid,etime,time,pcpu,rss", "--no-headers"])
        results[name] = {
            "pid": pid,
            "alive": bool(out and "NOT FOUND" not in out and "ERROR" not in out),
            "info": out,
        }
    return results


def gpu_state() -> str:
    return run([
        "nvidia-smi",
        "--query-gpu=index,utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader",
    ])


def manuscript_tables_comparison() -> Dict[str, object]:
    """Compare original vs P1-baseline manuscript tables for reproducibility."""
    orig_manifest = ROOT / "results" / "manuscript_tables_pc_cng_v3" / "manifest.json"
    new_manifest = ROOT / "results" / "manuscript_tables_p1_baseline_20260719" / "manifest.json"
    if not orig_manifest.exists() or not new_manifest.exists():
        return {"status": "manifests_missing"}
    with orig_manifest.open() as f:
        orig = json.load(f)
    with new_manifest.open() as f:
        new = json.load(f)
    # Compare row counts for each table
    diffs = []
    matches = []
    for key, orig_entry in orig.items():
        new_entry = new.get(key, {})
        orig_rows = orig_entry.get("rows", 0)
        new_rows = new_entry.get("rows", 0)
        if orig_rows != new_rows:
            diffs.append({"table": key, "orig_rows": orig_rows, "new_rows": new_rows})
        else:
            matches.append(key)
    return {
        "orig_table_count": len(orig),
        "new_table_count": len(new),
        "matching_tables": matches,
        "diff_tables": diffs,
        "reproducible": len(diffs) == 0,
    }


def main() -> None:
    manifest = {
        "manifest_version": "1.0",
        "generated_at": datetime.now().astimezone().isoformat(),
        "project_root": str(ROOT),
        "p1_phase": "P1 startup baseline",
        "constraints": {
            "no_delete_results": True,
            "no_modify_existing_docs": True,
            "no_gpu4_use": True,
            "all_new_code_with_tests": True,
            "all_perf_claims_need_10seed_paired_significance": True,
        },
        "results_inventory": count_result_subdirs(),
        "key_artifact_hashes": key_summary_hashes(),
        "best_models": best_models(),
        "manuscript_tables_reproducibility": manuscript_tables_comparison(),
        "test_status": test_status(),
        "active_processes": active_processes(),
        "gpu_state": gpu_state(),
        "file_reorganization": {
            "manifest_path": "docs/file_reorganization_manifest_20260719.md",
            "manifest_sha256": sha256_file(ROOT / "docs" / "file_reorganization_manifest_20260719.md"),
            "scripts_moved_to_scripts_dir": 21,
            "docs_moved_to_chem_negative_sampling_docs": 5,
            "archive_dir_created": "docs/archive_20260719/ (empty)",
        },
        "p1_tasks_status": {
            "P1-00": "in_progress (this manifest)",
            "P1-01": "pending",
            "P1-02": "pending",
            "P1-03": "pending",
            "P1-04": "pending",
            "P1-05": "pending",
            "P1-06": "pending",
            "P1-07": "pending",
            "P1-08": "pending",
            "P1-09": "pending",
            "P1-10": "pending",
            "P1-11": "pending",
            "P1-12": "pending",
            "P1-13": "partially_done (file moves complete; README + deprecated tags pending)",
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"Manifest written to {OUTPUT}")
    print(f"  result subdirs: {manifest['results_inventory']['total_subdirs']}")
    print(f"  key artifacts hashed: {len(manifest['key_artifact_hashes'])}")
    print(f"  best models recorded: {len(manifest['best_models'])}")
    print(f"  manuscript tables reproducible: {manifest['manuscript_tables_reproducibility']['reproducible']}")
    print(f"  tests passed: {manifest['test_status']['passed']}")
    print(f"  active processes alive: {sum(1 for v in manifest['active_processes'].values() if v['alive'])}")


if __name__ == "__main__":
    main()
