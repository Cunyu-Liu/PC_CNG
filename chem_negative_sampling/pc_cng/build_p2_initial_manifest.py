"""Generate P2 initial state manifest for PC-CNG project.

Captures the project state at P2 startup (2026-07-20) for later comparison.
Records: P1 artifact SHA-256 inventory, P1 test suite status,
manuscript v1 SHA-256, active processes, GPU state, P2 task plan.

Outputs:
- docs/P2_initial_state_manifest_20260720.json
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
OUTPUT = ROOT / "docs" / "P2_initial_state_manifest_20260720.json"

# P1 result subdirs (full inventory from Section 26 P2 inputs)
P1_RESULT_SUBDIRS = [
    "cross_dataset_transfer_20260719",
    "calibration_error_10seed_20260719",
    "ood_scaffold_template_split_20260719",
    "retrosynthesis_route_ranking_20260719",
    "false_negative_three_layer_20260719",
    "ord_data_quality_audit_20260719",
    "xtb_dft_validation_20260719",
    "external_calibration_heldout_full_beam_5k_20260719",
    "external_calibration_heldout_full_beam_mlp_apply_5k_20260719",
    "external_calibration_heldout_full_beam_paired_significance_5k_20260719",
    "expert_review_20260719",
    "manuscript_tables_p1_baseline_20260719",
    "P1_initial_state_20260719",
    "semi_hard_curriculum_10seed_20260719",
    "semi_hard_curriculum_smoke_20260719",
    "failure_prototype_calibration_smoke_20260719",
    "cross_dataset_transfer_smoke_20260719",
    "cross_dataset_transfer_smoke2_20260719",
]

# P1 key artifacts (top-level summaries for SHA-256 locking)
P1_KEY_ARTIFACTS = [
    "results/cross_dataset_transfer_20260719/multiseed_paired_significance_summary.json",
    "results/calibration_error_10seed_20260719/calibration_error_summary.json",
    "results/ood_scaffold_template_split_20260719/ood_summary.json",
    "results/retrosynthesis_route_ranking_20260719/route_ranking_summary.json",
    "results/false_negative_three_layer_20260719/three_layer_summary.json",
    "results/ord_data_quality_audit_20260719/ord_audit_summary.json",
    "results/xtb_dft_validation_20260719/xtb_dft_summary.json",
    "results/external_calibration_heldout_full_beam_5k_20260719/benchmark_summary.json",
    "results/external_calibration_heldout_full_beam_mlp_apply_5k_20260719/apply_summary.json",
    "results/external_calibration_heldout_full_beam_paired_significance_5k_20260719/paired_significance.json",
    "results/expert_review_20260719/sampled_for_review.csv",
    "results/manuscript_tables_p1_baseline_20260719/manifest.json",
    "results/P1_initial_state_20260719/p1_summary.json",
    "results/semi_hard_curriculum_10seed_20260719/curriculum_summary.json",
    "data/processed/ni_coupling_supplement.csv",
    "docs/manuscript_v1_20260719.md",
    "docs/manuscript_supplementary_v1_20260719.md",
    "docs/00_当前有效文档/顶刊论文核心思想与从0到1落地方案.md",
    "docs/P1_initial_state_manifest_20260719.json",
    "docs/file_reorganization_manifest_20260719.md",
    "docs/PC-CNG-v3-reproducibility-manifest-20260712.json",
    "docs/PC-CNG-v3-benchmark-manifest-20260712.json",
]


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


def p1_artifact_inventory() -> Dict[str, object]:
    """SHA-256 lock for every P1 result subdir + key artifact."""
    inventory: List[Dict[str, object]] = []
    for subdir in P1_RESULT_SUBDIRS:
        path = ROOT / "results" / subdir
        entry: Dict[str, object] = {
            "name": subdir,
            "exists": path.exists(),
            "is_dir": path.is_dir() if path.exists() else False,
        }
        if path.exists() and path.is_dir():
            files = sorted([p for p in path.rglob("*") if p.is_file()])
            entry["file_count"] = len(files)
            entry["size_bytes_total"] = sum(p.stat().st_size for p in files)
            # SHA-256 of every .json/.csv/.md/.log under subdir
            locked_files = []
            for f in files:
                if f.suffix in {".json", ".csv", ".md", ".log", ".yaml", ".yml"}:
                    locked_files.append({
                        "rel_path": str(f.relative_to(path)),
                        "sha256": sha256_file(f),
                        "size_bytes": f.stat().st_size,
                    })
            entry["locked_files"] = locked_files
        inventory.append(entry)
    return {
        "p1_result_subdir_count": len(inventory),
        "subdirs": inventory,
    }


def p1_key_artifact_hashes() -> List[Dict[str, str]]:
    out = []
    for rel in P1_KEY_ARTIFACTS:
        p = ROOT / rel
        sha = sha256_file(p)
        out.append({
            "path": rel,
            "exists": p.exists(),
            "sha256": sha,
            "size_bytes": p.stat().st_size if p.exists() and p.is_file() else 0,
        })
    return out


def p1_main_claims() -> List[Dict[str, str]]:
    """Main claims from manuscript v1 (locked for P2 comparison)."""
    return [
        {
            "id": "C1",
            "claim": "PC-CNG delivers statistically significant cross-dataset migration gain from RegioSQM20 to USPTO",
            "metric": "delta Top-1 = 1.63 pp",
            "ci": "[0.59, 2.72]",
            "pvalue": "0.0028",
            "n_seeds": 10,
            "status": "P1 main claim",
        },
        {
            "id": "C2",
            "claim": "PC-CNG improves retrosynthesis route ranking",
            "metric": "MRR 24.24% -> 54.87%, delta = 30.63 pp",
            "ci": "[29.23, 32.05] pp",
            "pvalue": "< 0.0001",
            "n_seeds": 10,
            "status": "P1 main claim",
        },
        {
            "id": "C3",
            "claim": "Calibration is acceptable",
            "metric": "ECE = 0.0889",
            "n_seeds": 10,
            "status": "P1 main claim",
        },
        {
            "id": "C4",
            "claim": "OOD scaffold/template splits show no significant degradation",
            "n_seeds": 10,
            "status": "P1 main claim",
        },
        {
            "id": "C5",
            "claim": "Reproducibility manifest covers 28 result artifacts + Ni-coupling 1688 reactions",
            "status": "P1 main claim",
        },
    ]


def p1_limitations() -> List[Dict[str, str]]:
    """P1 limitations to be addressed by P2 tasks (L1-L8 from Section 26)."""
    return [
        {"id": "L1", "name": "External bridge NO-GO", "issue": "MLP calibrator underperforms Chemformer LL by 10.56 pp", "p2_task": "P2-04", "status": "to_fix"},
        {"id": "L2", "name": "Retrosynthesis pseudo-route fallback", "issue": "P1-04 used pseudo-route instead of AiZynthFinder real routes", "p2_task": "P2-01", "status": "to_fix"},
        {"id": "L3", "name": "DFT partial support", "issue": "MMFF94 support rate 0.48 < 0.6 threshold", "p2_task": "P2-02", "status": "to_fix"},
        {"id": "L4", "name": "Expert review not executed", "issue": "P1-08 Layer 3 used rule-based fallback", "p2_task": "P2-03", "status": "to_fix"},
        {"id": "L5", "name": "Cross-dataset transfer weak", "issue": "Only 1/4 migration pairs significant", "p2_task": "P2-05", "status": "to_fix"},
        {"id": "L6", "name": "No SOTA direct comparison", "issue": "Missing LocalRetro/Graph2SMILES/Molecular Transformer baselines", "p2_task": "P2-06", "status": "to_fix"},
        {"id": "L7", "name": "GNN decoder not better than rules", "issue": "GNN learned decoder did not exceed rule-based", "p2_task": "P2-07", "status": "to_fix"},
        {"id": "L8", "name": "Downstream coverage insufficient", "issue": "No condition prediction eval", "p2_task": "P2-08", "status": "to_fix"},
    ]


def p2_task_plan() -> List[Dict[str, str]]:
    """Full P2 task plan with Go/No-Go criteria."""
    return [
        {"task": "P2-00", "title": "P1 结果整合与 manuscript v2 基线锁定", "priority": "must", "status": "in_progress"},
        {"task": "P2-01", "title": "AiZynthFinder 真实路线对比 (L2 最高优先级)", "priority": "high", "status": "pending"},
        {"task": "P2-02", "title": "DFT 验证 chemoselectivity_error 子集 (L3)", "priority": "high", "status": "pending"},
        {"task": "P2-03", "title": "专家审查实际执行 (L4)", "priority": "medium", "status": "pending", "note": "Requires user to recruit 2-3 chemistry experts"},
        {"task": "P2-04", "title": "MLP calibrator v2 chemformer-aware (L1)", "priority": "high", "status": "pending"},
        {"task": "P2-05", "title": "Cross-dataset 迁移扩大 (L5)", "priority": "medium", "status": "pending"},
        {"task": "P2-06", "title": "SOTA 多基线对比 (L6)", "priority": "high", "status": "pending"},
        {"task": "P2-07", "title": "Transformer-based generator 消融 (L7)", "priority": "medium", "status": "pending"},
        {"task": "P2-08", "title": "反应条件预测下游任务 (L8)", "priority": "medium", "status": "pending"},
        {"task": "P2-09", "title": "Manuscript v2 + 投稿准备", "priority": "high", "status": "pending"},
    ]


def test_status() -> Dict[str, object]:
    out = run(
        [PY, "-m", "pytest", "tests/", "-q", "--tb=no", "--disable-warnings"],
        cwd=ROOT / "chem_negative_sampling",
        timeout=180,
    )
    return {
        "command": "pytest tests/ -q --tb=no",
        "output_tail": out[-1500:],
        "passed": bool(out and "passed" in out.lower() and "failed" not in out.lower()),
    }


def active_processes() -> Dict[str, object]:
    checks = {
        "calibrate_pid_2544995": 2544995,
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


def venv_inventory() -> Dict[str, object]:
    """Check if P2-required venvs exist (aizynthfinder / dft / sota)."""
    venvs_root = Path("/home/cunyuliu/venvs")
    expected = ["aizynthfinder", "dft", "sota"]
    out = {}
    for name in expected:
        path = venvs_root / name
        out[name] = {
            "expected_path": str(path),
            "exists": path.exists(),
            "has_python": (path / "bin" / "python").exists() if path.exists() else False,
        }
    return {
        "venvs_root": str(venvs_root),
        "venvs_root_exists": venvs_root.exists(),
        "venvs": out,
    }


def manuscript_v1_lock() -> Dict[str, object]:
    """Lock manuscript v1 (main + supplementary + figures dir)."""
    paths = {
        "manuscript_main": "docs/manuscript_v1_20260719.md",
        "manuscript_supp": "docs/manuscript_supplementary_v1_20260719.md",
        "manuscript_dir": "docs/manuscript_v1_20260719",
    }
    out = {}
    for key, rel in paths.items():
        p = ROOT / rel
        if p.is_file():
            out[key] = {
                "path": rel,
                "sha256": sha256_file(p),
                "size_bytes": p.stat().st_size,
            }
        elif p.is_dir():
            files = sorted([x for x in p.rglob("*") if x.is_file()])
            out[key] = {
                "path": rel,
                "is_dir": True,
                "file_count": len(files),
                "size_bytes_total": sum(x.stat().st_size for x in files),
            }
        else:
            out[key] = {"path": rel, "exists": False}
    return out


def main() -> None:
    manifest = {
        "manifest_version": "1.0",
        "manifest_type": "P2_initial_state",
        "generated_at": datetime.now().astimezone().isoformat(),
        "project_root": str(ROOT),
        "p2_phase": "P2 startup baseline",
        "constraints": {
            "no_delete_results_subdirs": True,
            "no_modify_existing_docs": True,
            "no_gpu4_use": True,
            "all_new_code_with_tests": True,
            "all_perf_claims_need_10seed_paired_significance": True,
            "aizynthfinder_dft_sota_in_isolated_venv": True,
        },
        "p1_artifact_inventory": p1_artifact_inventory(),
        "p1_key_artifact_hashes": p1_key_artifact_hashes(),
        "p1_main_claims": p1_main_claims(),
        "p1_limitations_to_fix": p1_limitations(),
        "manuscript_v1_lock": manuscript_v1_lock(),
        "p2_task_plan": p2_task_plan(),
        "test_status": test_status(),
        "active_processes": active_processes(),
        "gpu_state": gpu_state(),
        "venv_inventory": venv_inventory(),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"Manifest written to {OUTPUT}")
    print(f"  p1 result subdirs locked: {manifest['p1_artifact_inventory']['p1_result_subdir_count']}")
    print(f"  p1 key artifacts hashed: {len(manifest['p1_key_artifact_hashes'])}")
    print(f"  p1 main claims recorded: {len(manifest['p1_main_claims'])}")
    print(f"  p1 limitations tracked: {len(manifest['p1_limitations_to_fix'])}")
    print(f"  p2 tasks planned: {len(manifest['p2_task_plan'])}")
    print(f"  tests passed: {manifest['test_status']['passed']}")
    alive = sum(1 for v in manifest["active_processes"].values() if v["alive"])
    print(f"  active processes alive: {alive}")


if __name__ == "__main__":
    main()
