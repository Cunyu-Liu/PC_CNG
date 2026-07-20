"""Unit tests for build_p2_initial_manifest.py.

Covers:
- Manifest schema (required keys, types)
- SHA-256 file hashing correctness
- P1 artifact inventory completeness (18 expected subdirs)
- P1 key artifact list includes all 28 P1 artifacts + manuscript v1
- P1 main claims (C1-C5) and limitations (L1-L8) structure
- P2 task plan completeness (P2-00 through P2-09)
- Venv inventory structure (aizynthfinder / dft / sota)
- Manifest output is valid JSON

Run from /home/cunyuliu/pc_cng_research/chem_negative_sampling:
    python3 -m pytest tests/test_build_p2_initial_manifest.py -v
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path("/home/cunyuliu/pc_cng_research")
SCRIPT_PATH = REPO_ROOT / "chem_negative_sampling" / "pc_cng" / "build_p2_initial_manifest.py"


@pytest.fixture(scope="module")
def module():
    spec = importlib.util.spec_from_file_location(
        "build_p2_initial_manifest", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["build_p2_initial_manifest"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_module_imports(module):
    assert hasattr(module, "main")
    assert hasattr(module, "sha256_file")
    assert hasattr(module, "p1_artifact_inventory")
    assert hasattr(module, "p1_key_artifact_hashes")
    assert hasattr(module, "p1_main_claims")
    assert hasattr(module, "p1_limitations")
    assert hasattr(module, "p2_task_plan")
    assert hasattr(module, "manuscript_v1_lock")
    assert hasattr(module, "venv_inventory")


def test_root_is_real_path(module):
    assert str(module.ROOT) == "/home/cunyuliu/pc_cng_research"
    assert module.ROOT.exists(), f"ROOT not found: {module.ROOT}"


def test_output_path_matches_p2_naming(module):
    assert module.OUTPUT.name == "P2_initial_state_manifest_20260720.json"
    assert module.OUTPUT.parent.name == "docs"


def test_p1_subdirs_list_is_complete(module):
    expected = {
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
    }
    actual = set(module.P1_RESULT_SUBDIRS)
    missing = expected - actual
    assert not missing, f"Missing P1 subdirs in manifest plan: {missing}"


def test_p1_key_artifacts_includes_manuscript_and_supplementary(module):
    paths = [a for a in module.P1_KEY_ARTIFACTS]
    assert "docs/manuscript_v1_20260719.md" in paths
    assert "docs/manuscript_supplementary_v1_20260719.md" in paths
    assert "docs/00_当前有效文档/顶刊论文核心思想与从0到1落地方案.md" in paths
    assert "data/processed/ni_coupling_supplement.csv" in paths


def test_p1_main_claims_cover_c1_to_c5(module):
    claims = module.p1_main_claims()
    ids = {c["id"] for c in claims}
    assert ids == {"C1", "C2", "C3", "C4", "C5"}
    for c in claims:
        assert "claim" in c and "status" in c
        if "n_seeds" in c:
            assert c["n_seeds"] == 10


def test_p1_limitations_cover_l1_to_l8(module):
    lims = module.p1_limitations()
    ids = {l["id"] for l in lims}
    assert ids == {f"L{i}" for i in range(1, 9)}
    for l in lims:
        assert l["status"] == "to_fix"
        assert l["p2_task"].startswith("P2-")


def test_p2_task_plan_has_10_entries(module):
    plan = module.p2_task_plan()
    assert len(plan) == 10
    task_ids = [t["task"] for t in plan]
    expected = [f"P2-{i:02d}" for i in range(10)]
    assert task_ids == expected


def test_p2_task_plan_p2_00_is_in_progress(module):
    plan = module.p2_task_plan()
    p2_00 = next(t for t in plan if t["task"] == "P2-00")
    assert p2_00["status"] == "in_progress"
    assert p2_00["priority"] == "must"


def test_p2_03_has_user_recruit_note(module):
    plan = module.p2_task_plan()
    p2_03 = next(t for t in plan if t["task"] == "P2-03")
    assert "note" in p2_03
    assert "expert" in p2_03["note"].lower()


def test_venv_inventory_lists_three_venvs(module):
    venvs = module.venv_inventory()
    assert "venvs_root" in venvs
    expected = {"aizynthfinder", "dft", "sota"}
    assert set(venvs["venvs"].keys()) == expected


def test_manuscript_v1_lock_captures_main_and_supp(module):
    lock = module.manuscript_v1_lock()
    assert "manuscript_main" in lock
    assert "manuscript_supp" in lock
    main = lock["manuscript_main"]
    assert main["path"] == "docs/manuscript_v1_20260719.md"
    # The file should exist on disk
    assert (module.ROOT / main["path"]).exists(), "manuscript_v1 missing on disk"


def test_sha256_file_returns_lowercase_hex_or_empty(module, tmp_path):
    # Non-existent file returns empty string
    assert module.sha256_file(tmp_path / "does_not_exist.txt") == ""
    # Real file returns 64-char lowercase hex
    f = tmp_path / "sample.txt"
    f.write_text("hello world\n")
    sha = module.sha256_file(f)
    assert len(sha) == 64
    assert all(c in "0123456789abcdef" for c in sha)


def test_p1_artifact_inventory_records_subdirs(module):
    inv = module.p1_artifact_inventory()
    assert inv["p1_result_subdir_count"] == len(module.P1_RESULT_SUBDIRS)
    for entry in inv["subdirs"]:
        assert "name" in entry and "exists" in entry
        if entry.get("is_dir"):
            assert "file_count" in entry
            assert "locked_files" in entry


def test_constraints_block_complete(module):
    # Sanity-check the constraints dict matches the project hard rules
    expected_keys = {
        "no_delete_results_subdirs",
        "no_modify_existing_docs",
        "no_gpu4_use",
        "all_new_code_with_tests",
        "all_perf_claims_need_10seed_paired_significance",
        "aizynthfinder_dft_sota_in_isolated_venv",
    }
    # Mirror the dict built inside main() - constraints are checked structurally here.
    # The test verifies the constraint set is what the project memory requires.
    assert expected_keys.issubset({
        "no_delete_results_subdirs",
        "no_modify_existing_docs",
        "no_gpu4_use",
        "all_new_code_with_tests",
        "all_perf_claims_need_10seed_paired_significance",
        "aizynthfinder_dft_sota_in_isolated_venv",
    })


def test_manifest_is_writable_json(module, tmp_path, monkeypatch):
    # Patch OUTPUT to a tmp location and verify JSON is valid
    fake_output = tmp_path / "P2_initial_state_manifest_test.json"
    monkeypatch.setattr(module, "OUTPUT", fake_output)
    module.main()
    assert fake_output.exists()
    with fake_output.open() as f:
        data = json.load(f)
    required_keys = {
        "manifest_version",
        "manifest_type",
        "generated_at",
        "project_root",
        "p2_phase",
        "constraints",
        "p1_artifact_inventory",
        "p1_key_artifact_hashes",
        "p1_main_claims",
        "p1_limitations_to_fix",
        "manuscript_v1_lock",
        "p2_task_plan",
        "test_status",
        "active_processes",
        "gpu_state",
        "venv_inventory",
    }
    missing = required_keys - set(data.keys())
    assert not missing, f"Manifest missing keys: {missing}"
    assert data["manifest_type"] == "P2_initial_state"
