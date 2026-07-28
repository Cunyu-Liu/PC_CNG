from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
# The benchmark is launched as ``pc_cng`` but imports the frozen Chemformer
# through the repository namespace.  Preserve both paths in the test so this
# catches the same integration contract as the formal module invocation.
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "chem_negative_sampling"))

from pc_cng.p4_g6_benchmark_v3 import (  # noqa: E402
    CumulativeLinkHead,
    FormalAnalysisPlan,
    T1_PRIMARY_LOW_YIELD_THRESHOLD,
    assert_matched_source_arms,
    _pair_indices,
    load_matched_source_arms,
    normalized_condition_fields,
    partition_context_complete_records,
    source_macro_auprc,
    source_macro_auprc_diagnostics,
    validate_formal_analysis_plan,
    validate_cluster_contract,
    validate_reaction_condition_records,
)
from pc_cng.p4_g6_inference_v3 import (  # noqa: E402
    effect_size_by_seed,
    run_preregistered_primary_inference,
    simulate_inference_operating_characteristics,
)
from pc_cng.paired_cluster_inference import holm_correction  # noqa: E402
from pc_cng.run_p4_g6_v3 import select_stratified_smoke_records  # noqa: E402
from pc_cng.verify_p4_g6_v3 import (  # noqa: E402
    canonicalize_legacy_json_scalars,
    verify_reconstruction,
)


def test_documented_phase_b_cli_imports_without_repository_parent_path():
    """The public module entrypoint must work from chem_negative_sampling/.

    The in-process tests add the repository parent to ``sys.path`` and would
    otherwise hide a broken production import.
    """
    package_root = REPO_ROOT / "chem_negative_sampling"
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import pc_cng.p4_g6_benchmark_v3 as benchmark; "
                "print(benchmark.FORMAL_SCHEMA_VERSION)"
            ),
        ],
        cwd=package_root,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "g6_v3_formal_20260728"


def _record(index: int, *, split: str = "train", yield_value: float = 80.0) -> dict:
    return {
        "record_id": f"R{index}",
        "reactants": "CCO.CN",
        "catalysts": "[Pd]",
        "solvents": "CCO",
        "reagents": "O",
        "products": "CCOC",
        "temperature": 25.0,
        "reaction_time_hrs": 12.0,
        "measured_yield": yield_value,
        "split": split,
        "experimental_group": f"P{index // 2}",
        "plate_id": f"P{index // 2}",
        "reaction_family": "TEST",
        "source_publication": "SOURCE_A",
    }


def test_frozen_plan_rejects_endpoint_or_threshold_changes():
    plan = FormalAnalysisPlan().to_dict()
    validate_formal_analysis_plan(plan)
    changed = dict(plan)
    changed["t1_low_yield_threshold"] = 20.0
    with pytest.raises(ValueError):
        validate_formal_analysis_plan(changed)


def test_condition_contract_keeps_all_fields_and_requires_temperature_time():
    record = _record(0)
    fields = normalized_condition_fields(record)
    assert fields == {"reactants": "CCO.CN", "catalysts": "[Pd]", "solvents": "CCO", "reagents": "O", "products": "CCOC"}
    availability = validate_reaction_condition_records([record] * 10, formal=True)
    assert availability["temperature_fraction"] == 1.0
    bad = dict(record)
    bad["temperature"] = None
    with pytest.raises(ValueError):
        validate_reaction_condition_records([bad] * 10, formal=True)


def test_missing_context_is_explicitly_excluded_not_zero_filled():
    valid = _record(0)
    missing = dict(_record(1))
    missing["reactants"] = ""
    included, excluded = partition_context_complete_records([valid, missing])
    assert [r["record_id"] for r in included] == ["R0"]
    assert excluded == [{"record_id": "R1", "split": "train", "reason": "missing_reactants"}]


def test_cumulative_link_head_returns_valid_ordered_probabilities():
    torch.manual_seed(3)
    head = CumulativeLinkHead(8, n_classes=5)
    features = torch.randn(4, 8)
    probabilities = head.class_probabilities(features)
    assert probabilities.shape == (4, 5)
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(4), atol=1e-5)
    assert torch.all(head.thresholds()[1:] > head.thresholds()[:-1])
    loss = head.loss(features, torch.tensor([0, 1, 2, 4]))
    assert torch.isfinite(loss)


def test_t4_pair_construction_is_within_plate_and_directional():
    records = [_record(0, yield_value=10.0), _record(1, yield_value=80.0)]
    records[0]["plate_id"] = records[1]["plate_id"] = "PLATE_A"
    records.extend([_record(2, yield_value=30.0), _record(3, yield_value=90.0)])
    records[2]["plate_id"] = records[3]["plate_id"] = "PLATE_B"
    pairs = _pair_indices(records, max_pairs=20, seed=7)
    assert pairs
    for high, low, weight in pairs:
        assert records[high]["plate_id"] == records[low]["plate_id"]
        assert records[high]["measured_yield"] > records[low]["measured_yield"]
        assert weight == 1.0


def test_single_source_macro_reports_no_cross_publication_replication():
    records = [
        {"source_publication": "ONE", "label": 1, "score": 0.9},
        {"source_publication": "ONE", "label": 0, "score": 0.1},
    ]
    diagnostics = source_macro_auprc_diagnostics(records)
    assert diagnostics["source_publication_slices_total"] == 1.0
    assert diagnostics["source_publication_slices_evaluable"] == 1.0
    assert diagnostics["source_macro_has_cross_publication_replication"] == 0.0


def test_smoke_sampling_is_deterministic_and_exercises_both_t5_classes():
    records = [
        _record(index, split="test", yield_value=80.0 if index % 5 else 10.0)
        for index in range(30)
    ]
    sampled_a = select_stratified_smoke_records(records, n_records=12, seed=17)
    sampled_b = select_stratified_smoke_records(records, n_records=12, seed=17)
    assert [row["record_id"] for row in sampled_a] == [row["record_id"] for row in sampled_b]
    assert {int(row["measured_yield"] >= 50.0) for row in sampled_a} == {0, 1}


def test_formal_cluster_contract_requires_preexisting_evaluable_groups():
    records = []
    for index in range(20):
        low = _record(index * 2, split="test", yield_value=10.0)
        high = _record(index * 2 + 1, split="test", yield_value=80.0)
        low["experimental_group"] = high["experimental_group"] = f"PLATE_{index}"
        records.extend([low, high])
    contract = validate_cluster_contract(records, formal=True)
    assert contract["cluster_count"] == 20.0
    assert contract["endpoint_evaluable_cluster_count"] == 20.0
    with pytest.raises(ValueError):
        validate_cluster_contract(records[:2], formal=True)


def test_operating_characteristics_reports_familywise_type_i_error():
    result = simulate_inference_operating_characteristics(
        n_simulations=3, n_bootstrap=20, n_permutations=30, seed=22
    )
    assert "familywise_type_i_error" in result
    assert 0.0 <= result["familywise_type_i_error"] <= 1.0


def test_legacy_numpy_boolean_strings_are_normalized_narrowly():
    value = {
        "ci_all_positive": "False",
        "ordinary_text": "False",
        "nested": [{"ci_all_positive": "True"}, "True", 1.0],
    }
    assert canonicalize_legacy_json_scalars(value) == {
        "ci_all_positive": False,
        "ordinary_text": "False",
        "nested": [{"ci_all_positive": True}, "True", 1.0],
    }


def test_complete_primary_inference_reconstruction_verifier():
    plan = FormalAnalysisPlan().to_dict()
    primary = {
        "endpoint": plan["primary_endpoint"],
        "comparisons": [
            {
                "comparison": name,
                "bootstrap": {"ci_all_positive": False},
                "superiority_confirmed": False,
            }
            for name in plan["primary_comparisons"]
        ],
        "baseline_selection": "fixed in analysis plan; no test-data best-baseline selection",
    }
    formal = {
        "scientific_status": "FORMAL",
        "analysis_plan": plan,
        "primary_inference": {
            **primary,
            "comparisons": [
                {
                    **item,
                    "bootstrap": {"ci_all_positive": "False"},
                }
                for item in primary["comparisons"]
            ],
        },
    }
    result = verify_reconstruction(formal, primary, plan)
    assert result["verified"] is True
    assert result["all_superiority_confirmed"] is False
    changed = json.loads(json.dumps(primary))
    changed["comparisons"][0]["superiority_confirmed"] = True
    with pytest.raises(AssertionError, match="differs"):
        verify_reconstruction(formal, changed, plan)


def test_matched_source_arms_share_parents_and_budget(tmp_path: Path):
    records = [_record(index) for index in range(24)]
    groups = []
    for index in range(24):
        candidates = []
        for rank, source in enumerate(("rule_pc_cng", "random_mismatch", "template_perturbation")):
            candidates.append({"candidate_id": f"c{index}_{source}", "candidate_source": source, "candidate_source_rank": rank, "canonical_smiles": f"CC{index % 5}O"})
        groups.append({"split": "train", "source_reaction_id": f"R{index}", "candidates": candidates})
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"groups": groups}))
    arms, audit = load_matched_source_arms(records, manifest)
    assert audit["budget_matched"]
    assert audit["parent_count"] == 24
    assert assert_matched_source_arms(arms)["counts"]["pc_cng"]["n_positive"] == 24


def _scored(seed: int, improvement: float) -> tuple[list[dict], list[dict]]:
    challenger, baseline = [], []
    rng = np.random.default_rng(seed)
    for group in range(6):
        for index in range(8):
            label = int(index % 2 == 0)
            base = float(np.clip(0.5 + (0.2 if label else -0.2) + rng.normal(0, 0.1), 0, 1))
            better = float(np.clip(base + (improvement if label else -improvement), 0, 1))
            row = {"record_id": f"{seed}_{group}_{index}", "label": label, "experimental_group": f"G{group}", "source_publication": "SRC"}
            challenger.append({**row, "score": better})
            baseline.append({**row, "score": base})
    return challenger, baseline


def test_preregistered_inference_has_fixed_comparisons_and_holm():
    ch0, bl0 = _scored(0, 0.08)
    ch1, bl1 = _scored(1, 0.08)
    challenger = {0: ch0, 1: ch1}
    baseline = {0: bl0, 1: bl1}
    result = run_preregistered_primary_inference(
        {"pc_cng": challenger, "random": baseline, "template_rule": baseline, "union": challenger},
        n_bootstrap=80,
        n_permutations=120,
    )
    assert [item["comparison"] for item in result["comparisons"]] == ["pc_cng_vs_random", "pc_cng_vs_template_rule", "union_vs_pc_cng"]
    assert all("holm" in item and "noninferior" in item for item in result["comparisons"])
    assert source_macro_auprc(ch0) > source_macro_auprc(bl0)


def test_holm_adjusted_p_values_use_ordered_cumulative_maximum():
    corrected = holm_correction([0.04, 0.01, 0.03])
    assert [item["adjusted_p"] for item in corrected] == pytest.approx([0.06, 0.03, 0.06])
    assert [item["rejected"] for item in corrected] == [False, True, False]


def test_single_seed_standardized_effect_is_json_null_not_infinity():
    challenger, baseline = _scored(0, 0.08)
    effect = effect_size_by_seed({0: challenger}, {0: baseline})
    assert effect["standardized_seed_effect"] is None
    encoded = json.dumps(effect, allow_nan=False)
    assert "Infinity" not in encoded
