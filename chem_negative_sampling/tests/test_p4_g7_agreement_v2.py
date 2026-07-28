import csv
import json

import pytest

from pc_cng.p4_g7_agreement_v2 import analyze_completed_forms
from pc_cng.p4_g7_sampling_v2 import SCHEMA, SCORING_DIMENSIONS


def _build_pilot(tmp_path):
    items = []
    strata = [
        "positive_control",
        "obvious_negative_control",
        "random_mismatch",
        "rule_pc_cng",
        "learned_structured",
        "shuffled_real",
        "uniform_union",
        "learned_source_gate",
    ]
    for stratum in strata:
        for index in range(10):
            items.append(
                {
                    "blinded_id": f"{stratum}-{index}",
                    "stratum": stratum,
                }
            )
    (tmp_path / "pilot_manifest.json").write_text(
        json.dumps({"schema": SCHEMA, "n_items": 80})
    )
    (tmp_path / "sampling_key_unblinded.json").write_text(
        json.dumps({"schema": SCHEMA, "items": items})
    )
    return items


def _write_completed_form(path, items, reviewer_index, missing=False):
    fields = ["blinded_id", *SCORING_DIMENSIONS, "reason_code", "notes"]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item_index, item in enumerate(items):
            if missing and item_index == 0:
                continue
            stratum = item["stratum"]
            if stratum == "positive_control":
                base = 5
            elif stratum == "obvious_negative_control":
                base = 1
            elif stratum == "learned_source_gate":
                base = 5
            elif stratum == "uniform_union":
                base = 3
            else:
                base = 2
            row = {
                "blinded_id": item["blinded_id"],
                **{dimension: base for dimension in SCORING_DIMENSIONS},
                "reason_code": "",
                "notes": "",
            }
            writer.writerow(row)


def test_complete_agreeing_forms_can_pass_pilot(tmp_path):
    items = _build_pilot(tmp_path)
    forms = []
    for reviewer in range(3):
        path = tmp_path / f"reviewer_{reviewer + 1}.csv"
        _write_completed_form(path, items, reviewer)
        forms.append(path)
    report = analyze_completed_forms(
        tmp_path,
        forms,
        n_bootstrap=300,
        seed=3,
    )
    assert report["verdict"]["pilot_exit_met"] is True
    assert report["verdict"]["main_review_allowed"] is True
    assert report["control_discrimination"]["structural_validity"][
        "ci_all_positive"
    ] is True
    assert report["source_comparisons"]["gate_vs_uniform_union"][
        "ci_all_positive"
    ] is True


def test_incomplete_or_too_few_forms_fail_closed(tmp_path):
    items = _build_pilot(tmp_path)
    forms = []
    for reviewer in range(3):
        path = tmp_path / f"reviewer_{reviewer + 1}.csv"
        _write_completed_form(path, items, reviewer, missing=reviewer == 0)
        forms.append(path)
    with pytest.raises(ValueError, match="form has"):
        analyze_completed_forms(tmp_path, forms, n_bootstrap=50)
    with pytest.raises(ValueError, match="at least three"):
        analyze_completed_forms(tmp_path, forms[1:], n_bootstrap=50)
