"""Tests for pc_cng.execute_expert_review (P2-03).

Covers:
- Module imports
- CLI args parsing
- Form generation (reviewer CSV files with correct columns)
- Likert rating parsing
- Cohen's kappa computation (mocked data)
- Fleiss' kappa computation (mocked data)
- Pass rate computation
- Failure mode distribution
- Deferred status documentation
- Synthetic small end-to-end test with 10 samples
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import List

import pytest

from pc_cng import execute_expert_review as ee


# ---------- Fixtures ----------

SAMPLE_HEADER = [
    "sample_id",
    "reaction_smiles",
    "parent_reaction_smiles",
    "failure_type",
    "task",
    "source_origin",
    "true_label",
    "chemical_validity",
    "mechanistic_plausibility",
    "side_product_likelihood",
    "feasibility_score",
    "overall_verdict",
    "comment",
    "reviewer_id",
    "review_timestamp",
]


def _make_sample_csv(path: Path, n: int = 10) -> Path:
    """Write a small synthetic sampled_for_review.csv."""
    rows = []
    for i in range(1, n + 1):
        rows.append({
            "sample_id": f"S{i:04d}",
            "reaction_smiles": f"CCO{i}>>CC(=O)O{i}",
            "parent_reaction_smiles": f"CCO{i}>>CC(=O)O{i}",
            "failure_type": "no_reaction" if i % 2 == 0 else "side_product",
            "task": "forward_outcome",
            "source_origin": "pc_cng_synthetic",
            "true_label": "0",
            "chemical_validity": "",
            "mechanistic_plausibility": "",
            "side_product_likelihood": "",
            "feasibility_score": "",
            "overall_verdict": "",
            "comment": "",
            "reviewer_id": "",
            "review_timestamp": "",
        })
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=SAMPLE_HEADER)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return path


@pytest.fixture
def sample_csv(tmp_path: Path) -> Path:
    return _make_sample_csv(tmp_path / "sampled_for_review.csv", n=10)


@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    d = tmp_path / "expert_review_out"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------- Module imports ----------

def test_module_imports():
    """Module exposes expected public API."""
    assert hasattr(ee, "main")
    assert hasattr(ee, "run_prepare")
    assert hasattr(ee, "run_aggregate")
    assert hasattr(ee, "build_arg_parser")
    assert hasattr(ee, "LIKERT_COLUMNS")


def test_constants():
    """Constants are correctly defined."""
    assert "chemical_validity" in ee.LIKERT_COLUMNS
    assert "mechanistic_plausibility" in ee.LIKERT_COLUMNS
    assert "side_product_likelihood" in ee.LIKERT_COLUMNS
    assert "feasibility" in ee.LIKERT_COLUMNS
    assert "overall_verdict" in ee.LIKERT_COLUMNS
    assert len(ee.LIKERT_COLUMNS) == 5
    assert "deferred" in ee.DEFERRED_STATUS.lower()
    assert "revision" in ee.DEFERRED_STATUS.lower()


# ---------- CLI args ----------

def test_cli_args_default():
    parser = ee.build_arg_parser()
    args = parser.parse_args(["--output-dir", "/tmp/out"])
    assert args.mode == "prepare"
    assert args.reviewer_count == 2
    assert args.output_dir == Path("/tmp/out")
    assert args.filled_forms_dir is None
    assert args.samples == Path("results/expert_review_20260719/sampled_for_review.csv")


def test_cli_args_aggregate_parses():
    parser = ee.build_arg_parser()
    args = parser.parse_args([
        "--output-dir", "/tmp/out",
        "--mode", "aggregate",
        "--filled-forms-dir", "/tmp/filled",
    ])
    assert args.mode == "aggregate"
    assert args.filled_forms_dir == Path("/tmp/filled")


def test_cli_args_invalid_choice_mode_exits():
    parser = ee.build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--output-dir", "/tmp/out", "--mode", "bogus"])


def test_cli_args_missing_output_dir_exits():
    parser = ee.build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_main_invalid_reviewer_count_exits(tmp_path):
    with pytest.raises(SystemExit):
        ee.main([
            "--output-dir", str(tmp_path / "out"),
            "--reviewer-count", "4",
        ])


def test_main_aggregate_without_filled_forms_dir_exits(tmp_path):
    with pytest.raises(SystemExit):
        ee.main([
            "--output-dir", str(tmp_path / "out"),
            "--mode", "aggregate",
        ])


# ---------- Form generation ----------

def test_prepare_generates_2_reviewer_forms(sample_csv, output_dir):
    result = ee.run_prepare(sample_csv, reviewer_count=2, output_dir=output_dir)
    assert result["mode"] == "prepare"
    assert result["n_samples"] == 10
    assert len(result["reviewer_forms"]) == 2
    forms_dir = output_dir / "reviewer_forms"
    assert (forms_dir / "reviewer_1_form.csv").exists()
    assert (forms_dir / "reviewer_2_form.csv").exists()
    # Check columns
    with open(forms_dir / "reviewer_1_form.csv", "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        assert len(rows) == 10
        for col in ee.FORM_COLUMNS:
            assert col in reader.fieldnames, f"missing column {col}"
        # Likert columns must be blank
        for row in rows:
            for col in ee.LIKERT_COLUMNS:
                assert row[col] == "", f"Likert column {col} should be blank"
        # sample_id preserved
        assert rows[0]["sample_id"] == "S0001"
        # reviewer_id column set
        assert rows[0]["reviewer_id"] == "reviewer_1"


def test_prepare_with_3_reviewers(sample_csv, output_dir):
    ee.run_prepare(sample_csv, reviewer_count=3, output_dir=output_dir)
    forms_dir = output_dir / "reviewer_forms"
    for r in range(1, 4):
        assert (forms_dir / f"reviewer_{r}_form.csv").exists()


def test_prepare_invalid_reviewer_count_raises(sample_csv, output_dir):
    with pytest.raises(ValueError):
        ee.run_prepare(sample_csv, reviewer_count=1, output_dir=output_dir)
    with pytest.raises(ValueError):
        ee.run_prepare(sample_csv, reviewer_count=4, output_dir=output_dir)


def test_prepare_writes_protocol_and_deferred(sample_csv, output_dir):
    ee.run_prepare(sample_csv, reviewer_count=2, output_dir=output_dir)
    assert (output_dir / "protocol.md").exists()
    assert (output_dir / "deferred_status.json").exists()
    # protocol.md content
    protocol_text = (output_dir / "protocol.md").read_text(encoding="utf-8")
    assert "deferred" in protocol_text.lower()
    assert "Likert" in protocol_text or "likert" in protocol_text.lower()
    # deferred_status content
    with open(output_dir / "deferred_status.json", "r", encoding="utf-8") as fh:
        data = json.load(fh)
    assert data["status"] == "deferred_to_revision"
    assert data["protocol_built"] is True
    assert data["execution_completed"] is False
    assert data["reviewer_count_filled"] == 0
    assert "deferred" in data["status_message"].lower()


def test_prepare_form_includes_reactants_and_products(sample_csv, output_dir):
    ee.run_prepare(sample_csv, reviewer_count=2, output_dir=output_dir)
    forms_dir = output_dir / "reviewer_forms"
    with open(forms_dir / "reviewer_1_form.csv", "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        # For "CCO1>>CC(=O)O1", reactants="CCO1", products="CC(=O)O1"
        assert rows[0]["reactants"] == "CCO1"
        assert rows[0]["products"] == "CC(=O)O1"
        assert rows[0]["candidate_reaction"] == "CCO1>>CC(=O)O1"


# ---------- Likert parsing ----------

def test_parse_likert_valid():
    assert ee.parse_likert("1") == 1
    assert ee.parse_likert("5") == 5
    assert ee.parse_likert("3") == 3
    assert ee.parse_likert(4) == 4
    assert ee.parse_likert("  2  ") == 2
    assert ee.parse_likert("3.0") == 3


def test_parse_likert_invalid():
    assert ee.parse_likert("") is None
    assert ee.parse_likert(None) is None
    assert ee.parse_likert("abc") is None
    assert ee.parse_likert("0") is None
    assert ee.parse_likert("6") is None
    assert ee.parse_likert("-1") is None
    assert ee.parse_likert("3.5") is None  # must be int after float conversion


def test_parse_form_ratings():
    rows = [
        {"sample_id": "S0001", "reviewer_id": "reviewer_1",
         "chemical_validity": "4", "mechanistic_plausibility": "5",
         "side_product_likelihood": "3", "feasibility": "4", "overall_verdict": "4",
         "comment": "ok"},
        {"sample_id": "S0002", "reviewer_id": "reviewer_1",
         "chemical_validity": "", "mechanistic_plausibility": "2",
         "side_product_likelihood": "", "feasibility": "1", "overall_verdict": "",
         "comment": ""},
    ]
    parsed = ee.parse_form_ratings(rows)
    assert parsed[0]["overall_verdict"] == 4
    assert parsed[0]["chemical_validity"] == 4
    assert parsed[1]["overall_verdict"] is None
    assert parsed[1]["mechanistic_plausibility"] == 2


# ---------- Cohen's kappa ----------

def test_cohen_kappa_perfect_agreement():
    a = [1, 2, 3, 4, 5, 1, 2, 3]
    b = list(a)
    k = ee.cohen_kappa(a, b)
    assert k == pytest.approx(1.0, abs=1e-6)


def test_cohen_kappa_worse_than_random():
    a = [1, 1, 1, 1, 2, 2, 2, 2]
    b = [5, 5, 5, 5, 4, 4, 4, 4]
    k = ee.cohen_kappa(a, b)
    assert k <= 0.0 + 1e-6


def test_cohen_kappa_moderate_agreement():
    a = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5]
    b = [1, 2, 2, 2, 3, 3, 4, 5, 5, 5, 1, 2, 3, 4, 4, 2, 2, 3, 4, 5]
    k = ee.cohen_kappa(a, b)
    assert 0.4 <= k <= 1.0


def test_cohen_kappa_empty_returns_zero():
    assert ee.cohen_kappa([], []) == 0.0


def test_cohen_kappa_none_values_filtered():
    a = [1, 2, None, 3, 4]
    b = [1, 2, 3, None, 4]
    k = ee.cohen_kappa(a, b)
    # Overlap is [1,2] from a, [1,2] from b -> perfect agreement
    assert k == pytest.approx(1.0, abs=1e-6)


def test_cohen_kappa_different_lengths():
    a = [1, 2, 3, 4, 5]
    b = [1, 2, 3]
    k = ee.cohen_kappa(a, b)
    assert isinstance(k, float)


# ---------- Fleiss' kappa ----------

def test_fleiss_kappa_perfect_agreement():
    ratings = [
        [1, 1, 1],
        [2, 2, 2],
        [3, 3, 3],
        [4, 4, 4],
        [5, 5, 5],
    ]
    k = ee.fleiss_kappa(ratings, n_categories=5)
    assert k == pytest.approx(1.0, abs=1e-6)


def test_fleiss_kappa_random_near_zero():
    ratings = [
        [1, 2, 3],
        [2, 3, 1],
        [3, 1, 2],
        [4, 5, 1],
        [5, 4, 2],
        [1, 5, 3],
        [2, 4, 1],
        [3, 5, 2],
    ]
    k = ee.fleiss_kappa(ratings, n_categories=5)
    assert -0.5 <= k <= 0.5


def test_fleiss_kappa_empty_returns_zero():
    assert ee.fleiss_kappa([], n_categories=5) == 0.0


def test_fleiss_kappa_three_raters_moderate():
    # Constructed to give moderate-to-substantial agreement
    ratings = [
        [4, 4, 4],
        [3, 3, 3],
        [5, 5, 5],
        [2, 2, 2],
        [1, 1, 1],
        [4, 4, 5],
        [3, 3, 4],
        [5, 5, 4],
    ]
    k = ee.fleiss_kappa(ratings, n_categories=5)
    assert 0.3 <= k <= 1.0


def test_fleiss_kappa_filters_none():
    ratings = [
        [1, 1, None],
        [2, 2, 2],
        [3, 3, 3],
    ]
    # Sample 0 has 2 non-None ratings; samples 1 and 2 have 3.
    # n = min(2, 3, 3) = 2, so we truncate all to 2 raters.
    k = ee.fleiss_kappa(ratings, n_categories=5)
    assert isinstance(k, float)


# ---------- compute_pairwise_agreement ----------

def test_pairwise_agreement_two_reviewers():
    parsed_per_reviewer = [
        [{"sample_id": "S1", "overall_verdict": 4},
         {"sample_id": "S2", "overall_verdict": 2}],
        [{"sample_id": "S1", "overall_verdict": 4},
         {"sample_id": "S2", "overall_verdict": 2}],
    ]
    result = ee.compute_pairwise_agreement(parsed_per_reviewer, column="overall_verdict")
    assert result["metric"] == "cohen_kappa"
    assert result["value"] == pytest.approx(1.0, abs=1e-6)
    assert result["n_common_samples"] == 2
    assert result["passes_threshold"] is True


def test_pairwise_agreement_three_reviewers_uses_fleiss():
    parsed_per_reviewer = [
        [{"sample_id": "S1", "overall_verdict": 4},
         {"sample_id": "S2", "overall_verdict": 2}],
        [{"sample_id": "S1", "overall_verdict": 4},
         {"sample_id": "S2", "overall_verdict": 2}],
        [{"sample_id": "S1", "overall_verdict": 4},
         {"sample_id": "S2", "overall_verdict": 2}],
    ]
    result = ee.compute_pairwise_agreement(parsed_per_reviewer, column="overall_verdict")
    assert result["metric"] == "fleiss_kappa"
    assert "pairwise_cohen_kappa" in result
    assert result["value"] == pytest.approx(1.0, abs=1e-6)


def test_pairwise_agreement_no_common_samples():
    parsed_per_reviewer = [
        [{"sample_id": "S1", "overall_verdict": 4}],
        [{"sample_id": "S2", "overall_verdict": 2}],
    ]
    result = ee.compute_pairwise_agreement(parsed_per_reviewer, column="overall_verdict")
    assert result["n_common_samples"] == 0
    assert result["passes_threshold"] is False


# ---------- Pass rate ----------

def test_compute_pass_rate_all_pass():
    parsed_per_reviewer = [
        [{"sample_id": "S1", "reviewer_id": "r1", "overall_verdict": 5},
         {"sample_id": "S2", "reviewer_id": "r1", "overall_verdict": 4}],
        [{"sample_id": "S1", "reviewer_id": "r2", "overall_verdict": 4},
         {"sample_id": "S2", "reviewer_id": "r2", "overall_verdict": 5}],
    ]
    result = ee.compute_pass_rate(parsed_per_reviewer, threshold=4)
    assert result["per_sample_majority"]["pass_rate"] == 1.0
    assert result["per_sample_majority"]["n_pass"] == 2
    assert result["passes_go_no_go"] is True


def test_compute_pass_rate_half_fail():
    parsed_per_reviewer = [
        [{"sample_id": "S1", "reviewer_id": "r1", "overall_verdict": 5},
         {"sample_id": "S2", "reviewer_id": "r1", "overall_verdict": 2}],
        [{"sample_id": "S1", "reviewer_id": "r2", "overall_verdict": 4},
         {"sample_id": "S2", "reviewer_id": "r2", "overall_verdict": 1}],
    ]
    result = ee.compute_pass_rate(parsed_per_reviewer, threshold=4)
    assert result["per_sample_majority"]["pass_rate"] == 0.5
    assert result["passes_go_no_go"] is False


def test_compute_pass_rate_per_reviewer_counts():
    parsed_per_reviewer = [
        [{"sample_id": "S1", "reviewer_id": "r1", "overall_verdict": 5},
         {"sample_id": "S2", "reviewer_id": "r1", "overall_verdict": 3}],
        [{"sample_id": "S1", "reviewer_id": "r2", "overall_verdict": 4},
         {"sample_id": "S2", "reviewer_id": "r2", "overall_verdict": 2}],
    ]
    result = ee.compute_pass_rate(parsed_per_reviewer, threshold=4)
    assert len(result["per_reviewer"]) == 2
    assert result["per_reviewer"][0]["n_pass"] == 1
    assert result["per_reviewer"][0]["n_total"] == 2


def test_compute_pass_rate_skips_none_verdicts():
    parsed_per_reviewer = [
        [{"sample_id": "S1", "reviewer_id": "r1", "overall_verdict": 5},
         {"sample_id": "S2", "reviewer_id": "r1", "overall_verdict": None}],
        [{"sample_id": "S1", "reviewer_id": "r2", "overall_verdict": 4},
         {"sample_id": "S2", "reviewer_id": "r2", "overall_verdict": None}],
    ]
    result = ee.compute_pass_rate(parsed_per_reviewer, threshold=4)
    # Only S1 has ratings
    assert result["per_sample_majority"]["n_total"] == 1
    assert result["per_sample_majority"]["pass_rate"] == 1.0


# ---------- Failure mode distribution ----------

def test_failure_mode_distribution():
    sample_rows = [
        {"sample_id": "S1", "failure_type": "no_reaction"},
        {"sample_id": "S2", "failure_type": "side_product"},
        {"sample_id": "S3", "failure_type": "no_reaction"},
    ]
    parsed_per_reviewer = [
        [{"sample_id": "S1", "overall_verdict": 1},
         {"sample_id": "S2", "overall_verdict": 2},
         {"sample_id": "S3", "overall_verdict": 1}],
        [{"sample_id": "S1", "overall_verdict": 2},
         {"sample_id": "S2", "overall_verdict": 1},
         {"sample_id": "S3", "overall_verdict": 2}],
    ]
    result = ee.compute_failure_mode_distribution(parsed_per_reviewer, sample_rows)
    assert result["n_fail_samples"] == 3
    assert result["failure_type_counts"]["no_reaction"] == 2
    assert result["failure_type_counts"]["side_product"] == 1
    assert pytest.approx(result["failure_type_distribution"]["no_reaction"], abs=1e-3) == 2 / 3


def test_failure_mode_distribution_no_fails():
    sample_rows = [{"sample_id": "S1", "failure_type": "no_reaction"}]
    parsed_per_reviewer = [
        [{"sample_id": "S1", "overall_verdict": 5}],
        [{"sample_id": "S1", "overall_verdict": 4}],
    ]
    result = ee.compute_failure_mode_distribution(parsed_per_reviewer, sample_rows)
    assert result["n_fail_samples"] == 0
    assert result["failure_type_counts"] == {}


def test_failure_mode_distribution_mixed():
    sample_rows = [
        {"sample_id": "S1", "failure_type": "no_reaction"},
        {"sample_id": "S2", "failure_type": "side_product"},
    ]
    # S1: both rate >= 4 (pass). S2: both rate < 4 (fail).
    parsed_per_reviewer = [
        [{"sample_id": "S1", "overall_verdict": 5},
         {"sample_id": "S2", "overall_verdict": 2}],
        [{"sample_id": "S1", "overall_verdict": 4},
         {"sample_id": "S2", "overall_verdict": 1}],
    ]
    result = ee.compute_failure_mode_distribution(parsed_per_reviewer, sample_rows)
    assert result["n_fail_samples"] == 1
    assert result["failure_type_counts"]["side_product"] == 1
    assert "no_reaction" not in result["failure_type_counts"]


# ---------- Deferred status ----------

def test_deferred_status_message():
    assert "deferred" in ee.DEFERRED_STATUS.lower()
    assert "revision" in ee.DEFERRED_STATUS.lower()
    assert "protocol" in ee.DEFERRED_STATUS.lower()


def test_prepare_marks_deferred(sample_csv, output_dir):
    ee.run_prepare(sample_csv, reviewer_count=2, output_dir=output_dir)
    with open(output_dir / "deferred_status.json", "r", encoding="utf-8") as fh:
        data = json.load(fh)
    assert data["execution_completed"] is False
    assert data["protocol_built"] is True
    assert data["reviewer_count_filled"] == 0
    assert "deferred" in data["status"].lower()
    assert data["limitation_preserved_in_paper"] is True
    assert "go_no_go_thresholds" in data
    assert data["go_no_go_thresholds"]["inter_annotator_agreement_kappa"] == 0.60
    assert data["go_no_go_thresholds"]["reviewer_pass_rate"] == 0.70


# ---------- Split reaction SMILES ----------

def test_split_reaction_smiles_forward():
    r, p = ee.split_reaction_smiles("A>>B")
    assert r == "A"
    assert p == "B"


def test_split_reaction_smiles_empty():
    r, p = ee.split_reaction_smiles("")
    assert r == ""
    assert p == ""


def test_split_reaction_smiles_reversible():
    r, p = ee.split_reaction_smiles("A>B<C")
    assert "A" in r
    assert "C" in p


def test_split_reaction_smiles_no_separator():
    r, p = ee.split_reaction_smiles("just_a_molecule")
    assert r == "just_a_molecule"
    assert p == ""


# ---------- build_reviewer_form ----------

def test_build_reviewer_form_columns():
    rows = [{
        "sample_id": "S1",
        "reaction_smiles": "A>>B",
        "parent_reaction_smiles": "C>>D",
        "failure_type": "no_reaction",
        "task": "forward_outcome",
        "source_origin": "pc_cng_synthetic",
        "true_label": "0",
    }]
    form = ee.build_reviewer_form(rows, reviewer_id=2)
    assert len(form) == 1
    row = form[0]
    assert row["sample_id"] == "S1"
    assert row["reactants"] == "A"
    assert row["products"] == "B"
    assert row["candidate_reaction"] == "A>>B"
    assert row["parent_reaction_smiles"] == "C>>D"
    assert row["failure_type"] == "no_reaction"
    assert row["reviewer_id"] == "reviewer_2"
    for col in ee.LIKERT_COLUMNS:
        assert row[col] == ""
    assert row["comment"] == ""
    assert row["review_timestamp"] == ""


# ---------- main() integration ----------

def test_main_prepare_mode(sample_csv, output_dir, capsys):
    rc = ee.main([
        "--samples", str(sample_csv),
        "--reviewer-count", "2",
        "--output-dir", str(output_dir),
        "--mode", "prepare",
    ])
    assert rc == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["mode"] == "prepare"
    assert data["n_samples"] == 10
    assert data["status"] == "deferred_to_revision"
    assert (output_dir / "reviewer_forms" / "reviewer_1_form.csv").exists()
    assert (output_dir / "reviewer_forms" / "reviewer_2_form.csv").exists()
    assert (output_dir / "protocol.md").exists()
    assert (output_dir / "deferred_status.json").exists()


# ---------- Synthetic small end-to-end (10 samples) ----------

def test_synthetic_10_samples_prepare_and_aggregate(tmp_path):
    """End-to-end test: prepare 10 samples with 2 reviewers, simulate filled
    forms, then aggregate and verify agreement + summary outputs."""
    samples_path = tmp_path / "sampled.csv"
    _make_sample_csv(samples_path, n=10)
    output_dir = tmp_path / "expert_review_out"
    # Prepare
    prep = ee.run_prepare(samples_path, reviewer_count=2, output_dir=output_dir)
    assert prep["n_samples"] == 10
    forms_dir = output_dir / "reviewer_forms"
    # Simulate filling forms: identical ratings for high agreement.
    for r in range(1, 3):
        form_path = forms_dir / f"reviewer_{r}_form.csv"
        with open(form_path, "r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            fieldnames = reader.fieldnames
            rows = list(reader)
        for i, row in enumerate(rows):
            row["chemical_validity"] = "4"
            row["mechanistic_plausibility"] = "4"
            row["side_product_likelihood"] = "3"
            row["feasibility"] = "4"
            row["overall_verdict"] = "4" if i % 2 == 0 else "2"
            row["comment"] = f"review_{r}_sample_{i}"
            row["review_timestamp"] = "2026-07-20T12:00:00"
        with open(form_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
    # Aggregate
    agg = ee.run_aggregate(forms_dir, output_dir, samples_path)
    assert (output_dir / "reviewer_ratings_raw.csv").exists()
    assert (output_dir / "inter_annotator_agreement.json").exists()
    assert (output_dir / "expert_review_summary.json").exists()
    summary = agg["summary"]
    # Both reviewers agreed perfectly -> Cohen's kappa = 1.0
    assert summary["agreement"]["value"] == pytest.approx(1.0, abs=1e-6)
    assert summary["agreement"]["metric"] == "cohen_kappa"
    # 5 samples pass (overall_verdict=4 on even indices), 5 fail
    assert summary["pass_rate"]["per_sample_majority"]["pass_rate"] == pytest.approx(0.5)
    assert summary["pass_rate"]["per_sample_majority"]["n_total"] == 10
    # Failure mode distribution: 5 fail samples, alternates no_reaction / side_product
    # Even indices (0,2,4,6,8) -> pass (verdict=4); odd indices (1,3,5,7,9) -> fail
    # Sample i+1 has failure_type "no_reaction" if (i+1) % 2 == 0 else "side_product"
    # i=1 -> S0002 -> no_reaction (fail)
    # i=3 -> S0004 -> no_reaction (fail)
    # i=5 -> S0006 -> no_reaction (fail)
    # i=7 -> S0008 -> no_reaction (fail)
    # i=9 -> S0010 -> no_reaction (fail)
    # All 5 fails are no_reaction
    assert summary["failure_mode_distribution"]["n_fail_samples"] == 5
    assert summary["failure_mode_distribution"]["failure_type_counts"]["no_reaction"] == 5
    # go_no_go: agreement passes (1.0 >= 0.6), pass rate fails (0.5 < 0.7)
    assert summary["go_no_go"]["agreement_passes"] is True
    assert summary["go_no_go"]["pass_rate_passes"] is False
    assert summary["go_no_go"]["overall_go"] is False


def test_synthetic_end_to_end_main_aggregate(tmp_path):
    """Test main() in aggregate mode end-to-end."""
    samples_path = tmp_path / "sampled.csv"
    _make_sample_csv(samples_path, n=5)
    output_dir = tmp_path / "expert_review_out"
    # Prepare via main()
    ee.main([
        "--samples", str(samples_path),
        "--reviewer-count", "2",
        "--output-dir", str(output_dir),
        "--mode", "prepare",
    ])
    forms_dir = output_dir / "reviewer_forms"
    # Fill forms with all-pass ratings
    for r in range(1, 3):
        form_path = forms_dir / f"reviewer_{r}_form.csv"
        with open(form_path, "r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            fieldnames = reader.fieldnames
            rows = list(reader)
        for row in rows:
            for col in ee.LIKERT_COLUMNS:
                row[col] = "5"
            row["review_timestamp"] = "2026-07-20T12:00:00"
        with open(form_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
    # Aggregate via main()
    rc = ee.main([
        "--samples", str(samples_path),
        "--output-dir", str(output_dir),
        "--mode", "aggregate",
        "--filled-forms-dir", str(forms_dir),
    ])
    assert rc == 0
    assert (output_dir / "expert_review_summary.json").exists()
    with open(output_dir / "expert_review_summary.json", "r", encoding="utf-8") as fh:
        summary = json.load(fh)
    # All pass, perfect agreement -> overall_go = True
    assert summary["agreement"]["value"] == pytest.approx(1.0, abs=1e-6)
    assert summary["pass_rate"]["per_sample_majority"]["pass_rate"] == 1.0
    assert summary["go_no_go"]["overall_go"] is True


def test_aggregate_missing_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        ee.run_aggregate(tmp_path / "does_not_exist", tmp_path / "out")


def test_aggregate_insufficient_forms_raises(tmp_path):
    forms_dir = tmp_path / "forms"
    forms_dir.mkdir(parents=True)
    # Only one form (need >= 2)
    (forms_dir / "reviewer_1_form.csv").write_text("sample_id\nS1\n", encoding="utf-8")
    with pytest.raises(ValueError):
        ee.run_aggregate(forms_dir, tmp_path / "out")
