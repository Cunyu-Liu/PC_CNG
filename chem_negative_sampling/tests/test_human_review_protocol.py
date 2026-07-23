"""Tests for P4-G7 Human Expert Calibration protocol.

Tests:
- Sampling covers all 12 strata
- Blinding: no source/fnr/score visible in forms
- Randomization is reproducible
- Form structure is correct
- Statistical functions work correctly
"""

from __future__ import annotations

import csv
import json
import math
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Add module path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pc_cng.p4_g7_sampling import (
    STRATA, SCORING_DIMENSIONS, REASON_CODES,
    PILOT_PER_STRATUM, MAIN_PER_STRATUM,
    _strip_atom_mapping, _blinded_id,
    stratified_sample, create_blinded_forms,
    write_sampling_manifest, write_samples_csv,
    run_pilot,
)
from pc_cng.p4_g7_agreement import (
    weighted_kappa, krippendorff_alpha,
    control_discrimination, source_level_effect,
    reviewer_effect, confidence_sensitivity,
    llm_human_agreement, _cliffs_delta,
)


# ---------------------------------------------------------------------------
# Atom mapping stripping
# ---------------------------------------------------------------------------

class TestStripAtomMapping:
    def test_simple_mapping(self):
        assert _strip_atom_mapping("[C:1]O") == "[C]O"

    def test_multiple_mappings(self):
        s = "[C:1][O:2][N:3]"
        result = _strip_atom_mapping(s)
        assert ":" not in result
        assert "1" not in result or result.count("1") == 0 or "[C1]" not in result

    def test_empty(self):
        assert _strip_atom_mapping("") == ""

    def test_no_mapping(self):
        assert _strip_atom_mapping("CCO") == "CCO"

    def test_complex_smiles(self):
        s = "[n:1]1[c:2]([CH3:3])[cH:4][n:5]1"
        result = _strip_atom_mapping(s)
        assert ":" not in result


# ---------------------------------------------------------------------------
# Blinded ID generation
# ---------------------------------------------------------------------------

class TestBlindedID:
    def test_format(self):
        bid = _blinded_id(42, 0, "random")
        assert bid.startswith("BLD-")
        assert len(bid) == 16  # BLD- (4) + 12 hex chars

    def test_deterministic(self):
        a = _blinded_id(42, 0, "random")
        b = _blinded_id(42, 0, "random")
        assert a == b

    def test_different_inputs(self):
        a = _blinded_id(42, 0, "random")
        b = _blinded_id(42, 1, "random")
        assert a != b


# ---------------------------------------------------------------------------
# Strata coverage
# ---------------------------------------------------------------------------

class TestStrataCoverage:
    def test_all_12_strata_defined(self):
        assert len(STRATA) == 12

    def test_strata_match_spec(self):
        expected = {
            "random", "tanimoto", "template", "pc_cng",
            "known_real_negative", "known_positive_control",
            "pc_cng_low_risk", "pc_cng_medium_risk", "pc_cng_high_risk",
            "llm_disagreement", "hte_false_positive", "hte_false_negative",
        }
        assert set(STRATA) == expected

    def test_pilot_counts_sum_to_80(self):
        assert sum(PILOT_PER_STRATUM.values()) == 80

    def test_main_counts_sum_to_250(self):
        assert sum(MAIN_PER_STRATUM.values()) == 250

    def test_pilot_all_strata_have_samples(self):
        for s in STRATA:
            assert PILOT_PER_STRATUM[s] > 0, f"Stratum {s} has 0 samples in pilot"

    def test_main_all_strata_have_samples(self):
        for s in STRATA:
            assert MAIN_PER_STRATUM[s] > 0, f"Stratum {s} has 0 samples in main"


# ---------------------------------------------------------------------------
# Scoring dimensions and reason codes
# ---------------------------------------------------------------------------

class TestScoringDimensions:
    def test_six_dimensions(self):
        assert len(SCORING_DIMENSIONS) == 6

    def test_expected_dimensions(self):
        expected = {
            "structural_validity",
            "mechanistic_plausibility",
            "plausible_competing_outcome",
            "likely_low_yield_failure",
            "likely_feasible_positive",
            "confidence",
        }
        assert set(SCORING_DIMENSIONS) == expected


class TestReasonCodes:
    def test_at_least_10_codes(self):
        assert len(REASON_CODES) >= 10

    def test_expected_codes(self):
        assert "wrong_reaction_center" in REASON_CODES
        assert "unlikely_bond_change" in REASON_CODES
        assert "chemoselectivity_issue" in REASON_CODES
        assert "other" in REASON_CODES


# ---------------------------------------------------------------------------
# Blinded forms
# ---------------------------------------------------------------------------

class TestBlindedForms:
    def test_form_has_no_source(self):
        """Blinded form must NOT contain candidate_source."""
        samples = [
            {"blinded_id": "S0001", "reaction_smiles": "CCO",
             "candidate_source": "rule_pc_cng", "stratum": "pc_cng",
             "fnr": 0.5, "model_score": 0.7},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            forms = create_blinded_forms(samples, Path(tmp), n_reviewers=1, seed=42)
            with open(forms[0]) as f:
                reader = csv.reader(f)
                header = next(reader)
                assert "candidate_source" not in header
                assert "stratum" not in header
                assert "fnr" not in header
                assert "model_score" not in header

    def test_form_has_scoring_columns(self):
        samples = [{"blinded_id": "S0001", "reaction_smiles": "CCO",
                     "candidate_source": "x", "stratum": "x"}]
        with tempfile.TemporaryDirectory() as tmp:
            forms = create_blinded_forms(samples, Path(tmp), n_reviewers=1, seed=42)
            with open(forms[0]) as f:
                header = next(csv.reader(f))
                for dim in SCORING_DIMENSIONS:
                    assert dim in header
                assert "reason_codes" in header
                assert "notes" in header

    def test_each_reviewer_gets_different_order(self):
        samples = [
            {"blinded_id": f"S{i:04d}", "reaction_smiles": f"C{i}",
             "candidate_source": "x", "stratum": "x"}
            for i in range(10)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            forms = create_blinded_forms(samples, Path(tmp), n_reviewers=2, seed=42)
            ids1 = []
            ids2 = []
            with open(forms[0]) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    ids1.append(row["blinded_id"])
            with open(forms[1]) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    ids2.append(row["blinded_id"])
            # Same set but different order
            assert set(ids1) == set(ids2)
            assert ids1 != ids2

    def test_all_samples_present(self):
        samples = [
            {"blinded_id": f"S{i:04d}", "reaction_smiles": f"C{i}",
             "candidate_source": "x", "stratum": "x"}
            for i in range(20)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            forms = create_blinded_forms(samples, Path(tmp), n_reviewers=1, seed=42)
            with open(forms[0]) as f:
                rows = list(csv.DictReader(f))
            assert len(rows) == 20


# ---------------------------------------------------------------------------
# Sampling manifest
# ---------------------------------------------------------------------------

class TestSamplingManifest:
    def test_manifest_contains_unblinded_info(self):
        samples = [
            {"blinded_id": "S0001", "reaction_smiles": "CCO",
             "candidate_source": "rule_pc_cng", "stratum": "pc_cng",
             "fnr": 0.5, "candidate_id": "c1"},
        ]
        summary = {"seed": 42, "n_total": 1, "stratum_counts": {"pc_cng": 1},
                   "all_strata_covered": False, "n_per_stratum_requested": {}}
        with tempfile.TemporaryDirectory() as tmp:
            path = write_sampling_manifest(samples, summary, Path(tmp) / "manifest.json")
            with open(path) as f:
                m = json.load(f)
            assert m["samples"][0]["candidate_source"] == "rule_pc_cng"
            assert m["samples"][0]["stratum"] == "pc_cng"
            assert m["samples"][0]["fnr"] == 0.5


class TestSamplesCSV:
    def test_csv_is_blinded(self):
        samples = [
            {"blinded_id": "S0001", "reaction_smiles": "CCO",
             "candidate_source": "secret", "stratum": "secret"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = write_samples_csv(samples, Path(tmp) / "samples.csv")
            with open(path) as f:
                reader = csv.DictReader(f)
                row = next(reader)
                assert "blinded_id" in row
                assert "reaction_smiles" in row
                assert "candidate_source" not in row
                assert "stratum" not in row


# ---------------------------------------------------------------------------
# Statistical functions
# ---------------------------------------------------------------------------

class TestWeightedKappa:
    def test_perfect_agreement(self):
        scores = [1, 2, 3, 4, 5]
        assert weighted_kappa(scores, scores) > 0.99

    def test_random_agreement(self):
        scores1 = [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]
        scores2 = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
        kappa = weighted_kappa(scores1, scores2)
        assert kappa < 0.3  # Should be low

    def test_empty(self):
        assert weighted_kappa([], []) == 0.0

    def test_different_lengths(self):
        assert weighted_kappa([1, 2], [1, 2, 3]) == 0.0


class TestKrippendorffAlpha:
    def test_perfect_agreement(self):
        data = [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]]
        alpha = krippendorff_alpha(data, level="ordinal")
        assert alpha > 0.9

    def test_no_agreement(self):
        data = [[1, 5], [2, 4], [3, 3], [4, 2], [5, 1]]
        alpha = krippendorff_alpha(data, level="ordinal")
        assert alpha < 0.3

    def test_missing_values(self):
        data = [[1, 1, None], [2, None, 2], [3, 3, 3]]
        alpha = krippendorff_alpha(data, level="ordinal")
        assert isinstance(alpha, float)

    def test_empty(self):
        assert krippendorff_alpha([], level="ordinal") == 0.0


class TestControlDiscrimination:
    def test_significant_discrimination(self):
        pos = [5.0, 5.0, 4.0, 5.0, 4.0, 5.0, 4.0, 5.0]
        neg = [1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0]
        result = control_discrimination(pos, neg)
        assert result["discrimination_significant"]
        assert result["mean_positive"] > result["mean_negative"]

    def test_no_discrimination(self):
        pos = [3.0, 3.0, 3.0, 3.0]
        neg = [3.0, 3.0, 3.0, 3.0]
        result = control_discrimination(pos, neg)
        assert not result["discrimination_significant"]

    def test_insufficient_samples(self):
        result = control_discrimination([1.0], [2.0])
        assert not result["discrimination_significant"]


class TestSourceLevelEffect:
    def test_significant_difference(self):
        scores = {
            "pc_cng": [5.0, 4.0, 5.0, 4.0, 5.0],
            "random": [1.0, 2.0, 1.0, 2.0, 1.0],
            "template": [2.0, 1.0, 2.0, 1.0, 2.0],
        }
        result = source_level_effect(scores)
        assert "pairwise" in result
        assert len(result["pairwise"]) > 0

    def test_no_difference(self):
        scores = {
            "pc_cng": [3.0, 3.0, 3.0, 3.0, 3.0],
            "random": [3.0, 3.0, 3.0, 3.0, 3.0],
        }
        result = source_level_effect(scores)
        # All same -> no significant difference
        for v in result.get("pairwise", {}).values():
            assert v["p_value"] > 0.05


class TestReviewerEffect:
    def test_no_effect(self):
        scores = {
            "r1": [3.0, 3.0, 3.0, 3.0, 3.0],
            "r2": [3.0, 3.0, 3.0, 3.0, 3.0],
        }
        result = reviewer_effect(scores)
        assert not result["reviewer_effect_significant"]

    def test_effect_present(self):
        scores = {
            "r1": [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
            "r2": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        }
        result = reviewer_effect(scores)
        assert result["reviewer_effect_significant"]


class TestConfidenceSensitivity:
    def test_correlation(self):
        scores = [1.0, 5.0, 1.0, 5.0, 1.0, 5.0, 1.0, 5.0]
        conf = [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]
        result = confidence_sensitivity(scores, conf)
        assert "correlation" in result

    def test_insufficient_data(self):
        result = confidence_sensitivity([1.0], [1.0])
        assert not result["significant"]


class TestCliffsDelta:
    def test_perfect_separation(self):
        assert _cliffs_delta([5, 5, 5], [1, 1, 1]) == 1.0

    def test_perfect_overlap(self):
        assert _cliffs_delta([3, 3, 3], [3, 3, 3]) == 0.0

    def test_inverse(self):
        assert _cliffs_delta([1, 1, 1], [5, 5, 5]) == -1.0


class TestLLMHumanAgreement:
    def test_perfect_agreement(self):
        result = llm_human_agreement([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
        assert result["spearman_rho"] > 0.9
        assert "Supplementary" in result["note"]

    def test_insufficient(self):
        result = llm_human_agreement([1], [1])
        assert result["agreement"] == 0.0


# ---------------------------------------------------------------------------
# Integration test (requires manifest data on server)
# ---------------------------------------------------------------------------

class TestPilotGeneration:
    """Integration test that runs pilot generation with real data."""

    @pytest.mark.skipif(
        not Path("/home/cunyuliu/pc_cng_research/data/p4/manifests/hte_feasibility_v2.json").exists(),
        reason="Manifest not found (server-only test)"
    )
    def test_pilot_generates_all_outputs(self):
        from pc_cng.run_p4_human_review import cmd_pilot, DEFAULT_MANIFEST, DEFAULT_RISK, DEFAULT_G6_PRED
        import argparse

        args = argparse.Namespace(
            manifest=str(DEFAULT_MANIFEST),
            risk=str(DEFAULT_RISK),
            g6_pred=str(DEFAULT_G6_PRED),
            output="/tmp/p4_g7_test_output",
            seed=20260723,
        )
        cmd_pilot(args)

        output = Path("/tmp/p4_g7_test_output")
        assert (output / "sampling_manifest.json").exists()
        assert (output / "samples.csv").exists()
        assert (output / "blinded_forms").exists()

        # Check manifest
        with open(output / "sampling_manifest.json") as f:
            m = json.load(f)
        assert m["n_total"] == 80
        assert m["all_strata_covered"]

        # Check blinding
        with open(output / "samples.csv") as f:
            reader = csv.DictReader(f)
            header = reader.fieldnames
            assert "candidate_source" not in header
            assert "stratum" not in header
            assert "fnr" not in header
