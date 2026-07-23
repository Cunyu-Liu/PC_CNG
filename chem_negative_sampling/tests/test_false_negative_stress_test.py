"""Tests for P4-G5 false-negative stress tests.

Run with::

    python3 -m pytest chem_negative_sampling/tests/test_false_negative_stress_test.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CNS_ROOT = _REPO_ROOT / "chem_negative_sampling"
if str(_CNS_ROOT) not in sys.path:
    sys.path.insert(0, str(_CNS_ROOT))

from pc_cng.evaluation.false_negative_stress_test import (
    build_known_positive_set,
    build_near_positive_set,
    build_ood_family_set,
    collision_sensitivity,
    coverage_risk_curve,
    ece_brier_nll,
    known_positive_metrics,
    near_positive_metrics,
    ood_metrics,
)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestBuildKnownPositiveSet:
    def _inputs(self):
        heldout = [
            {"smiles": f"CC{'O' * i}", "reaction_family": "famA"} for i in range(1, 6)
        ] + [
            {"smiles": f"CN{i}", "reaction_family": "famB"} for i in range(5)
        ]
        by_family = {
            "famA": [{"smiles": f"c1ccccc1C{i}", "reaction_family": "famA"} for i in range(10)],
        }
        fallback = [{"smiles": f"CCCl{i}", "reaction_family": "famX"} for i in range(20)]
        return heldout, by_family, fallback

    def test_group_shape(self):
        heldout, by_family, fallback = self._inputs()
        groups = build_known_positive_set(heldout, by_family, fallback, n=5, k_neg=3, seed=1)
        assert len(groups) == 5
        for g in groups:
            assert len(g["candidates"]) == 4  # 1 pos + 3 neg
            pos = [c for c in g["candidates"] if c["is_disguised_positive"]]
            assert len(pos) == 1
            assert pos[0]["label"] == 1
            assert sum(c["label"] for c in g["candidates"]) == 1

    def test_family_match_preferred(self):
        heldout, by_family, fallback = self._inputs()
        groups = build_known_positive_set(heldout[:5], by_family, fallback, n=3, k_neg=3, seed=2)
        for g in groups:
            negs = [c for c in g["candidates"] if not c["is_disguised_positive"]]
            assert all(c["smiles"].startswith("c1ccccc1") for c in negs)

    def test_fallback_when_family_missing(self):
        heldout, by_family, fallback = self._inputs()
        groups = build_known_positive_set(heldout[5:], by_family, fallback, n=3, k_neg=3, seed=3)
        assert len(groups) == 3
        for g in groups:
            negs = [c for c in g["candidates"] if not c["is_disguised_positive"]]
            assert all(c["smiles"].startswith("CCCl") for c in negs)

    def test_deterministic_seed(self):
        heldout, by_family, fallback = self._inputs()
        g1 = build_known_positive_set(heldout, by_family, fallback, n=4, k_neg=2, seed=7)
        g2 = build_known_positive_set(heldout, by_family, fallback, n=4, k_neg=2, seed=7)
        assert g1 == g2


class TestBuildNearPositiveSet:
    def test_threshold_and_exclusions(self):
        cands = [
            {"smiles": "A", "candidate_id": "c1", "nearest_train_similarity": 0.9,
             "gold_candidate": False, "train_overlap": False},
            {"smiles": "B", "candidate_id": "c2", "nearest_train_similarity": 0.5,
             "gold_candidate": False, "train_overlap": False},
            {"smiles": "C", "candidate_id": "c3", "nearest_train_similarity": 0.95,
             "gold_candidate": True, "train_overlap": False},
            {"smiles": "D", "candidate_id": "c4", "nearest_train_similarity": 0.99,
             "gold_candidate": False, "train_overlap": True},
        ]
        out = build_near_positive_set(cands, min_sim=0.7)
        ids = [c["candidate_id"] for c in out]
        assert ids == ["c1"]

    def test_label_one_excluded(self):
        cands = [{"smiles": "A", "candidate_id": "c1", "nearest_train_similarity": 0.9,
                  "label": 1, "train_overlap": False}]
        assert build_near_positive_set(cands) == []


class TestBuildOodFamilySet:
    def test_only_unseen_families(self):
        rows = [
            {"smiles": "A", "label": 1, "reaction_family": "seen"},
            {"smiles": "B", "label": 0, "reaction_family": "unseen1"},
            {"smiles": "C", "label": 1, "reaction_family": "unseen1"},
            {"smiles": "D", "label": 1, "reaction_family": "unseen2"},
        ]
        out = build_ood_family_set(rows, ["seen"], seed=1)
        fams = {r["reaction_family"] for r in out}
        assert fams == {"unseen1", "unseen2"}
        assert len(out) == 3

    def test_cap_per_family(self):
        rows = [{"smiles": f"S{i}", "label": 1, "reaction_family": "u"} for i in range(100)]
        out = build_ood_family_set(rows, [], max_per_family=10, seed=1)
        assert len(out) == 10


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

class TestKnownPositiveMetrics:
    def test_perfect_recovery(self):
        groups = [
            [{"score": 5.0, "is_disguised_positive": True},
             {"score": -5.0, "is_disguised_positive": False}],
            [{"score": 4.0, "is_disguised_positive": True},
             {"score": -4.0, "is_disguised_positive": False}],
        ]
        m = known_positive_metrics(groups)
        assert m["n"] == 2
        assert m["recovery_top1"] == 1.0
        assert m["mean_prob"] > 0.9

    def test_full_rejection(self):
        groups = [
            [{"score": -5.0, "is_disguised_positive": True},
             {"score": 5.0, "is_disguised_positive": False}],
        ]
        m = known_positive_metrics(groups)
        assert m["recovery_top1"] == 0.0
        assert m["mean_prob"] < 0.1

    def test_empty(self):
        m = known_positive_metrics([])
        assert m["n"] == 0 and m["recovery_top1"] == 0.0


class TestNearPositiveMetrics:
    def test_hard_reject_rate(self):
        scored = [
            {"score": -10.0, "fnr": 0.9},   # prob ~ 0
            {"score": -10.0, "fnr": 0.8},
            {"score": 10.0, "fnr": 0.1},    # prob ~ 1
        ]
        m = near_positive_metrics(scored)
        assert m["n"] == 3
        assert m["hard_reject_rate"] == pytest.approx(2 / 3)
        assert m["fnr_corr"] < 0  # prob anti-correlated with fnr here

    def test_empty(self):
        m = near_positive_metrics([])
        assert m["n"] == 0


class TestEceBrierNll:
    def test_perfect_calibration(self):
        probs = [0.0, 0.0, 1.0, 1.0]
        labels = [0, 0, 1, 1]
        m = ece_brier_nll(probs, labels)
        assert m["brier"] == pytest.approx(0.0)
        assert m["ece"] == pytest.approx(0.0)
        assert m["nll"] == pytest.approx(0.0, abs=1e-6)

    def test_worst_case(self):
        probs = [1.0, 1.0, 0.0, 0.0]
        labels = [0, 0, 1, 1]
        m = ece_brier_nll(probs, labels)
        assert m["brier"] == pytest.approx(1.0)
        assert m["ece"] == pytest.approx(1.0)

    def test_empty(self):
        m = ece_brier_nll([], [])
        assert m["n"] == 0


class TestOodMetrics:
    def test_scored_rows(self):
        scored = [{"score": 10.0, "label": 1}, {"score": -10.0, "label": 0}]
        m = ood_metrics(scored)
        assert m["n"] == 2
        assert m["brier"] < 0.01

    def test_empty(self):
        assert ood_metrics([])["n"] == 0


class TestCollisionSensitivity:
    def test_filters_flagged_ids(self):
        preds = [
            {"candidate_id": "a", "score": -10.0},
            {"candidate_id": "b", "score": 10.0},
            {"candidate_id": "c", "score": 10.0},
        ]
        m = collision_sensitivity(preds, ["a", "b"])
        assert m["n"] == 2
        assert m["hard_reject_rate"] == pytest.approx(0.5)

    def test_no_matches(self):
        m = collision_sensitivity([{"candidate_id": "x", "score": 0.0}], ["zzz"])
        assert m["n"] == 0


class TestCoverageRiskCurve:
    def test_abstention_reduces_risk_when_uncertainty_informative(self):
        # High-uncertainty items are the wrong ones
        probs = [0.9, 0.1, 0.9, 0.1]
        labels = [1, 0, 0, 1]
        unc = [0.01, 0.01, 0.9, 0.9]
        cr = coverage_risk_curve(probs, labels, unc, coverages=[0.5, 1.0])
        assert cr["risk"][0] < cr["risk"][1]
        assert cr["auc"] > 0

    def test_risk_at_0p8(self):
        probs = [0.5] * 10
        labels = [0] * 10
        unc = [0.1] * 10
        cr = coverage_risk_curve(probs, labels, unc)
        assert 0.8 in cr["coverage"]
        assert cr["risk_at_0p8"] == pytest.approx(0.25)

    def test_empty(self):
        cr = coverage_risk_curve([], [], [])
        assert cr["risk_at_0p8"] == 0.0
