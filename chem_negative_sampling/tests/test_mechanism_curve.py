"""Tests for P4-G8A mechanism curve analysis.

Covers: curve fitting, binning, difficulty metrics, verdict computation.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

# Suppress RDKit warnings
os.environ["RDKitRDLogger"] = "0"

# Ensure chem_negative_sampling is importable
_THIS = Path(__file__).resolve()
for _cand in (_THIS.parents[1], _THIS.parents[2] / "chem_negative_sampling"):
    if (_cand / "pc_cng").is_dir() and str(_cand) not in sys.path:
        sys.path.insert(0, str(_cand))

from pc_cng.p4_g8a_mechanism_curve import (
    N_BINS,
    DIFFICULTY_METRICS,
    CURVE_SHAPES,
    bin_values,
    fit_curve,
    analyze_curve,
    compute_verdict as compute_g8a_verdict,
    compute_scoring_margin,
    compute_downstream_loss,
)


# ---------------------------------------------------------------------------
# Binning Tests
# ---------------------------------------------------------------------------

class TestBinValues:
    """Test binning functionality."""

    def test_basic_binning(self):
        values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        bins = bin_values(values, n_bins=5)
        assert len(bins) == 10
        assert all(0 <= b < 5 for b in bins)

    def test_single_value(self):
        bins = bin_values([0.5], n_bins=5)
        assert bins == [0]

    def test_empty(self):
        assert bin_values([], n_bins=5) == []

    def test_all_same(self):
        bins = bin_values([0.5, 0.5, 0.5], n_bins=5)
        assert all(b == 0 for b in bins)

    def test_binary_values(self):
        """Database collision is binary (0 or 1)."""
        values = [0, 0, 0, 1, 1, 0, 1, 0, 1, 1]
        bins = bin_values(values, n_bins=2)
        assert len(bins) == 10
        assert all(0 <= b < 2 for b in bins)


# ---------------------------------------------------------------------------
# Curve Fitting Tests
# ---------------------------------------------------------------------------

class TestFitCurve:
    """Test curve shape fitting."""

    def test_monotonic_increasing(self):
        x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
        y = np.array([0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 0.95])
        result = fit_curve(x, y)
        assert result["best_shape"] in ("monotonic_increasing", "inverted_u")
        assert result["r2"]["monotonic_increasing"] > 0.5

    def test_monotonic_decreasing(self):
        x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
        y = np.array([0.95, 0.85, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.15, 0.1])
        result = fit_curve(x, y)
        assert result["best_shape"] in ("monotonic_decreasing", "inverted_u")
        assert result["r2"]["monotonic_decreasing"] > 0.5

    def test_inverted_u(self):
        x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
        y = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 0.85, 0.7, 0.5, 0.3, 0.1])
        result = fit_curve(x, y)
        assert result["best_shape"] in ("inverted_u", "threshold")
        assert result["r2"]["inverted_u"] > 0.3

    def test_flat(self):
        x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
        y = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        result = fit_curve(x, y)
        assert result["best_shape"] == "flat"

    def test_insufficient_data(self):
        result = fit_curve(np.array([1.0]), np.array([0.5]))
        assert result["best_shape"] == "insufficient_data"


# ---------------------------------------------------------------------------
# Curve Analysis Tests
# ---------------------------------------------------------------------------

class TestAnalyzeCurve:
    """Test full curve analysis."""

    def test_basic_analysis(self):
        difficulty = list(np.random.rand(100))
        utility = list(1.0 - np.array(difficulty) + np.random.rand(100) * 0.1)
        risk = list(np.array(difficulty) * 0.5 + np.random.rand(100) * 0.1)
        result = analyze_curve(difficulty, utility, risk, "test_metric")
        assert result["metric"] == "test_metric"
        assert result["n_candidates"] == 100
        assert result["n_bins"] <= N_BINS
        assert "utility_curve_shape" in result
        assert "risk_curve_shape" in result
        assert "utility_r2" in result
        assert "risk_r2" in result


class TestScoringMargin:
    """Test scoring margin computation."""

    def test_basic_margin(self):
        preds = [
            {"group_id": "g1", "candidate_id": "c1", "score": 0.9, "label": 1},
            {"group_id": "g1", "candidate_id": "c2", "score": 0.3, "label": 0},
            {"group_id": "g2", "candidate_id": "c3", "score": 0.8, "label": 1},
            {"group_id": "g2", "candidate_id": "c4", "score": 0.75, "label": 0},
        ]
        margins = compute_scoring_margin(preds)
        assert margins["c1"] == pytest.approx(0.6, abs=0.01)
        assert margins["c2"] == pytest.approx(0.6, abs=0.01)
        assert margins["c3"] == pytest.approx(0.05, abs=0.01)

    def test_single_candidate_group(self):
        preds = [{"group_id": "g1", "candidate_id": "c1", "score": 0.9, "label": 1}]
        margins = compute_scoring_margin(preds)
        assert margins["c1"] == 0.0


class TestDownstreamLoss:
    """Test downstream loss computation."""

    def test_basic_loss(self):
        preds = [
            {"group_id": "g1", "candidate_id": "c1", "score": 0.9, "label": 1},
            {"group_id": "g1", "candidate_id": "c2", "score": 0.3, "label": 0},
        ]
        candidates = {"c1": {"gold_candidate": True}, "c2": {"gold_candidate": False}}
        losses = compute_downstream_loss(preds, candidates)
        # c1 is rank 1, label 1 -> MRR=1, loss=0
        assert losses["c1"] == 0.0
        # c2 is rank 2, label 0 -> MRR=0, loss=1
        assert losses["c2"] == 1.0


# ---------------------------------------------------------------------------
# Verdict Tests
# ---------------------------------------------------------------------------

class TestG8AVerdict:
    """Test G8-A verdict computation."""

    def test_go_reproduced(self):
        """GO: >=2 metrics reproduced across >=2 datasets and >=2 scorers."""
        curves = []
        for metric in ["edit_distance", "positive_similarity"]:
            for dataset in ["g3_v2_test", "g6_hte"]:
                for scorer in ["chemformer", "gnn"]:
                    curves.append({
                        "metric": metric,
                        "dataset": dataset,
                        "scorer": scorer,
                        "utility_curve_shape": "inverted_u",
                    })
        verdict = compute_g8a_verdict(curves)
        assert verdict["verdict"] == "GO"
        assert verdict["n_reproduced_metrics"] >= 2

    def test_partial_go_single_metric(self):
        curves = []
        for dataset in ["g3_v2_test", "g6_hte"]:
            for scorer in ["chemformer", "gnn"]:
                curves.append({
                    "metric": "edit_distance",
                    "dataset": dataset,
                    "scorer": scorer,
                    "utility_curve_shape": "inverted_u",
                })
        verdict = compute_g8a_verdict(curves)
        assert verdict["verdict"] == "PARTIAL_GO"

    def test_no_go_no_relationship(self):
        curves = [
            {"metric": "edit_distance", "dataset": "g3_v2_test", "scorer": "chemformer",
             "utility_curve_shape": "flat"},
        ]
        verdict = compute_g8a_verdict(curves)
        assert verdict["verdict"] == "NO_GO"

    def test_no_go_empty(self):
        verdict = compute_g8a_verdict([])
        assert verdict["verdict"] == "NO_GO"


# ---------------------------------------------------------------------------
# Constants Tests
# ---------------------------------------------------------------------------

class TestConstants:
    """Test constants are properly defined."""

    def test_g8a_metrics(self):
        assert len(DIFFICULTY_METRICS) == 7
        assert "edit_distance" in DIFFICULTY_METRICS
        assert "false_negative_risk" in DIFFICULTY_METRICS

    def test_g8a_shapes(self):
        assert len(CURVE_SHAPES) == 5
        assert "inverted_u" in CURVE_SHAPES
        assert "monotonic_decreasing" in CURVE_SHAPES
