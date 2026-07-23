"""Tests for P4-G8B cross-family transfer experiments.

Covers: MLP model, MRR/AUPRC/ECE metrics, verdict computation.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# Suppress RDKit warnings
os.environ["RDKitRDLogger"] = "0"

# Ensure chem_negative_sampling is importable
_THIS = Path(__file__).resolve()
for _cand in (_THIS.parents[1], _THIS.parents[2] / "chem_negative_sampling"):
    if (_cand / "pc_cng").is_dir() and str(_cand) not in sys.path:
        sys.path.insert(0, str(_cand))

from pc_cng.p4_g8b_cross_family_transfer import (
    MorganMLPScorer,
    compute_mrr,
    compute_auprc,
    compute_ece,
    compute_verdict as compute_g8b_verdict,
    MIN_FAMILY_SIZE,
)


# ---------------------------------------------------------------------------
# Model Tests
# ---------------------------------------------------------------------------

class TestMorganMLPScorer:
    """Test MLP model."""

    def test_forward(self):
        model = MorganMLPScorer(n_bits=128, hidden=32)
        x = torch.randn(4, 128)
        out = model(x)
        assert out.shape == (4,)

    def test_parameters(self):
        model = MorganMLPScorer(n_bits=128, hidden=32)
        params = sum(p.numel() for p in model.parameters())
        assert params > 0


# ---------------------------------------------------------------------------
# Metric Tests
# ---------------------------------------------------------------------------

class TestComputeMRR:
    """Test MRR computation."""

    def test_perfect_ranking(self):
        scores = np.array([0.9, 0.1])
        labels = np.array([1, 0])
        assert compute_mrr(scores, labels) == 1.0

    def test_worst_ranking(self):
        scores = np.array([0.1, 0.9])
        labels = np.array([1, 0])
        assert compute_mrr(scores, labels) == 0.5

    def test_no_positives(self):
        scores = np.array([0.9, 0.1])
        labels = np.array([0, 0])
        assert compute_mrr(scores, labels) == 0.0

    def test_group_based(self):
        scores = np.array([0.9, 0.1, 0.8, 0.2])
        labels = np.array([1, 0, 1, 0])
        mrr = compute_mrr(scores, labels, group_size=2)
        assert mrr == 1.0  # both groups have positive at rank 1


class TestComputeAUPRC:
    """Test AUPRC computation."""

    def test_perfect(self):
        scores = np.array([0.9, 0.8, 0.1, 0.05])
        labels = np.array([1, 1, 0, 0])
        assert compute_auprc(scores, labels) == 1.0

    def test_random(self):
        scores = np.array([0.5, 0.5, 0.5, 0.5])
        labels = np.array([1, 0, 1, 0])
        auprc = compute_auprc(scores, labels)
        assert 0.0 < auprc < 1.0

    def test_single_class(self):
        scores = np.array([0.9, 0.1])
        labels = np.array([1, 1])
        assert compute_auprc(scores, labels) == 0.0


class TestComputeECE:
    """Test ECE computation."""

    def test_perfect_calibration(self):
        scores = np.array([0.0, 0.0, 1.0, 1.0])
        labels = np.array([0, 0, 1, 1])
        assert compute_ece(scores, labels) == 0.0

    def test_miscalibrated(self):
        scores = np.array([0.9, 0.9, 0.1, 0.1])
        labels = np.array([0, 0, 1, 1])  # completely wrong
        ece = compute_ece(scores, labels)
        assert ece > 0.5


# ---------------------------------------------------------------------------
# Verdict Tests
# ---------------------------------------------------------------------------

class TestG8BVerdict:
    """Test G8-B verdict computation."""

    def test_go_multiple_positive(self):
        results = [
            # Baselines
            {"source_family": "A", "target_family": "A", "method": "baseline",
             "target_mrr": 0.5, "seed": 1},
            {"source_family": "B", "target_family": "B", "method": "baseline",
             "target_mrr": 0.4, "seed": 1},
            # Positive transfers
            {"source_family": "A", "target_family": "B", "method": "direct",
             "target_mrr": 0.45, "seed": 1},
            {"source_family": "A", "target_family": "B", "method": "direct",
             "target_mrr": 0.43, "seed": 2},
            {"source_family": "B", "target_family": "A", "method": "direct",
             "target_mrr": 0.52, "seed": 1},
            {"source_family": "B", "target_family": "A", "method": "direct",
             "target_mrr": 0.51, "seed": 2},
        ]
        verdict = compute_g8b_verdict(results)
        assert verdict["verdict"] in ("GO", "PARTIAL_GO")
        assert len(verdict["positive_directions"]) >= 2

    def test_no_go_all_negative(self):
        results = [
            {"source_family": "A", "target_family": "A", "method": "baseline",
             "target_mrr": 0.5, "seed": 1},
            {"source_family": "B", "target_family": "B", "method": "baseline",
             "target_mrr": 0.4, "seed": 1},
            {"source_family": "A", "target_family": "B", "method": "direct",
             "target_mrr": 0.3, "seed": 1},
            {"source_family": "B", "target_family": "A", "method": "direct",
             "target_mrr": 0.2, "seed": 1},
        ]
        verdict = compute_g8b_verdict(results)
        assert verdict["verdict"] == "NO_GO"
        assert len(verdict["negative_directions"]) >= 2

    def test_partial_go_one_positive(self):
        results = [
            {"source_family": "A", "target_family": "A", "method": "baseline",
             "target_mrr": 0.5, "seed": 1},
            {"source_family": "B", "target_family": "B", "method": "baseline",
             "target_mrr": 0.4, "seed": 1},
            {"source_family": "A", "target_family": "B", "method": "direct",
             "target_mrr": 0.45, "seed": 1},  # positive
            {"source_family": "B", "target_family": "A", "method": "direct",
             "target_mrr": 0.3, "seed": 1},  # negative
        ]
        verdict = compute_g8b_verdict(results)
        assert verdict["verdict"] == "PARTIAL_GO"
        assert len(verdict["positive_directions"]) >= 1
        assert len(verdict["negative_directions"]) >= 1


# ---------------------------------------------------------------------------
# Constants Tests
# ---------------------------------------------------------------------------

class TestConstants:
    """Test constants are properly defined."""

    def test_g8b_min_family_size(self):
        assert MIN_FAMILY_SIZE > 0
