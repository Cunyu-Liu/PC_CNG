#!/usr/bin/env python3
"""Quick sanity tests (reduced simulations for fast validation)."""
from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "chem_negative_sampling"))

from pc_cng.paired_cluster_inference import (
    auprc_metric,
    holm_correction,
    mrr_metric,
    paired_cluster_bootstrap,
    paired_permutation_test,
)


def _make_paired(n_clusters=10, n_per=10, delta=0.0, seed=42):
    """delta controls class separation improvement for challenger.

    delta=0: same separation (no signal)
    delta>0: challenger has better positive/negative separation (higher AUPRC)
    """
    rng = random.Random(seed)
    ch, bl = [], []
    for c in range(n_clusters):
        for i in range(n_per):
            label = 1 if rng.random() < 0.5 else 0
            # Baseline: significant overlap (AUPRC ~0.55-0.65)
            base_pos = 0.35 + rng.random() * 0.5   # positive ~0.35-0.85
            base_neg = 0.15 + rng.random() * 0.5   # negative ~0.15-0.65
            base_score = base_pos if label else base_neg
            # Challenger: improved separation by delta (positive shifted up, negative stays)
            ch_pos = min(1.0, base_pos + delta)    # positive shifted up
            ch_neg = base_neg                      # negative unchanged
            ch_score = ch_pos if label else ch_neg
            ch.append({"label": label, "score": ch_score, "experimental_group": f"c{c}"})
            bl.append({"label": label, "score": base_score, "experimental_group": f"c{c}"})
    return ch, bl


class TestLabelDirection:
    def test_correct_direction(self):
        recs = [{"label": 1, "score": 0.9}, {"label": 1, "score": 0.8},
                {"label": 0, "score": 0.1}, {"label": 0, "score": 0.2}] * 50
        assert auprc_metric(recs) > 0.7

    def test_wrong_direction(self):
        recs = [{"label": 1, "score": 0.1}, {"label": 1, "score": 0.2},
                {"label": 0, "score": 0.9}, {"label": 0, "score": 0.8}] * 50
        assert auprc_metric(recs) < 0.5


class TestMonotonicity:
    def test_auprc_increases(self):
        auprcs = []
        for s in [0.0, 0.5, 0.9]:
            rng = random.Random(42)
            recs = []
            for _ in range(200):
                label = 1 if rng.random() < 0.5 else 0
                recs.append({"label": label, "score": label * s + rng.random() * (1 - s)})
            auprcs.append(auprc_metric(recs))
        assert auprcs[2] > auprcs[0]


class TestTypeIError:
    def test_no_signal(self):
        fp = 0
        for sim in range(50):  # reduced from 200
            ch, bl = _make_paired(delta=0.0, seed=sim)
            ci = paired_cluster_bootstrap(ch, bl, auprc_metric, n_bootstrap=200, seed=sim)
            if ci["ci_all_positive"]:
                fp += 1
        fpr = fp / 50
        assert fpr < 0.20, f"FPR={fpr:.2%}"


class TestPower:
    def test_strong_effect(self):
        det = 0
        for sim in range(50):
            ch, bl = _make_paired(n_clusters=20, n_per=10, delta=0.3, seed=sim)
            ci = paired_cluster_bootstrap(ch, bl, auprc_metric, n_bootstrap=200, seed=sim)
            if ci["ci_all_positive"]:
                det += 1
        power = det / 50
        assert power > 0.30, f"Power={power:.2%}"


class TestHolm:
    def test_controls_fwer(self):
        pvals = []
        for i in range(5):
            ch, bl = _make_paired(delta=0.0, seed=i)
            ci = paired_cluster_bootstrap(ch, bl, auprc_metric, n_bootstrap=200, seed=i)
            pvals.append(ci["p_value"])
        holm = holm_correction(pvals)
        assert sum(1 for h in holm if h["rejected"]) <= 1


class TestPermutation:
    def test_detects_signal(self):
        ch, bl = _make_paired(delta=0.3, seed=42)
        r = paired_permutation_test([r["score"] for r in ch], [r["score"] for r in bl], n_permutations=500)
        assert r["p_value"] < 0.05

    def test_no_signal(self):
        ch, bl = _make_paired(delta=0.0, seed=42)
        r = paired_permutation_test([r["score"] for r in ch], [r["score"] for r in bl], n_permutations=500)
        assert r["p_value"] > 0.05


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
