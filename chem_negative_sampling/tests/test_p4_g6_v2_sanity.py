#!/usr/bin/env python3
"""Toy-data sanity tests for G3/G6 v2 statistical infrastructure.

Validates:
1. Label direction: T1 (low-yield positive) produces correct AUPRC direction
2. Metric monotonicity: better scores produce better metrics
3. CI coverage: 95% CI covers true effect in ~95% of simulations
4. Type-I error: under no signal, false positive rate ~ alpha
5. Power: under known effect, detection rate meets expectation
"""
from __future__ import annotations

import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "chem_negative_sampling"))

from pc_cng.paired_cluster_inference import (
    auprc_metric,
    ece_metric,
    holm_correction,
    mae_metric,
    mrr_metric,
    paired_cluster_bootstrap,
    paired_permutation_test,
)


def _make_toy_records(
    n_clusters: int = 20,
    n_per_cluster: int = 10,
    signal_strength: float = 0.5,
    seed: int = 42,
) -> list[dict]:
    """Generate toy records with known signal.

    signal_strength=0 -> no signal (for type-I error test)
    signal_strength>0 -> positive class has higher score (for power test)
    """
    rng = random.Random(seed)
    records = []
    for c in range(n_clusters):
        for i in range(n_per_cluster):
            label = 1 if rng.random() < 0.5 else 0
            noise = rng.random()
            score = label * signal_strength + noise * (1 - signal_strength)
            records.append({
                "label": label,
                "score": score,
                "experimental_group": f"cluster_{c}",
                "measured_yield": 80 if label == 1 else 20,
                "products": f"C{'C' * (i % 5)}O" if label == 1 else f"C{'N' * (i % 5)}O",
            })
    return records


def _make_paired_toy(
    n_clusters: int = 20,
    n_per_cluster: int = 10,
    delta: float = 0.0,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Generate paired toy records for challenger vs baseline.

    delta=0 -> no difference (type-I error test)
    delta>0 -> challenger better (power test)
    """
    rng = random.Random(seed)
    challenger = []
    baseline = []
    for c in range(n_clusters):
        for i in range(n_per_cluster):
            label = 1 if rng.random() < 0.5 else 0
            base_score = label * 0.5 + rng.random() * 0.5
            ch_score = base_score + delta + rng.random() * 0.1 - 0.05
            rec_ch = {"label": label, "score": ch_score, "experimental_group": f"cluster_{c}",
                      "measured_yield": 80 if label == 1 else 20, "products": "CCO"}
            rec_bl = {"label": label, "score": base_score, "experimental_group": f"cluster_{c}",
                      "measured_yield": 80 if label == 1 else 20, "products": "CCO"}
            challenger.append(rec_ch)
            baseline.append(rec_bl)
    return challenger, baseline


class TestLabelDirection:
    """Test 1: T1 (low-yield positive) produces correct AUPRC direction."""

    def test_low_yield_positive_direction(self):
        """AUPRC should be > 0.5 when low-yield (label=1) has higher score."""
        records = []
        rng = random.Random(42)
        for i in range(200):
            is_low_yield = rng.random() < 0.5
            # Low yield gets HIGHER score (correct direction for T1)
            score = 0.8 if is_low_yield else 0.2
            records.append({
                "label": 1 if is_low_yield else 0,
                "score": score + rng.random() * 0.1,
            })
        auprc = auprc_metric(records)
        assert auprc > 0.7, f"AUPRC should be high when direction correct: {auprc}"

    def test_wrong_direction_fails(self):
        """AUPRC should be < 0.5 when direction is wrong (high yield gets higher score)."""
        records = []
        rng = random.Random(42)
        for i in range(200):
            is_high_yield = rng.random() < 0.5
            # High yield gets higher score (WRONG for T1 where low yield is positive)
            score = 0.8 if is_high_yield else 0.2
            records.append({
                "label": 1 if not is_high_yield else 0,  # low yield = 1
                "score": score + rng.random() * 0.1,
            })
        auprc = auprc_metric(records)
        assert auprc < 0.5, f"AUPRC should be low when direction wrong: {auprc}"


class TestMetricMonotonicity:
    """Test 2: Better scores produce better metrics."""

    def test_auprc_monotonic(self):
        """AUPRC should increase as signal strength increases."""
        auprcs = []
        for strength in [0.0, 0.3, 0.6, 0.9]:
            records = _make_toy_records(signal_strength=strength, seed=42)
            auprcs.append(auprc_metric(records))
        for i in range(len(auprcs) - 1):
            assert auprcs[i + 1] >= auprcs[i] - 0.05, (
                f"AUPRC should be monotonic: {auprcs}"
            )

    def test_mrr_monotonic(self):
        """MRR should increase as signal strength increases."""
        mrrs = []
        for strength in [0.0, 0.3, 0.6, 0.9]:
            records = _make_toy_records(signal_strength=strength, seed=42)
            # Add group_id for MRR
            for r in records:
                r["group_id"] = r["experimental_group"]
            mrrs.append(mrr_metric(records))
        assert mrrs[-1] > mrrs[0], f"MRR should increase: {mrrs}"


class TestCICoverage:
    """Test 3: 95% CI should cover true effect in ~95% of simulations."""

    def test_coverage_95(self):
        """Run 200 simulations, check that 95% CI covers true delta."""
        true_delta = 0.1
        n_sims = 200
        covered = 0
        for sim in range(n_sims):
            ch, bl = _make_paired_toy(delta=true_delta, seed=sim)
            ci = paired_cluster_bootstrap(
                ch, bl, auprc_metric,
                n_bootstrap=500, seed=sim,
            )
            if ci["delta_ci_low"] <= true_delta <= ci["delta_ci_high"]:
                covered += 1
        coverage_rate = covered / n_sims
        # Should be approximately 95% (allow some slack)
        assert coverage_rate > 0.85, f"CI coverage too low: {coverage_rate:.2%}, expected ~95%"


class TestTypeIError:
    """Test 4: Under no signal (delta=0), false positive rate should be ~alpha."""

    def test_type_i_error_no_signal(self):
        """Under null (delta=0), false positive rate should be ~5%."""
        n_sims = 200
        false_positives = 0
        for sim in range(n_sims):
            ch, bl = _make_paired_toy(delta=0.0, seed=sim)
            ci = paired_cluster_bootstrap(
                ch, bl, auprc_metric,
                n_bootstrap=500, seed=sim,
            )
            # False positive: CI excludes 0 when true delta is 0
            if ci["ci_all_positive"]:
                false_positives += 1
        fpr = false_positives / n_sims
        # Type-I error should be approximately 5% (allow slack for bootstrap)
        assert fpr < 0.15, f"Type-I error too high: {fpr:.2%}, expected ~5%"

    def test_negative_control_no_signal(self):
        """Negative control (randomized labels) should show no signal."""
        records = _make_toy_records(signal_strength=0.0, seed=42)
        # Randomize labels
        labels = [r["label"] for r in records]
        random.Random(123).shuffle(labels)
        for r, l in zip(records, labels):
            r["label"] = l
        auprc = auprc_metric(records)
        # AUPRC should be near random (~0.5 for balanced)
        assert 0.3 < auprc < 0.7, f"Randomized labels should give ~0.5 AUPRC: {auprc}"


class TestPower:
    """Test 5: Under known effect, detection rate should meet expectation."""

    def test_power_known_effect(self):
        """With a strong effect (delta=0.2), power should be high (>70%).."""
        n_sims = 200
        detected = 0
        for sim in range(n_sims):
            ch, bl = _make_paired_toy(delta=0.2, seed=sim)
            ci = paired_cluster_bootstrap(
                ch, bl, auprc_metric,
                n_bootstrap=500, seed=sim,
            )
            if ci["ci_all_positive"]:
                detected += 1
        power = detected / n_sims
        assert power > 0.70, f"Power too low with strong effect: {power:.2%}, expected >70%"

    def test_power_increases_with_effect(self):
        """Power should increase with effect size."""
        powers = []
        for delta in [0.05, 0.10, 0.20, 0.30]:
            detected = 0
            n_sims = 100
            for sim in range(n_sims):
                ch, bl = _make_paired_toy(delta=delta, seed=sim)
                ci = paired_cluster_bootstrap(
                    ch, bl, auprc_metric,
                    n_bootstrap=300, seed=sim,
                )
                if ci["ci_all_positive"]:
                    detected += 1
            powers.append(detected / n_sims)
        # Power should generally increase
        assert powers[-1] > powers[0], f"Power should increase with effect: {powers}"


class TestHolmCorrection:
    """Test Holm multiple comparison correction."""

    def test_holm_corrects_familywise_error(self):
        """With 10 tests at alpha=0.05, Holm should control FWER."""
        # Simulate 10 tests with no true effect
        p_values = []
        for i in range(10):
            ch, bl = _make_paired_toy(delta=0.0, seed=i)
            ci = paired_cluster_bootstrap(
                ch, bl, auprc_metric,
                n_bootstrap=300, seed=i,
            )
            p_values.append(ci["p_value"])
        holm = holm_correction(p_values, alpha=0.05)
        # At least some should be non-rejected (since no true effect)
        n_rejected = sum(1 for h in holm if h["rejected"])
        assert n_rejected <= 2, f"Too many rejections under null: {n_rejected}/10"

    def test_holm_preserves_true_positives(self):
        """With one true effect, Holm should detect it."""
        p_values = []
        # 9 null tests + 1 with strong effect
        for i in range(9):
            ch, bl = _make_paired_toy(delta=0.0, seed=i)
            ci = paired_cluster_bootstrap(ch, bl, auprc_metric, n_bootstrap=300, seed=i)
            p_values.append(ci["p_value"])
        # Strong effect test
        ch, bl = _make_paired_toy(delta=0.5, seed=99)
        ci = paired_cluster_bootstrap(ch, bl, auprc_metric, n_bootstrap=300, seed=99)
        p_values.append(ci["p_value"])
        holm = holm_correction(p_values, alpha=0.05)
        # The last one (true effect) should be rejected
        assert holm[-1]["rejected"], "True effect should be detected after Holm correction"


class TestPermutationTest:
    """Test paired permutation test."""

    def test_permutation_detects_signal(self):
        """Permutation test should detect a real signal."""
        ch, bl = _make_paired_toy(delta=0.3, seed=42)
        result = paired_permutation_test(
            [r["score"] for r in ch],
            [r["score"] for r in bl],
            n_permutations=1000,
        )
        assert result["p_value"] < 0.05, f"Permutation test should detect signal: p={result['p_value']}"

    def test_permutation_no_signal(self):
        """Permutation test should not detect signal under null."""
        ch, bl = _make_paired_toy(delta=0.0, seed=42)
        result = paired_permutation_test(
            [r["score"] for r in ch],
            [r["score"] for r in bl],
            n_permutations=1000,
        )
        assert result["p_value"] > 0.05, f"Permutation test should not detect signal under null: p={result['p_value']}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
