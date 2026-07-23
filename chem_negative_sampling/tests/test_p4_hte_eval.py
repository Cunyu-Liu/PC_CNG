"""Tests for P4-G6 HTE evaluation (tasks, methods, metrics).

Run with::

    python3 -m pytest chem_negative_sampling/tests/test_p4_hte_eval.py -v
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import List

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CNS_ROOT = _REPO_ROOT / "chem_negative_sampling"
if str(_CNS_ROOT) not in sys.path:
    sys.path.insert(0, str(_CNS_ROOT))

from pc_cng.p4_g6_hte_eval import (
    METHODS,
    LOW_YIELD_THRESHOLDS,
    YIELD_BINS,
    _auprc,
    _ece,
    _brier,
    _spearman,
    _ndcg,
    _morgan_fp,
    compute_task_metrics,
    cluster_bootstrap_ci,
    train_and_score,
)


# ---------------------------------------------------------------------------
# Metric unit tests
# ---------------------------------------------------------------------------

class TestAUPRC:
    def test_perfect_ranking(self):
        assert _auprc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)

    def test_worst_ranking(self):
        # AUPRC for worst case is the positive rate
        labels = [1, 1, 0, 0]
        scores = [0.9, 0.8, 0.7, 0.6]  # reversed
        result = _auprc(labels, scores)
        assert 0.0 <= result <= 1.0

    def test_single_class(self):
        assert _auprc([1, 1, 1], [0.5, 0.6, 0.7]) == 0.0


class TestECE:
    def test_perfect_calibration(self):
        # All predictions match labels
        ece = _ece([1, 1, 0, 0], [1.0, 1.0, 0.0, 0.0])
        assert ece == pytest.approx(0.0, abs=1e-6)

    def test_worst_calibration(self):
        ece = _ece([0, 0, 1, 1], [1.0, 1.0, 0.0, 0.0])
        assert ece >= 0.5

    def test_range(self):
        ece = _ece([1, 0, 1, 0], [0.6, 0.4, 0.7, 0.3])
        assert 0.0 <= ece <= 1.0


class TestBrier:
    def test_perfect(self):
        assert _brier([1, 0], [1.0, 0.0]) == pytest.approx(0.0)

    def test_worst(self):
        assert _brier([1, 0], [0.0, 1.0]) == pytest.approx(1.0)


class TestSpearman:
    def test_perfect_correlation(self):
        assert _spearman([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)

    def test_perfect_anti_correlation(self):
        assert _spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)

    def test_no_correlation(self):
        # Use data with near-zero correlation
        r = _spearman([1, 2, 3, 4, 5], [3, 1, 5, 2, 4])
        assert -0.5 < r < 0.5

    def test_too_short(self):
        assert _spearman([1], [2]) == 0.0


class TestNDCG:
    def test_perfect_ranking(self):
        assert _ndcg([10, 5, 1], [0.9, 0.5, 0.1]) == pytest.approx(1.0)

    def test_worst_ranking(self):
        result = _ndcg([10, 5, 1], [0.1, 0.5, 0.9])
        assert 0.0 <= result < 1.0

    def test_single_item(self):
        assert _ndcg([5], [0.5]) == 0.0


class TestMorganFP:
    def test_valid_smiles(self):
        fp = _morgan_fp("CCO")
        assert len(fp) == 2048
        assert fp.sum() > 0

    def test_invalid_smiles(self):
        fp = _morgan_fp("not_smiles###")
        assert fp.sum() == 0

    def test_empty_smiles(self):
        fp = _morgan_fp("")
        assert fp.sum() == 0


# ---------------------------------------------------------------------------
# Task metrics tests
# ---------------------------------------------------------------------------

class TestComputeTaskMetrics:
    @staticmethod
    def _make_records(n: int = 50) -> list:
        records = []
        for i in range(n):
            yield_val = (i / n) * 100
            records.append({
                "measured_yield": yield_val,
                "experimental_group": f"SCRN_{i // 10}",
                "reaction_family": "TEST",
            })
        return records

    def test_all_metrics_present(self):
        records = self._make_records()
        scores = [r["measured_yield"] / 100 for r in records]
        metrics = compute_task_metrics(scores, records)
        expected_keys = [
            "t1_low_yield_auprc_5", "t1_low_yield_auprc_10",
            "t2_macro_auprc", "t3_yield_mae", "t3_spearman",
            "t4_plate_ndcg", "t5_condition_feasibility_auprc",
            "ece", "brier", "family_macro_auprc",
            "collision_sensitivity",
        ]
        for key in expected_keys:
            assert key in metrics, f"Missing metric: {key}"

    def test_perfect_scores(self):
        records = self._make_records()
        scores = [r["measured_yield"] / 100 for r in records]
        metrics = compute_task_metrics(scores, records)
        # Perfect ranking → high NDCG
        assert metrics["t4_plate_ndcg"] == pytest.approx(1.0, abs=0.01)
        # Perfect correlation → Spearman ~ 1
        assert metrics["t3_spearman"] > 0.9

    def test_constant_scores(self):
        records = self._make_records()
        scores = [0.5] * len(records)
        metrics = compute_task_metrics(scores, records)
        # Constant scores → low NDCG (but 0.0 for ties is implementation-dependent)
        assert metrics["t4_plate_ndcg"] >= 0.0


# ---------------------------------------------------------------------------
# Cluster bootstrap tests
# ---------------------------------------------------------------------------

class TestClusterBootstrap:
    def test_ci_bounds(self):
        records = [
            {"measured_yield": float(i % 10), "experimental_group": f"SCRN_{i // 5}"}
            for i in range(50)
        ]
        scores = [r["measured_yield"] / 10 for r in records]
        ci = cluster_bootstrap_ci(records, scores,
                                  lambda s, r: sum(s) / len(s),
                                  n_bootstrap=100, seed=42)
        assert "mean" in ci
        assert "ci_low" in ci
        assert "ci_high" in ci
        assert ci["ci_low"] <= ci["mean"] <= ci["ci_high"]

    def test_single_screen(self):
        records = [{"measured_yield": 1.0, "experimental_group": "SCRN_1"}]
        ci = cluster_bootstrap_ci(records, [0.5],
                                  lambda s, r: 1.0,
                                  n_bootstrap=10, seed=42)
        assert ci["mean"] == 0.0  # single screen → returns 0


# ---------------------------------------------------------------------------
# Methods tests
# ---------------------------------------------------------------------------

class TestMethods:
    @staticmethod
    def _make_train_test():
        train = [
            {"products": "c1ccccc1", "measured_yield": 50.0},
            {"products": "CCO", "measured_yield": 0.0},
            {"products": "CC(=O)O", "measured_yield": 80.0},
            {"products": "CCN", "measured_yield": 0.0},
        ]
        test = [
            {"products": "c1ccccc1", "measured_yield": 60.0},
            {"products": "CCO", "measured_yield": 1.0},
        ]
        return train, test

    def test_all_methods_present(self):
        assert set(METHODS) == {
            "positive_only", "tanimoto_baseline",
            "hard_label_pc_cng", "risk_aware_pc_cng",
            "observed_negative_upper_bound",
        }

    def test_positive_only(self):
        train, test = self._make_train_test()
        scores = train_and_score("positive_only", train, test)
        assert len(scores) == len(test)
        assert all(0.0 <= s <= 1.0 for s in scores)

    def test_tanimoto_baseline(self):
        train, test = self._make_train_test()
        scores = train_and_score("tanimoto_baseline", train, test)
        assert len(scores) == len(test)
        # c1ccccc1 is in train positives → high similarity
        assert scores[0] > scores[1]

    def test_observed_negative(self):
        train, test = self._make_train_test()
        scores = train_and_score("observed_negative_upper_bound", train, test)
        assert len(scores) == len(test)
        assert all(0.0 <= s <= 1.0 for s in scores)

    def test_hard_label_pc_cng(self):
        train, test = self._make_train_test()
        pc_cng_neg = ["CCC", "CCCC"]
        scores = train_and_score("hard_label_pc_cng", train, test,
                                 pc_cng_neg_smiles=pc_cng_neg)
        assert len(scores) == len(test)

    def test_risk_aware_pc_cng(self):
        train, test = self._make_train_test()
        pc_cng_neg = ["CCC", "CCCC"]
        fnr = {"c1": 0.2, "c2": 0.3}
        scores = train_and_score("risk_aware_pc_cng", train, test,
                                 pc_cng_neg_smiles=pc_cng_neg,
                                 fnr_by_cand=fnr)
        assert len(scores) == len(test)

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            train_and_score("not_a_method", [], [])
