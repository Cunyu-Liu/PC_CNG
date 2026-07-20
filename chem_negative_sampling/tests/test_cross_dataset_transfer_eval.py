"""Unit tests for chem_negative_sampling.pc_cng.run_cross_dataset_transfer_eval.

Tests cover data loading, transfer pair configuration, paired significance
calculation, and the overflow-safe sign-test.  No GPU or training is required.
"""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import unittest
from typing import List

# Ensure the chem_negative_sampling directory is on sys.path when tests are
# run from the repo root without installation.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(THIS_DIR)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

from pc_cng.run_cross_dataset_transfer_eval import (  # noqa: E402
    DATASET_REGISTRY,
    DATASET_SYNTHETIC_REGISTRY,
    TRANSFER_PAIRS,
    parse_seeds,
    paired_significance,
    sign_test_p_value_safe,
    per_group_top1,
    resolve_dataset_path,
)


class TransferPairConfigTest(unittest.TestCase):
    def test_four_transfer_pairs_defined(self) -> None:
        self.assertEqual(len(TRANSFER_PAIRS), 4)
        expected = {
            ("regiosqm20", "hitea"),
            ("hitea", "regiosqm20"),
            ("regiosqm20", "uspto"),
            ("hitea", "uspto"),
        }
        self.assertEqual(set(TRANSFER_PAIRS), expected)

    def test_dataset_registry_has_all_sources(self) -> None:
        for name in ["regiosqm20", "hitea", "uspto"]:
            self.assertIn(name, DATASET_REGISTRY)
            self.assertTrue(DATASET_REGISTRY[name].endswith(".csv"))

    def test_uspto_has_synthetic_fallback(self) -> None:
        # USPTO has no real_negative labels, so it must have a synthetic fallback.
        self.assertIn("uspto", DATASET_SYNTHETIC_REGISTRY)
        pos_csv, synth_csv = DATASET_SYNTHETIC_REGISTRY["uspto"]
        self.assertTrue(pos_csv.endswith(".csv"))
        self.assertTrue(synth_csv.endswith(".csv"))

    def test_regiosqm20_and_hitea_have_no_synthetic_fallback(self) -> None:
        # These datasets have real_negative labels and don't need synthetic fallback.
        self.assertNotIn("regiosqm20", DATASET_SYNTHETIC_REGISTRY)
        self.assertNotIn("hitea", DATASET_SYNTHETIC_REGISTRY)

    def test_resolve_dataset_path_relative(self) -> None:
        path = resolve_dataset_path("regiosqm20", "/fake/root")
        self.assertTrue(path.endswith("data/processed/regiosqm20_normalized.csv"))

    def test_resolve_dataset_path_unknown_raises(self) -> None:
        with self.assertRaises(ValueError):
            resolve_dataset_path("unknown_dataset", "/fake/root")


class ParseSeedsTest(unittest.TestCase):
    def test_parse_ten_seeds(self) -> None:
        raw = "20260710,20260711,20260712,20260713,20260714,20260715,20260716,20260717,20260718,20260719"
        seeds = parse_seeds(raw)
        self.assertEqual(len(seeds), 10)
        self.assertEqual(seeds[0], 20260710)
        self.assertEqual(seeds[-1], 20260719)

    def test_parse_handles_whitespace(self) -> None:
        seeds = parse_seeds(" 1 , 2 , 3 ")
        self.assertEqual(seeds, [1, 2, 3])


class SignTestSafeTest(unittest.TestCase):
    def test_all_positive_is_significant(self) -> None:
        values = [1.0] * 20
        p = sign_test_p_value_safe(values)
        self.assertLess(p, 0.01)

    def test_all_zero_returns_one(self) -> None:
        values = [0.0] * 10
        p = sign_test_p_value_safe(values)
        self.assertEqual(p, 1.0)

    def test_balanced_returns_one(self) -> None:
        values = [1.0, -1.0] * 10
        p = sign_test_p_value_safe(values)
        self.assertGreater(p, 0.5)

    def test_large_n_does_not_overflow(self) -> None:
        # The original sign_test_p_value overflows for n > ~1000 because of
        # math.comb.  The safe version must handle thousands of deltas.
        values = [1.0] * 2000 + [-1.0] * 500
        p = sign_test_p_value_safe(values)
        self.assertGreaterEqual(p, 0.0)
        self.assertLessEqual(p, 1.0)
        self.assertLess(p, 0.01)  # strongly significant


class PairedSignificanceTest(unittest.TestCase):
    def test_empty_deltas(self) -> None:
        result = paired_significance([], bootstrap_iterations=100, seed=42)
        self.assertEqual(result["n"], 0)
        self.assertEqual(result["delta_mean"], 0.0)
        self.assertEqual(result["paired_permutation_p"], 1.0)

    def test_positive_deltas_significant(self) -> None:
        # treatment consistently beats baseline
        deltas = [1.0] * 50
        result = paired_significance(deltas, bootstrap_iterations=500, seed=42)
        self.assertGreater(result["delta_mean"], 0.99)
        self.assertGreater(result["delta_ci95_low"], 0.5)
        self.assertLess(result["sign_test_p"], 0.01)

    def test_mixed_deltas_ci_brackets_zero(self) -> None:
        deltas = [1.0, -1.0] * 25
        result = paired_significance(deltas, bootstrap_iterations=500, seed=42)
        self.assertAlmostEqual(result["delta_mean"], 0.0, places=5)
        self.assertLessEqual(result["delta_ci95_low"], 0.0)
        self.assertGreaterEqual(result["delta_ci95_high"], 0.0)
        self.assertGreater(result["sign_test_p"], 0.3)


class PerGroupTop1Test(unittest.TestCase):
    def test_positive_ranked_first(self) -> None:
        rows = [
            {"group_id": "g1", "label": 1, "score": 0.9},
            {"group_id": "g1", "label": 0, "score": 0.1},
        ]
        result = per_group_top1(rows)
        self.assertEqual(result["g1"], 1.0)

    def test_positive_not_ranked_first(self) -> None:
        rows = [
            {"group_id": "g1", "label": 1, "score": 0.2},
            {"group_id": "g1", "label": 0, "score": 0.8},
        ]
        result = per_group_top1(rows)
        self.assertEqual(result["g1"], 0.0)

    def test_group_without_both_labels_skipped(self) -> None:
        rows = [
            {"group_id": "g1", "label": 1, "score": 0.9},
            {"group_id": "g2", "label": 0, "score": 0.1},
        ]
        result = per_group_top1(rows)
        self.assertEqual(result, {})

    def test_multiple_groups(self) -> None:
        rows = [
            {"group_id": "g1", "label": 1, "score": 0.9},
            {"group_id": "g1", "label": 0, "score": 0.1},
            {"group_id": "g2", "label": 1, "score": 0.3},
            {"group_id": "g2", "label": 0, "score": 0.7},
        ]
        result = per_group_top1(rows)
        self.assertEqual(result["g1"], 1.0)
        self.assertEqual(result["g2"], 0.0)


if __name__ == "__main__":
    unittest.main()
