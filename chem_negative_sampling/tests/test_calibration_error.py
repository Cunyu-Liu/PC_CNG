"""Unit tests for chem_negative_sampling.pc_cng.evaluate_calibration_error.

Tests ECE, MCE, and Brier score against hand-computed values to guarantee
correctness.  Uses numpy arrays as inputs; no model training or GPU needed.
"""

from __future__ import annotations

import csv
import math
import os
import sys
import tempfile
import unittest

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(THIS_DIR)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

from pc_cng.evaluate_calibration_error import (  # noqa: E402
    compute_all_calibration,
    compute_brier_score,
    compute_ece,
    compute_mce,
    discover_seed_dirs,
    evaluate_seed,
    load_predictions,
)


class ECETest(unittest.TestCase):
    def test_perfect_calibration(self) -> None:
        # All predictions are exactly correct and confident at 1.0 or 0.0.
        scores = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float64)
        labels = np.array([1, 1, 0, 0], dtype=np.int64)
        ece = compute_ece(scores, labels, n_bins=10)
        self.assertAlmostEqual(ece, 0.0, places=6)

    def test_uniform_mis_calibration(self) -> None:
        # 4 predictions in bin [0.1, 0.2): confidence=mean(scores)=0.16, accuracy=1.0
        # ECE = (4/4) * |1.0 - 0.16| = 0.84 (ECE uses mean of scores, not bin midpoint)
        scores = np.array([0.1, 0.15, 0.19, 0.2 - 1e-9], dtype=np.float64)
        labels = np.array([1, 1, 1, 1], dtype=np.int64)
        ece = compute_ece(scores, labels, n_bins=10)
        self.assertAlmostEqual(ece, 0.84, places=4)

    def test_empty_returns_zero(self) -> None:
        ece = compute_ece(np.array([]), np.array([], dtype=np.int64))
        self.assertEqual(ece, 0.0)

    def test_two_bin_hand_computed(self) -> None:
        # bin0 [0, 0.5): scores 0.2, 0.4; labels 0, 1 -> conf=0.3, acc=0.5, gap=0.2
        # bin1 [0.5, 1.0]: scores 0.6, 0.8; labels 0, 1 -> conf=0.7, acc=0.5, gap=0.2
        # n=4; ECE = (2/4)*0.2 + (2/4)*0.2 = 0.2
        scores = np.array([0.2, 0.4, 0.6, 0.8], dtype=np.float64)
        labels = np.array([0, 1, 0, 1], dtype=np.int64)
        ece = compute_ece(scores, labels, n_bins=2)
        self.assertAlmostEqual(ece, 0.2, places=6)


class MCETest(unittest.TestCase):
    def test_perfect_calibration(self) -> None:
        scores = np.array([1.0, 0.0], dtype=np.float64)
        labels = np.array([1, 0], dtype=np.int64)
        mce = compute_mce(scores, labels, n_bins=10)
        self.assertAlmostEqual(mce, 0.0, places=6)

    def test_max_gap(self) -> None:
        # bin [0.1, 0.2): conf=0.15, acc=1.0 -> gap 0.85
        scores = np.array([0.15, 0.15], dtype=np.float64)
        labels = np.array([1, 1], dtype=np.int64)
        mce = compute_mce(scores, labels, n_bins=10)
        self.assertAlmostEqual(mce, 0.85, places=4)

    def test_empty_returns_zero(self) -> None:
        mce = compute_mce(np.array([]), np.array([], dtype=np.int64))
        self.assertEqual(mce, 0.0)


class BrierTest(unittest.TestCase):
    def test_perfect(self) -> None:
        scores = np.array([1.0, 0.0], dtype=np.float64)
        labels = np.array([1, 0], dtype=np.int64)
        self.assertAlmostEqual(compute_brier_score(scores, labels), 0.0, places=6)

    def test_worst_case(self) -> None:
        scores = np.array([1.0, 0.0], dtype=np.float64)
        labels = np.array([0, 1], dtype=np.int64)
        self.assertAlmostEqual(compute_brier_score(scores, labels), 1.0, places=6)

    def test_hand_computed(self) -> None:
        # (0.8-1)^2 + (0.3-0)^2 = 0.04 + 0.09 = 0.13 / 2 = 0.065
        scores = np.array([0.8, 0.3], dtype=np.float64)
        labels = np.array([1, 0], dtype=np.int64)
        self.assertAlmostEqual(compute_brier_score(scores, labels), 0.065, places=6)

    def test_empty(self) -> None:
        self.assertEqual(compute_brier_score(np.array([]), np.array([], dtype=np.int64)), 0.0)


class ComputeAllTest(unittest.TestCase):
    def test_returns_all_keys(self) -> None:
        scores = np.array([0.2, 0.7, 0.9], dtype=np.float64)
        labels = np.array([0, 1, 1], dtype=np.int64)
        result = compute_all_calibration(scores, labels, n_bins=5)
        for key in ["n", "n_bins", "ece", "mce", "brier", "mean_score", "positive_rate", "bins"]:
            self.assertIn(key, result)
        self.assertEqual(result["n"], 3)
        self.assertEqual(result["n_bins"], 5)
        self.assertEqual(len(result["bins"]), 5)
        self.assertAlmostEqual(result["brier"], compute_brier_score(scores, labels))

    def test_bins_cover_full_range(self) -> None:
        scores = np.array([0.0, 0.5, 1.0], dtype=np.float64)
        labels = np.array([0, 1, 1], dtype=np.int64)
        result = compute_all_calibration(scores, labels, n_bins=2)
        # score=1.0 must land in the last bin (inclusive)
        total_count = sum(b.get("count", 0) for b in result["bins"])
        self.assertEqual(total_count, 3)


class LoadPredictionsTest(unittest.TestCase):
    def test_load_valid_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "test_predictions.csv")
            with open(path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["source_id", "dataset", "reaction_class", "label", "score", "reaction_smiles"])
                writer.writeheader()
                writer.writerow({"source_id": "a", "dataset": "x", "reaction_class": "", "label": "1", "score": "0.9", "reaction_smiles": "A>>B"})
                writer.writerow({"source_id": "b", "dataset": "x", "reaction_class": "", "label": "0", "score": "0.1", "reaction_smiles": "A>>C"})
            scores, labels, rows = load_predictions(path)
            self.assertEqual(len(scores), 2)
            self.assertEqual(len(labels), 2)
            self.assertAlmostEqual(scores[0], 0.9)
            self.assertEqual(labels[0], 1)
            self.assertEqual(labels[1], 0)

    def test_missing_columns_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bad.csv")
            with open(path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["source_id", "foo"])
                writer.writeheader()
                writer.writerow({"source_id": "a", "foo": "1"})
            with self.assertRaises(ValueError):
                load_predictions(path)

    def test_skips_bad_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "test_predictions.csv")
            with open(path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["source_id", "label", "score", "reaction_smiles"])
                writer.writeheader()
                writer.writerow({"source_id": "a", "label": "1", "score": "0.9", "reaction_smiles": "A>>B"})
                writer.writerow({"source_id": "b", "label": "NaN", "score": "0.1", "reaction_smiles": "A>>C"})
                writer.writerow({"source_id": "c", "label": "0", "score": "not_a_number", "reaction_smiles": "A>>D"})
            scores, labels, rows = load_predictions(path)
            self.assertEqual(len(scores), 1)  # only the first row is valid
            self.assertAlmostEqual(scores[0], 0.9)


class DiscoverSeedDirsTest(unittest.TestCase):
    def test_finds_unreacted_pattern(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            seed_dir = os.path.join(tmp, "unreacted_augmented_pairwise_seed20260710")
            os.makedirs(seed_dir)
            result = discover_seed_dirs(tmp, [20260710])
            self.assertIn(20260710, result)

    def test_finds_seedN_pattern(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            seed_dir = os.path.join(tmp, "seed20260710")
            os.makedirs(seed_dir)
            result = discover_seed_dirs(tmp, [20260710])
            self.assertIn(20260710, result)

    def test_missing_seed_not_returned(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = discover_seed_dirs(tmp, [20260710, 20260711])
            self.assertEqual(result, {})


class EvaluateSeedTest(unittest.TestCase):
    def test_evaluates_seed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            seed_dir = os.path.join(tmp, "unreacted_augmented_pairwise_seed20260710")
            os.makedirs(seed_dir)
            path = os.path.join(seed_dir, "test_predictions.csv")
            with open(path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["source_id", "dataset", "reaction_class", "label", "score", "reaction_smiles"])
                writer.writeheader()
                writer.writerow({"source_id": "a", "dataset": "x", "reaction_class": "", "label": "1", "score": "0.9", "reaction_smiles": "A>>B"})
                writer.writerow({"source_id": "b", "dataset": "x", "reaction_class": "", "label": "0", "score": "0.1", "reaction_smiles": "A>>C"})
            metrics = evaluate_seed(seed_dir, 20260710, n_bins=10)
            self.assertEqual(metrics["seed"], 20260710)
            self.assertEqual(metrics["n"], 2)
            self.assertIn("ece", metrics)
            self.assertIn("mce", metrics)
            self.assertIn("brier", metrics)


if __name__ == "__main__":
    unittest.main()
