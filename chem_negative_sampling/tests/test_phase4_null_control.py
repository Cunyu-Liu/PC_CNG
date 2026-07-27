"""Unit tests for run_phase4_null_control (randomized-label null arm).

The null arm reuses the EXACT training pairs of the shuffled_parent arm
(via ``build_main_arm_train(METHOD_SHUFFLED_PARENT, ...)``) and only
permutes the label vector.  These tests pin down the two properties that
make it a valid "~0.5 hard control":

1. the permutation preserves the class balance (same #pos/#neg) while
   destroying the (X -> y) pairing;
2. a classifier trained on such permuted labels cannot exceed ~0.5 AUPRC
   on held-out data drawn from the same (separable) distribution - i.e.
   any AUPRC >> 0.5 for the real pipeline would indicate leakage, not
   signal.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


class TestLabelPermutation(unittest.TestCase):
    """Property tests for the null-arm label permutation."""

    def test_permutation_preserves_class_balance(self):
        y = np.array([1, 0] * 500, dtype=np.float32)
        rng = np.random.default_rng(20260726 + 777)
        y_null = rng.permutation(y)
        self.assertEqual(int(y_null.sum()), int(y.sum()))
        self.assertEqual(len(y_null), len(y))

    def test_permutation_is_seeded_and_reproducible(self):
        y = np.arange(100, dtype=np.float32)
        a = np.random.default_rng(20260726 + 777).permutation(y)
        b = np.random.default_rng(20260726 + 777).permutation(y)
        np.testing.assert_array_equal(a, b)

    def test_permuted_labels_destroy_signal(self):
        """Separable data + permuted labels -> AUPRC ~ base rate."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import average_precision_score
        rng = np.random.default_rng(0)
        n = 400
        # Two well-separated clusters; true labels align with clusters.
        X_pos = rng.normal(loc=+2.0, scale=0.5, size=(n // 2, 8))
        X_neg = rng.normal(loc=-2.0, scale=0.5, size=(n // 2, 8))
        X = np.vstack([X_pos, X_neg])
        y = np.array([1] * (n // 2) + [0] * (n // 2), dtype=np.float32)
        # Sanity: true labels are learnable (AUPRC ~ 1).
        clf = LogisticRegression(max_iter=200).fit(X, y)
        true_auprc = average_precision_score(y, clf.predict_proba(X)[:, 1])
        self.assertGreater(true_auprc, 0.95)
        # Permuted labels: the model can only learn noise.
        y_null = np.random.default_rng(20260726 + 777).permutation(y)
        clf_null = LogisticRegression(max_iter=200).fit(X, y_null)
        null_auprc = average_precision_score(
            y, clf_null.predict_proba(X)[:, 1])
        # Base rate is 0.5; allow generous slack for finite-sample noise.
        self.assertLess(abs(null_auprc - 0.5), 0.15,
                        msg=f"null AUPRC {null_auprc:.3f} too far from 0.5")


class TestNullArmWiring(unittest.TestCase):
    """The null arm must reuse the shuffled_parent training pairs exactly."""

    def test_imports_and_constant(self):
        from pc_cng import run_phase4_null_control as mod
        self.assertEqual(mod.METHOD_NULL, "null_randomized_label")
        # Reuses the main-run builder for identical pairs.
        from pc_cng.run_phase4_fixed_testset import build_main_arm_train
        self.assertIs(mod.build_main_arm_train, build_main_arm_train)

    def test_output_csv_name(self):
        from pc_cng import run_phase4_null_control as mod
        name = f"author_lab__{mod.METHOD_NULL}__semi_hard.csv"
        self.assertEqual(name,
                         "author_lab__null_randomized_label__semi_hard.csv")


if __name__ == "__main__":
    unittest.main()
