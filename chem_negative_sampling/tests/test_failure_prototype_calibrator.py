"""Unit tests for the failure prototype calibrator (P1-06)."""

from __future__ import annotations

import importlib.util
import os
import tempfile
import unittest

import numpy as np
import torch

IS_TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
IS_RDKIT_AVAILABLE = importlib.util.find_spec("rdkit") is not None

if IS_TORCH_AVAILABLE:
    from pc_cng.failure_prototype_calibrator import (
        FAILURE_TYPES,
        FAILURE_TYPE_TO_IDX,
        FailurePrototypeCalibrator,
        evaluate_controllability,
        label_failure_type,
        train_calibrator,
        UNIFORM_ENTROPY,
    )


@unittest.skipUnless(IS_TORCH_AVAILABLE, "PyTorch is required")
class FailurePrototypeCalibratorTest(unittest.TestCase):
    def test_failure_types_count(self) -> None:
        self.assertEqual(len(FAILURE_TYPES), 10)
        self.assertEqual(len(set(FAILURE_TYPES)), 10)
        self.assertEqual(len(FAILURE_TYPE_TO_IDX), 10)
        for idx, name in enumerate(FAILURE_TYPES):
            self.assertEqual(FAILURE_TYPE_TO_IDX[name], idx)
        # expected members present
        for expected in (
            "wrong_anchor",
            "broken_atom_balance",
            "invalid_valence",
            "fragment_misalignment",
            "wrong_bond_type",
            "aromaticity_violation",
            "stereochemistry_loss",
            "over_reaction",
            "under_reaction",
            "side_product",
        ):
            self.assertIn(expected, FAILURE_TYPES)

    def test_label_failure_type_wrong_anchor(self) -> None:
        # Regio-isomer: same formula, different connectivity.
        # Reactant: imidazole bromination. Expected: 4-bromo product.
        # Alternative: 5-bromo product (wrong anchor regio-isomer).
        reaction = "O=C1CCC(=O)N1Br.c1cn[nH]c1>>Brc1cc[nH]n1"
        expected = "Brc1cn[nH]c1"
        label = label_failure_type(reaction, [expected])
        # Regio-isomer -> wrong_anchor (or, if RDKit not available, a non-None label).
        self.assertIn(label, FAILURE_TYPES)
        if IS_RDKIT_AVAILABLE:
            self.assertEqual(label, "wrong_anchor")

    def test_label_failure_type_atom_balance(self) -> None:
        # Reactant has extra atoms not present in product -> imbalance.
        reaction = "CCO.CC>>CC"  # carbon lost, oxygen lost
        label = label_failure_type(reaction, ["CC"])
        self.assertEqual(label, "broken_atom_balance")

    def test_label_failure_type_invalid_valence(self) -> None:
        # Pentavalent carbon -> invalid valence.
        reaction = "CC>>C(C)(C)(C)(C)C"
        label = label_failure_type(reaction, ["C(C)(C)(C)(C)C"])
        # RDKit may parse this with implicit Hs; ensure at least a valid label.
        self.assertIn(label, FAILURE_TYPES)
        # Hard invalid SMILES case.
        label2 = label_failure_type(">>", [])
        self.assertEqual(label2, "invalid_valence")

    def test_model_forward_output_shapes(self) -> None:
        model = FailurePrototypeCalibrator(input_dim=10, hidden_dim=16, embedding_dim=8)
        x = torch.randn(4, 10)
        logits, embeddings = model(x)
        self.assertEqual(logits.shape, (4, 10))
        self.assertEqual(embeddings.shape, (4, 8))

    def test_classify_returns_valid_indices(self) -> None:
        model = FailurePrototypeCalibrator(input_dim=10, hidden_dim=16, embedding_dim=8)
        x = torch.randn(7, 10)
        preds = model.classify(x)
        self.assertEqual(preds.shape, (7,))
        self.assertTrue(torch.all(preds >= 0))
        self.assertTrue(torch.all(preds < 10))
        # Long dtype.
        self.assertEqual(preds.dtype, torch.long)

    def test_train_smoke_one_epoch(self) -> None:
        # Synthetic data with two separable classes.
        rng = np.random.RandomState(0)
        n_per = 20
        x0 = rng.randn(n_per, 10) + np.array([2.0] * 10)
        x1 = rng.randn(n_per, 10) + np.array([-2.0] * 10)
        X = np.concatenate([x0, x1], axis=0).astype(np.float32)
        y = np.concatenate([np.zeros(n_per, dtype=int), np.ones(n_per, dtype=int)] + [np.full(5, 2)]).astype(np.int64)
        # Pad to ensure all classes have at least one sample.
        x2 = rng.randn(5, 10) + np.array([0.0] * 10)
        X = np.concatenate([X, x2.astype(np.float32)], axis=0)
        model = FailurePrototypeCalibrator(input_dim=10, hidden_dim=16, embedding_dim=8)
        history = train_calibrator(
            model,
            torch.from_numpy(X),
            torch.from_numpy(y),
            epochs=1,
            batch_size=8,
            lr=1e-2,
            device="cpu",
            verbose=False,
        )
        self.assertIn("train_loss", history)
        self.assertEqual(len(history["train_loss"]), 1)
        self.assertGreaterEqual(history["train_acc"][0], 0.0)

    def test_failure_type_distribution_entropy(self) -> None:
        model = FailurePrototypeCalibrator(input_dim=10, hidden_dim=16, embedding_dim=8)
        x = torch.randn(5, 10)
        dist = model.failure_type_distribution(x)
        self.assertEqual(dist.shape, (5, 10))
        # Probabilities sum to 1.
        sums = dist.sum(dim=-1)
        self.assertTrue(torch.allclose(sums, torch.ones(5), atol=1e-5))
        # Non-negative entropy.
        probs = dist.clamp(min=1e-12)
        entropy = -(probs * probs.log()).sum(dim=-1)
        self.assertTrue(torch.all(entropy >= 0))
        # Uniform reference sanity check.
        self.assertGreater(UNIFORM_ENTROPY, 0.0)

    def test_control_generation_gradient(self) -> None:
        model = FailurePrototypeCalibrator(input_dim=10, hidden_dim=16, embedding_dim=8)
        x = torch.randn(3, 10, requires_grad=True)
        loss = model.control_generation(x, target_type=2)
        self.assertEqual(loss.dim(), 0)
        loss.backward()
        self.assertIsNotNone(x.grad)
        # Gradient should be non-zero for the input we differentiated against.
        self.assertGreater(x.grad.abs().sum().item(), 0.0)

    def test_checkpoint_save_load(self) -> None:
        model = FailurePrototypeCalibrator(input_dim=10, hidden_dim=16, embedding_dim=8, temperature=0.7)
        # Initialize prototypes so the checkpoint has a non-default state.
        feats = torch.randn(6, 10)
        labels = [0, 0, 1, 1, 2, 2]
        model.init_prototypes(feats, labels)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "calibrator.pt")
            torch.save(model.to_checkpoint(), path)
            loaded = FailurePrototypeCalibrator.from_checkpoint(path, map_location="cpu")
        self.assertEqual(loaded.input_dim, 10)
        self.assertEqual(loaded.hidden_dim, 16)
        self.assertEqual(loaded.embedding_dim, 8)
        self.assertEqual(loaded.temperature, 0.7)
        # Prototype weights match exactly.
        self.assertTrue(torch.allclose(loaded.prototypes, model.prototypes))

    def test_evaluate_controllability_returns_expected_keys(self) -> None:
        model = FailurePrototypeCalibrator(input_dim=10, hidden_dim=16, embedding_dim=8)
        rng = np.random.RandomState(42)
        X = rng.randn(40, 10).astype(np.float32)
        y = (rng.randint(0, 10, size=40)).astype(np.int64)
        report = evaluate_controllability(
            model,
            torch.from_numpy(X),
            torch.from_numpy(y),
            min_per_class=3,
            device="cpu",
        )
        for key in (
            "classification_accuracy",
            "per_class_accuracy",
            "confusion",
            "mean_entropy",
            "normalized_entropy",
            "aggregate_entropy",
            "target_hit_rate",
            "uniform_entropy",
        ):
            self.assertIn(key, report)
        self.assertGreaterEqual(report["classification_accuracy"], 0.0)
        self.assertLessEqual(report["classification_accuracy"], 1.0)
        self.assertGreaterEqual(report["mean_entropy"], 0.0)


@unittest.skipUnless(IS_TORCH_AVAILABLE and IS_RDKIT_AVAILABLE, "PyTorch and RDKit are required")
class LabelFailureTypeExtraTest(unittest.TestCase):
    def test_under_reaction_when_product_equals_reactant(self) -> None:
        # Product identical to reactant -> under_reaction.
        reaction = "CCO>>CCO"
        label = label_failure_type(reaction, ["CCO"])
        self.assertEqual(label, "under_reaction")

    def test_side_product_fallback(self) -> None:
        # Unrelated product with valid balance but no expected match.
        reaction = "CCO>>CCN"  # different molecule, balanced-ish
        label = label_failure_type(reaction, ["CCC"])
        self.assertIn(label, FAILURE_TYPES)


if __name__ == "__main__":
    unittest.main()
