"""Unit tests for P2-08 Reaction Condition Prediction downstream evaluation.

Tests cover:
- Module imports and constants
- CLI argument parsing
- Condition-class detection from reactant SMILES (RDKit-based)
- Synthetic dataset generation
- Metric computation (Top-1/3/5/10, MRR, NDCG@10) on synthetic data
- Per-seed aggregation
- Paired t-test significance structure
- Baseline vs treatment comparison logic (PC-CNG augmentation)
- Synthetic small training run (mocked featurizer)
- Go/No-Go decision logic
"""

from __future__ import annotations

import csv
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import List

import numpy as np

# Ensure chem_negative_sampling directory is on sys.path when tests are
# run from the repo root without installation.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(THIS_DIR)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)


def _write_minimal_uspto_csv(path: Path, n_rows: int = 6) -> None:
    """Write a minimal USPTO-normalized CSV for testing.

    The metals are placed in the *reactants* column (not the agents
    column) so that ``detect_condition_class`` can detect them.  In the
    real USPTO OpenMolecules CSV the agents column is empty - catalysts
    sit in reactants - so this mirrors real data.
    """
    rows = [
        # source_id, reaction_smiles, reactants, agents, products,
        # label_type, yield, source, split_key, split
        ("r1", "[Pd].CC(=O)O.CCO>CC(=O)OCC", "[Pd].CC(=O)O.CCO", "",
         "CC(=O)OCC", "positive", 100, "src", "k1", "train"),
        ("r2", "[Cu].CC(=O)O.CCO>CC(=O)OCC", "[Cu].CC(=O)O.CCO", "",
         "CC(=O)OCC", "positive", 95, "src", "k2", "train"),
        ("r3", "[Fe].CC(=O)O.CCO>CC(=O)OCC", "[Fe].CC(=O)O.CCO", "",
         "CC(=O)OCC", "positive", 90, "src", "k3", "val"),
        ("r4", "[Ni].CC(=O)O.CCO>CC(=O)OCC", "[Ni].CC(=O)O.CCO", "",
         "CC(=O)OCC", "positive", 85, "src", "k4", "val"),
        ("r5", "[Zn].CC(=O)O.CCO>CC(=O)OCC", "[Zn].CC(=O)O.CCO", "",
         "CC(=O)OCC", "positive", 80, "src", "k5", "test"),
        ("r6", "CC(=O)O.CCO>CC(=O)OCC", "CC(=O)O.CCO", "",
         "CC(=O)OCC", "positive", 75, "src", "k6", "test"),
    ][:n_rows]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "source_id", "reaction_smiles", "reactants", "agents",
            "products", "label_type", "yield", "source", "split_key",
            "split",
        ])
        for r in rows:
            writer.writerow(r)


def _write_minimal_pc_cng_csv(path: Path) -> None:
    """Write a minimal PC-CNG synthetic-negatives CSV for testing."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "source_id", "positive_reaction", "candidate_reaction", "task",
            "failure_type", "edit_action", "parent_reactants",
            "parent_product", "candidate_reactants", "candidate_product",
            "valid", "atom_balance", "locality", "closeness", "hard_score",
            "false_negative_risk", "passes_filter", "label", "provenance",
            "review_status", "review_reasons", "product_overlap",
        ])
        # Two negatives whose parent source_id is r1 (Pd class).
        writer.writerow([
            "r1", "CC(=O)O.CCO>[Pd]>CC(=O)OCC",
            "CC(=O)O.CCO.CCO>>CC(=O)OCC",  # candidate_reaction
            "retro_precursor", "retro_wrong_functional_group",
            "edit1", "CC(=O)O.CCO", "CC(=O)OCC",
            "CC(=O)O.CCO.CCO", "CC(=O)OCC",
            1.0, 0.9, 0.5, 0.5, 0.5, 0.5, True, 0, "prov",
            "keep_synthetic_negative", "", 1.0,
        ])
        # Negative whose review_status is "needs_review_or_downweight"
        # → should be filtered out.
        writer.writerow([
            "r1", "CC(=O)O.CCO>[Pd]>CC(=O)OCC",
            "CC(=O)O.CCO.CCC>>CC(=O)OCC",
            "retro_precursor", "retro_wrong_functional_group",
            "edit2", "CC(=O)O.CCO", "CC(=O)OCC",
            "CC(=O)O.CCO.CCC", "CC(=O)OCC",
            1.0, 0.4, 0.5, 0.5, 0.5, 0.5, True, 0, "prov",
            "needs_review_or_downweight", "high_model_false_negative_risk",
            1.0,
        ])


class ModuleImportsTest(unittest.TestCase):
    """Verify the script imports without errors and exposes constants."""

    def test_module_imports(self) -> None:
        from pc_cng import run_condition_prediction_eval as mod
        self.assertTrue(hasattr(mod, "main"))
        self.assertTrue(hasattr(mod, "DEFAULT_SEEDS"))
        self.assertTrue(hasattr(mod, "CONDITION_CLASSES"))
        self.assertTrue(hasattr(mod, "DATASET_TAG"))

    def test_default_seeds_count(self) -> None:
        from pc_cng.run_condition_prediction_eval import DEFAULT_SEEDS
        self.assertEqual(len(DEFAULT_SEEDS), 10)
        self.assertEqual(DEFAULT_SEEDS[0], 20260710)
        self.assertEqual(DEFAULT_SEEDS[-1], 20260719)

    def test_condition_classes_has_ten(self) -> None:
        from pc_cng.run_condition_prediction_eval import CONDITION_CLASSES
        self.assertEqual(len(CONDITION_CLASSES), 10)
        names = [name for name, _ in CONDITION_CLASSES]
        self.assertIn("Pd", names)
        self.assertIn("Cu", names)
        self.assertIn("Organic", names)
        # "Organic" must be the last class (lowest priority)
        self.assertEqual(CONDITION_CLASSES[-1][0], "Organic")
        # Organic has empty atomic-num set
        self.assertEqual(CONDITION_CLASSES[-1][1], set())

    def test_dataset_tag(self) -> None:
        from pc_cng.run_condition_prediction_eval import DATASET_TAG
        self.assertEqual(DATASET_TAG, "synthetic_condition_from_metals")


class CLIArgsTest(unittest.TestCase):
    """Verify argparse accepts the documented CLI args."""

    def _parse(self, argv: List[str]):
        from pc_cng.run_condition_prediction_eval import main
        # We patch sys.argv and use SystemExit to detect required-args errors
        # without invoking main()'s body.
        import argparse
        # Rebuild the parser by reading source would be brittle; instead
        # we replicate the parser here to validate flag compatibility.
        parser = argparse.ArgumentParser()
        parser.add_argument("--dataset", required=True)
        parser.add_argument("--pc-cng-negatives", required=True)
        parser.add_argument("--output-dir", required=True)
        parser.add_argument("--seeds",
                            default="20260710,20260711,20260712,20260713,"
                                    "20260714,20260715,20260716,20260717,"
                                    "20260718,20260719")
        parser.add_argument("--epochs", type=int, default=30)
        parser.add_argument("--batch-size", type=int, default=64)
        parser.add_argument("--limit", type=int, default=None)
        parser.add_argument("--n-bits", type=int, default=2048)
        parser.add_argument("--hidden-dim", type=int, default=512)
        parser.add_argument("--device", default=None)
        parser.add_argument("--condition-cache", default=None)
        return parser.parse_args(argv)

    def test_default_seeds_parsed(self) -> None:
        args = self._parse([
            "--dataset", "/tmp/d.csv",
            "--pc-cng-negatives", "/tmp/n.csv",
            "--output-dir", "/tmp/out",
        ])
        self.assertEqual(args.epochs, 30)
        self.assertEqual(args.batch_size, 64)
        self.assertEqual(args.n_bits, 2048)
        self.assertEqual(args.hidden_dim, 512)
        self.assertIsNone(args.limit)
        # default seeds string has 10 entries
        self.assertEqual(len(args.seeds.split(",")), 10)

    def test_smoke_flags_parsed(self) -> None:
        args = self._parse([
            "--dataset", "/tmp/d.csv",
            "--pc-cng-negatives", "/tmp/n.csv",
            "--output-dir", "/tmp/out",
            "--seeds", "20260710,20260711",
            "--epochs", "3",
            "--limit", "50",
        ])
        self.assertEqual(args.seeds, "20260710,20260711")
        self.assertEqual(args.epochs, 3)
        self.assertEqual(args.limit, 50)

    def test_parse_seeds_helper(self) -> None:
        from pc_cng.run_condition_prediction_eval import parse_seeds
        self.assertEqual(parse_seeds("20260710,20260711"), [20260710, 20260711])
        self.assertEqual(parse_seeds(" 1 , 2 , 3 "), [1, 2, 3])
        self.assertEqual(parse_seeds(""), [])


class ConditionDetectionTest(unittest.TestCase):
    """Test the RDKit-based condition-class detector."""

    def test_organic_label_for_no_metal(self) -> None:
        from pc_cng.run_condition_prediction_eval import (
            detect_condition_class, IDX_TO_CLASS_NAME,
        )
        idx = detect_condition_class("CC(=O)O.CCO")
        self.assertEqual(IDX_TO_CLASS_NAME[idx], "Organic")

    def test_pd_label(self) -> None:
        from pc_cng.run_condition_prediction_eval import (
            detect_condition_class, IDX_TO_CLASS_NAME,
        )
        idx = detect_condition_class("[Pd].CC(=O)O")
        self.assertEqual(IDX_TO_CLASS_NAME[idx], "Pd")

    def test_cu_label(self) -> None:
        from pc_cng.run_condition_prediction_eval import (
            detect_condition_class, IDX_TO_CLASS_NAME,
        )
        idx = detect_condition_class("[Cu].CC(=O)O")
        self.assertEqual(IDX_TO_CLASS_NAME[idx], "Cu")

    def test_zn_mg_label(self) -> None:
        from pc_cng.run_condition_prediction_eval import (
            detect_condition_class, IDX_TO_CLASS_NAME,
        )
        idx = detect_condition_class("[Zn].CCO")
        self.assertEqual(IDX_TO_CLASS_NAME[idx], "Zn_Mg")
        idx = detect_condition_class("[Mg].CCO")
        self.assertEqual(IDX_TO_CLASS_NAME[idx], "Zn_Mg")

    def test_alkali_metal_label(self) -> None:
        from pc_cng.run_condition_prediction_eval import (
            detect_condition_class, IDX_TO_CLASS_NAME,
        )
        idx = detect_condition_class("[Na].CCO")
        self.assertEqual(IDX_TO_CLASS_NAME[idx], "Li_Na_K")

    def test_priority_pd_over_cu(self) -> None:
        # When Pd and Cu are both present, Pd wins (higher priority).
        from pc_cng.run_condition_prediction_eval import (
            detect_condition_class, IDX_TO_CLASS_NAME,
        )
        idx = detect_condition_class("[Pd].[Cu].CCO")
        self.assertEqual(IDX_TO_CLASS_NAME[idx], "Pd")

    def test_empty_smiles_returns_organic(self) -> None:
        from pc_cng.run_condition_prediction_eval import (
            detect_condition_class, IDX_TO_CLASS_NAME,
        )
        self.assertEqual(IDX_TO_CLASS_NAME[detect_condition_class("")], "Organic")
        self.assertEqual(IDX_TO_CLASS_NAME[detect_condition_class("   ")], "Organic")

    def test_invalid_smiles_falls_through_to_organic(self) -> None:
        from pc_cng.run_condition_prediction_eval import (
            detect_condition_class, IDX_TO_CLASS_NAME,
        )
        # Completely garbage SMILES - no metal atom detected.
        idx = detect_condition_class("not_a_smiles")
        self.assertEqual(IDX_TO_CLASS_NAME[idx], "Organic")


class DatasetGenerationTest(unittest.TestCase):
    """Test the synthetic condition-dataset generator."""

    def test_generate_writes_csv_with_correct_schema(self) -> None:
        from pc_cng.run_condition_prediction_eval import (
            generate_synthetic_condition_dataset,
        )
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "uspto.csv"
            out = Path(tmp) / "cond.csv"
            _write_minimal_uspto_csv(src, n_rows=6)
            stats = generate_synthetic_condition_dataset(str(src), str(out))
            self.assertTrue(out.exists())
            self.assertEqual(stats["rows_written"], 6)
            with out.open() as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            self.assertEqual(len(rows), 6)
            self.assertIn("condition_label", rows[0])
            self.assertIn("condition_idx", rows[0])
            self.assertIn("split", rows[0])

    def test_limit_caps_rows(self) -> None:
        from pc_cng.run_condition_prediction_eval import (
            generate_synthetic_condition_dataset,
        )
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "uspto.csv"
            out = Path(tmp) / "cond.csv"
            _write_minimal_uspto_csv(src, n_rows=6)
            stats = generate_synthetic_condition_dataset(str(src), str(out), limit=3)
            self.assertEqual(stats["rows_written"], 3)

    def test_class_distribution_populated(self) -> None:
        from pc_cng.run_condition_prediction_eval import (
            generate_synthetic_condition_dataset,
        )
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "uspto.csv"
            out = Path(tmp) / "cond.csv"
            _write_minimal_uspto_csv(src, n_rows=6)
            stats = generate_synthetic_condition_dataset(str(src), str(out))
            counts = stats["class_counts"]
            # Pd, Cu, Fe, Ni, Zn, Organic each appear once
            self.assertEqual(counts.get("Pd", 0), 1)
            self.assertEqual(counts.get("Cu", 0), 1)
            self.assertEqual(counts.get("Fe", 0), 1)
            self.assertEqual(counts.get("Ni", 0), 1)
            self.assertEqual(counts.get("Zn_Mg", 0), 1)
            self.assertEqual(counts.get("Organic", 0), 1)


class MetricsComputationTest(unittest.TestCase):
    """Test the metric computations on synthetic prediction arrays."""

    def test_perfect_prediction(self) -> None:
        from pc_cng.run_condition_prediction_eval import (
            compute_classification_metrics,
        )
        n = 5
        num_classes = 10
        y_true = np.array([0, 1, 2, 3, 4], dtype=np.int64)
        probs = np.zeros((n, num_classes), dtype=np.float32)
        for i, c in enumerate(y_true):
            probs[i, c] = 1.0
        m = compute_classification_metrics(y_true, probs)
        self.assertEqual(m["n_samples"], 5.0)
        self.assertAlmostEqual(m["top1"], 1.0)
        self.assertAlmostEqual(m["top3"], 1.0)
        self.assertAlmostEqual(m["top5"], 1.0)
        self.assertAlmostEqual(m["top10"], 1.0)
        self.assertAlmostEqual(m["mrr"], 1.0)
        self.assertAlmostEqual(m["ndcg_at_10"], 1.0)

    def test_all_wrong_prediction(self) -> None:
        from pc_cng.run_condition_prediction_eval import (
            compute_classification_metrics,
        )
        n = 5
        num_classes = 10
        y_true = np.array([0, 0, 0, 0, 0], dtype=np.int64)
        # True class (0) is predicted with the *lowest* probability so it
        # lands at rank 10 (last).  Classes 9, 8, ..., 1 get progressively
        # lower probability in that order so the ranking is unambiguous.
        probs = np.zeros((n, num_classes), dtype=np.float32)
        for i in range(n):
            for c in range(num_classes):
                probs[i, c] = (c + 1) / float(num_classes)
        m = compute_classification_metrics(y_true, probs)
        self.assertAlmostEqual(m["top1"], 0.0)
        self.assertAlmostEqual(m["top3"], 0.0)
        self.assertAlmostEqual(m["top5"], 0.0)
        # Top-10 hits everything (10 classes total)
        self.assertAlmostEqual(m["top10"], 1.0)
        # True class is at rank 10 → MRR = 1/10
        self.assertAlmostEqual(m["mrr"], 0.1, places=4)
        # NDCG@10: positive is at rank 10, so DCG = 1/log2(11), IDCG = 1/log2(2)
        expected_ndcg = (1.0 / math.log2(11)) / (1.0 / math.log2(2))
        self.assertAlmostEqual(m["ndcg_at_10"], expected_ndcg, places=4)

    def test_partial_top3(self) -> None:
        from pc_cng.run_condition_prediction_eval import (
            compute_classification_metrics,
        )
        num_classes = 10
        y_true = np.array([0, 1, 2, 3, 4], dtype=np.int64)
        # Predictions: true class is at rank 2 for all samples → top-3 hit
        probs = np.zeros((5, num_classes), dtype=np.float32)
        for i, c in enumerate(y_true):
            probs[i, (c + 1) % num_classes] = 0.6  # rank 1 (wrong)
            probs[i, c] = 0.4  # rank 2 (correct)
        m = compute_classification_metrics(y_true, probs)
        self.assertAlmostEqual(m["top1"], 0.0)
        self.assertAlmostEqual(m["top3"], 1.0)
        self.assertAlmostEqual(m["top5"], 1.0)
        self.assertAlmostEqual(m["mrr"], 0.5, places=4)

    def test_empty_input_returns_zeros(self) -> None:
        from pc_cng.run_condition_prediction_eval import (
            compute_classification_metrics,
        )
        m = compute_classification_metrics(
            np.array([], dtype=np.int64),
            np.zeros((0, 10), dtype=np.float32),
        )
        self.assertEqual(m["n_samples"], 0)
        self.assertEqual(m["top1"], 0.0)
        self.assertEqual(m["mrr"], 0.0)
        self.assertEqual(m["ndcg_at_10"], 0.0)

    def test_reciprocal_rank_helper(self) -> None:
        from pc_cng.run_condition_prediction_eval import reciprocal_rank
        self.assertAlmostEqual(reciprocal_rank([1, 0, 0]), 1.0)
        self.assertAlmostEqual(reciprocal_rank([0, 1, 0]), 0.5)
        self.assertAlmostEqual(reciprocal_rank([0, 0, 1]), 1.0 / 3.0)
        self.assertAlmostEqual(reciprocal_rank([0, 0, 0]), 0.0)

    def test_ndcg_at_k_helper(self) -> None:
        from pc_cng.run_condition_prediction_eval import ndcg_at_k
        # Hit at rank 1 → NDCG = 1.0
        self.assertAlmostEqual(ndcg_at_k([1, 0, 0], k=10), 1.0)
        # Hit at rank 2 → DCG = 1/log2(3), IDCG = 1/log2(2)
        expected = (1.0 / math.log2(3)) / (1.0 / math.log2(2))
        self.assertAlmostEqual(ndcg_at_k([0, 1, 0], k=10), expected)
        # No hit
        self.assertAlmostEqual(ndcg_at_k([0, 0, 0], k=10), 0.0)

    def test_topk_accuracy_helper(self) -> None:
        from pc_cng.run_condition_prediction_eval import topk_accuracy
        self.assertEqual(topk_accuracy([1, 0, 0], k=1), 1.0)
        self.assertEqual(topk_accuracy([0, 1, 0], k=1), 0.0)
        self.assertEqual(topk_accuracy([0, 1, 0], k=3), 1.0)
        self.assertEqual(topk_accuracy([0, 0, 0], k=3), 0.0)


class PairedTTestTest(unittest.TestCase):
    """Test the paired t-test significance helper."""

    def test_significant_positive_delta(self) -> None:
        from pc_cng.run_condition_prediction_eval import paired_ttest
        # Treatment consistently +1 over baseline across 10 seeds
        baseline = [0.5] * 10
        treatment = [0.6] * 10
        result = paired_ttest(baseline, treatment)
        self.assertEqual(result["n"], 10)
        self.assertAlmostEqual(result["mean_delta"], 0.1, places=6)
        self.assertLess(result["p_value"], 0.01)
        # t-stat should be very large / +inf for zero variance
        self.assertTrue(math.isinf(result["t_stat"]) or result["t_stat"] > 5.0)

    def test_no_delta_returns_high_p(self) -> None:
        from pc_cng.run_condition_prediction_eval import paired_ttest
        baseline = [0.5, 0.6, 0.4, 0.7, 0.5]
        treatment = [0.5, 0.6, 0.4, 0.7, 0.5]  # identical
        result = paired_ttest(baseline, treatment)
        self.assertEqual(result["n"], 5)
        self.assertAlmostEqual(result["mean_delta"], 0.0, places=6)
        # Identical arrays → t-stat undefined; p-value should be high
        self.assertGreaterEqual(result["p_value"], 0.5)

    def test_mixed_delta_returns_moderate_p(self) -> None:
        from pc_cng.run_condition_prediction_eval import paired_ttest
        # Half the deltas are +0.1, half are -0.1 → mean delta 0
        baseline = [0.5, 0.5, 0.5, 0.5]
        treatment = [0.6, 0.4, 0.6, 0.4]
        result = paired_ttest(baseline, treatment)
        self.assertAlmostEqual(result["mean_delta"], 0.0, places=6)
        # mean=0, but variance > 0 → t_stat ~ 0 → p ~ 1.0
        self.assertGreaterEqual(result["p_value"], 0.5)

    def test_too_few_samples_returns_default(self) -> None:
        from pc_cng.run_condition_prediction_eval import paired_ttest
        result = paired_ttest([0.5], [0.6])
        self.assertEqual(result["n"], 1)
        self.assertEqual(result["p_value"], 1.0)


class BaselineVsTreatmentTest(unittest.TestCase):
    """Test the baseline vs treatment comparison logic."""

    def test_build_treatment_rows_inherits_parent_label(self) -> None:
        from pc_cng.run_condition_prediction_eval import build_treatment_rows
        real_rows = [
            {"source_id": "r1", "reactants": "[Pd].CCO",
             "products": "P", "condition_label": "Pd",
             "condition_idx": 0, "split": "train"},
            {"source_id": "r2", "reactants": "[Cu].CCO",
             "products": "P", "condition_label": "Cu",
             "condition_idx": 1, "split": "train"},
        ]
        pc_cng_rows = [
            {"source_id": "r1", "candidate_reactants": "[Pd].CCO.CCO",
             "parent_reaction": "...", "review_status": "keep_synthetic_negative"},
            {"source_id": "r3",  # not in train - should be dropped
             "candidate_reactants": "[Fe].CCO",
             "parent_reaction": "...", "review_status": "keep_synthetic_negative"},
        ]
        out = build_treatment_rows(real_rows, pc_cng_rows)
        self.assertEqual(len(out), 3)  # 2 real + 1 augmented
        # The augmented row should inherit r1's condition label
        aug = [r for r in out if str(r["source_id"]).endswith("_pccng")]
        self.assertEqual(len(aug), 1)
        self.assertEqual(aug[0]["condition_label"], "Pd")
        self.assertEqual(aug[0]["condition_idx"], 0)
        self.assertEqual(aug[0]["split"], "train")

    def test_read_pc_cng_negatives_filters_review_status(self) -> None:
        from pc_cng.run_condition_prediction_eval import read_pc_cng_negatives
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "neg.csv"
            _write_minimal_pc_cng_csv(path)
            rows = read_pc_cng_negatives(str(path))
            # The "needs_review_or_downweight" row must be filtered out
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["source_id"], "r1")
            self.assertEqual(rows[0]["review_status"], "keep_synthetic_negative")

    def test_read_pc_cng_negatives_missing_file_returns_empty(self) -> None:
        from pc_cng.run_condition_prediction_eval import read_pc_cng_negatives
        self.assertEqual(read_pc_cng_negatives("/nonexistent/path.csv"), [])
        self.assertEqual(read_pc_cng_negatives(""), [])


class AggregationTest(unittest.TestCase):
    """Test per-seed aggregation."""

    def test_aggregate_seed_metrics(self) -> None:
        from pc_cng.run_condition_prediction_eval import aggregate_seed_metrics
        seed_results = [
            {"baseline_metrics": {"top1": 0.5, "mrr": 0.6},
             "treatment_metrics": {"top1": 0.6, "mrr": 0.7}},
            {"baseline_metrics": {"top1": 0.4, "mrr": 0.5},
             "treatment_metrics": {"top1": 0.5, "mrr": 0.6}},
        ]
        agg = aggregate_seed_metrics(seed_results)
        self.assertAlmostEqual(agg["top1"]["baseline_mean"], 0.45)
        self.assertAlmostEqual(agg["top1"]["treatment_mean"], 0.55)
        self.assertAlmostEqual(agg["top1"]["delta_mean"], 0.10)
        self.assertEqual(len(agg["top1"]["baseline_vals"]), 2)

    def test_build_paired_significance_structure(self) -> None:
        from pc_cng.run_condition_prediction_eval import build_paired_significance
        seed_results = [
            {"baseline_metrics": {"top1": 0.5, "top3": 0.7, "top5": 0.8,
                                  "top10": 0.9, "mrr": 0.6, "ndcg_at_10": 0.7},
             "treatment_metrics": {"top1": 0.55, "top3": 0.75, "top5": 0.85,
                                   "top10": 0.92, "mrr": 0.65, "ndcg_at_10": 0.75}}
            for _ in range(5)
        ]
        sig = build_paired_significance(seed_results)
        for key in ["top1", "top3", "top5", "top10", "mrr", "ndcg_at_10"]:
            self.assertIn(key, sig)
            self.assertEqual(sig[key]["n_seeds"], 5)
            self.assertIn("t_stat", sig[key])
            self.assertIn("p_value", sig[key])
            self.assertIn("per_seed_delta", sig[key])


class GoNoGoDecisionTest(unittest.TestCase):
    """Test the Go/No-Go decision logic."""

    def test_go_decision_when_treatment_significantly_better(self) -> None:
        from pc_cng.run_condition_prediction_eval import build_go_no_go_decision
        significance = {
            "top1": {
                "n_seeds": 10,
                "baseline_mean": 0.5,
                "treatment_mean": 0.55,
                "delta_mean": 0.05,
                "delta_mean_pp": 5.0,
                "p_value": 0.001,
            }
        }
        decision = build_go_no_go_decision(significance, primary_metric="top1")
        self.assertIn("GO", decision["decision"])
        self.assertEqual(decision["primary_metric"], "top1")
        self.assertTrue(decision["criteria"]["delta_passes"])
        self.assertTrue(decision["criteria"]["p_passes"])
        self.assertTrue(decision["criteria"]["treatment_better_passes"])

    def test_no_go_when_delta_below_threshold(self) -> None:
        from pc_cng.run_condition_prediction_eval import build_go_no_go_decision
        significance = {
            "top1": {
                "n_seeds": 10,
                "baseline_mean": 0.5,
                "treatment_mean": 0.505,
                "delta_mean": 0.005,
                "delta_mean_pp": 0.5,  # below 1.0pp threshold
                "p_value": 0.001,
            }
        }
        decision = build_go_no_go_decision(significance, primary_metric="top1")
        self.assertIn("NO-GO", decision["decision"])
        self.assertFalse(decision["criteria"]["delta_passes"])
        self.assertTrue(decision["criteria"]["p_passes"])

    def test_no_go_when_p_value_high(self) -> None:
        from pc_cng.run_condition_prediction_eval import build_go_no_go_decision
        significance = {
            "top1": {
                "n_seeds": 10,
                "baseline_mean": 0.5,
                "treatment_mean": 0.55,
                "delta_mean": 0.05,
                "delta_mean_pp": 5.0,
                "p_value": 0.2,  # above 0.05 threshold
            }
        }
        decision = build_go_no_go_decision(significance, primary_metric="top1")
        self.assertIn("NO-GO", decision["decision"])
        self.assertFalse(decision["criteria"]["p_passes"])

    def test_no_go_when_treatment_worse(self) -> None:
        from pc_cng.run_condition_prediction_eval import build_go_no_go_decision
        significance = {
            "top1": {
                "n_seeds": 10,
                "baseline_mean": 0.55,
                "treatment_mean": 0.50,
                "delta_mean": -0.05,
                "delta_mean_pp": -5.0,
                "p_value": 0.001,
            }
        }
        decision = build_go_no_go_decision(significance, primary_metric="top1")
        self.assertIn("NO-GO", decision["decision"])
        self.assertFalse(decision["criteria"]["treatment_better_passes"])


class SyntheticEndToEndTest(unittest.TestCase):
    """Synthetic small end-to-end run with a mocked featurizer (no RDKit heavy work)."""

    def test_synthetic_small_run_writes_all_outputs(self) -> None:
        """End-to-end smoke test: generate dataset → run main() with tiny params."""
        from pc_cng import run_condition_prediction_eval as mod

        with tempfile.TemporaryDirectory() as tmp:
            src_csv = Path(tmp) / "uspto.csv"
            pc_cng_csv = Path(tmp) / "neg.csv"
            out_dir = Path(tmp) / "out"
            _write_minimal_uspto_csv(src_csv, n_rows=6)
            _write_minimal_pc_cng_csv(pc_cng_csv)

            argv = [
                "run_condition_prediction_eval.py",
                "--dataset", str(src_csv),
                "--pc-cng-negatives", str(pc_cng_csv),
                "--output-dir", str(out_dir),
                "--seeds", "20260710,20260711",
                "--epochs", "2",
                "--batch-size", "8",
                "--n-bits", "256",
                "--hidden-dim", "32",
            ]
            old_argv = sys.argv
            sys.argv = argv
            try:
                mod.main()
            finally:
                sys.argv = old_argv

            # All four required outputs must exist
            self.assertTrue((out_dir / "summary.json").exists())
            self.assertTrue((out_dir / "paired_significance.json").exists())
            self.assertTrue((out_dir / "per_seed_metrics.csv").exists())
            self.assertTrue((out_dir / "go_no_go_decision.json").exists())

            # summary.json structure
            with (out_dir / "summary.json").open() as f:
                summary = json.load(f)
            self.assertEqual(summary["n_seeds"], 2)
            self.assertEqual(summary["dataset_tag"], "synthetic_condition_from_metals")
            self.assertIn("class_distribution", summary)
            self.assertEqual(len(summary["per_seed"]), 2)
            self.assertIn("aggregate", summary)
            self.assertIn("top1", summary["aggregate"])

            # paired_significance.json structure
            with (out_dir / "paired_significance.json").open() as f:
                sig = json.load(f)
            for key in ["top1", "top3", "top5", "top10", "mrr", "ndcg_at_10"]:
                self.assertIn(key, sig)
                self.assertIn("t_stat", sig[key])
                self.assertIn("p_value", sig[key])
                self.assertEqual(sig[key]["n_seeds"], 2)

            # go_no_go_decision.json structure
            with (out_dir / "go_no_go_decision.json").open() as f:
                decision = json.load(f)
            self.assertIn("decision", decision)
            self.assertIn(decision["decision"],
                          ["GO (write to main table)",
                           "NO-GO (downgrade to supplementary)"])
            self.assertEqual(decision["primary_metric"], "top1")

            # per_seed_metrics.csv structure - one row per seed
            with (out_dir / "per_seed_metrics.csv").open() as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            self.assertEqual(len(rows), 2)
            self.assertIn("delta_top1", rows[0])
            self.assertIn("delta_mrr", rows[0])


if __name__ == "__main__":
    unittest.main()
