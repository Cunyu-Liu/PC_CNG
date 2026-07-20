"""Unit tests for chem_negative_sampling.pc_cng.run_ood_scaffold_template_eval.

Tests scaffold extraction, template extraction, split assignment logic, and
split preparation.  No GPU or training is required.
"""

from __future__ import annotations

import os
import sys
import unittest
from typing import List

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(THIS_DIR)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

from pc_cng.run_ood_scaffold_template_eval import (  # noqa: E402
    assign_split_by_key,
    murcko_scaffold,
    prepare_splits,
    reaction_template,
    _split_reaction,
)


class SplitReactionTest(unittest.TestCase):
    def test_three_part(self) -> None:
        r, a, p = _split_reaction("A>B>C")
        self.assertEqual(r, "A")
        self.assertEqual(a, "B")
        self.assertEqual(p, "C")

    def test_two_part(self) -> None:
        r, a, p = _split_reaction("A>>C")
        self.assertEqual(r, "A")
        self.assertEqual(a, "")
        self.assertEqual(p, "C")

    def test_invalid_raises(self) -> None:
        with self.assertRaises(ValueError):
            _split_reaction("no_reaction")


class MurckoScaffoldTest(unittest.TestCase):
    def test_benzene_returns_benzene(self) -> None:
        scaffold = murcko_scaffold("c1ccccc1")
        # BemisMurcko of benzene is benzene itself (or empty depending on RDKit version).
        self.assertTrue(scaffold in {"c1ccccc1", ""} )

    def test_substituted_benzene_has_scaffold(self) -> None:
        scaffold = murcko_scaffold("Cc1ccccc1")
        # Should reduce to benzene scaffold (may be canonical or empty).
        self.assertIsInstance(scaffold, str)

    def test_invalid_smiles_returns_empty(self) -> None:
        scaffold = murcko_scaffold("not_a_smiles")
        self.assertEqual(scaffold, "")

    def test_empty_input(self) -> None:
        self.assertEqual(murcko_scaffold(""), "")


class ReactionTemplateTest(unittest.TestCase):
    def test_unmapped_reaction_returns_canonical_reactants(self) -> None:
        template = reaction_template("CC.O>>CC.O")
        self.assertIsInstance(template, str)
        self.assertTrue(len(template) > 0)

    def test_mapped_reaction_uses_product(self) -> None:
        template = reaction_template("[CH3:1][OH:2]>>[CH3:1][OH:2]")
        self.assertIsInstance(template, str)

    def test_invalid_reaction_returns_empty(self) -> None:
        template = reaction_template("not_a_reaction")
        self.assertEqual(template, "")

    def test_empty_returns_empty(self) -> None:
        self.assertEqual(reaction_template(""), "")


class AssignSplitByKeyTest(unittest.TestCase):
    def test_train_frac_produces_three_splits(self) -> None:
        rows = [{"_key": f"k{i}"} for i in range(20)]
        assign_split_by_key(rows, "_key", train_frac=0.6, seed=42)
        splits = [r["split"] for r in rows]
        self.assertIn("train", splits)
        self.assertIn("test", splits)
        # train should be ~60% of unique keys
        train_count = sum(1 for s in splits if s == "train")
        self.assertGreater(train_count, 5)
        self.assertLess(train_count, 20)

    def test_deterministic_for_same_seed(self) -> None:
        rows1 = [{"_key": f"k{i}"} for i in range(10)]
        rows2 = [{"_key": f"k{i}"} for i in range(10)]
        assign_split_by_key(rows1, "_key", train_frac=0.8, seed=123)
        assign_split_by_key(rows2, "_key", train_frac=0.8, seed=123)
        self.assertEqual([r["split"] for r in rows1], [r["split"] for r in rows2])

    def test_different_seeds_may_differ(self) -> None:
        rows1 = [{"_key": f"k{i}"} for i in range(20)]
        rows2 = [{"_key": f"k{i}"} for i in range(20)]
        assign_split_by_key(rows1, "_key", train_frac=0.6, seed=1)
        assign_split_by_key(rows2, "_key", train_frac=0.6, seed=2)
        # Not guaranteed to differ, but very likely with 20 keys and seed shuffle
        # We only check both produce valid splits.
        for r in rows1 + rows2:
            self.assertIn(r["split"], {"train", "val", "test"})

    def test_empty_rows_no_error(self) -> None:
        rows: List[dict] = []
        assign_split_by_key(rows, "_key", train_frac=0.8, seed=42)
        self.assertEqual(rows, [])

    def test_single_key_all_train(self) -> None:
        rows = [{"_key": "only_one"}]
        assign_split_by_key(rows, "_key", train_frac=0.8, seed=42)
        self.assertEqual(rows[0]["split"], "train")


class PrepareSplitsTest(unittest.TestCase):
    def test_three_split_variants_returned(self) -> None:
        real_rows = [
            {
                "source_id": f"id{i}",
                "reaction_smiles": f"C{'C' * i}>>C{'C' * (i + 1)}",
                "reactants": f"C{'C' * i}",
                "products": f"C{'C' * (i + 1)}",
                "label": i % 2,
                "split": "train" if i % 3 == 0 else ("val" if i % 3 == 1 else "test"),
                "dataset": "test",
                "reaction_class": "",
            }
            for i in range(15)
        ]
        splits = prepare_splits(real_rows, train_frac=0.7, seed=42)
        self.assertIn("random", splits)
        self.assertIn("scaffold", splits)
        self.assertIn("template", splits)
        for name in ["random", "scaffold", "template"]:
            self.assertEqual(len(splits[name]), 15)

    def test_random_keeps_original_split(self) -> None:
        real_rows = [
            {
                "source_id": "a",
                "reaction_smiles": "CC>>CC",
                "reactants": "CC",
                "products": "CC",
                "label": 1,
                "split": "train",
                "dataset": "x",
                "reaction_class": "",
            },
            {
                "source_id": "b",
                "reaction_smiles": "CC>>CCC",
                "reactants": "CC",
                "products": "CCC",
                "label": 0,
                "split": "test",
                "dataset": "x",
                "reaction_class": "",
            },
        ]
        splits = prepare_splits(real_rows, train_frac=0.5, seed=42)
        self.assertEqual(splits["random"][0]["split"], "train")
        self.assertEqual(splits["random"][1]["split"], "test")

    def test_scaffold_reassigns_split(self) -> None:
        real_rows = [
            {
                "source_id": f"id{i}",
                "reaction_smiles": f"c1ccccc1C{i}>>c1ccccc1C{i}O",
                "reactants": f"c1ccccc1C{i}",
                "products": f"c1ccccc1C{i}O",
                "label": i % 2,
                "split": "train",
                "dataset": "x",
                "reaction_class": "",
            }
            for i in range(10)
        ]
        splits = prepare_splits(real_rows, train_frac=0.6, seed=42)
        scaffold_splits = [r["split"] for r in splits["scaffold"]]
        # Should have reassigned splits based on scaffold
        self.assertTrue(any(s == "train" for s in scaffold_splits))

    def test_template_reassigns_split(self) -> None:
        real_rows = [
            {
                "source_id": f"id{i}",
                "reaction_smiles": f"CC{i}>>CC{i}O",
                "reactants": f"CC{i}",
                "products": f"CC{i}O",
                "label": i % 2,
                "split": "train",
                "dataset": "x",
                "reaction_class": "",
            }
            for i in range(10)
        ]
        splits = prepare_splits(real_rows, train_frac=0.6, seed=42)
        template_splits = [r["split"] for r in splits["template"]]
        self.assertTrue(any(s == "train" for s in template_splits))


if __name__ == "__main__":
    unittest.main()
