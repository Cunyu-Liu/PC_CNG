"""Unit tests for P2-01 AiZynthFinder Route Ranking.

Tests cover:
- Module imports
- CLI args parsing
- Metric computation (top-k recall, MRR, NDCG) on synthetic data
- Paired significance test structure (4 rankers, all pairwise deltas)
- Synthetic small test with mock AiZynthFinder output (monkeypatched)
- False-positive route detection logic
- Ground-truth oracle ranker
- Heuristic baseline ranker
- PC-CNG augmented ranker training
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
from typing import Dict, List, Sequence
from unittest.mock import patch


def _write_pc_cng_csv(path: Path) -> None:
    """Write a minimal PC-CNG synthetic negatives CSV for testing."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "source_id", "positive_reaction", "candidate_reaction", "task",
            "failure_type", "edit_action", "parent_reactants", "parent_product",
            "candidate_reactants", "candidate_product", "valid", "atom_balance",
            "locality", "closeness", "hard_score", "false_negative_risk",
            "passes_filter", "label", "provenance", "review_status",
            "review_reasons", "product_overlap",
        ])
        # group g1: positive + 2 negatives, target = CC(=O)OCC
        writer.writerow([
            "g1", "CC(=O)O.CCO>>CC(=O)OCC",
            "CC(=O)O.CCO>>CC(=O)OCC.O",
            "retro_precursor", "retro_wrong_product", "edit1",
            "CC(=O)O.CCO", "CC(=O)OCC",
            "CC(=O)O.CCO", "CC(=O)OCC.O",
            1.0, 0.9, 0.5, 0.5, 0.5, 0.5, True, 0, "prov", "keep", "", 1.0,
        ])
        writer.writerow([
            "g1", "CC(=O)O.CCO>>CC(=O)OCC",
            "CC(=O)O.CCO>>CCC",
            "retro_precursor", "retro_wrong_product", "edit2",
            "CC(=O)O.CCO", "CC(=O)OCC",
            "CC(=O)O.CCO", "CCC",
            1.0, 0.4, 0.5, 0.5, 0.5, 0.5, True, 0, "prov", "keep", "", 1.0,
        ])
        # group g2: positive + 2 negatives, target = C(=O)ONCC
        writer.writerow([
            "g2", "C(=O)O.NCC>>C(=O)ONCC",
            "C(=O)O.NCC>>C(=O)ONCC.O",
            "retro_precursor", "retro_wrong_product", "edit1",
            "C(=O)O.NCC", "C(=O)ONCC",
            "C(=O)O.NCC", "C(=O)ONCC.O",
            1.0, 0.9, 0.5, 0.5, 0.5, 0.5, True, 0, "prov", "keep", "", 1.0,
        ])
        writer.writerow([
            "g2", "C(=O)O.NCC>>C(=O)ONCC",
            "C(=O)O.NCC>>NCC",
            "retro_precursor", "retro_wrong_product", "edit2",
            "C(=O)O.NCC", "C(=O)ONCC",
            "C(=O)O.NCC", "NCC",
            1.0, 0.4, 0.5, 0.5, 0.5, 0.5, True, 0, "prov", "keep", "", 1.0,
        ])


class AizynthfinderRouteRankingImportTest(unittest.TestCase):
    """Test that the module can be imported and exposes expected symbols."""

    def test_module_imports(self) -> None:
        from pc_cng import run_aizynthfinder_route_ranking as mod
        self.assertTrue(hasattr(mod, "main"))
        self.assertTrue(hasattr(mod, "run_seed"))
        self.assertTrue(hasattr(mod, "evaluate"))
        self.assertTrue(hasattr(mod, "paired_significance"))
        self.assertTrue(hasattr(mod, "try_aizynthfinder_search"))

    def test_default_seeds(self) -> None:
        from pc_cng.run_aizynthfinder_route_ranking import DEFAULT_SEEDS
        self.assertEqual(len(DEFAULT_SEEDS), 10)
        self.assertEqual(DEFAULT_SEEDS[0], 20260710)
        self.assertEqual(DEFAULT_SEEDS[-1], 20260719)

    def test_ranker_names(self) -> None:
        from pc_cng.run_aizynthfinder_route_ranking import RANKER_NAMES
        self.assertEqual(len(RANKER_NAMES), 4)
        self.assertIn("aizynthfinder_baseline", RANKER_NAMES)
        self.assertIn("aizynthfinder_chemformer", RANKER_NAMES)
        self.assertIn("aizynthfinder_pc_cng", RANKER_NAMES)
        self.assertIn("ground_truth", RANKER_NAMES)


class AizynthfinderRouteRankingCLITest(unittest.TestCase):
    """Test CLI argument parsing."""

    def test_parse_args_defaults(self) -> None:
        from pc_cng.run_aizynthfinder_route_ranking import _parse_args
        args = _parse_args([
            "--pc-cng-negatives", "/tmp/neg.csv",
            "--output-dir", "/tmp/out",
        ])
        self.assertEqual(args.pc_cng_negatives, "/tmp/neg.csv")
        self.assertEqual(args.output_dir, "/tmp/out")
        self.assertEqual(args.max_candidates_per_source, 10)
        self.assertEqual(args.top_k, 10)
        self.assertEqual(args.epochs, 200)
        self.assertIsNone(args.limit)

    def test_parse_args_limit_smoke(self) -> None:
        from pc_cng.run_aizynthfinder_route_ranking import _parse_args
        args = _parse_args([
            "--pc-cng-negatives", "/tmp/neg.csv",
            "--output-dir", "/tmp/out",
            "--limit", "10",
            "--seeds", "20260710",
            "--no-chemformer",
            "--no-aizynthfinder",
        ])
        self.assertEqual(args.limit, 10)
        self.assertEqual(args.seeds, "20260710")
        self.assertTrue(args.no_chemformer)
        self.assertTrue(args.no_aizynthfinder)

    def test_parse_args_aizynthfinder_python(self) -> None:
        from pc_cng.run_aizynthfinder_route_ranking import _parse_args, AIZYNTHFINDER_PYTHON_DEFAULT
        args = _parse_args([
            "--pc-cng-negatives", "/tmp/neg.csv",
            "--output-dir", "/tmp/out",
        ])
        self.assertEqual(args.aizynthfinder_python, AIZYNTHFINDER_PYTHON_DEFAULT)


class AizynthfinderRouteRankingMetricsTest(unittest.TestCase):
    """Test metric computations on synthetic data."""

    def test_topk_route_recall(self) -> None:
        from pc_cng.run_aizynthfinder_route_ranking import topk_route_recall
        rows = [
            {"group_id": "g1", "label": 1, "score": 0.9},
            {"group_id": "g1", "label": 0, "score": 0.1},
            {"group_id": "g2", "label": 0, "score": 0.8},
            {"group_id": "g2", "label": 1, "score": 0.3},
        ]
        # top1: g1 hit, g2 miss -> 0.5
        self.assertAlmostEqual(topk_route_recall(rows, 1), 0.5)
        # top3: both hit -> 1.0
        self.assertAlmostEqual(topk_route_recall(rows, 3), 1.0)

    def test_topk_route_recall_empty(self) -> None:
        from pc_cng.run_aizynthfinder_route_ranking import topk_route_recall
        # No evaluable groups (all label=1)
        rows = [
            {"group_id": "g1", "label": 1, "score": 0.9},
            {"group_id": "g1", "label": 1, "score": 0.1},
        ]
        self.assertAlmostEqual(topk_route_recall(rows, 1), 0.0)

    def test_mrr(self) -> None:
        from pc_cng.run_aizynthfinder_route_ranking import mrr
        rows = [
            {"group_id": "g1", "label": 1, "score": 0.9},
            {"group_id": "g1", "label": 0, "score": 0.1},
            {"group_id": "g2", "label": 0, "score": 0.8},
            {"group_id": "g2", "label": 1, "score": 0.3},
        ]
        # g1: 1/1 = 1.0; g2: 1/2 = 0.5; mean = 0.75
        self.assertAlmostEqual(mrr(rows), 0.75)

    def test_ndcg_at_k_perfect(self) -> None:
        from pc_cng.run_aizynthfinder_route_ranking import ndcg_at_k
        rows = [
            {"group_id": "g1", "label": 1, "score": 0.9},
            {"group_id": "g1", "label": 0, "score": 0.1},
        ]
        # Perfect ranking: gold at #1 -> NDCG = 1.0
        self.assertAlmostEqual(ndcg_at_k(rows, 10), 1.0)

    def test_ndcg_at_k_imperfect(self) -> None:
        from pc_cng.run_aizynthfinder_route_ranking import ndcg_at_k
        rows = [
            {"group_id": "g1", "label": 0, "score": 0.9},
            {"group_id": "g1", "label": 1, "score": 0.1},
        ]
        # Gold at #2: DCG = 0 + 1/log2(3) = 0.6309; IDCG = 1.0
        # NDCG = 0.6309
        expected = 1.0 / math.log2(3)
        self.assertAlmostEqual(ndcg_at_k(rows, 10), expected, places=4)

    def test_false_positive_route_rate(self) -> None:
        from pc_cng.run_aizynthfinder_route_ranking import false_positive_route_rate
        rows = [
            {"group_id": "g1", "label": 1, "score": 0.9},  # gold at #1 (no FP)
            {"group_id": "g1", "label": 0, "score": 0.1},
            {"group_id": "g2", "label": 0, "score": 0.8},  # neg at #1 (FP)
            {"group_id": "g2", "label": 1, "score": 0.3},
        ]
        # 1 FP / 2 groups = 0.5
        self.assertAlmostEqual(false_positive_route_rate(rows), 0.5)

    def test_evaluate_keys(self) -> None:
        from pc_cng.run_aizynthfinder_route_ranking import evaluate
        rows = [
            {"group_id": "g1", "label": 1, "score": 0.9},
            {"group_id": "g1", "label": 0, "score": 0.1},
        ]
        metrics = evaluate(rows)
        for key in [
            "top1_route_recall", "top3_route_recall", "top5_route_recall",
            "top10_route_recall", "mrr", "ndcg_at_10",
            "false_positive_route_rate",
        ]:
            self.assertIn(key, metrics)


class AizynthfinderRouteRankingRankersTest(unittest.TestCase):
    """Test individual ranker scoring functions."""

    def test_score_rows_ground_truth(self) -> None:
        from pc_cng.run_aizynthfinder_route_ranking import score_rows_ground_truth
        rows = [
            {"group_id": "g1", "label": 1, "reaction_smiles": "A>>B", "hard_score": 1.0},
            {"group_id": "g1", "label": 0, "reaction_smiles": "C>>B", "hard_score": 0.5},
        ]
        scored = score_rows_ground_truth(rows)
        self.assertEqual(len(scored), 2)
        # label=1 gets higher score than label=0
        self.assertGreater(scored[0]["score"], scored[1]["score"])
        self.assertAlmostEqual(scored[0]["score"], 1.0 + 1e-3, places=5)

    def test_score_rows_heuristic(self) -> None:
        from pc_cng.run_aizynthfinder_route_ranking import score_rows_heuristic
        rows = [
            {"group_id": "g1", "label": 1, "reaction_smiles": "CC(=O)O.CCO>>CC(=O)OCC"},
            {"group_id": "g1", "label": 0, "reaction_smiles": "CC(=O)O.CCO>>CCC"},
        ]
        scored = score_rows_heuristic(rows)
        self.assertEqual(len(scored), 2)
        for row in scored:
            self.assertIn("score", row)
            self.assertIsInstance(row["score"], float)

    def test_score_rows_pc_cng(self) -> None:
        from pc_cng.run_aizynthfinder_route_ranking import (
            FeatureCache, CachedLogisticReactionRanker, score_rows_pc_cng,
        )
        cache = FeatureCache()
        model = CachedLogisticReactionRanker(cache=cache, epochs=5, n_features=10)
        train_rows = [
            {"reaction_smiles": "CC(=O)O.CCO>>CC(=O)OCC", "label": 1},
            {"reaction_smiles": "CC(=O)O.CCO>>CCC", "label": 0},
        ]
        model.fit(train_rows)
        test_rows = [
            {"group_id": "g1", "label": 1, "reaction_smiles": "CC(=O)O.CCO>>CC(=O)OCC"},
            {"group_id": "g1", "label": 0, "reaction_smiles": "CC(=O)O.CCO>>CCC"},
        ]
        scored = score_rows_pc_cng(model, test_rows)
        self.assertEqual(len(scored), 2)
        for row in scored:
            self.assertIn("score", row)
            self.assertGreaterEqual(row["score"], 0.0)
            self.assertLessEqual(row["score"], 1.0)


class AizynthfinderRouteRankingDataLoadingTest(unittest.TestCase):
    """Test PC-CNG negatives loading."""

    def test_load_pc_cng_negatives(self) -> None:
        from pc_cng.run_aizynthfinder_route_ranking import load_pc_cng_negatives
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "neg.csv"
            _write_pc_cng_csv(csv_path)
            rows = load_pc_cng_negatives(str(csv_path), max_candidates_per_source=10)
            # 2 groups × (1 positive + 2 negatives) = 6 rows
            self.assertEqual(len(rows), 6)
            pos = [r for r in rows if int(r["label"]) == 1]
            neg = [r for r in rows if int(r["label"]) == 0]
            self.assertEqual(len(pos), 2)
            self.assertEqual(len(neg), 4)
            self.assertEqual({str(r["group_id"]) for r in rows}, {"g1", "g2"})
            # parent_product captured
            for row in rows:
                self.assertTrue(row.get("parent_product"))
            self.assertEqual(
                {str(r["parent_product"]) for r in rows},
                {"CC(=O)OCC", "C(=O)ONCC"},
            )

    def test_load_pc_cng_negatives_caps_candidates(self) -> None:
        from pc_cng.run_aizynthfinder_route_ranking import load_pc_cng_negatives
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "neg.csv"
            _write_pc_cng_csv(csv_path)
            rows = load_pc_cng_negatives(str(csv_path), max_candidates_per_source=1)
            self.assertEqual(len(rows), 4)  # 2 groups × (1 pos + 1 neg)

    def test_load_uspto_mit_50k_routes_missing(self) -> None:
        from pc_cng.run_aizynthfinder_route_ranking import load_uspto_mit_50k_routes
        # Missing file returns empty list (no exception)
        rows = load_uspto_mit_50k_routes("/nonexistent/path.csv")
        self.assertEqual(rows, [])

    def test_load_uspto_mit_50k_routes_parses(self) -> None:
        from pc_cng.run_aizynthfinder_route_ranking import load_uspto_mit_50k_routes
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "routes.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as h:
                writer = csv.writer(h)
                writer.writerow(["product_smiles", "route_smiles", "route_id", "is_gold"])
                writer.writerow(["CC(=O)OCC", "CC(=O)O.CCO>>CC(=O)OCC", "r1", 1])
                writer.writerow(["CC(=O)OCC", "CC(=O)O.CCO>>CCC", "r2", 0])
            rows = load_uspto_mit_50k_routes(str(csv_path))
            self.assertEqual(len(rows), 2)
            self.assertEqual(int(rows[0]["label"]), 1)
            self.assertEqual(int(rows[1]["label"]), 0)


class AizynthfinderRouteRankingTemplateRetroTest(unittest.TestCase):
    """Test RDKit template-based retrosynthesis fallback."""

    def test_generate_template_routes_invalid_smiles(self) -> None:
        from pc_cng.run_aizynthfinder_route_ranking import generate_template_routes
        routes = generate_template_routes("not_a_smiles", max_routes=3)
        self.assertEqual(routes, [])

    def test_generate_template_routes_simple_target(self) -> None:
        from pc_cng.run_aizynthfinder_route_ranking import generate_template_routes
        # Ethyl acetate: CC(=O)OCC — ester hydrolysis template should apply
        routes = generate_template_routes("CC(=O)OCC", max_routes=3)
        # May or may not produce routes depending on RDKit's template matching,
        # but the function should not raise
        self.assertIsInstance(routes, list)
        for route in routes:
            self.assertIn("reaction_smiles", route)
            self.assertIn("score", route)
            self.assertIn("source", route)


class AizynthfinderRouteRankingMockedAizynthfinderTest(unittest.TestCase):
    """Test AiZynthFinder subprocess integration with monkeypatched subprocess."""

    def test_try_aizynthfinder_search_env_missing(self) -> None:
        from pc_cng.run_aizynthfinder_route_ranking import try_aizynthfinder_search
        routes, status = try_aizynthfinder_search(
            "CCO", "/nonexistent/python/bin/python",
        )
        self.assertEqual(routes, [])
        self.assertEqual(status, "env_missing")

    def test_try_aizynthfinder_search_mocked_ok(self) -> None:
        """Mock subprocess.run to simulate AiZynthFinder returning routes."""
        from pc_cng.run_aizynthfinder_route_ranking import try_aizynthfinder_search

        class FakeCompletedProcess:
            def __init__(self, stdout: str, returncode: int = 0) -> None:
                self.stdout = stdout
                self.returncode = returncode
                self.stderr = ""

        fake_output = json.dumps({
            "status": "ok",
            "target": "CC(=O)OCC",
            "routes": [
                {"reaction_smiles": "CC(=O)O.CCO>>CC(=O)OCC", "score": 0.9, "source": "aizynthfinder"},
                {"reaction_smiles": "CC(=O)O.CO>>CC(=O)OC", "score": 0.7, "source": "aizynthfinder"},
            ],
            "n_routes": 2,
        })
        with patch("pc_cng.run_aizynthfinder_route_ranking.subprocess.run") as mock_run:
            mock_run.return_value = FakeCompletedProcess(fake_output)
            # Create a fake python path that exists
            with tempfile.NamedTemporaryFile(suffix="python", delete=False) as fp:
                fp.write(b"#!/bin/bash\n")
                fake_python = fp.name
            try:
                os.chmod(fake_python, 0o755)
                routes, status = try_aizynthfinder_search(
                    "CC(=O)OCC", fake_python, time_limit=5,
                )
                self.assertEqual(status, "ok")
                self.assertEqual(len(routes), 2)
                self.assertEqual(routes[0]["reaction_smiles"], "CC(=O)O.CCO>>CC(=O)OCC")
            finally:
                os.unlink(fake_python)

    def test_try_aizynthfinder_search_mocked_error(self) -> None:
        from pc_cng.run_aizynthfinder_route_ranking import try_aizynthfinder_search

        class FakeCompletedProcess:
            def __init__(self) -> None:
                self.stdout = ""
                self.returncode = 1
                self.stderr = "ImportError"

        with patch("pc_cng.run_aizynthfinder_route_ranking.subprocess.run") as mock_run:
            mock_run.return_value = FakeCompletedProcess()
            with tempfile.NamedTemporaryFile(suffix="python", delete=False) as fp:
                fake_python = fp.name
            try:
                os.chmod(fake_python, 0o755)
                routes, status = try_aizynthfinder_search(
                    "CCO", fake_python, time_limit=5,
                )
                self.assertEqual(routes, [])
                self.assertEqual(status, "error")
            finally:
                os.unlink(fake_python)


class AizynthfinderRouteRankingSeedRunnerTest(unittest.TestCase):
    """Test the seed runner with a synthetic small dataset."""

    def test_run_seed_four_rankers(self) -> None:
        from pc_cng.run_aizynthfinder_route_ranking import run_seed
        rows = [
            {"group_id": "g1", "source_id": "g1",
             "reaction_smiles": "CC(=O)O.CCO>>CC(=O)OCC",
             "label": 1, "candidate_source": "positive_reaction",
             "hard_score": 1.0, "parent_product": "CC(=O)OCC"},
            {"group_id": "g1", "source_id": "g1",
             "reaction_smiles": "CC(=O)O.CCO>>CCC",
             "label": 0, "candidate_source": "pc_cng_synthetic",
             "hard_score": 0.5, "parent_product": "CC(=O)OCC"},
            {"group_id": "g2", "source_id": "g2",
             "reaction_smiles": "C(=O)O.NCC>>C(=O)ONCC",
             "label": 1, "candidate_source": "positive_reaction",
             "hard_score": 1.0, "parent_product": "C(=O)ONCC"},
            {"group_id": "g2", "source_id": "g2",
             "reaction_smiles": "C(=O)O.NCC>>NCC",
             "label": 0, "candidate_source": "pc_cng_synthetic",
             "hard_score": 0.5, "parent_product": "C(=O)ONCC"},
        ]
        result = run_seed(
            rows, seed=20260710, train_fraction=0.5, epochs=10,
            use_chemformer=False,  # skip chemformer subprocess for speed
        )
        # All 4 rankers must produce metrics
        for ranker in ["r1", "r2", "r3", "r4"]:
            self.assertIn(f"{ranker}_metrics", result)
            self.assertIn(f"{ranker}_per_group", result)
            metrics = result[f"{ranker}_metrics"]
            for key in ["top1_route_recall", "mrr", "ndcg_at_10",
                        "false_positive_route_rate"]:
                self.assertIn(key, metrics)
        # Ground-truth oracle (R4) should have perfect MRR
        self.assertAlmostEqual(result["r4_metrics"]["mrr"], 1.0)
        # R4 top1 recall should be 1.0 (gold always ranks first)
        self.assertAlmostEqual(result["r4_metrics"]["top1_route_recall"], 1.0)

    def test_run_seed_with_mocked_aizynthfinder_routes(self) -> None:
        """Test that AiZynthFinder routes are integrated into candidate set."""
        from pc_cng.run_aizynthfinder_route_ranking import run_seed
        rows = [
            {"group_id": "g1", "source_id": "g1",
             "reaction_smiles": "CC(=O)O.CCO>>CC(=O)OCC",
             "label": 1, "candidate_source": "positive_reaction",
             "hard_score": 1.0, "parent_product": "CC(=O)OCC"},
            {"group_id": "g1", "source_id": "g1",
             "reaction_smiles": "CC(=O)O.CCO>>CCC",
             "label": 0, "candidate_source": "pc_cng_synthetic",
             "hard_score": 0.5, "parent_product": "CC(=O)OCC"},
        ]
        # Provide a mock AiZynthFinder route that matches the gold route's reactants
        af_routes = {
            "g1": [
                {"reaction_smiles": "CC(=O)O.CCO>>CC(=O)OCC",
                 "score": 0.95, "source": "aizynthfinder"},
                {"reaction_smiles": "CC(=O)O.CCO>>CC",
                 "score": 0.5, "source": "aizynthfinder"},
            ],
        }
        result = run_seed(
            rows, seed=20260710, train_fraction=0.5, epochs=5,
            aizynthfinder_routes_by_group=af_routes,
            use_chemformer=False,
        )
        # The augmented test set should have more rows than original
        self.assertGreater(result["n_test_augmented"], result["n_test"])


class AizynthfinderRouteRankingPairedSignificanceTest(unittest.TestCase):
    """Test paired significance test structure with 4 rankers."""

    def test_paired_significance_all_pairs(self) -> None:
        from pc_cng.run_aizynthfinder_route_ranking import paired_significance
        # Synthetic seed_results: 2 seeds, 2 groups
        seed_results = [
            {
                "seed": 20260710,
                "r1_per_group": {"g1": {"mrr": 0.5, "top1": 0.0, "ndcg": 0.5},
                                  "g2": {"mrr": 0.3, "top1": 0.0, "ndcg": 0.3}},
                "r2_per_group": {"g1": {"mrr": 0.7, "top1": 1.0, "ndcg": 0.7},
                                  "g2": {"mrr": 0.5, "top1": 0.0, "ndcg": 0.5}},
                "r3_per_group": {"g1": {"mrr": 0.9, "top1": 1.0, "ndcg": 0.9},
                                  "g2": {"mrr": 0.8, "top1": 1.0, "ndcg": 0.8}},
                "r4_per_group": {"g1": {"mrr": 1.0, "top1": 1.0, "ndcg": 1.0},
                                  "g2": {"mrr": 1.0, "top1": 1.0, "ndcg": 1.0}},
            },
            {
                "seed": 20260711,
                "r1_per_group": {"g1": {"mrr": 0.4, "top1": 0.0, "ndcg": 0.4},
                                  "g2": {"mrr": 0.2, "top1": 0.0, "ndcg": 0.2}},
                "r2_per_group": {"g1": {"mrr": 0.6, "top1": 1.0, "ndcg": 0.6},
                                  "g2": {"mrr": 0.4, "top1": 0.0, "ndcg": 0.4}},
                "r3_per_group": {"g1": {"mrr": 0.85, "top1": 1.0, "ndcg": 0.85},
                                  "g2": {"mrr": 0.75, "top1": 1.0, "ndcg": 0.75}},
                "r4_per_group": {"g1": {"mrr": 1.0, "top1": 1.0, "ndcg": 1.0},
                                  "g2": {"mrr": 1.0, "top1": 1.0, "ndcg": 1.0}},
            },
        ]
        sig = paired_significance(seed_results, bootstrap_iterations=100, seed=20260710)
        # All 4 pairwise comparisons must be present
        for key in ["r3_vs_r1", "r3_vs_r2", "r2_vs_r1", "r3_vs_r4"]:
            self.assertIn(key, sig)
            pair_sig = sig[key]
            for field in [
                "n_seeds", "n_common_groups", "metric", "ranker_a", "ranker_b",
                "ranker_a_mean", "ranker_b_mean", "delta_mean", "delta_pp",
                "group_level_ci95_low", "group_level_ci95_high",
                "seed_level_ci95_low", "seed_level_ci95_high",
                "paired_permutation_p", "sign_test_p",
                "ranker_b_better_groups", "ranker_a_better_groups", "tie_groups",
            ]:
                self.assertIn(field, pair_sig, f"Missing {field} in {key}")
        # R3 should be better than R1 (positive delta)
        self.assertGreater(sig["r3_vs_r1"]["delta_mean"], 0.0)
        # R3 vs R4 (oracle) should be negative (R3 cannot beat oracle)
        self.assertLessEqual(sig["r3_vs_r4"]["delta_mean"], 0.0)


class AizynthfinderRouteRankingFalsePositiveTest(unittest.TestCase):
    """Test false-positive route detection logic."""

    def test_false_positive_routes_csv_written(self) -> None:
        from pc_cng.run_aizynthfinder_route_ranking import write_false_positive_routes
        seed_results = [
            {
                "seed": 20260710,
                "r1_per_group": {
                    "g1": {"mrr": 0.5, "top1": 0.0, "ndcg": 0.5},  # FP (top1=0)
                    "g2": {"mrr": 1.0, "top1": 1.0, "ndcg": 1.0},  # no FP
                },
                "r2_per_group": {
                    "g1": {"mrr": 1.0, "top1": 1.0, "ndcg": 1.0},
                    "g2": {"mrr": 1.0, "top1": 1.0, "ndcg": 1.0},
                },
                "r3_per_group": {
                    "g1": {"mrr": 1.0, "top1": 1.0, "ndcg": 1.0},
                    "g2": {"mrr": 0.3, "top1": 0.0, "ndcg": 0.3},  # FP
                },
                "r4_per_group": {
                    "g1": {"mrr": 1.0, "top1": 1.0, "ndcg": 1.0},
                    "g2": {"mrr": 1.0, "top1": 1.0, "ndcg": 1.0},
                },
            },
        ]
        all_rows = [
            {"group_id": "g1", "label": 1, "reaction_smiles": "A>>B"},
            {"group_id": "g1", "label": 0, "reaction_smiles": "C>>B"},
            {"group_id": "g2", "label": 1, "reaction_smiles": "D>>E"},
            {"group_id": "g2", "label": 0, "reaction_smiles": "F>>E"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "fp.csv"
            write_false_positive_routes(str(out_path), seed_results, all_rows)
            self.assertTrue(out_path.exists())
            with out_path.open() as h:
                reader = csv.DictReader(h)
                rows = list(reader)
            # Expect 2 FP rows: r1/g1 and r3/g2
            self.assertEqual(len(rows), 2)
            group_ranker_pairs = {(r["group_id"], r["ranker"]) for r in rows}
            self.assertIn(("g1", "aizynthfinder_baseline"), group_ranker_pairs)
            self.assertIn(("g2", "aizynthfinder_pc_cng"), group_ranker_pairs)


class AizynthfinderRouteRankingOutputWritersTest(unittest.TestCase):
    """Test output file writers."""

    def test_write_route_ranking_summary(self) -> None:
        from pc_cng.run_aizynthfinder_route_ranking import write_route_ranking_summary
        ranker_metrics = {
            "aizynthfinder_baseline": {"mrr": 0.3, "top1_route_recall": 0.2},
            "aizynthfinder_chemformer": {"mrr": 0.5, "top1_route_recall": 0.4},
            "aizynthfinder_pc_cng": {"mrr": 0.7, "top1_route_recall": 0.6},
            "ground_truth": {"mrr": 1.0, "top1_route_recall": 1.0},
        }
        sig = {"r3_vs_r1": {"delta_pp": 40.0}}
        manifest_meta = {
            "fallback_path": "template_fallback",
            "n_seeds": 10,
            "n_source_ids": 100,
        }
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "summary.json"
            write_route_ranking_summary(
                str(out_path), ranker_metrics, sig, manifest_meta,
            )
            self.assertTrue(out_path.exists())
            data = json.loads(out_path.read_text())
            self.assertEqual(data["task"], "P2-01 AiZynthFinder Route Ranking")
            self.assertIn("rankers", data)
            self.assertEqual(len(data["rankers"]), 4)
            self.assertIn("metrics", data)
            self.assertIn("paired_significance", data)

    def test_write_per_target_metrics(self) -> None:
        from pc_cng.run_aizynthfinder_route_ranking import write_per_target_metrics
        seed_results = [
            {
                "seed": 20260710,
                "r1_per_group": {"g1": {"mrr": 0.5, "top1": 0.0, "ndcg": 0.5}},
                "r2_per_group": {"g1": {"mrr": 0.7, "top1": 1.0, "ndcg": 0.7}},
                "r3_per_group": {"g1": {"mrr": 0.9, "top1": 1.0, "ndcg": 0.9}},
                "r4_per_group": {"g1": {"mrr": 1.0, "top1": 1.0, "ndcg": 1.0}},
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "per_target.csv"
            write_per_target_metrics(str(out_path), seed_results)
            self.assertTrue(out_path.exists())
            with out_path.open() as h:
                reader = csv.DictReader(h)
                rows = list(reader)
            # 1 seed × 1 group × 4 rankers = 4 rows
            self.assertEqual(len(rows), 4)
            expected_rankers = {
                "aizynthfinder_baseline", "aizynthfinder_chemformer",
                "aizynthfinder_pc_cng", "ground_truth",
            }
            self.assertEqual({r["ranker"] for r in rows}, expected_rankers)


class AizynthfinderRouteRankingEndToEndTest(unittest.TestCase):
    """End-to-end CLI smoke test (no AiZynthFinder, no Chemformer)."""

    def test_cli_smoke_no_aizynthfinder_no_chemformer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = root / "neg.csv"
            out_dir = root / "out"
            _write_pc_cng_csv(csv_path)
            result = subprocess.run(
                [
                    sys.executable,
                    "-m", "pc_cng.run_aizynthfinder_route_ranking",
                    "--routes-data", str(root / "nonexistent.csv"),
                    "--pc-cng-negatives", str(csv_path),
                    "--output-dir", str(out_dir),
                    "--seeds", "20260710",
                    "--max-sources", "10",
                    "--max-candidates-per-source", "5",
                    "--bootstrap-iterations", "50",
                    "--epochs", "20",
                    "--no-chemformer",
                    "--no-aizynthfinder",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            # All output files must exist
            for fname in [
                "route_ranking_summary.json",
                "per_target_metrics.csv",
                "paired_significance.json",
                "false_positive_routes.csv",
                "per_seed_detail.json",
            ]:
                self.assertTrue(
                    (out_dir / fname).exists(),
                    f"Missing {fname}; stderr=\n{result.stderr}",
                )
            # Manifest content
            summary = json.loads((out_dir / "route_ranking_summary.json").read_text())
            self.assertEqual(summary["task"], "P2-01 AiZynthFinder Route Ranking")
            self.assertEqual(summary["n_seeds"], 1)
            self.assertEqual(summary["aizynthfinder_status"], "skipped")
            self.assertEqual(summary["fallback_path"], "pseudo_route_only")
            self.assertIn("go_no_go", summary)
            self.assertIn("rankers", summary)
            self.assertEqual(len(summary["rankers"]), 4)
            # paired_significance.json structure
            sig = json.loads((out_dir / "paired_significance.json").read_text())
            for key in ["r3_vs_r1", "r3_vs_r2", "r2_vs_r1", "r3_vs_r4"]:
                self.assertIn(key, sig)
                for field in [
                    "ranker_a_mean", "ranker_b_mean",
                    "delta_mean", "delta_pp",
                    "group_level_ci95_low", "group_level_ci95_high",
                    "seed_level_ci95_low", "seed_level_ci95_high",
                    "paired_permutation_p", "sign_test_p",
                ]:
                    self.assertIn(field, sig[key], f"Missing {field} in {key}")
            # per_target_metrics.csv has rows for all 4 rankers
            with (out_dir / "per_target_metrics.csv").open() as h:
                reader = csv.DictReader(h)
                rows = list(reader)
            rankers_seen = {r["ranker"] for r in rows}
            self.assertEqual(rankers_seen, {
                "aizynthfinder_baseline", "aizynthfinder_chemformer",
                "aizynthfinder_pc_cng", "ground_truth",
            })
            # Ground-truth ranker should have perfect MRR
            gt_rows = [r for r in rows if r["ranker"] == "ground_truth"]
            for r in gt_rows:
                self.assertAlmostEqual(float(r["mrr"]), 1.0)


# Allow running tests with `python -m unittest` without pytest
if __name__ == "__main__":
    unittest.main()
