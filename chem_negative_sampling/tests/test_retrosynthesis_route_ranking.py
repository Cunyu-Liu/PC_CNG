"""Unit tests for P1-04 Retrosynthesis Route Ranking.

Tests cover:
- PC-CNG negatives / pseudo-route loading and parsing
- Top-K route recall computation
- MRR computation
- NDCG@10 computation
- False-positive route rate computation
- 10-seed paired significance (bootstrap CI, permutation p, sign-test p)
- CLI smoke test (full pipeline end-to-end)
"""

from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


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
        # group g1: positive + 2 negatives
        writer.writerow([
            "g1", "CC(=O)O.CCO>>CC(=O)OCC",
            "CC(=O)O.CCO>>CC(=O)OCC.O",  # added extra product (imbalanced)
            "retro_precursor", "retro_wrong_product", "edit1",
            "CC(=O)O.CCO", "CC(=O)OCC",
            "CC(=O)O.CCO", "CC(=O)OCC.O",
            1.0, 0.9, 0.5, 0.5, 0.5, 0.5, True, 0, "prov", "keep", "", 1.0,
        ])
        writer.writerow([
            "g1", "CC(=O)O.CCO>>CC(=O)OCC",
            "CC(=O)O.CCO>>CCC",  # different product entirely
            "retro_precursor", "retro_wrong_product", "edit2",
            "CC(=O)O.CCO", "CC(=O)OCC",
            "CC(=O)O.CCO", "CCC",
            1.0, 0.4, 0.5, 0.5, 0.5, 0.5, True, 0, "prov", "keep", "", 1.0,
        ])
        # group g2: positive + 2 negatives
        writer.writerow([
            "g2", "C(=O)O.NCC>>C(=O)ONCC",
            "C(=O)O.NCC>>C(=O)ONCC.O",  # imbalanced
            "retro_precursor", "retro_wrong_product", "edit1",
            "C(=O)O.NCC", "C(=O)ONCC",
            "C(=O)O.NCC", "C(=O)ONCC.O",
            1.0, 0.9, 0.5, 0.5, 0.5, 0.5, True, 0, "prov", "keep", "", 1.0,
        ])
        writer.writerow([
            "g2", "C(=O)O.NCC>>C(=O)ONCC",
            "C(=O)O.NCC>>NCC",  # missing product fragment
            "retro_precursor", "retro_wrong_product", "edit2",
            "C(=O)O.NCC", "C(=O)ONCC",
            "C(=O)O.NCC", "NCC",
            1.0, 0.4, 0.5, 0.5, 0.5, 0.5, True, 0, "prov", "keep", "", 1.0,
        ])


class RetrosynthesisRouteRankingTest(unittest.TestCase):
    def test_load_pc_cng_negatives(self) -> None:
        from pc_cng.run_retrosynthesis_route_ranking import load_pc_cng_negatives
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
            # group_id matches source_id
            self.assertEqual({str(r["group_id"]) for r in rows}, {"g1", "g2"})
            # candidate_source labeling
            self.assertEqual(
                {str(r["candidate_source"]) for r in pos},
                {"positive_reaction"},
            )
            self.assertEqual(
                {str(r["candidate_source"]) for r in neg},
                {"pc_cng_synthetic"},
            )

    def test_load_pc_cng_negatives_caps_candidates(self) -> None:
        from pc_cng.run_retrosynthesis_route_ranking import load_pc_cng_negatives
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "neg.csv"
            _write_pc_cng_csv(csv_path)
            # Cap at 1 candidate per source → 2 groups × (1 pos + 1 neg) = 4 rows
            rows = load_pc_cng_negatives(str(csv_path), max_candidates_per_source=1)
            self.assertEqual(len(rows), 4)
            self.assertEqual(
                sum(1 for r in rows if int(r["label"]) == 1), 2
            )
            self.assertEqual(
                sum(1 for r in rows if int(r["label"]) == 0), 2
            )

    def test_topk_route_recall(self) -> None:
        from pc_cng.run_retrosynthesis_route_ranking import topk_route_recall
        rows = [
            {"group_id": "g1", "label": 1, "score": 0.9},  # rank 1 (hit)
            {"group_id": "g1", "label": 0, "score": 0.1},
            {"group_id": "g2", "label": 0, "score": 0.8},  # rank 1 (miss)
            {"group_id": "g2", "label": 1, "score": 0.3},  # rank 2
        ]
        # top1: g1 hit, g2 miss → 0.5
        self.assertAlmostEqual(topk_route_recall(rows, 1), 0.5)
        # top3: both hit (each group has 2 candidates) → 1.0
        self.assertAlmostEqual(topk_route_recall(rows, 3), 1.0)

    def test_topk_route_recall_empty(self) -> None:
        from pc_cng.run_retrosynthesis_route_ranking import topk_route_recall
        # No evaluable groups (all label=1 or all label=0)
        rows = [
            {"group_id": "g1", "label": 1, "score": 0.9},
            {"group_id": "g1", "label": 1, "score": 0.1},
        ]
        self.assertAlmostEqual(topk_route_recall(rows, 1), 0.0)

    def test_mrr(self) -> None:
        from pc_cng.run_retrosynthesis_route_ranking import mrr
        rows = [
            {"group_id": "g1", "label": 1, "score": 0.9},  # rank 1
            {"group_id": "g1", "label": 0, "score": 0.1},
            {"group_id": "g2", "label": 0, "score": 0.8},  # rank 1
            {"group_id": "g2", "label": 1, "score": 0.3},  # rank 2
        ]
        # g1: 1/1 = 1.0; g2: 1/2 = 0.5; mean = 0.75
        self.assertAlmostEqual(mrr(rows), 0.75)

    def test_ndcg_at_k_perfect(self) -> None:
        from pc_cng.run_retrosynthesis_route_ranking import ndcg_at_k
        rows = [
            {"group_id": "g1", "label": 1, "score": 0.9},  # rank 1
            {"group_id": "g1", "label": 0, "score": 0.1},
        ]
        # DCG = 1/log2(2) = 1.0; IDCG = 1.0; NDCG = 1.0
        self.assertAlmostEqual(ndcg_at_k(rows, 10), 1.0)

    def test_ndcg_at_k_worst(self) -> None:
        from pc_cng.run_retrosynthesis_route_ranking import ndcg_at_k
        rows = [
            {"group_id": "g1", "label": 0, "score": 0.9},  # rank 1
            {"group_id": "g1", "label": 1, "score": 0.1},  # rank 2
        ]
        # DCG = 0 + 1/log2(3) = 1/1.585 = 0.6309
        # IDCG = 1/log2(2) = 1.0
        # NDCG = 0.6309
        self.assertAlmostEqual(ndcg_at_k(rows, 10), 1.0 / math.log2(3), places=4)

    def test_false_positive_route_rate(self) -> None:
        from pc_cng.run_retrosynthesis_route_ranking import (
            false_positive_route_rate,
        )
        rows = [
            {"group_id": "g1", "label": 1, "score": 0.9},  # rank 1 (TP)
            {"group_id": "g1", "label": 0, "score": 0.1},
            {"group_id": "g2", "label": 0, "score": 0.8},  # rank 1 (FP)
            {"group_id": "g2", "label": 1, "score": 0.3},
        ]
        self.assertAlmostEqual(false_positive_route_rate(rows), 0.5)

    def test_heuristic_score_is_deterministic(self) -> None:
        from pc_cng.run_retrosynthesis_route_ranking import heuristic_score
        rxn = "CC(=O)O.CCO>>CC(=O)OCC"
        s1 = heuristic_score(rxn)
        s2 = heuristic_score(rxn)
        self.assertEqual(s1, s2)
        # Should be in [0, 1]
        self.assertGreaterEqual(s1, 0.0)
        self.assertLessEqual(s1, 1.0)

    def test_per_group_metrics(self) -> None:
        from pc_cng.run_retrosynthesis_route_ranking import per_group_metrics
        rows = [
            {"group_id": "g1", "label": 1, "score": 0.9, "reaction_smiles": "A>>B"},
            {"group_id": "g1", "label": 0, "score": 0.1, "reaction_smiles": "A>>C"},
        ]
        pg = per_group_metrics(rows)
        self.assertIn("g1", pg)
        self.assertAlmostEqual(pg["g1"]["top1"], 1.0)
        self.assertAlmostEqual(pg["g1"]["mrr"], 1.0)
        self.assertAlmostEqual(pg["g1"]["ndcg"], 1.0)

    def test_paired_significance_clear_improvement(self) -> None:
        from pc_cng.run_retrosynthesis_route_ranking import paired_significance
        # 3 seeds with PC-CNG >> baseline on every group
        seed_results = []
        for s in range(3):
            baseline_pg = {
                f"g{i}": {"top1": 0.0, "mrr": 0.3, "ndcg": 0.3} for i in range(10)
            }
            pc_cng_pg = {
                f"g{i}": {"top1": 1.0, "mrr": 0.9, "ndcg": 0.9} for i in range(10)
            }
            seed_results.append({
                "seed": s,
                "baseline_per_group": baseline_pg,
                "pc_cng_per_group": pc_cng_pg,
            })
        sig = paired_significance(seed_results, bootstrap_iterations=500, seed=20260710)
        self.assertEqual(sig["n_seeds"], 3)
        self.assertEqual(sig["n_common_groups"], 10)
        self.assertGreater(sig["delta_mean"], 0.0)
        self.assertGreater(sig["delta_pp"], 50.0)
        # CI should be entirely positive
        self.assertGreater(sig["group_level_ci95_low"], 0.0)
        self.assertGreater(sig["seed_level_ci95_low"], 0.0)
        # Permutation p should be small (clear improvement)
        self.assertLess(sig["paired_permutation_p"], 0.1)

    def test_paired_significance_no_difference(self) -> None:
        from pc_cng.run_retrosynthesis_route_ranking import paired_significance
        # 3 seeds with PC-CNG == baseline (delta = 0)
        seed_results = []
        for s in range(3):
            pg = {
                f"g{i}": {"top1": 0.5, "mrr": 0.5, "ndcg": 0.5} for i in range(10)
            }
            seed_results.append({
                "seed": s,
                "baseline_per_group": pg,
                "pc_cng_per_group": pg,
            })
        sig = paired_significance(seed_results, bootstrap_iterations=500, seed=20260710)
        self.assertAlmostEqual(sig["delta_mean"], 0.0)
        # CI should include 0
        self.assertLessEqual(sig["group_level_ci95_low"], 0.0 + 1e-9)
        self.assertGreaterEqual(sig["group_level_ci95_high"], 0.0 - 1e-9)
        # Sign-test p should be 1.0 (all ties)
        self.assertGreaterEqual(sig["sign_test_p"], 0.5)

    def test_run_seed_returns_metrics(self) -> None:
        from pc_cng.run_retrosynthesis_route_ranking import run_seed
        rows = [
            {"group_id": "g1", "source_id": "g1",
             "reaction_smiles": "CC(=O)O.CCO>>CC(=O)OCC", "label": 1},
            {"group_id": "g1", "source_id": "g1",
             "reaction_smiles": "CC(=O)O.CCO>>CCC", "label": 0},
            {"group_id": "g2", "source_id": "g2",
             "reaction_smiles": "C(=O)O.NCC>>C(=O)ONCC", "label": 1},
            {"group_id": "g2", "source_id": "g2",
             "reaction_smiles": "C(=O)O.NCC>>NCC", "label": 0},
        ]
        result = run_seed(rows, seed=20260710, train_fraction=0.5, epochs=20)
        self.assertIn("baseline_metrics", result)
        self.assertIn("pc_cng_metrics", result)
        self.assertIn("baseline_per_group", result)
        self.assertIn("pc_cng_per_group", result)
        # MRR should be in [0, 1]
        self.assertGreaterEqual(result["baseline_metrics"]["mrr"], 0.0)
        self.assertLessEqual(result["baseline_metrics"]["mrr"], 1.0)
        self.assertGreaterEqual(result["pc_cng_metrics"]["mrr"], 0.0)
        self.assertLessEqual(result["pc_cng_metrics"]["mrr"], 1.0)

    def test_cli_smoke(self) -> None:
        """End-to-end CLI smoke test with tiny PC-CNG CSV and 2 seeds."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = root / "neg.csv"
            out_dir = root / "out"
            _write_pc_cng_csv(csv_path)
            result = subprocess.run(
                [
                    sys.executable,
                    "-m", "pc_cng.run_retrosynthesis_route_ranking",
                    "--pc-cng-negatives", str(csv_path),
                    "--output-dir", str(out_dir),
                    "--seeds", "20260710,20260711",
                    "--max-sources", "10",
                    "--max-candidates-per-source", "5",
                    "--bootstrap-iterations", "100",
                    "--epochs", "30",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            # All output files must exist
            for fname in [
                "topk_route_recall.json",
                "mrr.json",
                "ndcg.json",
                "false_positive_route_rate.json",
                "paired_significance.json",
                "manifest.json",
                "per_seed_detail.json",
            ]:
                self.assertTrue(
                    (out_dir / fname).exists(),
                    f"Missing {fname}; stderr=\n{result.stderr}",
                )
            # Manifest content
            manifest = json.loads((out_dir / "manifest.json").read_text())
            self.assertEqual(manifest["task"], "P1-04 Retrosynthesis Route Ranking")
            self.assertEqual(manifest["fallback_path"], "pseudo-route")
            self.assertEqual(manifest["n_seeds"], 2)
            self.assertIn("go_no_go", manifest)
            self.assertIn("paired_significance", manifest)
            # paired_significance.json structure
            sig = json.loads((out_dir / "paired_significance.json").read_text())
            self.assertEqual(sig["n_seeds"], 2)
            for key in [
                "baseline_mean", "pc_cng_mean", "delta_mean", "delta_pp",
                "group_level_ci95_low", "group_level_ci95_high",
                "seed_level_ci95_low", "seed_level_ci95_high",
                "paired_permutation_p", "sign_test_p",
            ]:
                self.assertIn(key, sig)

    def test_cli_routes_data_missing_falls_back(self) -> None:
        """If --routes-data points to a non-existent file, must fall back to
        pseudo-routes from --pc-cng-negatives."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = root / "neg.csv"
            out_dir = root / "out"
            _write_pc_cng_csv(csv_path)
            result = subprocess.run(
                [
                    sys.executable,
                    "-m", "pc_cng.run_retrosynthesis_route_ranking",
                    "--routes-data", str(root / "nonexistent.csv"),
                    "--pc-cng-negatives", str(csv_path),
                    "--output-dir", str(out_dir),
                    "--seeds", "20260710",
                    "--max-sources", "10",
                    "--max-candidates-per-source", "5",
                    "--bootstrap-iterations", "50",
                    "--epochs", "20",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            manifest = json.loads((out_dir / "manifest.json").read_text())
            self.assertEqual(manifest["fallback_path"], "pseudo-route")
            self.assertIn("AiZynthFinder", manifest["fallback_reason"])


if __name__ == "__main__":
    unittest.main()
