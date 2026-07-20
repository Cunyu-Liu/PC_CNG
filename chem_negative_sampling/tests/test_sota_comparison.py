"""Unit tests for P2-06 SOTA multi-baseline comparison (L6).

Tests cover:
- Module imports
- CLI args parsing
- Method selection (rdkit_template / heuristic_validator / tanimoto_nn / pc_cng)
- Metric computation (Top-k, MRR, NDCG) on synthetic data
- Paired significance test structure (PC-CNG vs each baseline)
- Synthetic small test (mocked baselines)
- Go/No-Go decision: PC-CNG Top-1 ≥ 3/3 baselines + 1.0 pp
- sota_installation_status.json documents deferred methods
"""

from __future__ import annotations

import csv
import json
import math
import os
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
        # group g1: positive + 2 negatives, target = CC(=O)OCC (ethyl acetate)
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
        # group g2: positive + 2 negatives, target = CC(=O)NCC (acetamide)
        writer.writerow([
            "g2", "CC(=O)O.NCC>>CC(=O)NCC",
            "CC(=O)O.NCC>>CC(=O)NCC.O",
            "retro_precursor", "retro_wrong_product", "edit1",
            "CC(=O)O.NCC", "CC(=O)NCC",
            "CC(=O)O.NCC", "CC(=O)NCC.O",
            1.0, 0.9, 0.5, 0.5, 0.5, 0.5, True, 0, "prov", "keep", "", 1.0,
        ])
        writer.writerow([
            "g2", "CC(=O)O.NCC>>CC(=O)NCC",
            "CC(=O)O.NCC>>NCC",
            "retro_precursor", "retro_wrong_product", "edit2",
            "CC(=O)O.NCC", "CC(=O)NCC",
            "CC(=O)O.NCC", "NCC",
            1.0, 0.4, 0.5, 0.5, 0.5, 0.5, True, 0, "prov", "keep", "", 1.0,
        ])
        # group g3: positive + 1 negative, target = CCOCC (diethyl ether)
        writer.writerow([
            "g3", "CCO.CCO>>CCOCC",
            "CCO.CCO>>CCO",
            "retro_precursor", "retro_wrong_product", "edit1",
            "CCO.CCO", "CCOCC",
            "CCO.CCO", "CCO",
            1.0, 0.5, 0.5, 0.5, 0.5, 0.5, True, 0, "prov", "keep", "", 1.0,
        ])


class SotaComparisonImportTest(unittest.TestCase):
    """Test that the module can be imported and exposes expected symbols."""

    def test_module_imports(self) -> None:
        from pc_cng import run_sota_comparison as mod
        self.assertTrue(hasattr(mod, "main"))
        self.assertTrue(hasattr(mod, "run_seed"))
        self.assertTrue(hasattr(mod, "evaluate"))
        self.assertTrue(hasattr(mod, "paired_significance"))
        self.assertTrue(hasattr(mod, "aggregate_metrics"))

    def test_default_seeds(self) -> None:
        from pc_cng.run_sota_comparison import DEFAULT_SEEDS
        self.assertEqual(len(DEFAULT_SEEDS), 10)
        self.assertEqual(DEFAULT_SEEDS[0], 20260710)
        self.assertEqual(DEFAULT_SEEDS[-1], 20260719)

    def test_method_names(self) -> None:
        from pc_cng.run_sota_comparison import METHOD_NAMES
        self.assertIn("rdkit_template", METHOD_NAMES)
        self.assertIn("heuristic_validator", METHOD_NAMES)
        self.assertIn("tanimoto_nn", METHOD_NAMES)
        self.assertIn("pc_cng", METHOD_NAMES)

    def test_deferred_sota_methods(self) -> None:
        from pc_cng.run_sota_comparison import DEFERRED_SOTA_METHODS
        self.assertIn("localretro", DEFERRED_SOTA_METHODS)
        self.assertIn("graph2smiles", DEFERRED_SOTA_METHODS)
        self.assertIn("molecular_transformer", DEFERRED_SOTA_METHODS)


class SotaComparisonCLITest(unittest.TestCase):
    """Test CLI argument parsing."""

    def test_parse_args_defaults(self) -> None:
        from pc_cng.run_sota_comparison import _parse_args, DEFAULT_METHODS
        args = _parse_args([
            "--pc-cng-negatives", "/tmp/neg.csv",
            "--output-dir", "/tmp/out",
        ])
        self.assertEqual(args.pc_cng_negatives, "/tmp/neg.csv")
        self.assertEqual(args.output_dir, "/tmp/out")
        self.assertEqual(args.methods, DEFAULT_METHODS)
        self.assertEqual(args.top_k, 10)
        self.assertEqual(args.max_candidates_per_source, 10)
        self.assertIsNone(args.limit)

    def test_parse_args_methods(self) -> None:
        from pc_cng.run_sota_comparison import _parse_args
        args = _parse_args([
            "--pc-cng-negatives", "/tmp/neg.csv",
            "--output-dir", "/tmp/out",
            "--methods", "rdkit_template,pc_cng",
        ])
        self.assertEqual(args.methods, "rdkit_template,pc_cng")

    def test_parse_args_limit_smoke(self) -> None:
        from pc_cng.run_sota_comparison import _parse_args
        args = _parse_args([
            "--pc-cng-negatives", "/tmp/neg.csv",
            "--output-dir", "/tmp/out",
            "--limit", "5",
            "--seeds", "20260710,20260711",
        ])
        self.assertEqual(args.limit, 5)
        self.assertEqual(args.seeds, "20260710,20260711")


class SotaComparisonMetricTest(unittest.TestCase):
    """Test metric computation (Top-k, MRR, NDCG) on synthetic data."""

    def test_topk_route_recall(self) -> None:
        from pc_cng.run_sota_comparison import topk_route_recall
        # Group with 1 positive (label=1) and 1 negative (label=0)
        rows = [
            {"group_id": "g1", "label": 1, "score": 0.9},
            {"group_id": "g1", "label": 0, "score": 0.1},
        ]
        self.assertAlmostEqual(topk_route_recall(rows, 1), 1.0)
        self.assertAlmostEqual(topk_route_recall(rows, 3), 1.0)

        # Reverse the order: negative on top
        rows_reversed = [
            {"group_id": "g1", "label": 1, "score": 0.1},
            {"group_id": "g1", "label": 0, "score": 0.9},
        ]
        self.assertAlmostEqual(topk_route_recall(rows_reversed, 1), 0.0)
        self.assertAlmostEqual(topk_route_recall(rows_reversed, 3), 1.0)

    def test_mrr(self) -> None:
        from pc_cng.run_sota_comparison import mrr
        # Positive at rank 1 -> MRR = 1.0
        rows = [
            {"group_id": "g1", "label": 1, "score": 0.9},
            {"group_id": "g1", "label": 0, "score": 0.1},
        ]
        self.assertAlmostEqual(mrr(rows), 1.0)

        # Positive at rank 2 -> MRR = 0.5
        rows2 = [
            {"group_id": "g1", "label": 1, "score": 0.1},
            {"group_id": "g1", "label": 0, "score": 0.9},
        ]
        self.assertAlmostEqual(mrr(rows2), 0.5)

    def test_ndcg_at_k(self) -> None:
        from pc_cng.run_sota_comparison import ndcg_at_k
        # Ideal ranking (positive on top) -> NDCG = 1.0
        rows = [
            {"group_id": "g1", "label": 1, "score": 0.9},
            {"group_id": "g1", "label": 0, "score": 0.1},
        ]
        self.assertAlmostEqual(ndcg_at_k(rows, 10), 1.0)

    def test_evaluate_returns_all_metrics(self) -> None:
        from pc_cng.run_sota_comparison import evaluate
        rows = [
            {"group_id": "g1", "label": 1, "score": 0.9},
            {"group_id": "g1", "label": 0, "score": 0.1},
        ]
        metrics = evaluate(rows)
        for key in ("top1", "top3", "top5", "top10", "mrr", "ndcg_at_10"):
            self.assertIn(key, metrics)
        self.assertAlmostEqual(metrics["top1"], 1.0)
        self.assertAlmostEqual(metrics["mrr"], 1.0)


class SotaComparisonPairedSignificanceTest(unittest.TestCase):
    """Test paired significance test structure (PC-CNG vs each baseline)."""

    def test_paired_significance_structure(self) -> None:
        from pc_cng.run_sota_comparison import paired_significance
        # Build minimal seed_results
        seed_results = []
        for seed in (20260710, 20260711):
            per_group = {
                "g1": {"top1": 1.0, "mrr": 1.0, "ndcg": 1.0},
                "g2": {"top1": 1.0, "mrr": 0.5, "ndcg": 0.7},
            }
            seed_results.append({
                "seed": seed,
                "rdkit_template_metrics": {"top1": 0.5, "mrr": 0.5, "ndcg": 0.5},
                "rdkit_template_per_group": {
                    "g1": {"top1": 0.0, "mrr": 0.5, "ndcg": 0.5},
                    "g2": {"top1": 1.0, "mrr": 0.5, "ndcg": 0.5},
                },
                "heuristic_validator_metrics": {"top1": 0.5, "mrr": 0.5, "ndcg": 0.5},
                "heuristic_validator_per_group": {
                    "g1": {"top1": 0.0, "mrr": 0.5, "ndcg": 0.5},
                    "g2": {"top1": 1.0, "mrr": 0.5, "ndcg": 0.5},
                },
                "tanimoto_nn_metrics": {"top1": 0.5, "mrr": 0.5, "ndcg": 0.5},
                "tanimoto_nn_per_group": {
                    "g1": {"top1": 0.0, "mrr": 0.5, "ndcg": 0.5},
                    "g2": {"top1": 1.0, "mrr": 0.5, "ndcg": 0.5},
                },
                "pc_cng_metrics": {"top1": 1.0, "mrr": 1.0, "ndcg": 1.0},
                "pc_cng_per_group": per_group,
            })
        methods = ("rdkit_template", "heuristic_validator", "tanimoto_nn", "pc_cng")
        sig = paired_significance(
            seed_results, methods, bootstrap_iterations=100, seed=20260710,
        )
        # Should produce one entry per baseline
        self.assertIn("pc_cng_vs_rdkit_template", sig)
        self.assertIn("pc_cng_vs_heuristic_validator", sig)
        self.assertIn("pc_cng_vs_tanimoto_nn", sig)
        # Each entry should have standard keys
        for key, pair in sig.items():
            self.assertIn("delta_pp", pair)
            self.assertIn("seed_level_ci95_low_pp", pair)
            self.assertIn("seed_level_ci95_high_pp", pair)
            self.assertIn("paired_permutation_p", pair)
            self.assertIn("sign_test_p", pair)
            self.assertIn("method_a", pair)
            self.assertIn("method_b", pair)


class SotaComparisonAggregationTest(unittest.TestCase):
    """Test aggregation of metrics across seeds (mean ± std)."""

    def test_aggregate_metrics(self) -> None:
        from pc_cng.run_sota_comparison import aggregate_metrics
        seed_results = [
            {
                "seed": 1,
                "pc_cng_metrics": {"top1": 0.5, "mrr": 0.6, "ndcg": 0.7},
            },
            {
                "seed": 2,
                "pc_cng_metrics": {"top1": 0.7, "mrr": 0.8, "ndcg": 0.9},
            },
        ]
        agg = aggregate_metrics(seed_results, ["pc_cng"])
        self.assertIn("pc_cng", agg)
        self.assertEqual(agg["pc_cng"]["n_seeds"], 2)
        # mean of 0.5 and 0.7 = 0.6
        self.assertAlmostEqual(agg["pc_cng"]["top1"]["mean"], 0.6)
        # std of 0.5, 0.7 (sample std) = sqrt(0.02) ≈ 0.1414
        self.assertAlmostEqual(
            agg["pc_cng"]["top1"]["std"], math.sqrt(0.02), places=4,
        )


class SotaComparisonRunSeedTest(unittest.TestCase):
    """Synthetic small test exercising run_seed with all 4 methods."""

    def test_run_seed_all_methods(self) -> None:
        from pc_cng.run_sota_comparison import run_seed, FeatureCache
        # Build minimal rows
        rows = [
            {"group_id": "g1", "source_id": "g1",
             "reaction_smiles": "CC(=O)O.CCO>>CC(=O)OCC",
             "label": 1, "parent_product": "CC(=O)OCC",
             "hard_score": 1.0},
            {"group_id": "g1", "source_id": "g1",
             "reaction_smiles": "CC(=O)O.CCO>>CCC",
             "label": 0, "parent_product": "CC(=O)OCC",
             "hard_score": 0.5},
            {"group_id": "g2", "source_id": "g2",
             "reaction_smiles": "CC(=O)O.NCC>>CC(=O)NCC",
             "label": 1, "parent_product": "CC(=O)NCC",
             "hard_score": 1.0},
            {"group_id": "g2", "source_id": "g2",
             "reaction_smiles": "CC(=O)O.NCC>>NCC",
             "label": 0, "parent_product": "CC(=O)NCC",
             "hard_score": 0.5},
        ]
        cache = FeatureCache()
        cache.precompute([str(r["reaction_smiles"]) for r in rows])
        result = run_seed(
            rows, seed=20260710,
            methods=("rdkit_template", "heuristic_validator",
                     "tanimoto_nn", "pc_cng"),
            train_fraction=0.5,  # 1 of 2 groups in train
            epochs=10,
            shared_cache=cache,
            tanimoto_k=1,
        )
        # All four methods should produce metrics
        self.assertIn("rdkit_template_metrics", result)
        self.assertIn("heuristic_validator_metrics", result)
        self.assertIn("tanimoto_nn_metrics", result)
        self.assertIn("pc_cng_metrics", result)
        # Each metrics dict should have top1 / mrr / ndcg
        for method in ("rdkit_template", "heuristic_validator",
                       "tanimoto_nn", "pc_cng"):
            metrics = result[f"{method}_metrics"]
            self.assertIn("top1", metrics)
            self.assertIn("mrr", metrics)
            self.assertIn("ndcg_at_10", metrics)
            # Scores should be in [0, 1]
            self.assertGreaterEqual(metrics["top1"], 0.0)
            self.assertLessEqual(metrics["top1"], 1.0)


class SotaComparisonGoNoGoTest(unittest.TestCase):
    """Test Go/No-Go decision: PC-CNG must beat 3/3 baselines by ≥ 1.0 pp."""

    def test_go_decision_when_pc_cng_beats_all(self) -> None:
        from pc_cng.run_sota_comparison import write_go_no_go_decision
        sig = {
            "pc_cng_vs_rdkit_template": {
                "delta_pp": 5.0,
                "seed_level_ci95_low_pp": 2.0,
                "seed_level_ci95_high_pp": 8.0,
                "paired_permutation_p": 0.01,
                "sign_test_p": 0.02,
            },
            "pc_cng_vs_heuristic_validator": {
                "delta_pp": 4.0,
                "seed_level_ci95_low_pp": 1.5,
                "seed_level_ci95_high_pp": 7.0,
                "paired_permutation_p": 0.02,
                "sign_test_p": 0.03,
            },
            "pc_cng_vs_tanimoto_nn": {
                "delta_pp": 3.0,
                "seed_level_ci95_low_pp": 1.0,
                "seed_level_ci95_high_pp": 5.0,
                "paired_permutation_p": 0.03,
                "sign_test_p": 0.04,
            },
        }
        methods = ("rdkit_template", "heuristic_validator", "tanimoto_nn", "pc_cng")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "go_no_go_decision.json")
            write_go_no_go_decision(path, sig, methods)
            with open(path) as fh:
                payload = json.load(fh)
            self.assertEqual(payload["n_baselines_evaluated"], 3)
            self.assertEqual(payload["n_baselines_pc_cng_beats"], 3)
            self.assertIn("GO", payload["overall_decision"])

    def test_no_go_decision_when_pc_cng_loses_one(self) -> None:
        from pc_cng.run_sota_comparison import write_go_no_go_decision
        sig = {
            "pc_cng_vs_rdkit_template": {
                "delta_pp": 5.0,
                "seed_level_ci95_low_pp": 2.0,
                "seed_level_ci95_high_pp": 8.0,
                "paired_permutation_p": 0.01,
                "sign_test_p": 0.02,
            },
            "pc_cng_vs_heuristic_validator": {
                "delta_pp": 0.5,  # Below 1.0 pp threshold
                "seed_level_ci95_low_pp": -1.0,
                "seed_level_ci95_high_pp": 2.0,
                "paired_permutation_p": 0.4,
                "sign_test_p": 0.5,
            },
            "pc_cng_vs_tanimoto_nn": {
                "delta_pp": 3.0,
                "seed_level_ci95_low_pp": 1.0,
                "seed_level_ci95_high_pp": 5.0,
                "paired_permutation_p": 0.03,
                "sign_test_p": 0.04,
            },
        }
        methods = ("rdkit_template", "heuristic_validator", "tanimoto_nn", "pc_cng")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "go_no_go_decision.json")
            write_go_no_go_decision(path, sig, methods)
            with open(path) as fh:
                payload = json.load(fh)
            self.assertEqual(payload["n_baselines_pc_cng_beats"], 2)
            self.assertIn("NO-GO", payload["overall_decision"])


class SotaInstallationStatusTest(unittest.TestCase):
    """Test sota_installation_status.json documents deferred methods."""

    def test_write_sota_installation_status(self) -> None:
        from pc_cng.run_sota_comparison import write_sota_installation_status
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "sota_installation_status.json")
            write_sota_installation_status(path)
            with open(path) as fh:
                payload = json.load(fh)
            self.assertEqual(payload["network_access"], "none")
            deferred_names = [m["name"] for m in payload["deferred_methods"]]
            self.assertIn("localretro", deferred_names)
            self.assertIn("graph2smiles", deferred_names)
            self.assertIn("molecular_transformer", deferred_names)
            # Each deferred method should have a reason mentioning installation failure
            for entry in payload["deferred_methods"]:
                self.assertEqual(entry["status"], "deferred")
                self.assertIn("installation failure", entry["reason"])
                self.assertTrue(entry["attempted_install"])

    def test_evaluated_methods_listed(self) -> None:
        from pc_cng.run_sota_comparison import write_sota_installation_status
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "sota_installation_status.json")
            write_sota_installation_status(path)
            with open(path) as fh:
                payload = json.load(fh)
            evaluated_keys = [m["key"] for m in payload["evaluated_methods"]]
            self.assertIn("rdkit_template", evaluated_keys)
            self.assertIn("heuristic_validator", evaluated_keys)
            self.assertIn("tanimoto_nn", evaluated_keys)
            self.assertIn("pc_cng", evaluated_keys)


class SotaComparisonMainSmokeTest(unittest.TestCase):
    """End-to-end smoke test of main() with a tiny PC-CNG CSV."""

    def test_main_runs_with_limit(self) -> None:
        from pc_cng import run_sota_comparison as mod
        with tempfile.TemporaryDirectory() as tmpdir:
            neg_path = os.path.join(tmpdir, "neg.csv")
            _write_pc_cng_csv(Path(neg_path))
            out_dir = os.path.join(tmpdir, "out")
            mod.main([
                "--pc-cng-negatives", neg_path,
                "--output-dir", out_dir,
                "--seeds", "20260710,20260711",
                "--limit", "5",
                "--methods", "rdkit_template,heuristic_validator,tanimoto_nn,pc_cng",
                "--bootstrap-iterations", "100",
                "--epochs", "10",
            ])
            # Verify all expected output files exist
            for fname in (
                "summary.json",
                "paired_significance.json",
                "per_target_metrics.csv",
                "sota_installation_status.json",
                "go_no_go_decision.json",
                "per_seed_detail.json",
            ):
                self.assertTrue(
                    os.path.exists(os.path.join(out_dir, fname)),
                    f"Missing output file: {fname}",
                )
            # Verify summary.json contents
            with open(os.path.join(out_dir, "summary.json")) as fh:
                summary = json.load(fh)
            self.assertIn("metrics", summary)
            self.assertIn("paired_significance", summary)
            self.assertIn("deferred_sota_methods", summary)
            self.assertIn("localretro", summary["deferred_sota_methods"])
            # Verify go_no_go_decision.json contents
            with open(os.path.join(out_dir, "go_no_go_decision.json")) as fh:
                go_payload = json.load(fh)
            self.assertIn("overall_decision", go_payload)
            self.assertIn(go_payload["overall_decision"],
                          ("GO (write to main table)", "NO-GO (downgrade to supplementary)"))
            # Verify per_target_metrics.csv has rows
            import csv as csv_mod
            with open(os.path.join(out_dir, "per_target_metrics.csv")) as fh:
                reader = csv_mod.reader(fh)
                header = next(reader)
                self.assertIn("seed", header)
                self.assertIn("group_id", header)
                self.assertIn("method", header)
                rows = list(reader)
                self.assertGreater(len(rows), 0)


if __name__ == "__main__":
    unittest.main()
