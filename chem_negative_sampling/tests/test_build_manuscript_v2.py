"""Unit tests for chem_negative_sampling.pc_cng.build_manuscript_v2 and
generate_manuscript_figures_v2 (P2-09 manuscript v2 builder).

Tests cover:
    - Module imports for both scripts.
    - P2 results loading with fallback for missing artifacts.
    - Manuscript v2 structure (all required sections present).
    - Journal positioning logic (top / strong / fallback tiers).
    - Go/No-Go aggregation across all eight P2 tasks.
    - Cover letter generation.
    - Journal decision document generation.
    - Pending-results JSON generation.
    - Figure generation (mocked matplotlib to keep tests CPU-only and headless).
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Dict
from unittest import mock

# Ensure chem_negative_sampling is on sys.path when tests run from the repo
# root without installation.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(THIS_DIR)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

from pc_cng.build_manuscript_v2 import (  # noqa: E402
    FALLBACK_P2,
    LIMITATIONS_V2,
    P2Results,
    aggregate_go_no_go,
    build_cover_letter,
    build_journal_decision,
    build_manuscript_v2,
    build_pending_results,
    build_supplementary_v2,
    decide_journal_tier,
    load_json,
    main as build_main,
)
from pc_cng.generate_manuscript_figures_v2 import (  # noqa: E402
    DEFAULT_DATA,
    FIGURE_NAMES,
    generate_all_figures,
    generate_figure_1_architecture,
    generate_figure_2_cross_dataset,
    generate_figure_3_route_ranking,
    generate_figure_4_external_bridge,
    generate_figure_5_dft_validation,
    generate_figure_6_sota_radar,
    main as figures_main,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_p2_results_dir(tmp: str) -> Path:
    """Create a minimal results directory mirroring the real layout."""
    root = Path(tmp)
    # P2-01
    p01 = root / "aizynthfinder_route_ranking_20260720"
    p01.mkdir(parents=True)
    (p01 / "route_ranking_summary.json").write_text(json.dumps({
        "metrics": {
            "aizynthfinder_baseline": {"mrr": 0.24},
            "aizynthfinder_pc_cng": {"mrr": 0.54},
        },
        "fallback_reason": "AiZynthFinder unavailable",
    }))
    (p01 / "paired_significance.json").write_text(json.dumps({
        "r3_vs_r1": {
            "n_seeds": 10, "n_common_groups": 150,
            "delta_pp": 30.0, "seed_level_ci95_low_pp": 28.0,
            "seed_level_ci95_high_pp": 32.0,
            "paired_permutation_p": 9.999e-05,
            "ranker_b_better_groups": 144,
        }
    }))
    # P2-02
    p02 = root / "dft_validation_chemoselectivity_20260720"
    p02.mkdir(parents=True)
    (p02 / "dft_validation_summary.json").write_text(json.dumps({
        "support_rate": 0.9, "n_supported": 27, "n_not_supported": 3,
        "n_computed": 30, "go_no_go_verdict": "GO", "xtb_method": "GFN2-xTB",
    }))
    # P2-04
    p04 = root / "external_score_mlp_calibrator_v2_chemformer_aware_20260720"
    p04.mkdir(parents=True)
    (p04 / "summary.json").write_text(json.dumps({
        "baseline_top1_mean": 0.525, "delta_top1_pp": 2.54, "decision": "GO",
        "n_seeds": 10,
        "metrics": {"top1": {"v2_mean": 0.5504,
                              "paired_test": {"ci_low": 0.0133, "ci_high": 0.0375,
                                              "p_value": 0.001}}},
    }))
    (p04 / "paired_significance.json").write_text(json.dumps({
        "baseline_mean_top1": 0.525, "v2_mean_top1": 0.5504,
        "mean_delta": 0.0254, "ci_low": 0.0133, "ci_high": 0.0375,
        "p_value": 0.001, "n_seeds": 10,
        "v2_score_name": "pc_cng_mlp_calibrator_v2",
    }))
    # P2-05
    p05 = root / "cross_dataset_transfer_v2_20260720"
    p05.mkdir(parents=True)
    (p05 / "aggregate_summary.json").write_text(json.dumps({
        "n_pairs_total": 2, "n_pairs_ci_all_positive": 0, "pairs": [],
    }))
    (p05 / "go_no_go_decision.json").write_text(json.dumps({
        "decision": "NO-GO", "count_ci_all_positive": 0,
        "n_pairs_total": 2, "threshold_for_go": 3,
    }))
    for pair_name, delta in [("regiosqm20_to_hitea", 0.0), ("regiosqm20_to_uspto", 0.011)]:
        pd = p05 / pair_name
        pd.mkdir()
        (pd / "paired_significance.json").write_text(json.dumps({
            "source": pair_name.split("_to_")[0],
            "target": pair_name.split("_to_")[1],
            "paired_significance_pooled": {
                "delta_mean": delta, "delta_ci95_low": -0.01,
                "delta_ci95_high": 0.02, "paired_permutation_p": 0.2, "n": 1000,
            },
            "seed_level_significance": {
                "n_seeds": 10, "ci95_low": 0.0, "ci95_high": 0.02,
            },
        }))
    # P2-06 smoke
    p06 = root / "sota_comparison_uspto_mit_50k_20260720_smoke"
    p06.mkdir(parents=True)
    (p06 / "go_no_go_decision.json").write_text(json.dumps({
        "overall_decision": "NO-GO (downgrade to supplementary)",
        "n_baselines_evaluated": 3, "n_baselines_pc_cng_beats": 2,
        "deferred_sota_methods": ["localretro", "graph2smiles", "molecular_transformer"],
        "deferred_reason": "no network access",
        "per_baseline": {
            "pc_cng_vs_rdkit_template": {"baseline": "rdkit_template", "delta_pp": 27.8,
                                          "ci_low_pp": 27.5, "ci_high_pp": 28.1,
                                          "pc_cng_better": True},
            "pc_cng_vs_heuristic_validator": {"baseline": "heuristic_validator", "delta_pp": 27.8,
                                               "ci_low_pp": 27.5, "ci_high_pp": 28.1,
                                               "pc_cng_better": True},
            "pc_cng_vs_tanimoto_nn": {"baseline": "tanimoto_nn", "delta_pp": -48.6,
                                       "ci_low_pp": -48.9, "ci_high_pp": -48.3,
                                       "pc_cng_better": False},
        },
    }))
    (p06 / "summary.json").write_text(json.dumps({"task": "P2-06 smoke"}))
    # P2-07 smoke
    p07 = root / "transformer_negative_generator_20260720_smoke"
    p07.mkdir(parents=True)
    (p07 / "go_no_go_decision.json").write_text(json.dumps({
        "decision": "NO-GO", "g1_top1_mean": 0.9, "g3_top1_mean": 0.485,
        "delta_pp": -41.5,
    }))
    (p07 / "summary.json").write_text(json.dumps({
        "degradation_path": "small_pytorch_transformer_from_scratch",
    }))
    # P2-08 smoke
    p08 = root / "condition_prediction_20260720_smoke"
    p08.mkdir(parents=True)
    (p08 / "go_no_go_decision.json").write_text(json.dumps({
        "decision": "NO-GO (downgrade to supplementary)",
        "delta_mean_pp": -5.56, "p_value": 0.5,
    }))
    (p08 / "summary.json").write_text(json.dumps({
        "fallback_reason": "agents column empty",
    }))
    return root


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class ImportTest(unittest.TestCase):
    """Verify both modules import cleanly."""

    def test_build_manuscript_v2_imports(self) -> None:
        from pc_cng import build_manuscript_v2
        self.assertTrue(hasattr(build_manuscript_v2, "main"))
        self.assertTrue(hasattr(build_manuscript_v2, "P2Results"))
        self.assertTrue(hasattr(build_manuscript_v2, "build_manuscript_v2"))

    def test_generate_manuscript_figures_v2_imports(self) -> None:
        from pc_cng import generate_manuscript_figures_v2
        self.assertTrue(hasattr(generate_manuscript_figures_v2, "main"))
        self.assertTrue(hasattr(generate_manuscript_figures_v2, "generate_all_figures"))
        self.assertEqual(len(generate_manuscript_figures_v2.FIGURE_NAMES), 6)


class LoadJsonTest(unittest.TestCase):
    def test_load_json_missing_returns_empty(self) -> None:
        self.assertEqual(load_json(Path("/nonexistent/file.json")), {})

    def test_load_json_invalid_returns_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "bad.json"
            p.write_text("{not valid", encoding="utf-8")
            self.assertEqual(load_json(p), {})

    def test_load_json_valid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "ok.json"
            p.write_text('{"a": 1}', encoding="utf-8")
            self.assertEqual(load_json(p), {"a": 1})


class FallbackNumbersTest(unittest.TestCase):
    def test_fallback_covers_all_p2_tasks(self) -> None:
        required = ["p2_01", "p2_02", "p2_03", "p2_04", "p2_05",
                    "p2_06", "p2_07", "p2_08"]
        for key in required:
            self.assertIn(key, FALLBACK_P2, f"missing fallback: {key}")

    def test_fallback_p2_01_decision_is_go(self) -> None:
        self.assertEqual(FALLBACK_P2["p2_01"]["decision"], "GO")

    def test_fallback_p2_02_support_above_threshold(self) -> None:
        self.assertGreaterEqual(FALLBACK_P2["p2_02"]["support_rate"], 0.6)

    def test_fallback_p2_04_decision_is_go(self) -> None:
        self.assertEqual(FALLBACK_P2["p2_04"]["decision"], "GO")

    def test_limitations_v2_has_eight_entries(self) -> None:
        self.assertEqual(len(LIMITATIONS_V2), 8)
        ids = [l["id"] for l in LIMITATIONS_V2]
        self.assertEqual(ids, ["L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8"])


class P2ResultsLoadingTest(unittest.TestCase):
    def test_load_all_with_empty_results_dir_uses_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data = P2Results(Path(tmp)).load_all()
        self.assertIn("p2_01", data)
        self.assertIn("p2_02", data)
        self.assertIn("p2_04", data)
        # P2-01 should fall back to the GO decision
        self.assertEqual(data["p2_01"]["decision"], "GO")
        # P2-03 is always deferred
        self.assertEqual(data["p2_03"]["decision"], "DEFERRED")
        # Pending list should include the missing tasks
        pending_tasks = [p["task"] for p in data["pending"]]
        self.assertIn("P2-03", pending_tasks)

    def test_load_all_with_real_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _make_p2_results_dir(tmp)
            data = P2Results(root).load_all()
        # P2-01 from artifact
        self.assertAlmostEqual(data["p2_01"]["baseline_mrr"], 0.24)
        self.assertAlmostEqual(data["p2_01"]["pc_cng_mrr"], 0.54)
        self.assertAlmostEqual(data["p2_01"]["delta_pp"], 30.0)
        # P2-02 from artifact
        self.assertEqual(data["p2_02"]["n_supported"], 27)
        self.assertAlmostEqual(data["p2_02"]["support_rate"], 0.9)
        # P2-04 from artifact
        self.assertAlmostEqual(data["p2_04"]["baseline_top1"], 0.525)
        self.assertAlmostEqual(data["p2_04"]["delta_pp"], 2.54)
        # P2-05 has 2 pairs
        self.assertEqual(data["p2_05"]["n_pairs_total"], 2)
        # P2-06 smoke
        self.assertEqual(data["p2_06"]["n_baselines_pc_cng_beats"], 2)
        self.assertTrue(data["p2_06"]["is_smoke"])
        # P2-07 / P2-08 smoke
        self.assertTrue(data["p2_07"]["is_smoke"])
        self.assertTrue(data["p2_08"]["is_smoke"])
        # Pending list should include P2-06, P2-07, P2-08 (smoke) and P2-03 (deferred)
        pending_tasks = {p["task"] for p in data["pending"]}
        self.assertIn("P2-03", pending_tasks)
        self.assertIn("P2-06", pending_tasks)
        self.assertIn("P2-07", pending_tasks)
        self.assertIn("P2-08", pending_tasks)


class GoNoGoAggregationTest(unittest.TestCase):
    def test_aggregate_counts_go_and_no_go(self) -> None:
        p2 = {
            "p2_01": {"decision": "GO"},
            "p2_02": {"decision": "GO"},
            "p2_03": {"decision": "DEFERRED"},
            "p2_04": {"decision": "GO"},
            "p2_05": {"decision": "NO-GO"},
            "p2_06": {"decision": "NO-GO (downgrade to supplementary)", "is_smoke": True},
            "p2_07": {"decision": "NO-GO", "is_smoke": True},
            "p2_08": {"decision": "NO-GO (downgrade to supplementary)", "is_smoke": True},
        }
        agg = aggregate_go_no_go(p2)
        self.assertEqual(agg["n_go"], 3)
        self.assertEqual(agg["n_no_go"], 4)
        self.assertEqual(agg["n_deferred"], 1)
        self.assertEqual(agg["n_smoke"], 3)
        self.assertEqual(agg["n_total"], 8)

    def test_aggregate_with_real_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _make_p2_results_dir(tmp)
            data = P2Results(root).load_all()
        agg = aggregate_go_no_go(data)
        # P2-01, P2-02, P2-04 are GO (3); P2-05, P2-06, P2-07, P2-08 are NO-GO (4);
        # P2-03 is DEFERRED (1)
        self.assertEqual(agg["n_go"], 3)
        self.assertEqual(agg["n_no_go"], 4)
        self.assertEqual(agg["n_deferred"], 1)


class JournalTierTest(unittest.TestCase):
    def test_strong_tier_when_p2_01_p2_04_go_and_p2_06_beats_one_third(self) -> None:
        # P2-01 GO, P2-04 GO, P2-06 beats 2/3 baselines -> strong tier
        p2 = {
            "p2_01": {"decision": "GO"},
            "p2_02": {"decision": "GO"},
            "p2_03": {"decision": "DEFERRED"},
            "p2_04": {"decision": "GO"},
            "p2_05": {"decision": "NO-GO"},
            "p2_06": {"decision": "NO-GO (downgrade to supplementary)",
                       "n_baselines_pc_cng_beats": 2, "n_baselines_evaluated": 3,
                       "is_smoke": True},
            "p2_07": {"decision": "NO-GO", "is_smoke": True},
            "p2_08": {"decision": "NO-GO (downgrade to supplementary)", "is_smoke": True},
        }
        result = decide_journal_tier(p2)
        self.assertEqual(result["tier"], "strong")
        self.assertIn("J. Chem. Inf. Model.", result["target_journals"])

    def test_top_tier_requires_p2_03_go(self) -> None:
        # P2-03 deferred -> cannot be top tier even if everything else is GO
        p2 = {
            "p2_01": {"decision": "GO"},
            "p2_02": {"decision": "GO"},
            "p2_03": {"decision": "DEFERRED"},
            "p2_04": {"decision": "GO"},
            "p2_05": {"decision": "GO"},
            "p2_06": {"decision": "GO", "n_baselines_pc_cng_beats": 3,
                       "n_baselines_evaluated": 3},
            "p2_07": {"decision": "GO"},
            "p2_08": {"decision": "GO"},
        }
        result = decide_journal_tier(p2)
        self.assertNotEqual(result["tier"], "top")

    def test_top_tier_when_all_required_pass(self) -> None:
        p2 = {
            "p2_01": {"decision": "GO"},
            "p2_02": {"decision": "GO"},
            "p2_03": {"decision": "GO"},
            "p2_04": {"decision": "GO"},
            "p2_05": {"decision": "GO"},
            "p2_06": {"decision": "GO", "n_baselines_pc_cng_beats": 3,
                       "n_baselines_evaluated": 3},
            "p2_07": {"decision": "GO"},
            "p2_08": {"decision": "GO"},
        }
        result = decide_journal_tier(p2)
        self.assertEqual(result["tier"], "top")
        self.assertIn("Nature Chemistry", result["target_journals"])

    def test_fallback_tier_when_p2_01_fails(self) -> None:
        p2 = {
            "p2_01": {"decision": "NO-GO"},
            "p2_02": {"decision": "GO"},
            "p2_03": {"decision": "DEFERRED"},
            "p2_04": {"decision": "NO-GO"},
            "p2_05": {"decision": "NO-GO"},
            "p2_06": {"decision": "NO-GO", "n_baselines_pc_cng_beats": 0,
                       "n_baselines_evaluated": 3},
            "p2_07": {"decision": "NO-GO"},
            "p2_08": {"decision": "NO-GO"},
        }
        result = decide_journal_tier(p2)
        self.assertEqual(result["tier"], "fallback")

    def test_real_artifacts_yield_strong_tier(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _make_p2_results_dir(tmp)
            data = P2Results(root).load_all()
        result = decide_journal_tier(data)
        # P2-01 GO, P2-04 GO, P2-06 beats 2/3 -> strong tier
        self.assertEqual(result["tier"], "strong")


class ManuscriptStructureTest(unittest.TestCase):
    def test_manuscript_has_required_sections(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _make_p2_results_dir(tmp)
            data = P2Results(root).load_all()
        manuscript = build_manuscript_v2("", data)
        required_sections = [
            "## Abstract",
            "## 8. P2 Validation Programme Overview",
            "## 9. E3 DFT Validation (P2-02, updated)",
            "## 10. External Bridge Calibration (P2-04, updated)",
            "## 11. SOTA Multi-Baseline Comparison (P2-06)",
            "## 12. Condition Prediction Downstream (P2-08, new)",
            "## 13. Transformer Negative Generator Ablation (P2-07, new)",
            "## 14. Cross-Dataset Transfer v2 (P2-05, updated)",
            "## 15. Retrosynthesis Route Ranking (P2-01, updated)",
            "## 16. Limitations (updated)",
            "## 17. Conclusion",
        ]
        for section in required_sections:
            self.assertIn(section, manuscript, f"missing section: {section}")

    def test_manuscript_contains_p2_numeric_claims(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _make_p2_results_dir(tmp)
            data = P2Results(root).load_all()
        manuscript = build_manuscript_v2("", data)
        # P2-01 delta
        self.assertIn("30.00", manuscript)
        # P2-02 support rate
        self.assertIn("90%", manuscript)
        # P2-04 delta
        self.assertIn("2.54", manuscript)
        # P2-03 deferred
        self.assertIn("DEFERRED", manuscript)

    def test_manuscript_mentions_10_seed_paired_significance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _make_p2_results_dir(tmp)
            data = P2Results(root).load_all()
        manuscript = build_manuscript_v2("", data)
        self.assertIn("10-seed paired", manuscript)

    def test_supplementary_has_provenance_table(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _make_p2_results_dir(tmp)
            data = P2Results(root).load_all()
        supp = build_supplementary_v2(data)
        self.assertIn("Provenance", supp)
        self.assertIn("p2_01", supp)
        self.assertIn("Journal Positioning", supp)


class CoverLetterTest(unittest.TestCase):
    def test_cover_letter_mentions_target_journal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _make_p2_results_dir(tmp)
            data = P2Results(root).load_all()
        cover = build_cover_letter(data)
        # Strong tier -> JCIM should be the first target
        self.assertIn("J. Chem. Inf. Model.", cover)
        self.assertIn("Dear Editor", cover)
        self.assertIn("P2-01", cover)
        self.assertIn("P2-04", cover)

    def test_cover_letter_includes_go_no_go_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _make_p2_results_dir(tmp)
            data = P2Results(root).load_all()
        cover = build_cover_letter(data)
        self.assertIn("GO", cover)
        self.assertIn("NO-GO", cover)


class JournalDecisionTest(unittest.TestCase):
    def test_journal_decision_documents_tier_rules(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _make_p2_results_dir(tmp)
            data = P2Results(root).load_all()
        doc = build_journal_decision(data)
        self.assertIn("Tier", doc)
        self.assertIn("Top tier", doc)
        self.assertIn("Strong tier", doc)
        self.assertIn("Fallback", doc)

    def test_journal_decision_strong_tier_with_real_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _make_p2_results_dir(tmp)
            data = P2Results(root).load_all()
        doc = build_journal_decision(data)
        self.assertIn("strong", doc)


class PendingResultsTest(unittest.TestCase):
    def test_pending_results_lists_incomplete_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _make_p2_results_dir(tmp)
            data = P2Results(root).load_all()
        pending = build_pending_results(data)
        self.assertIn("pending_tasks", pending)
        tasks = {p["task"] for p in pending["pending_tasks"]}
        # P2-03 (deferred), P2-06/07/08 (smoke) should be pending
        self.assertIn("P2-03", tasks)
        self.assertIn("P2-06", tasks)

    def test_pending_results_json_serialisable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _make_p2_results_dir(tmp)
            data = P2Results(root).load_all()
        pending = build_pending_results(data)
        # Should not raise
        json.dumps(pending)


class FiguresTest(unittest.TestCase):
    """Test figure generation with mocked matplotlib to keep tests headless."""

    def test_generate_all_figures_with_mocked_matplotlib(self) -> None:
        # Mock matplotlib so the test does not require a display and stays CPU-only.
        with mock.patch("pc_cng.generate_manuscript_figures_v2._try_import_matplotlib",
                        return_value=None):
            with tempfile.TemporaryDirectory() as tmp:
                paths = generate_all_figures(Path(tmp), results_dir=None)
                self.assertEqual(len(paths), 6)
                for p in paths:
                    self.assertTrue(p.exists(), f"figure file missing: {p}")
                    self.assertEqual(p.suffix, ".txt")  # ASCII fallback

    def test_generate_all_figures_names(self) -> None:
        self.assertEqual(len(FIGURE_NAMES), 6)
        expected = [
            "figure_1_architecture_overview",
            "figure_2_cross_dataset_migration",
            "figure_3_route_ranking",
            "figure_4_external_bridge_calibration",
            "figure_5_dft_validation_support",
            "figure_6_sota_radar",
        ]
        self.assertEqual(FIGURE_NAMES, expected)

    def test_figure_manifest_written(self) -> None:
        with mock.patch("pc_cng.generate_manuscript_figures_v2._try_import_matplotlib",
                        return_value=None):
            with tempfile.TemporaryDirectory() as tmp:
                generate_all_figures(Path(tmp), results_dir=None)
                manifest_path = Path(tmp) / "figures_manifest.json"
                self.assertTrue(manifest_path.exists())
                manifest = json.loads(manifest_path.read_text())
                self.assertEqual(manifest["n_figures"], 6)
                self.assertEqual(manifest["backend"], "ascii")

    def test_individual_figure_generators_return_path(self) -> None:
        with mock.patch("pc_cng.generate_manuscript_figures_v2._try_import_matplotlib",
                        return_value=None):
            with tempfile.TemporaryDirectory() as tmp:
                p1 = generate_figure_1_architecture(None, Path(tmp))
                p2 = generate_figure_2_cross_dataset(None, Path(tmp), DEFAULT_DATA["p2_05"])
                p3 = generate_figure_3_route_ranking(None, Path(tmp), DEFAULT_DATA["p2_01"])
                p4 = generate_figure_4_external_bridge(None, Path(tmp), DEFAULT_DATA["p2_04"])
                p5 = generate_figure_5_dft_validation(None, Path(tmp), DEFAULT_DATA["p2_02"])
                p6 = generate_figure_6_sota_radar(None, Path(tmp), DEFAULT_DATA["p2_06"])
                for p in [p1, p2, p3, p4, p5, p6]:
                    self.assertTrue(p.exists())


class MainCliTest(unittest.TestCase):
    def test_build_main_writes_all_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _make_p2_results_dir(tmp)
            docs = Path(tmp) / "docs"
            docs.mkdir()
            # Write a minimal P1 manuscript
            p1 = docs / "manuscript_v1_20260719.md"
            p1.write_text("# PC-CNG v1\n\n## 1. Introduction\n\nv1 body.\n", encoding="utf-8")
            rc = build_main([
                "--results-dir", str(root),
                "--p1-manuscript", str(p1),
                "--output-dir", str(docs),
            ])
            self.assertEqual(rc, 0)
            self.assertTrue((docs / "manuscript_v2_20260720.md").exists())
            self.assertTrue((docs / "manuscript_supplementary_v2_20260720.md").exists())
            self.assertTrue((docs / "cover_letter_20260720.md").exists())
            self.assertTrue((docs / "target_journal_decision_20260720.md").exists())
            self.assertTrue((docs / "pending_results.json").exists())
            # pending_results.json should be valid JSON
            pending = json.loads((docs / "pending_results.json").read_text())
            self.assertIn("pending_tasks", pending)

    def test_figures_main_writes_figures(self) -> None:
        with mock.patch("pc_cng.generate_manuscript_figures_v2._try_import_matplotlib",
                        return_value=None):
            with tempfile.TemporaryDirectory() as tmp:
                root = _make_p2_results_dir(tmp)
                figs = Path(tmp) / "figs"
                rc = figures_main([
                    "--output-dir", str(figs),
                    "--results-dir", str(root),
                ])
                self.assertEqual(rc, 0)
                self.assertEqual(len(list(figs.glob("*.txt"))), 6)
                self.assertTrue((figs / "figures_manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
