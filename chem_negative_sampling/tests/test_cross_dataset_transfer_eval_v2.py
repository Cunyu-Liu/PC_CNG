"""Unit tests for chem_negative_sampling.pc_cng.run_cross_dataset_transfer_eval_v2.

Tests cover the v2 pair configuration (10 default pairs), CLI parsing
(``--pairs`` and ``--pair-only``), aggregate summary construction, GO/NO-GO
decision logic, paired-significance key presence, and a synthetic end-to-end
run that monkeypatches v1's ``run_transfer_pair`` to avoid real training.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from typing import Dict, List
from unittest import mock

# Ensure the chem_negative_sampling directory is on sys.path when tests are
# run from the repo root without installation.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(THIS_DIR)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

from pc_cng import run_cross_dataset_transfer_eval_v2 as v2  # noqa: E402
from pc_cng.run_cross_dataset_transfer_eval_v2 import (  # noqa: E402
    DEFAULT_PAIRS,
    DEFAULT_SEEDS,
    GO_THRESHOLD,
    V2_DATASET_REGISTRY,
    build_aggregate_summary,
    build_go_no_go,
    is_ci_all_positive,
    parse_pairs,
    write_per_pair_summary,
)


class ImportTest(unittest.TestCase):
    """Verify that the v2 module imports cleanly and re-exports v1 helpers."""

    def test_module_imports(self) -> None:
        self.assertTrue(hasattr(v2, "main"))
        self.assertTrue(hasattr(v2, "run_transfer_pair"))
        self.assertTrue(hasattr(v2, "parse_seeds"))

    def test_v1_registry_patched_with_ord_and_nicolit(self) -> None:
        """v1's DATASET_REGISTRY should now include ord and nicolit after v2 import."""
        from pc_cng.run_cross_dataset_transfer_eval import DATASET_REGISTRY as v1_reg
        self.assertIn("ord", v1_reg)
        self.assertIn("nicolit", v1_reg)
        self.assertTrue(v1_reg["ord"].endswith("ord_normalized.csv"))
        self.assertTrue(v1_reg["nicolit"].endswith("ni_coupling_supplement.csv"))


class DefaultPairsTest(unittest.TestCase):
    """The 10 default migration pairs must match the P2-05 spec exactly."""

    def test_ten_default_pairs(self) -> None:
        self.assertEqual(len(DEFAULT_PAIRS), 10)

    def test_no_self_migration(self) -> None:
        for src, tgt in DEFAULT_PAIRS:
            self.assertNotEqual(src, tgt, f"self-migration found: {src}->{tgt}")

    def test_all_datasets_in_five_set(self) -> None:
        valid = {"regiosqm20", "hitea", "uspto", "ord", "nicolit"}
        for src, tgt in DEFAULT_PAIRS:
            self.assertIn(src, valid)
            self.assertIn(tgt, valid)

    def test_expected_ten_pairs_exact(self) -> None:
        expected = {
            ("regiosqm20", "hitea"),
            ("regiosqm20", "uspto"),
            ("regiosqm20", "ord"),
            ("regiosqm20", "nicolit"),
            ("hitea", "uspto"),
            ("hitea", "ord"),
            ("hitea", "nicolit"),
            ("uspto", "ord"),
            ("uspto", "nicolit"),
            ("ord", "nicolit"),
        }
        self.assertEqual(set(DEFAULT_PAIRS), expected)

    def test_pairs_unique(self) -> None:
        self.assertEqual(len(DEFAULT_PAIRS), len(set(DEFAULT_PAIRS)))

    def test_default_seeds_count(self) -> None:
        seeds = v2.parse_seeds(DEFAULT_SEEDS)
        self.assertEqual(len(seeds), 10)
        self.assertEqual(seeds[0], 20260710)
        self.assertEqual(seeds[-1], 20260719)
        # All unique
        self.assertEqual(len(seeds), len(set(seeds)))


class DatasetRegistryTest(unittest.TestCase):
    def test_five_datasets_registered(self) -> None:
        for name in ["regiosqm20", "hitea", "uspto", "ord", "nicolit"]:
            self.assertIn(name, V2_DATASET_REGISTRY)
            self.assertTrue(V2_DATASET_REGISTRY[name].endswith(".csv"))

    def test_ord_path_correct(self) -> None:
        self.assertTrue(V2_DATASET_REGISTRY["ord"].endswith("ord_normalized.csv"))

    def test_nicolit_path_correct(self) -> None:
        # NiCOlit uses the ni_coupling_supplement.csv file (1688 reactions,
        # mostly NiCOlit literature).
        self.assertTrue(V2_DATASET_REGISTRY["nicolit"].endswith("ni_coupling_supplement.csv"))


class ParsePairsTest(unittest.TestCase):
    def test_parse_single_pair(self) -> None:
        self.assertEqual(parse_pairs("regiosqm20->hitea"),
                         [("regiosqm20", "hitea")])

    def test_parse_multiple_pairs(self) -> None:
        result = parse_pairs("regiosqm20->hitea,hitea->uspto")
        self.assertEqual(result, [("regiosqm20", "hitea"), ("hitea", "uspto")])

    def test_parse_invalid_pair_raises(self) -> None:
        with self.assertRaises(ValueError):
            parse_pairs("regiosqm20:hitea")

    def test_parse_handles_whitespace(self) -> None:
        result = parse_pairs(" regiosqm20 -> hitea , hitea -> uspto ")
        self.assertEqual(result, [("regiosqm20", "hitea"), ("hitea", "uspto")])

    def test_parse_empty_string(self) -> None:
        self.assertEqual(parse_pairs(""), [])

    def test_parse_skips_empty_items(self) -> None:
        result = parse_pairs("regiosqm20->hitea,,hitea->uspto,")
        self.assertEqual(result, [("regiosqm20", "hitea"), ("hitea", "uspto")])


class IsCiAllPositiveTest(unittest.TestCase):
    def test_positive_ci_low(self) -> None:
        payload = {"paired_significance_pooled": {"delta_ci95_low": 0.05, "delta_ci95_high": 0.20}}
        self.assertTrue(is_ci_all_positive(payload))

    def test_zero_ci_low(self) -> None:
        payload = {"paired_significance_pooled": {"delta_ci95_low": 0.0, "delta_ci95_high": 0.20}}
        self.assertFalse(is_ci_all_positive(payload))

    def test_negative_ci_low(self) -> None:
        payload = {"paired_significance_pooled": {"delta_ci95_low": -0.10, "delta_ci95_high": 0.20}}
        self.assertFalse(is_ci_all_positive(payload))

    def test_missing_pooled(self) -> None:
        self.assertFalse(is_ci_all_positive({}))

    def test_missing_ci_low(self) -> None:
        payload = {"paired_significance_pooled": {}}
        self.assertFalse(is_ci_all_positive(payload))

    def test_non_numeric_ci_low(self) -> None:
        payload = {"paired_significance_pooled": {"delta_ci95_low": "not-a-number"}}
        self.assertFalse(is_ci_all_positive(payload))


class BuildAggregateSummaryTest(unittest.TestCase):
    def _fake_pair(self, src: str, tgt: str, ci_low: float, ci_high: float = 0.3) -> Dict[str, object]:
        return {
            "source": src,
            "target": tgt,
            "paired_significance_pooled": {
                "n": 10,
                "delta_mean": 0.1,
                "delta_ci95_low": ci_low,
                "delta_ci95_high": ci_high,
                "paired_permutation_p": 0.01,
                "sign_test_p": 0.02,
            },
        }

    def test_aggregate_counts_ci_positive(self) -> None:
        per_pair = [self._fake_pair("a", "b", 0.05), self._fake_pair("c", "d", -0.05)]
        agg = build_aggregate_summary(per_pair)
        self.assertEqual(agg["n_pairs_total"], 2)
        self.assertEqual(agg["n_pairs_ci_all_positive"], 1)
        self.assertEqual(len(agg["pairs"]), 2)
        self.assertTrue(agg["pairs"][0]["ci_all_positive"])
        self.assertFalse(agg["pairs"][1]["ci_all_positive"])

    def test_aggregate_has_all_ten_pairs_when_provided(self) -> None:
        per_pair = [self._fake_pair(s, t, 0.05) for s, t in DEFAULT_PAIRS]
        agg = build_aggregate_summary(per_pair)
        self.assertEqual(agg["n_pairs_total"], 10)
        self.assertEqual(agg["n_pairs_ci_all_positive"], 10)
        self.assertEqual(len(agg["pairs"]), 10)

    def test_aggregate_pair_names(self) -> None:
        per_pair = [self._fake_pair("regiosqm20", "hitea", 0.05)]
        agg = build_aggregate_summary(per_pair)
        self.assertEqual(agg["pairs"][0]["pair"], "regiosqm20_to_hitea")
        self.assertEqual(agg["pairs"][0]["source"], "regiosqm20")
        self.assertEqual(agg["pairs"][0]["target"], "hitea")

    def test_aggregate_zero_positive(self) -> None:
        per_pair = [self._fake_pair(s, t, -0.1) for s, t in DEFAULT_PAIRS]
        agg = build_aggregate_summary(per_pair)
        self.assertEqual(agg["n_pairs_ci_all_positive"], 0)

    def test_aggregate_missing_pooled_treated_as_zero(self) -> None:
        per_pair = [{"source": "a", "target": "b"}]
        agg = build_aggregate_summary(per_pair)
        self.assertEqual(agg["n_pairs_ci_all_positive"], 0)
        self.assertFalse(agg["pairs"][0]["ci_all_positive"])

    def test_aggregate_ci_high_zero_excluded(self) -> None:
        # CI low > 0 but CI high == 0 should NOT count (CI not "all" positive).
        per_pair = [self._fake_pair("a", "b", ci_low=0.05, ci_high=0.0)]
        agg = build_aggregate_summary(per_pair)
        self.assertEqual(agg["n_pairs_ci_all_positive"], 0)


class BuildGoNoGoTest(unittest.TestCase):
    def test_go_when_three_or_more(self) -> None:
        agg = {"n_pairs_ci_all_positive": 3, "n_pairs_total": 10}
        decision = build_go_no_go(agg, threshold=3)
        self.assertEqual(decision["decision"], "GO")
        self.assertEqual(decision["count_ci_all_positive"], 3)
        self.assertTrue(decision["fixes_L5_NOGO"])

    def test_go_when_above_threshold(self) -> None:
        agg = {"n_pairs_ci_all_positive": 7, "n_pairs_total": 10}
        decision = build_go_no_go(agg, threshold=3)
        self.assertEqual(decision["decision"], "GO")

    def test_no_go_when_below_threshold(self) -> None:
        agg = {"n_pairs_ci_all_positive": 2, "n_pairs_total": 10}
        decision = build_go_no_go(agg, threshold=3)
        self.assertEqual(decision["decision"], "NO-GO")
        self.assertEqual(decision["count_ci_all_positive"], 2)
        self.assertFalse(decision["fixes_L5_NOGO"])

    def test_no_go_when_zero(self) -> None:
        agg = {"n_pairs_ci_all_positive": 0, "n_pairs_total": 10}
        decision = build_go_no_go(agg, threshold=3)
        self.assertEqual(decision["decision"], "NO-GO")

    def test_has_required_fields(self) -> None:
        agg = {"n_pairs_ci_all_positive": 0, "n_pairs_total": 10}
        decision = build_go_no_go(agg, threshold=3)
        self.assertIn("count_ci_all_positive", decision)
        self.assertIn("decision", decision)
        self.assertIn("threshold_for_go", decision)
        self.assertIn("n_pairs_total", decision)
        self.assertIn("rule", decision)
        self.assertIn("fixes_L5_NOGO", decision)

    def test_default_threshold_is_three(self) -> None:
        self.assertEqual(GO_THRESHOLD, 3)


class PairedSignificanceKeysTest(unittest.TestCase):
    """Verify v1's paired_significance (reused by v2) has the expected keys."""

    def test_paired_significance_has_required_keys(self) -> None:
        from pc_cng.run_cross_dataset_transfer_eval import paired_significance
        sig = paired_significance([0.1, 0.2, 0.3, -0.1, 0.5],
                                  bootstrap_iterations=100, seed=42)
        for key in ["n", "delta_mean", "delta_ci95_low", "delta_ci95_high",
                    "paired_permutation_p", "sign_test_p"]:
            self.assertIn(key, sig, f"missing key {key!r}")

    def test_paired_significance_empty_deltas(self) -> None:
        from pc_cng.run_cross_dataset_transfer_eval import paired_significance
        sig = paired_significance([], bootstrap_iterations=100, seed=42)
        self.assertEqual(sig["n"], 0)
        self.assertEqual(sig["delta_mean"], 0.0)


class WritePerPairSummaryTest(unittest.TestCase):
    def test_summary_json_has_required_keys(self) -> None:
        payload = {
            "source": "regiosqm20",
            "target": "hitea",
            "seeds": [20260710, 20260711],
            "config": {"epochs": 2, "pccng_limit": 50},
            "per_seed": [{"seed": 20260710, "baseline_target_top1": 0.4}],
            "paired_significance_pooled": {
                "n": 10, "delta_mean": 0.1,
                "delta_ci95_low": 0.05, "delta_ci95_high": 0.2,
            },
            "seed_level_significance": {"n_seeds": 2},
        }
        with tempfile.TemporaryDirectory() as tmp:
            pair_dir = os.path.join(tmp, "regiosqm20_to_hitea")
            write_per_pair_summary(pair_dir, payload)
            summary_path = os.path.join(pair_dir, "summary.json")
            self.assertTrue(os.path.exists(summary_path))
            with open(summary_path) as h:
                s = json.load(h)
            for key in ["source", "target", "seeds", "config", "per_seed",
                        "paired_significance_pooled", "seed_level_significance"]:
                self.assertIn(key, s)


class SyntheticRunWithMonkeypatchTest(unittest.TestCase):
    """End-to-end test that runs v2.main() with a fake run_transfer_pair."""

    def _fake_pair_result(self, source: str, target: str, ci_low: float = 0.05) -> Dict[str, object]:
        return {
            "source": source,
            "target": target,
            "source_csv": f"/fake/{source}.csv",
            "target_csv": f"/fake/{target}.csv",
            "seeds": [20260710, 20260711],
            "pccng_negatives_csv": f"/fake/pccng_{source}.csv",
            "config": {"epochs": 2, "pccng_limit": 50,
                       "bootstrap_iterations": 100, "smoke": False},
            "per_seed": [{"seed": 20260710, "baseline_target_top1": 0.4,
                          "treatment_target_top1": 0.6}],
            "paired_significance_pooled": {
                "n": 10,
                "delta_mean": 0.1,
                "delta_ci95_low": ci_low,
                "delta_ci95_high": 0.3,
                "paired_permutation_p": 0.01,
                "sign_test_p": 0.02,
                "positive_deltas": 8,
                "negative_deltas": 2,
                "zero_deltas": 0,
            },
            "seed_level_significance": {
                "n_seeds": 2, "mean_delta": 0.1,
                "ci95_low": 0.0, "ci95_high": 0.2,
                "sign_test_p": 0.5, "per_seed_deltas": [0.2, 0.2],
            },
            "decision_rule": {
                "go_to_main_paper": "CI95_low > 0 AND sign_test_p < 0.05",
                "supplementary_only": "CI crosses 0 OR sign_test_p >= 0.05",
            },
        }

    def _run_v2_main(self, argv: List[str]) -> str:
        """Run v2.main() with the given argv (without program name)."""
        argv_backup = sys.argv
        sys.argv = ["v2"] + argv
        try:
            v2.main()
        finally:
            sys.argv = argv_backup

    def test_smoke_run_writes_all_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = os.path.join(tmp, "v2_out")
            with mock.patch(
                "pc_cng.run_cross_dataset_transfer_eval_v2.run_transfer_pair"
            ) as fake:
                # First pair: CI_low > 0 (counts toward GO). Second pair: CI_low < 0.
                fake.side_effect = lambda source, target, **kwargs: self._fake_pair_result(
                    source, target,
                    ci_low=0.05 if source == "regiosqm20" else -0.05,
                )
                self._run_v2_main([
                    "--output-dir", output_dir,
                    "--pairs", "regiosqm20->hitea,hitea->uspto",
                    "--pccng-limit", "50",
                    "--epochs", "2",
                    "--seeds", "20260710,20260711",
                ])

            # Per-pair summary.json + paired_significance.json
            for pair in ["regiosqm20_to_hitea", "hitea_to_uspto"]:
                pair_dir = os.path.join(output_dir, pair)
                self.assertTrue(
                    os.path.exists(os.path.join(pair_dir, "summary.json")),
                    f"summary.json missing for {pair}",
                )
                with open(os.path.join(pair_dir, "summary.json")) as h:
                    s = json.load(h)
                self.assertIn("paired_significance_pooled", s)
                self.assertIn("config", s)

            # aggregate_summary.json
            agg_path = os.path.join(output_dir, "aggregate_summary.json")
            self.assertTrue(os.path.exists(agg_path))
            with open(agg_path) as h:
                agg = json.load(h)
            self.assertEqual(agg["n_pairs_total"], 2)
            self.assertEqual(agg["n_pairs_ci_all_positive"], 1)
            self.assertEqual(len(agg["pairs"]), 2)

            # go_no_go_decision.json
            dec_path = os.path.join(output_dir, "go_no_go_decision.json")
            self.assertTrue(os.path.exists(dec_path))
            with open(dec_path) as h:
                dec = json.load(h)
            self.assertIn("count_ci_all_positive", dec)
            self.assertIn("decision", dec)
            self.assertEqual(dec["count_ci_all_positive"], 1)
            self.assertEqual(dec["decision"], "NO-GO")  # 1 < 3 threshold

    def test_pair_only_filter_runs_single_pair(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = os.path.join(tmp, "v2_out_pair_only")
            with mock.patch(
                "pc_cng.run_cross_dataset_transfer_eval_v2.run_transfer_pair"
            ) as fake:
                fake.side_effect = lambda source, target, **kwargs: self._fake_pair_result(
                    source, target,
                )
                self._run_v2_main([
                    "--output-dir", output_dir,
                    "--pair-only", "regiosqm20->hitea",
                    "--pccng-limit", "50",
                    "--epochs", "2",
                    "--seeds", "20260710",
                ])
            # Only 1 pair should be in aggregate
            with open(os.path.join(output_dir, "aggregate_summary.json")) as h:
                agg = json.load(h)
            self.assertEqual(agg["n_pairs_total"], 1)
            self.assertEqual(agg["pairs"][0]["source"], "regiosqm20")
            self.assertEqual(agg["pairs"][0]["target"], "hitea")

    def test_pairs_override_runs_all_specified_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = os.path.join(tmp, "v2_out_pairs")
            with mock.patch(
                "pc_cng.run_cross_dataset_transfer_eval_v2.run_transfer_pair"
            ) as fake:
                fake.side_effect = lambda source, target, **kwargs: self._fake_pair_result(
                    source, target, ci_low=0.05,
                )
                self._run_v2_main([
                    "--output-dir", output_dir,
                    "--pairs", "regiosqm20->hitea,regiosqm20->uspto,regiosqm20->ord",
                    "--pccng-limit", "50",
                    "--epochs", "2",
                    "--seeds", "20260710",
                ])
            with open(os.path.join(output_dir, "aggregate_summary.json")) as h:
                agg = json.load(h)
            self.assertEqual(agg["n_pairs_total"], 3)
            self.assertEqual(agg["n_pairs_ci_all_positive"], 3)
            self.assertEqual(agg["pairs"][0]["decision"] if "decision" in agg["pairs"][0] else None, None)

    def test_default_pairs_aggregate_has_all_ten_when_all_run(self) -> None:
        """If we ran all 10 default pairs (monkeypatched), aggregate has 10."""
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = os.path.join(tmp, "v2_out_default")
            with mock.patch(
                "pc_cng.run_cross_dataset_transfer_eval_v2.run_transfer_pair"
            ) as fake:
                fake.side_effect = lambda source, target, **kwargs: self._fake_pair_result(
                    source, target, ci_low=0.05,
                )
                self._run_v2_main([
                    "--output-dir", output_dir,
                    "--pccng-limit", "10",
                    "--epochs", "1",
                    "--seeds", "20260710",
                ])
            with open(os.path.join(output_dir, "aggregate_summary.json")) as h:
                agg = json.load(h)
            self.assertEqual(agg["n_pairs_total"], 10)
            self.assertEqual(agg["n_pairs_ci_all_positive"], 10)
            with open(os.path.join(output_dir, "go_no_go_decision.json")) as h:
                dec = json.load(h)
            self.assertEqual(dec["decision"], "GO")
            self.assertEqual(dec["count_ci_all_positive"], 10)
            self.assertTrue(dec["fixes_L5_NOGO"])

    def test_default_seeds_used_when_not_specified(self) -> None:
        """When --seeds is not given, the 10 default seeds are used."""
        captured: Dict[str, object] = {}

        def fake_runner(source, target, **kwargs):
            captured["seeds"] = kwargs.get("seeds")
            return self._fake_pair_result(source, target)

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = os.path.join(tmp, "v2_out_default_seeds")
            with mock.patch(
                "pc_cng.run_cross_dataset_transfer_eval_v2.run_transfer_pair"
            ) as fake:
                fake.side_effect = fake_runner
                self._run_v2_main([
                    "--output-dir", output_dir,
                    "--pair-only", "regiosqm20->hitea",
                    "--pccng-limit", "10",
                    "--epochs", "1",
                ])
            seeds = captured.get("seeds")
            self.assertEqual(len(seeds), 10)
            self.assertEqual(seeds[0], 20260710)
            self.assertEqual(seeds[-1], 20260719)

    def test_default_pccng_limit_is_1000(self) -> None:
        """Default --pccng-limit should be 1000 (up from v1's 200 in P1-02)."""
        captured: Dict[str, object] = {}

        def fake_runner(source, target, **kwargs):
            captured["pccng_limit"] = kwargs.get("pccng_limit")
            return self._fake_pair_result(source, target)

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = os.path.join(tmp, "v2_out_default_limit")
            with mock.patch(
                "pc_cng.run_cross_dataset_transfer_eval_v2.run_transfer_pair"
            ) as fake:
                fake.side_effect = fake_runner
                self._run_v2_main([
                    "--output-dir", output_dir,
                    "--pair-only", "regiosqm20->hitea",
                    "--epochs", "1",
                ])
            self.assertEqual(captured.get("pccng_limit"), 1000)

    def test_default_epochs_is_15(self) -> None:
        """Default --epochs should be 15 (per P2-05 spec)."""
        captured: Dict[str, object] = {}

        def fake_runner(source, target, **kwargs):
            captured["epochs"] = kwargs.get("epochs")
            return self._fake_pair_result(source, target)

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = os.path.join(tmp, "v2_out_default_epochs")
            with mock.patch(
                "pc_cng.run_cross_dataset_transfer_eval_v2.run_transfer_pair"
            ) as fake:
                fake.side_effect = fake_runner
                self._run_v2_main([
                    "--output-dir", output_dir,
                    "--pair-only", "regiosqm20->hitea",
                ])
            self.assertEqual(captured.get("epochs"), 15)


if __name__ == "__main__":
    unittest.main()
