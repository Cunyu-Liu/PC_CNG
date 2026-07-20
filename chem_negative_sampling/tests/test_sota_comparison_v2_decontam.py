"""Unit tests for P3-02 decontamination filter (Tanimoto-NN leakage fix).

The Tanimoto-NN baseline achieves MRR=1.0 on the contaminated test set
because test-set products appear verbatim in the train set (Tanimoto=1.0
nearest neighbour trivially recovers the correct label).  These tests
verify the decontamination filter that excludes such leaked test
candidates so PC-CNG vs Tanimoto-NN is a fair comparison.

Covers
------
* ``_canonical_smiles`` — robust canonicalisation (CCO == OCC) + fallbacks
* ``filter_leaked_test_rows`` — excludes leaked candidates, reports n_leaked
* ``run_seed(decontaminate=True)`` — stores both decontam (primary) and
  ``_contam`` reference metrics; records ``n_leaked``
* ``run_seed(decontaminate=False)`` — unchanged behaviour (no ``_contam`` keys)
* ``_parse_args`` — accepts the new ``--decontaminate`` flag
* ``aggregate_metrics`` / ``paired_significance`` — ``suffix`` parameter
* End-to-end: decontamination removes the leakage artifact so PC-CNG is
  no longer unfairly compared against a Tanimoto-NN baseline that
  trivially memorised the test products.

Hard constraint compliance
--------------------------
* HC #4: every new module / function has unit tests.
* HC #5: tests verify the significance-test plumbing is wired up
  (suffix forwarding for contaminated reference).
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Dict, List


def _ensure_importable() -> None:
    """Add chem_negative_sampling to sys.path so we can import pc_cng."""
    here = Path(__file__).resolve()
    cns_root = here.parents[1]
    if str(cns_root) not in sys.path:
        sys.path.insert(0, str(cns_root))


_ensure_importable()

from pc_cng.run_sota_comparison_v2 import (  # type: ignore
    _canonical_smiles,
    _parse_args,
    aggregate_metrics,
    filter_leaked_test_rows,
    paired_significance,
    run_seed,
)
import pc_cng.run_sota_comparison_v2 as _SOTA  # type: ignore


def _row(
    group_id: str,
    source_id: str,
    rxn: str,
    label: int,
    parent_product: str,
) -> Dict[str, object]:
    """Build a minimal route-candidate row for tests."""
    return {
        "group_id": group_id,
        "source_id": source_id,
        "reaction_smiles": rxn,
        "label": label,
        "candidate_source": "positive_reaction" if label else "pc_cng_synthetic",
        "failure_type": "gold" if label else "synthetic",
        "edit_action": "",
        "hard_score": 1.0 if label else 0.0,
        "false_negative_risk": 0.0,
        "parent_product": parent_product,
    }


class TestCanonicalSmiles(unittest.TestCase):
    """``_canonical_smiles`` must canonicalise robustly and fall back safely."""

    def test_canonicalises_ethanol(self):
        # CCO and OCC are the same molecule (ethanol).
        self.assertEqual(_canonical_smiles("CCO"), _canonical_smiles("OCC"))

    def test_empty_string_returns_empty(self):
        self.assertEqual(_canonical_smiles(""), "")

    def test_invalid_smiles_falls_back_to_stripped_input(self):
        # An unparseable SMILES must not raise; it falls back to the
        # stripped input so the filter degrades to exact-string matching.
        invalid = "not_a_smiles$$$"
        self.assertEqual(_canonical_smiles(invalid), invalid.strip())

    def test_canonical_form_is_stable(self):
        # Repeated calls return the same canonical string.
        a = _canonical_smiles("CCO")
        b = _canonical_smiles("CCO")
        self.assertEqual(a, b)


class TestFilterLeakedTestRows(unittest.TestCase):
    """``filter_leaked_test_rows`` excludes test products seen in train."""

    def test_excludes_leaked_candidates(self):
        train = [
            _row("g1", "s1", "A>>B", 1, "B"),
            _row("g1", "s1", "A>>B", 0, "B"),
        ]
        test = [
            # Leaked: product "B" appears in train.
            _row("g9", "s9", "X>>B", 1, "B"),
            _row("g9", "s9", "X>>B", 0, "B"),
            # Not leaked: product "Z" is novel.
            _row("g10", "s10", "Y>>Z", 1, "Z"),
            _row("g10", "s10", "Y>>Z", 0, "Z"),
        ]
        decontam, n_leaked, leaked_ids = filter_leaked_test_rows(train, test)
        self.assertEqual(n_leaked, 2)
        self.assertEqual(leaked_ids, ["s9"])
        # Only the two non-leaked rows survive.
        self.assertEqual(len(decontam), 2)
        self.assertTrue(all(r["parent_product"] == "Z" for r in decontam))

    def test_canonical_matching_ethanol(self):
        # Train product "CCO" should match test product "OCC" (same molecule).
        train = [_row("g1", "s1", "A>>CCO", 1, "CCO")]
        test = [_row("g2", "s2", "X>>OCC", 1, "OCC")]
        _, n_leaked, _ = filter_leaked_test_rows(train, test)
        self.assertEqual(n_leaked, 1)

    def test_no_leakage_returns_all_rows(self):
        train = [_row("g1", "s1", "A>>B", 1, "B")]
        test = [
            _row("g2", "s2", "Y>>Z", 1, "Z"),
            _row("g3", "s3", "Y>>W", 1, "W"),
        ]
        decontam, n_leaked, leaked_ids = filter_leaked_test_rows(train, test)
        self.assertEqual(n_leaked, 0)
        self.assertEqual(leaked_ids, [])
        self.assertEqual(len(decontam), 2)

    def test_empty_train_set_leaks_nothing(self):
        test = [_row("g1", "s1", "A>>B", 1, "B")]
        decontam, n_leaked, _ = filter_leaked_test_rows([], test)
        self.assertEqual(n_leaked, 0)
        self.assertEqual(len(decontam), 1)

    def test_missing_parent_product_treated_as_non_leaked(self):
        # A row with no parent_product must not match anything (and not crash).
        train = [_row("g1", "s1", "A>>B", 1, "B")]
        weird = _row("g2", "s2", "X>>Y", 1, "")
        decontam, n_leaked, _ = filter_leaked_test_rows(train, [weird])
        self.assertEqual(n_leaked, 0)
        self.assertEqual(len(decontam), 1)


class TestRunSeedDecontaminate(unittest.TestCase):
    """``run_seed`` must store both decontam (primary) and _contam metrics."""

    def _build_rows(self) -> List[Dict[str, object]]:
        # Two source_ids: s_train (goes to train) and s_test (goes to test).
        # The test source's product *matches* the train source's product,
        # which is the leakage artifact we want to filter.
        return [
            # --- train source (s1 < s2 so s1 lands in train under 0.5 split) ---
            _row("g1", "s1", "A>>B", 1, "B"),
            _row("g1", "s1", "A>>B", 0, "B"),
            # --- test source whose product "B" leaks into train ---
            _row("g2", "s2", "X>>B", 1, "B"),
            _row("g2", "s2", "X>>B", 0, "B"),
            # --- test source with a genuinely novel product "Z" ---
            _row("g3", "s3", "Y>>Z", 1, "Z"),
            _row("g3", "s3", "Y>>Z", 0, "Z"),
        ]

    def test_decontaminate_records_n_leaked_and_both_metric_sets(self):
        rows = self._build_rows()
        result = run_seed(
            rows, seed=42, methods=["tanimoto_nn", "pc_cng"],
            train_fraction=0.5, decontaminate=True,
        )
        # Bookkeeping fields present.
        self.assertIn("n_leaked", result)
        self.assertIn("n_test_contam", result)
        self.assertIn("n_test_decontam", result)
        self.assertIn("leaked_source_ids", result)
        self.assertTrue(result["decontaminate"])
        # The leaked source s2 (product "B") must be reported.
        self.assertGreater(result["n_leaked"], 0)
        self.assertIn("s2", result["leaked_source_ids"])
        self.assertLess(result["n_test_decontam"], result["n_test_contam"])
        # Primary (decontaminated) metrics under standard keys.
        self.assertIn("tanimoto_nn_metrics", result)
        self.assertIn("tanimoto_nn_per_group", result)
        self.assertIn("pc_cng_metrics", result)
        # Contaminated reference metrics under _contam suffix.
        self.assertIn("tanimoto_nn_metrics_contam", result)
        self.assertIn("tanimoto_nn_per_group_contam", result)
        self.assertIn("pc_cng_metrics_contam", result)

    def test_decontaminate_off_keeps_legacy_behaviour(self):
        rows = self._build_rows()
        result = run_seed(
            rows, seed=42, methods=["tanimoto_nn"],
            train_fraction=0.5, decontaminate=False,
        )
        # No leakage bookkeeping beyond the zero defaults.
        self.assertEqual(result["n_leaked"], 0)
        self.assertEqual(result["n_test_contam"], result["n_test_decontam"])
        self.assertEqual(result["leaked_source_ids"], [])
        self.assertFalse(result["decontaminate"])
        # No _contam reference keys when the flag is off.
        self.assertNotIn("tanimoto_nn_metrics_contam", result)
        self.assertNotIn("tanimoto_nn_per_group_contam", result)

    def test_decontaminate_drops_leaked_group_from_primary_per_group(self):
        rows = self._build_rows()
        result = run_seed(
            rows, seed=42, methods=["tanimoto_nn"],
            train_fraction=0.5, decontaminate=True,
        )
        primary_groups = set(result["tanimoto_nn_per_group"].keys())
        contam_groups = set(result["tanimoto_nn_per_group_contam"].keys())
        # The leaked group g2 (product "B") must be absent from the
        # primary (decontaminated) per-group dict but present in the
        # contaminated reference.
        self.assertNotIn("g2", primary_groups)
        self.assertIn("g2", contam_groups)
        # The non-leaked group g3 survives in both.
        self.assertIn("g3", primary_groups)
        self.assertIn("g3", contam_groups)


class TestAggregateAndSignificanceSuffix(unittest.TestCase):
    """``aggregate_metrics`` and ``paired_significance`` honour ``suffix``."""

    def test_aggregate_metrics_suffix_contam(self):
        rows = [
            _row("g1", "s1", "A>>B", 1, "B"),
            _row("g1", "s1", "A>>B", 0, "B"),
            _row("g2", "s2", "X>>Z", 1, "Z"),
            _row("g2", "s2", "X>>Z", 0, "Z"),
        ]
        r1 = run_seed(rows, seed=1, methods=["tanimoto_nn", "pc_cng"],
                      train_fraction=0.5, decontaminate=True)
        r2 = run_seed(rows, seed=2, methods=["tanimoto_nn", "pc_cng"],
                      train_fraction=0.5, decontaminate=True)
        agg_primary = aggregate_metrics([r1, r2], ["tanimoto_nn", "pc_cng"])
        agg_contam = aggregate_metrics(
            [r1, r2], ["tanimoto_nn", "pc_cng"], suffix="_contam",
        )
        # Both suffixes must return entries for the requested methods.
        self.assertIn("tanimoto_nn", agg_primary)
        self.assertIn("tanimoto_nn", agg_contam)
        self.assertIn("pc_cng", agg_primary)
        self.assertIn("pc_cng", agg_contam)
        # Each aggregated entry must report n_seeds.
        self.assertEqual(agg_primary["tanimoto_nn"]["n_seeds"], 2)
        self.assertEqual(agg_contam["tanimoto_nn"]["n_seeds"], 2)

    def test_paired_significance_suffix_contam(self):
        rows = [
            _row("g1", "s1", "A>>B", 1, "B"),
            _row("g1", "s1", "A>>B", 0, "B"),
            _row("g2", "s2", "X>>Z", 1, "Z"),
            _row("g2", "s2", "X>>Z", 0, "Z"),
        ]
        seed_results = [
            run_seed(rows, seed=s, methods=["tanimoto_nn", "pc_cng"],
                     train_fraction=0.5, decontaminate=True)
            for s in (1, 2, 3)
        ]
        sig_primary = paired_significance(
            seed_results, ["tanimoto_nn", "pc_cng"],
            bootstrap_iterations=200, seed=1,
        )
        sig_contam = paired_significance(
            seed_results, ["tanimoto_nn", "pc_cng"],
            bootstrap_iterations=200, seed=1, suffix="_contam",
        )
        # The PC-CNG vs Tanimoto-NN pair must be present in both.
        self.assertIn("pc_cng_vs_tanimoto_nn", sig_primary)
        self.assertIn("pc_cng_vs_tanimoto_nn", sig_contam)
        # Both must report a non-None delta.
        self.assertIsNotNone(sig_primary["pc_cng_vs_tanimoto_nn"]["delta_pp"])
        self.assertIsNotNone(sig_contam["pc_cng_vs_tanimoto_nn"]["delta_pp"])


class TestDecontaminateCLIArg(unittest.TestCase):
    """The CLI must accept the new ``--decontaminate`` flag."""

    def test_flag_defaults_off(self):
        with tempfile.TemporaryDirectory() as tmp:
            neg_path = Path(tmp) / "negatives.csv"
            neg_path.write_text(
                "source_id,positive_reaction,candidate_reaction,parent_product,label\n"
                "g1,A>>B,E>>F,B,1\n",
                encoding="utf-8",
            )
            args = _parse_args([
                "--pc-cng-negatives", str(neg_path),
                "--output-dir", str(Path(tmp) / "out"),
            ])
            self.assertFalse(args.decontaminate)

    def test_flag_turns_on(self):
        with tempfile.TemporaryDirectory() as tmp:
            neg_path = Path(tmp) / "negatives.csv"
            neg_path.write_text(
                "source_id,positive_reaction,candidate_reaction,parent_product,label\n"
                "g1,A>>B,E>>F,B,1\n",
                encoding="utf-8",
            )
            args = _parse_args([
                "--pc-cng-negatives", str(neg_path),
                "--output-dir", str(Path(tmp) / "out"),
                "--decontaminate",
            ])
            self.assertTrue(args.decontaminate)


class TestDecontaminationReversesLeakageArtifact(unittest.TestCase):
    """End-to-end: decontamination removes the Tanimoto-NN MRR=1.0 artifact.

    On the contaminated test set Tanimoto-NN gets a perfect score for any
    group whose product appears in train (nearest neighbour is the exact
    same product, Tanimoto=1.0).  After decontamination those leaked
    groups are dropped, so Tanimoto-NN no longer benefits from the
    artifact on the remaining (novel-product) groups.
    """

    def test_tanimoto_nn_mrr_drops_after_decontamination(self):
        rows = [
            # train source
            _row("g1", "s1", "A>>B", 1, "B"),
            _row("g1", "s1", "A>>B", 0, "B"),
            # test source that LEAKS (product "B" in train)
            _row("g2", "s2", "X>>B", 1, "B"),
            _row("g2", "s2", "X>>B", 0, "B"),
            # test source with a NOVEL product "Z"
            _row("g3", "s3", "Y>>Z", 1, "Z"),
            _row("g3", "s3", "Y>>Z", 0, "Z"),
        ]
        result = run_seed(
            rows, seed=42, methods=["tanimoto_nn"],
            train_fraction=0.5, decontaminate=True,
        )
        contam_mrr = result["tanimoto_nn_metrics_contam"]["mrr"]
        decontam_mrr = result["tanimoto_nn_metrics"]["mrr"]
        # On the contaminated set the leaked group g2 inflates Tanimoto-NN
        # (it matches train product "B" exactly).  After decontamination
        # g2 is dropped, so the decontaminated MRR must be <= the
        # contaminated MRR.
        self.assertLessEqual(
            decontam_mrr, contam_mrr + 1e-9,
            "Decontamination must not increase Tanimoto-NN MRR; the "
            "leakage artifact can only inflate the contaminated score.",
        )
        # And the leaked group must be gone from the primary per-group dict.
        self.assertNotIn("g2", result["tanimoto_nn_per_group"])


if __name__ == "__main__":
    unittest.main()
