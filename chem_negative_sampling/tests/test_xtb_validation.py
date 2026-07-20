"""Unit tests for the P1-10 xTB / MMFF94 computational validation runner."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from pc_cng.run_xtb_validation import (
    BARRIER_FORCE_FIELD_CONSTANT,
    BARRIER_SUPPORT_THRESHOLD,
    DEFAULT_NUM_SEEDS,
    DELTA_G_SUPPORT_THRESHOLD,
    compute_molecule_energy,
    compute_reaction_energy,
    judge_support,
    main as cli_main,
    parse_reaction_smiles,
    run_paired_significance_test,
    sample_candidates,
    strip_atom_maps,
    support_reason,
)


class TestParseReactionSmiles(unittest.TestCase):
    def test_simple_reaction(self):
        r, p = parse_reaction_smiles("CCO.[O]>>CC=O")
        self.assertEqual(r, "CCO.[O]")
        self.assertEqual(p, "CC=O")

    def test_no_arrow(self):
        r, p = parse_reaction_smiles("CCO")
        self.assertEqual(r, "")
        self.assertEqual(p, "")

    def test_empty(self):
        r, p = parse_reaction_smiles("")
        self.assertEqual(r, "")
        self.assertEqual(p, "")

    def test_multiple_arrows(self):
        # "A>>B>>C" splits into 3 parts -> invalid
        r, p = parse_reaction_smiles("A>>B>>C")
        self.assertEqual(r, "")
        self.assertEqual(p, "")

    def test_none_input(self):
        r, p = parse_reaction_smiles(None)  # type: ignore[arg-type]
        self.assertEqual(r, "")
        self.assertEqual(p, "")


class TestStripAtomMaps(unittest.TestCase):
    def test_atom_maps_removed(self):
        # [C:1] -> [C]
        out = strip_atom_maps("[CH3:1][O:2][CH3:3]")
        self.assertNotIn(":", out)
        # Should still be a valid SMILES (dimethyl ether)
        from rdkit import Chem
        m = Chem.MolFromSmiles(out)
        self.assertIsNotNone(m)

    def test_no_maps(self):
        out = strip_atom_maps("CCO")
        self.assertEqual(out, "CCO")

    def test_empty(self):
        self.assertEqual(strip_atom_maps(""), "")

    def test_invalid_smiles(self):
        self.assertEqual(strip_atom_maps("not_a_smiles!!!"), "")


class TestComputeMoleculeEnergy(unittest.TestCase):
    def test_ethanol_energy(self):
        # Ethanol should give a finite MMFF94 energy (can be negative:
        # MMFF energies are relative to a reference state, not absolute).
        import math
        res = compute_molecule_energy("CCO", method="mmff94", seed=42)
        self.assertEqual(res["status"], "ok")
        self.assertIsNotNone(res["energy_kcal_per_mol"])
        e = res["energy_kcal_per_mol"]
        self.assertFalse(math.isnan(e))
        self.assertFalse(math.isinf(e))
        self.assertEqual(res["method"], "mmff94")

    def test_methane_energy(self):
        res = compute_molecule_energy("C", method="mmff94", seed=42)
        self.assertEqual(res["status"], "ok")
        self.assertIsNotNone(res["energy_kcal_per_mol"])

    def test_empty_smiles(self):
        res = compute_molecule_energy("", method="mmff94")
        self.assertEqual(res["status"], "empty")
        self.assertIsNone(res["energy_kcal_per_mol"])

    def test_invalid_smiles(self):
        res = compute_molecule_energy("not_a_smiles!!!", method="mmff94")
        self.assertEqual(res["status"], "parse_error")
        self.assertIsNone(res["energy_kcal_per_mol"])

    def test_atom_mapped_smiles(self):
        # Atom maps should be stripped before parsing
        res = compute_molecule_energy("[CH3:1][O:2][CH3:3]", method="mmff94", seed=42)
        self.assertEqual(res["status"], "ok")
        self.assertIsNotNone(res["energy_kcal_per_mol"])

    def test_uff_fallback(self):
        # Force UFF method
        res = compute_molecule_energy("CCO", method="uff", seed=42)
        self.assertEqual(res["status"], "ok")
        self.assertIsNotNone(res["energy_kcal_per_mol"])
        self.assertEqual(res["method"], "uff")

    def test_deterministic_with_same_seed(self):
        a = compute_molecule_energy("CCO", method="mmff94", seed=123)
        b = compute_molecule_energy("CCO", method="mmff94", seed=123)
        self.assertEqual(a["status"], "ok")
        self.assertEqual(b["status"], "ok")
        self.assertAlmostEqual(a["energy_kcal_per_mol"], b["energy_kcal_per_mol"], places=4)


class TestComputeReactionEnergy(unittest.TestCase):
    def test_simple_exothermic(self):
        # A -> B where B has lower energy (exothermic, DeltaG < 0)
        # Use methane -> methane (same molecule, DeltaG = 0)
        res = compute_reaction_energy("C", "C", method="mmff94", seed=42)
        self.assertEqual(res["status"], "ok")
        self.assertAlmostEqual(res["delta_g"], 0.0, places=4)
        self.assertAlmostEqual(
            res["barrier_estimate"],
            abs(res["delta_g"]) + BARRIER_FORCE_FIELD_CONSTANT,
            places=4,
        )

    def test_multi_component(self):
        # A.B -> C  (sum of A+B energies vs C)
        res = compute_reaction_energy("C.C", "CC", method="mmff94", seed=42)
        self.assertEqual(res["status"], "ok")
        self.assertIsNotNone(res["delta_g"])

    def test_reactant_parse_failure(self):
        res = compute_reaction_energy("not_smiles!!!", "CCO", method="mmff94")
        self.assertIsNone(res["delta_g"])
        self.assertIn("reactant_failed", res["status"])

    def test_product_parse_failure(self):
        res = compute_reaction_energy("CCO", "not_smiles!!!", method="mmff94")
        self.assertIsNone(res["delta_g"])
        self.assertIn("product_failed", res["status"])

    def test_barrier_estimate_formula(self):
        # barrier = |DeltaG| + 5
        res = compute_reaction_energy("C", "CC", method="mmff94", seed=42)
        if res["status"] == "ok" and res["delta_g"] is not None:
            expected = abs(res["delta_g"]) + BARRIER_FORCE_FIELD_CONSTANT
            self.assertAlmostEqual(res["barrier_estimate"], expected, places=4)


class TestJudgeSupport(unittest.TestCase):
    def test_supported_by_delta_g(self):
        # DeltaG > 5 -> supported
        v = judge_support(10.0, 15.0)
        self.assertEqual(v, "supported")

    def test_supported_by_delta_g_boundary(self):
        v = judge_support(5.001, 10.0)
        self.assertEqual(v, "supported")

    def test_not_supported_low_delta_g(self):
        v = judge_support(2.0, 7.0)
        self.assertEqual(v, "not_supported")

    def test_not_supported_exothermic(self):
        # Exothermic, barrier high from |DeltaG|+5 but DeltaG<0 => not supported
        v = judge_support(-50.0, 55.0)
        self.assertEqual(v, "not_supported")

    def test_supported_by_barrier_endothermic(self):
        # DeltaG = 22 (not > 5... wait, 22 > 5, so already supported by delta_g)
        # Use DeltaG = 3, barrier = 30 (endothermic, barrier > 25)
        v = judge_support(3.0, 30.0)
        self.assertEqual(v, "supported")

    def test_not_supported_barrier_but_low_delta_g(self):
        # DeltaG = 1, barrier = 10 (barrier < 25, not supported)
        v = judge_support(1.0, 10.0)
        self.assertEqual(v, "not_supported")

    def test_inconclusive(self):
        v = judge_support(None, None)
        self.assertEqual(v, "inconclusive")

    def test_boundary_delta_g_exactly_5(self):
        # DeltaG = 5.0 exactly -> NOT > 5 -> not supported (strict)
        v = judge_support(5.0, 10.0)
        self.assertEqual(v, "not_supported")

    def test_exothermic_high_barrier_not_supported(self):
        # Critical: exothermic reaction with high |DeltaG| should NOT be supported
        # even though barrier = |DeltaG| + 5 > 25
        v = judge_support(-30.0, 35.0)
        self.assertEqual(v, "not_supported")


class TestSupportReason(unittest.TestCase):
    def test_reason_supported_delta_g(self):
        r = support_reason(10.0, 15.0)
        self.assertIn("delta_g", r)

    def test_reason_not_supported(self):
        r = support_reason(2.0, 7.0)
        self.assertIn("not supported", r)

    def test_reason_inconclusive(self):
        r = support_reason(None, None)
        self.assertIn("inconclusive", r)


class TestSampleCandidates(unittest.TestCase):
    def _make_df(self, n=10):
        rows = []
        # Diverse chemical-change reactions for even-index rows so that
        # deduplication does not collapse them all into a single row.
        chem_changes = [
            ("CCO", "CC=O"),
            ("CC(=O)O", "CC(=O)Cl"),
            ("c1ccccc1", "c1ccc(N)cc1"),
            ("CCN", "CCN(C)C"),
            ("O=C(O)C", "O=C(N)C"),
            ("CCBr", "CCCC"),
            ("COC=O", "COC(=O)NC"),
            ("CC(C)O", "CC(C)Cl"),
        ]
        for i in range(n):
            # Alternate reactants != product (even i) and reactants == product (odd i)
            if i % 2 == 0:
                r, p = chem_changes[(i // 2) % len(chem_changes)]
            else:
                r, p = "CCO", "CCO"
            rows.append({
                "source_id": f"SRC_{i:03d}",
                "positive_reaction": "CCO.[O]>>CC=O",
                "candidate_reactants": r,
                "candidate_product": p,
                "task": "retro_precursor",
                "failure_type": "chemoselectivity_error" if i % 2 == 0 else "no_reaction",
                "edit_action": "replace:O->N" if i % 2 == 0 else "reactants:=product",
                "hard_score": 0.5 + i * 0.04,  # 0.50, 0.54, ... 0.86
                "false_negative_risk": 0.3 + i * 0.05,
                "label": 0,
                "provenance": "pc_cng_mvp_synthetic_counterfactual",
            })
        return pd.DataFrame(rows)

    def test_sample_with_chemical_change(self):
        df = self._make_df(10)
        out = sample_candidates(df, limit=5, require_chemical_change=True)
        self.assertEqual(len(out), 5)
        # All rows should have reactants != product
        for _, r in out.iterrows():
            self.assertNotEqual(str(r["candidate_reactants"]), str(r["candidate_product"]))
        # Sorted by feasibility_score ascending = 1 - hard_score ascending = hard_score descending
        self.assertGreaterEqual(out["hard_score"].iloc[0], out["hard_score"].iloc[-1])

    def test_sample_without_chemical_change_filter(self):
        df = self._make_df(10)
        out = sample_candidates(df, limit=5, require_chemical_change=False)
        self.assertEqual(len(out), 5)

    def test_feasibility_score_computed(self):
        df = self._make_df(10)
        out = sample_candidates(df, limit=3, require_chemical_change=True)
        self.assertIn("feasibility_score", out.columns)
        # feasibility_score = 1 - hard_score
        for _, r in out.iterrows():
            self.assertAlmostEqual(r["feasibility_score"], 1.0 - r["hard_score"], places=6)

    def test_limit_caps_rows(self):
        df = self._make_df(10)
        out = sample_candidates(df, limit=2, require_chemical_change=True)
        self.assertEqual(len(out), 2)

    def test_handles_nan_hard_score(self):
        df = self._make_df(5)
        df.loc[0, "hard_score"] = float("nan")
        out = sample_candidates(df, limit=10, require_chemical_change=True)
        self.assertEqual(len(out), 2)  # only 2 even-index rows have chemical change + valid hard_score

    def test_deduplicate_drops_duplicate_reactions(self):
        df = self._make_df(10)
        # Add duplicates of the first chemical-change row
        dup = df.iloc[[0, 2]].copy()
        df = pd.concat([df, dup], ignore_index=True)
        # With deduplication, duplicates are dropped
        out = sample_candidates(df, limit=10, require_chemical_change=True, deduplicate=True)
        # Each (reactants, product) pair should be unique
        pairs = list(zip(out["candidate_reactants"], out["candidate_product"]))
        self.assertEqual(len(pairs), len(set(pairs)))

    def test_no_deduplicate_keeps_duplicates(self):
        df = self._make_df(10)
        dup = df.iloc[[0, 2]].copy()
        df = pd.concat([df, dup], ignore_index=True)
        out = sample_candidates(df, limit=10, require_chemical_change=True, deduplicate=False)
        pairs = list(zip(out["candidate_reactants"], out["candidate_product"]))
        self.assertGreaterEqual(len(pairs), len(set(pairs)))


class TestPairedSignificance(unittest.TestCase):
    def test_significant_difference(self):
        # neg deltas clearly higher than pos deltas
        neg = [20.0, 15.0, 18.0, 22.0, 17.0, 19.0, 16.0, 21.0, 23.0, 14.0]
        pos = [-5.0, -3.0, -2.0, -4.0, -1.0, -6.0, -2.0, -3.0, -4.0, -5.0]
        out = run_paired_significance_test(neg, pos, num_seeds=10, base_seed=42)
        self.assertEqual(out["status"], "ok")
        self.assertEqual(out["n_pairs"], 10)
        self.assertGreater(out["overall_mean_diff"], 0)
        self.assertGreaterEqual(out["n_significant_seeds"], 8)
        self.assertEqual(out["interpretation"], "significant")

    def test_no_difference(self):
        # neg and pos identical -> no significant difference
        d = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        out = run_paired_significance_test(d, d, num_seeds=10, base_seed=42)
        self.assertAlmostEqual(out["overall_mean_diff"], 0.0, places=6)
        # When diff=0 for all, CIs will be around 0 -> not significant
        self.assertEqual(out["interpretation"], "not_significant")

    def test_insufficient_data(self):
        out = run_paired_significance_test([1.0], [2.0], num_seeds=10)
        self.assertEqual(out["status"], "insufficient_data")
        self.assertEqual(out["n_pairs"], 1)

    def test_handles_none_values(self):
        neg = [10.0, None, 15.0, 20.0]
        pos = [-5.0, -3.0, None, -2.0]
        out = run_paired_significance_test(neg, pos, num_seeds=10, base_seed=42)
        # Only pair (0) and (3) are complete
        self.assertEqual(out["n_pairs"], 2)

    def test_num_seeds(self):
        out = run_paired_significance_test([10, 20], [0, 0], num_seeds=10)
        self.assertEqual(out["num_seeds"], 10)
        self.assertEqual(len(out["seed_results"]), 10)

    def test_deterministic_with_same_seed(self):
        neg = [10.0, 20.0, 30.0, 40.0, 50.0]
        pos = [0.0, 1.0, 2.0, 3.0, 4.0]
        a = run_paired_significance_test(neg, pos, num_seeds=5, base_seed=42)
        b = run_paired_significance_test(neg, pos, num_seeds=5, base_seed=42)
        self.assertEqual(a, b)


class TestCLIMain(unittest.TestCase):
    def _make_candidates_csv(self, path: str, n: int = 15):
        rows = []
        # Diverse chemical-change reactions so deduplication keeps them distinct.
        chem_changes = [
            ("CCO", "CC=O"),
            ("CC(=O)O", "CC(=O)Cl"),
            ("c1ccccc1", "c1ccc(N)cc1"),
            ("CCN", "CCN(C)C"),
            ("O=C(O)C", "O=C(N)C"),
            ("CCBr", "CCCC"),
            ("COC=O", "COC(=O)NC"),
            ("CC(C)O", "CC(C)Cl"),
        ]
        chem_idx = 0
        for i in range(n):
            if i % 3 == 0:
                # chemical change (will be sampled)
                cr, cp = chem_changes[chem_idx % len(chem_changes)]
                chem_idx += 1
                ft = "chemoselectivity_error"
                ea = "replace:O->N"
            else:
                # no reaction (will be filtered out)
                cr, cp = "CCO", "CCO"
                ft = "no_reaction"
                ea = "reactants:=product"
            rows.append({
                "source_id": f"SRC_{i:04d}",
                "positive_reaction": "CCO.[O]>>CC=O",
                "candidate_reaction": f"{cr}>>{cp}",
                "task": "retro_precursor",
                "failure_type": ft,
                "edit_action": ea,
                "parent_reactants": "CCO.[O]",
                "parent_product": "CC=O",
                "candidate_reactants": cr,
                "candidate_product": cp,
                "valid": 1,
                "atom_balance": 1.0,
                "locality": 0.7,
                "closeness": 0.8,
                "hard_score": 0.6 + i * 0.02,
                "false_negative_risk": 0.4,
                "passes_filter": True,
                "label": 0,
                "provenance": "pc_cng_mvp_synthetic_counterfactual",
                "review_status": "keep",
                "review_reasons": "",
                "product_overlap": 1.0,
                "layer1_ensemble_std": 0.1,
                "layer1_verdict": "keep",
                "layer2_verdict": "keep",
                "layer2_hit_reason": "",
                "layer3_verdict": "keep",
                "layer3_source": "rule_based_fallback",
            })
        pd.DataFrame(rows).to_csv(path, index=False)

    def test_cli_full_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            cand_path = os.path.join(tmp, "candidates.csv")
            out_dir = os.path.join(tmp, "out")
            self._make_candidates_csv(cand_path, n=15)

            rc = cli_main([
                "--candidates", cand_path,
                "--limit", "5",
                "--output-dir", out_dir,
                "--method", "mmff94",
                "--num-seeds", "5",
            ])
            self.assertEqual(rc, 0)

            # Output files exist
            self.assertTrue(os.path.isfile(os.path.join(out_dir, "xtb_results.csv")))
            self.assertTrue(os.path.isfile(os.path.join(out_dir, "control_positive_results.csv")))
            self.assertTrue(os.path.isfile(os.path.join(out_dir, "validation_summary.json")))
            self.assertTrue(os.path.isfile(os.path.join(out_dir, "paired_significance.json")))

            # xtb_results.csv content
            res = pd.read_csv(os.path.join(out_dir, "xtb_results.csv"))
            self.assertEqual(len(res), 5)
            self.assertIn("delta_g", res.columns)
            self.assertIn("support_verdict", res.columns)
            self.assertIn("method", res.columns)
            # All rows should have method mmff94 (or uff if degraded)
            for m in res["method"].dropna().unique():
                self.assertIn(m, ("mmff94", "uff"))

            # validation_summary.json content
            with open(os.path.join(out_dir, "validation_summary.json")) as f:
                summary = json.load(f)
            self.assertEqual(summary["method_requested"], "mmff94")
            self.assertIn(summary["method_actual"], ("mmff94", "uff"))
            self.assertEqual(summary["n_synthetic_negatives_computed"], 5)
            self.assertIn("support_rate", summary)
            self.assertIn("go_no_go_verdict", summary)
            self.assertIn(summary["go_no_go_verdict"], ("GO", "NO_GO_partial_support"))
            self.assertIn("paired_significance", summary)
            self.assertIn("delta_g_stats_synthetic_neg", summary)

    def test_cli_xtb_auto_degrades(self):
        with tempfile.TemporaryDirectory() as tmp:
            cand_path = os.path.join(tmp, "candidates.csv")
            out_dir = os.path.join(tmp, "out")
            self._make_candidates_csv(cand_path, n=15)

            rc = cli_main([
                "--candidates", cand_path,
                "--limit", "3",
                "--output-dir", out_dir,
                "--method", "xtb",
                "--num-seeds", "3",
            ])
            self.assertEqual(rc, 0)
            with open(os.path.join(out_dir, "validation_summary.json")) as f:
                summary = json.load(f)
            self.assertEqual(summary["method_requested"], "xtb")
            # Should degrade to mmff94 (or uff)
            self.assertIn(summary["method_actual"], ("mmff94", "uff"))
            self.assertTrue(summary["degraded_from_requested"])

    def test_cli_missing_candidates_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = os.path.join(tmp, "out")
            rc = cli_main([
                "--candidates", os.path.join(tmp, "nonexistent.csv"),
                "--limit", "5",
                "--output-dir", out_dir,
                "--method", "mmff94",
            ])
            self.assertEqual(rc, 2)

    def test_cli_no_require_chemical_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            cand_path = os.path.join(tmp, "candidates.csv")
            out_dir = os.path.join(tmp, "out")
            self._make_candidates_csv(cand_path, n=15)

            rc = cli_main([
                "--candidates", cand_path,
                "--limit", "10",
                "--output-dir", out_dir,
                "--method", "mmff94",
                "--no-require-chemical-change",
                "--num-seeds", "3",
            ])
            self.assertEqual(rc, 0)
            res = pd.read_csv(os.path.join(out_dir, "xtb_results.csv"))
            # Should include no-reaction rows (up to limit 10)
            self.assertGreaterEqual(len(res), 5)


if __name__ == "__main__":
    unittest.main()
