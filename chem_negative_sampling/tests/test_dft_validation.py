"""Unit tests for the P2-02 DFT validation runner (chemoselectivity_error subset).

Covers:
  * Module imports
  * CSV parsing and chemoselectivity_error filter
  * DeltaG calculation correctness (sum of components, formula)
  * Support decision logic (DeltaG > 0 => supports negative)
  * Output file structure (CSV + JSON)
  * Subprocess call to the dft venv (mocked with monkeypatch)
  * Method selection (xtb / mmff94 / orca dispatch)
  * Synthetic small end-to-end run
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from typing import Dict
from unittest import mock

import pandas as pd

from pc_cng.run_dft_validation import (
    BARRIER_CONSTANT,
    DEFAULT_DFT_PYTHON,
    DFT_SUPPORT_THRESHOLD,
    HARTREE_TO_KCAL_MOL,
    XTB_WORKER_SCRIPT,
    _build_summary,
    _collect_unique_component_smiles,
    compute_molecule_energy,
    compute_molecule_energy_mmff94,
    compute_molecule_energy_orca,
    compute_reaction_energy,
    embed_smiles_to_xyz,
    filter_by_failure_type,
    judge_support,
    main as cli_main,
    mol_to_xyz_block,
    parse_reaction_smiles,
    run_xtb_batch,
    sample_candidates,
    split_components,
    strip_atom_maps,
    support_reason,
    write_xtb_worker_script,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _make_candidates_csv(path: str, n_chem: int = 6, n_other: int = 4) -> None:
    """Write a synthetic high_confidence_negatives-style CSV.

    ``n_chem`` rows have ``failure_type=chemoselectivity_error`` with diverse
    chemical-change reactions; ``n_other`` rows have ``failure_type=no_reaction``
    so the filter is exercised.
    """
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
    rows = []
    for i in range(n_chem):
        cr, cp = chem_changes[i % len(chem_changes)]
        rows.append({
            "source_id": f"CHEM_{i:04d}",
            "positive_reaction": f"{cr}>>{cp}",
            "candidate_reaction": f"{cr}>>{cp}",
            "task": "forward_outcome",
            "failure_type": "chemoselectivity_error",
            "edit_action": "replace:Cl->Br",
            "parent_reactants": cr,
            "parent_product": cp,
            "candidate_reactants": cr,
            "candidate_product": cp,
            "valid": 1,
            "atom_balance": 1.0,
            "locality": 0.7,
            "closeness": 0.8,
            "hard_score": 0.7 + i * 0.02,
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
    for i in range(n_other):
        rows.append({
            "source_id": f"OTHER_{i:04d}",
            "positive_reaction": "CCO>>CCO",
            "candidate_reaction": "CCO>>CCO",
            "task": "retro_precursor",
            "failure_type": "no_reaction",
            "edit_action": "reactants:=product",
            "parent_reactants": "CCO",
            "parent_product": "CCO",
            "candidate_reactants": "CCO",
            "candidate_product": "CCO",
            "valid": 1,
            "atom_balance": 1.0,
            "locality": 0.7,
            "closeness": 0.8,
            "hard_score": 0.5,
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


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
class TestModuleImports(unittest.TestCase):
    """Verify the module and key symbols import cleanly."""

    def test_module_imports(self):
        import pc_cng.run_dft_validation as mod
        self.assertTrue(hasattr(mod, "main"))
        self.assertTrue(hasattr(mod, "compute_reaction_energy"))
        self.assertTrue(hasattr(mod, "judge_support"))

    def test_constants_present(self):
        self.assertEqual(DFT_SUPPORT_THRESHOLD, 0.0)
        self.assertEqual(BARRIER_CONSTANT, 5.0)
        self.assertGreater(HARTREE_TO_KCAL_MOL, 600.0)
        self.assertIsInstance(DEFAULT_DFT_PYTHON, str)
        self.assertIn("dft", DEFAULT_DFT_PYTHON)

    def test_xtb_worker_script_constant(self):
        self.assertIsInstance(XTB_WORKER_SCRIPT, str)
        self.assertIn("from xtb.ase.calculator import XTB", XTB_WORKER_SCRIPT)
        self.assertIn("def main", XTB_WORKER_SCRIPT)


class TestParseReactionSmiles(unittest.TestCase):
    def test_simple_reaction(self):
        r, p = parse_reaction_smiles("CCO.[O]>>CC=O")
        self.assertEqual(r, "CCO.[O]")
        self.assertEqual(p, "CC=O")

    def test_no_arrow(self):
        r, p = parse_reaction_smiles("CCO")
        self.assertEqual(r, "")
        self.assertEqual(p, "")

    def test_multiple_arrows(self):
        r, p = parse_reaction_smiles("A>>B>>C")
        self.assertEqual(r, "")
        self.assertEqual(p, "")

    def test_empty(self):
        self.assertEqual(parse_reaction_smiles(""), ("", ""))


class TestStripAtomMaps(unittest.TestCase):
    def test_atom_maps_removed(self):
        out = strip_atom_maps("[CH3:1][O:2][CH3:3]")
        self.assertNotIn(":", out)
        from rdkit import Chem
        self.assertIsNotNone(Chem.MolFromSmiles(out))

    def test_no_maps(self):
        self.assertEqual(strip_atom_maps("CCO"), "CCO")

    def test_empty(self):
        self.assertEqual(strip_atom_maps(""), "")

    def test_invalid(self):
        self.assertEqual(strip_atom_maps("not_a_smiles!!!"), "")


class TestSplitComponents(unittest.TestCase):
    def test_multi_component(self):
        self.assertEqual(split_components("A.B.C"), ["A", "B", "C"])

    def test_single(self):
        self.assertEqual(split_components("CCO"), ["CCO"])

    def test_empty_components_dropped(self):
        self.assertEqual(split_components("A..B."), ["A", "B"])

    def test_non_string(self):
        self.assertEqual(split_components(None), [])


class TestMolToXYZ(unittest.TestCase):
    def test_xyz_block_format(self):
        from rdkit import Chem
        from rdkit.Chem import AllChem
        mol = Chem.MolFromSmiles("CCO")
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        block = mol_to_xyz_block(mol)
        lines = block.strip().split("\n")
        self.assertEqual(int(lines[0]), mol.GetNumAtoms())
        self.assertEqual(lines[1], "rdkit embedded")
        # First atom symbol should be present
        parts = lines[2].split()
        self.assertEqual(len(parts), 4)


class TestEmbedSmilesToXYZ(unittest.TestCase):
    def test_ethanol_xyz(self):
        xyz, status = embed_smiles_to_xyz("CCO", seed=42)
        self.assertEqual(status, "ok")
        self.assertIsNotNone(xyz)
        self.assertIn("C", xyz)
        # First line is the atom count
        n_atoms = int(xyz.strip().split("\n")[0])
        self.assertGreater(n_atoms, 0)

    def test_atom_mapped_smiles(self):
        xyz, status = embed_smiles_to_xyz("[CH3:1][O:2][CH3:3]", seed=42)
        self.assertEqual(status, "ok")
        self.assertIsNotNone(xyz)

    def test_empty(self):
        xyz, status = embed_smiles_to_xyz("")
        self.assertIsNone(xyz)
        self.assertEqual(status, "empty")

    def test_invalid_smiles(self):
        xyz, status = embed_smiles_to_xyz("not_a_smiles!!!")
        self.assertIsNone(xyz)
        self.assertEqual(status, "parse_error")


class TestFilterByFailureType(unittest.TestCase):
    def test_filter_chemoselectivity(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "cands.csv")
            _make_candidates_csv(csv_path, n_chem=6, n_other=4)
            df = pd.read_csv(csv_path)
            chem = filter_by_failure_type(df, "chemoselectivity_error")
            self.assertEqual(len(chem), 6)
            for ft in chem["failure_type"]:
                self.assertEqual(ft, "chemoselectivity_error")

    def test_filter_no_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "cands.csv")
            _make_candidates_csv(csv_path, n_chem=3, n_other=2)
            df = pd.read_csv(csv_path)
            empty = filter_by_failure_type(df, "nonexistent_type")
            self.assertEqual(len(empty), 0)

    def test_missing_column_raises(self):
        df = pd.DataFrame({"not_failure_type": ["x"]})
        with self.assertRaises(KeyError):
            filter_by_failure_type(df, "chemoselectivity_error")


class TestSampleCandidates(unittest.TestCase):
    def test_limit_caps_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "cands.csv")
            _make_candidates_csv(csv_path, n_chem=6, n_other=0)
            df = pd.read_csv(csv_path)
            chem = filter_by_failure_type(df, "chemoselectivity_error")
            out = sample_candidates(chem, limit=3)
            self.assertEqual(len(out), 3)

    def test_sorted_by_hard_score_desc(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "cands.csv")
            _make_candidates_csv(csv_path, n_chem=6, n_other=0)
            df = pd.read_csv(csv_path)
            chem = filter_by_failure_type(df, "chemoselectivity_error")
            out = sample_candidates(chem, limit=10)
            self.assertEqual(len(out), 6)
            self.assertGreaterEqual(out["hard_score"].iloc[0], out["hard_score"].iloc[-1])

    def test_require_chemical_change_filters_no_reaction(self):
        df = pd.DataFrame({
            "candidate_reactants": ["CCO", "CCO"],
            "candidate_product": ["CC=O", "CCO"],  # one chemical change, one no-reaction
            "hard_score": [0.7, 0.5],
        })
        out = sample_candidates(df, limit=10, require_chemical_change=True)
        self.assertEqual(len(out), 1)

    def test_deduplicate(self):
        df = pd.DataFrame({
            "candidate_reactants": ["CCO", "CCO", "CCN"],
            "candidate_product": ["CC=O", "CC=O", "CCN(C)C"],
            "hard_score": [0.7, 0.6, 0.5],
        })
        out = sample_candidates(df, limit=10, require_chemical_change=True, deduplicate=True)
        self.assertEqual(len(out), 2)


class TestComputeMoleculeEnergyMMFF94(unittest.TestCase):
    def test_ethanol(self):
        res = compute_molecule_energy_mmff94("CCO", seed=42)
        self.assertEqual(res["status"], "ok")
        self.assertIsNotNone(res["energy_kcal_per_mol"])
        self.assertEqual(res["method"], "mmff94")

    def test_empty(self):
        res = compute_molecule_energy_mmff94("")
        self.assertEqual(res["status"], "empty")
        self.assertIsNone(res["energy_kcal_per_mol"])

    def test_invalid(self):
        res = compute_molecule_energy_mmff94("not_a_smiles!!!")
        self.assertEqual(res["status"], "parse_error")


class TestComputeMoleculeEnergyDispatch(unittest.TestCase):
    def test_xtb_uses_cache(self):
        cache = {"CCO": {"energy_kcal_per_mol": -10.5, "status": "ok", "method": "xtb"}}
        res = compute_molecule_energy("CCO", method="xtb", xtb_cache=cache)
        self.assertEqual(res["status"], "ok")
        self.assertAlmostEqual(res["energy_kcal_per_mol"], -10.5)
        self.assertEqual(res["method"], "xtb")

    def test_xtb_strips_atom_maps_before_cache_lookup(self):
        cache = {"CCO": {"energy_kcal_per_mol": -10.5, "status": "ok", "method": "xtb"}}
        res = compute_molecule_energy("[CH3:1][CH2:2][OH:3]", method="xtb", xtb_cache=cache)
        # Atom maps stripped to canonical "CCO" -> cache hit
        self.assertEqual(res["status"], "ok")
        self.assertAlmostEqual(res["energy_kcal_per_mol"], -10.5)

    def test_xtb_no_cache(self):
        res = compute_molecule_energy("CCO", method="xtb", xtb_cache=None)
        self.assertEqual(res["status"], "no_xtb_cache")
        self.assertIsNone(res["energy_kcal_per_mol"])

    def test_xtb_not_in_cache(self):
        cache = {"CC": {"energy_kcal_per_mol": -5.0, "status": "ok"}}
        res = compute_molecule_energy("CCO", method="xtb", xtb_cache=cache)
        self.assertEqual(res["status"], "not_in_cache")
        self.assertIsNone(res["energy_kcal_per_mol"])

    def test_mmff94_dispatch(self):
        res = compute_molecule_energy("CCO", method="mmff94")
        self.assertEqual(res["status"], "ok")
        self.assertEqual(res["method"], "mmff94")

    def test_orca_dispatch_returns_stub(self):
        res = compute_molecule_energy_orca("CCO")
        self.assertEqual(res["status"], "not_implemented")
        self.assertEqual(res["method"], "orca")
        self.assertIsNone(res["energy_kcal_per_mol"])

    def test_unknown_method(self):
        res = compute_molecule_energy("CCO", method="bogus")
        self.assertIn("unknown_method", res["status"])

    def test_empty_smiles(self):
        res = compute_molecule_energy("", method="xtb")
        self.assertEqual(res["status"], "empty")


class TestComputeReactionEnergy(unittest.TestCase):
    def test_delta_g_formula(self):
        # reactants energy = 10.0, products energy = 25.0 -> delta_g = 15.0
        cache = {
            "CCO": {"energy_kcal_per_mol": 10.0, "status": "ok", "method": "xtb"},
            "CC=O": {"energy_kcal_per_mol": 25.0, "status": "ok", "method": "xtb"},
        }
        res = compute_reaction_energy("CCO", "CC=O", method="xtb", xtb_cache=cache)
        self.assertEqual(res["status"], "ok")
        self.assertAlmostEqual(res["reactant_energy"], 10.0)
        self.assertAlmostEqual(res["product_energy"], 25.0)
        self.assertAlmostEqual(res["delta_g"], 15.0)
        self.assertAlmostEqual(res["barrier_estimate"], 15.0 + BARRIER_CONSTANT)

    def test_multi_component_sum(self):
        # reactants C.N -> C.energy + N.energy = 5 + 7 = 12
        # product O.energy = 20 -> delta_g = 8
        cache = {
            "C": {"energy_kcal_per_mol": 5.0, "status": "ok"},
            "N": {"energy_kcal_per_mol": 7.0, "status": "ok"},
            "O": {"energy_kcal_per_mol": 20.0, "status": "ok"},
        }
        res = compute_reaction_energy("C.N", "O", method="xtb", xtb_cache=cache)
        self.assertEqual(res["status"], "ok")
        self.assertAlmostEqual(res["reactant_energy"], 12.0)
        self.assertAlmostEqual(res["product_energy"], 20.0)
        self.assertAlmostEqual(res["delta_g"], 8.0)

    def test_reactant_failure_propagates(self):
        cache = {
            "CCbad!!!": {"energy_kcal_per_mol": None, "status": "xtb_failed:Err"},
            "CC=O": {"energy_kcal_per_mol": 25.0, "status": "ok"},
        }
        # After strip_atom_maps the SMILES is unchanged because it can't be parsed.
        # compute_molecule_energy uses the stripped form for the cache lookup, so
        # the bad SMILES lookup will miss the cache entirely. Use a SMILES that
        # is parseable but explicitly marked as failed in the cache.
        cache["CCO"] = {"energy_kcal_per_mol": None, "status": "xtb_failed:Err"}
        res = compute_reaction_energy("CCO", "CC=O", method="xtb", xtb_cache=cache)
        self.assertIsNone(res["delta_g"])
        self.assertIn("reactant_failed", res["status"])

    def test_product_failure_propagates(self):
        cache = {
            "CCO": {"energy_kcal_per_mol": 10.0, "status": "ok"},
            "CC=O": {"energy_kcal_per_mol": None, "status": "xtb_failed:Err"},
        }
        res = compute_reaction_energy("CCO", "CC=O", method="xtb", xtb_cache=cache)
        self.assertIsNone(res["delta_g"])
        self.assertIn("product_failed", res["status"])

    def test_mmff94_end_to_end(self):
        # Methane -> methane should give DeltaG = 0
        res = compute_reaction_energy("C", "C", method="mmff94", seed=42)
        self.assertEqual(res["status"], "ok")
        self.assertAlmostEqual(res["delta_g"], 0.0, places=4)


class TestJudgeSupport(unittest.TestCase):
    def test_supported_positive_delta_g(self):
        # DeltaG > 0 -> supported
        self.assertEqual(judge_support(0.1, 5.0), "supported")
        self.assertEqual(judge_support(15.0, 20.0), "supported")
        self.assertEqual(judge_support(100.0, 105.0), "supported")

    def test_not_supported_negative_delta_g(self):
        # DeltaG <= 0 -> not supported
        self.assertEqual(judge_support(-0.1, 5.0), "not_supported")
        self.assertEqual(judge_support(-50.0, 55.0), "not_supported")

    def test_boundary_zero(self):
        # DeltaG == 0 -> not supported (strict >)
        self.assertEqual(judge_support(0.0, 5.0), "not_supported")

    def test_inconclusive(self):
        self.assertEqual(judge_support(None, None), "inconclusive")
        self.assertEqual(judge_support(None, 10.0), "inconclusive")

    def test_threshold_is_zero(self):
        # P2-02 threshold is DeltaG > 0 (NOT > 5 like P1-10)
        self.assertEqual(DFT_SUPPORT_THRESHOLD, 0.0)
        self.assertEqual(judge_support(1.0, 6.0), "supported")
        # The barrier signal is intentionally ignored in P2-02
        self.assertEqual(judge_support(-1.0, 100.0), "not_supported")


class TestSupportReason(unittest.TestCase):
    def test_supported_reason(self):
        r = support_reason(15.0, 20.0)
        self.assertIn("delta_g > 0", r)
        self.assertIn("supports chemoselectivity_error", r)

    def test_not_supported_reason(self):
        r = support_reason(-5.0, 10.0)
        self.assertIn("not supported", r)

    def test_inconclusive_reason(self):
        r = support_reason(None, None)
        self.assertIn("inconclusive", r)


class TestCollectUniqueComponentSmiles(unittest.TestCase):
    def test_dedup_and_split(self):
        df = pd.DataFrame({
            "candidate_reactants": ["CCO.[O]", "CCO"],
            "candidate_product": ["CC=O", "CC=O"],
        })
        out = _collect_unique_component_smiles(df)
        # "CCO", atomic oxygen "[O]" (canonical form of [O] — note the
        # brackets are preserved because they signal "no implicit H"),
        # and "CC=O" form the unique set.
        self.assertIn("CCO", out)
        self.assertIn("CC=O", out)
        # Atomic oxygen keeps its brackets in canonical SMILES
        self.assertIn("[O]", out)
        # Each entry is unique
        self.assertEqual(len(out), len(set(out)))

    def test_atom_maps_stripped(self):
        df = pd.DataFrame({
            "candidate_reactants": ["[CH3:1][O:2][CH3:3]"],
            "candidate_product": ["CCO"],
        })
        out = _collect_unique_component_smiles(df)
        # The first SMILES strips to "COC" (dimethyl ether)
        self.assertTrue(any(s in out for s in ["COC", "COC"]))


class TestWriteXtbWorkerScript(unittest.TestCase):
    def test_worker_script_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "worker.py")
            write_xtb_worker_script(path)
            self.assertTrue(os.path.isfile(path))
            with open(path) as f:
                content = f.read()
            self.assertIn("from xtb.ase.calculator import XTB", content)
            self.assertIn("def main", content)
            self.assertIn("smiles_to_xyz", content)


class TestRunXtbBatchMocked(unittest.TestCase):
    """Mock subprocess.run to verify the dft-python subprocess contract."""

    def _fake_worker(self, in_path: str, out_path: str) -> None:
        """Write a fake output JSON mimicking the real worker's contract."""
        with open(in_path) as f:
            cfg = json.load(f)
        results = {}
        for smiles, _xyz in cfg["smiles_to_xyz"].items():
            results[smiles] = {
                "energy_hartree": -0.5,
                "energy_kcal_per_mol": -0.5 * HARTREE_TO_KCAL_MOL,
                "status": "ok",
                "method": "xtb",
            }
        with open(out_path, "w") as f:
            json.dump({"results": results, "worker_error": None}, f)

    def test_batch_returns_one_entry_per_smiles(self):
        smiles_list = ["CCO", "CC=O", "CCN"]
        with tempfile.TemporaryDirectory() as tmp:
            worker_path = os.path.join(tmp, "worker.py")
            write_xtb_worker_script(worker_path)

            def fake_run(cmd, **kwargs):
                # cmd = [dft_python, worker_path, in_path, out_path]
                in_path = cmd[2]
                out_path = cmd[3]
                self._fake_worker(in_path, out_path)
                return subprocess.CompletedProcess(cmd, returncode=0)

            with mock.patch("pc_cng.run_dft_validation.subprocess.run", side_effect=fake_run) as m_run:
                out = run_xtb_batch(
                    smiles_list,
                    dft_python="/fake/dft/python",
                    worker_script_path=worker_path,
                )
        self.assertEqual(len(out), 3)
        for s in smiles_list:
            self.assertIn(s, out)
            self.assertEqual(out[s]["status"], "ok")
            self.assertIsNotNone(out[s]["energy_kcal_per_mol"])
            self.assertEqual(out[s]["method"], "xtb")
        self.assertEqual(m_run.call_count, 1)

    def test_batch_clears_cuda_visible_devices(self):
        """CPU-only: the subprocess env must unset CUDA_VISIBLE_DEVICES."""
        with tempfile.TemporaryDirectory() as tmp:
            worker_path = os.path.join(tmp, "worker.py")
            write_xtb_worker_script(worker_path)
            captured_env = {}

            def fake_run(cmd, **kwargs):
                in_path = cmd[2]
                out_path = cmd[3]
                self._fake_worker(in_path, out_path)
                captured_env.update(kwargs.get("env", {}))
                return subprocess.CompletedProcess(cmd, returncode=0)

            with mock.patch("pc_cng.run_dft_validation.subprocess.run", side_effect=fake_run):
                run_xtb_batch(
                    ["CCO"],
                    dft_python="/fake/dft/python",
                    worker_script_path=worker_path,
                )
        self.assertIn("CUDA_VISIBLE_DEVICES", captured_env)
        self.assertEqual(captured_env["CUDA_VISIBLE_DEVICES"], "")

    def test_batch_empty_list_returns_empty(self):
        out = run_xtb_batch([], dft_python="/fake/dft/python")
        self.assertEqual(out, {})

    def test_batch_missing_dft_python(self):
        out = run_xtb_batch(
            ["CCO"],
            dft_python="/definitely/not/a/real/python",
            worker_script_path="/tmp/unused.py",
        )
        self.assertEqual(len(out), 1)
        self.assertEqual(out["CCO"]["status"], "dft_python_missing")
        self.assertIsNone(out["CCO"]["energy_kcal_per_mol"])

    def test_batch_all_embed_failures(self):
        # Invalid SMILES -> embed failure -> returned in results
        out = run_xtb_batch(
            ["not_a_smiles!!!"],
            dft_python="/fake/dft/python",
            worker_script_path="/tmp/unused.py",
        )
        self.assertEqual(len(out), 1)
        self.assertIn("embed_failed", next(iter(out.values()))["status"])

    def test_batch_timeout(self):
        with tempfile.TemporaryDirectory() as tmp:
            worker_path = os.path.join(tmp, "worker.py")
            write_xtb_worker_script(worker_path)

            def fake_run(cmd, **kwargs):
                raise subprocess.TimeoutExpired(cmd, 1.0)

            with mock.patch("pc_cng.run_dft_validation.subprocess.run", side_effect=fake_run):
                out = run_xtb_batch(
                    ["CCO"],
                    dft_python="/fake/dft/python",
                    worker_script_path=worker_path,
                    timeout_per_molecule=0.01,
                )
        self.assertEqual(out["CCO"]["status"], "xtb_timeout")


class TestBuildSummary(unittest.TestCase):
    def test_summary_structure(self):
        rows = [
            {
                "delta_g": 10.0,
                "support_verdict": "supported",
            },
            {
                "delta_g": -5.0,
                "support_verdict": "not_supported",
            },
            {
                "delta_g": None,
                "support_verdict": "inconclusive",
            },
        ]
        s = _build_summary(
            method_requested="xtb",
            method_actual="xtb",
            candidates_path="/tmp/c.csv",
            failure_type="chemoselectivity_error",
            total_loaded=100,
            total_filtered=10,
            n_computed=3,
            rows=rows,
            go_no_go_threshold=0.6,
            seed=42,
            notes=["test note"],
        )
        self.assertEqual(s["method_requested"], "xtb")
        self.assertEqual(s["method_actual"], "xtb")
        self.assertFalse(s["degraded_from_requested"])
        self.assertEqual(s["failure_type"], "chemoselectivity_error")
        self.assertEqual(s["n_computed"], 3)
        self.assertEqual(s["n_supported"], 1)
        self.assertEqual(s["n_not_supported"], 1)
        self.assertEqual(s["n_inconclusive"], 1)
        self.assertAlmostEqual(s["support_rate"], 1.0 / 3.0)
        self.assertEqual(s["go_no_go_verdict"], "NO_GO_partial_support")
        self.assertAlmostEqual(s["mean_dg"], 2.5)  # mean(10, -5)
        self.assertEqual(s["notes"], ["test note"])
        self.assertIn("timestamp", s)
        self.assertIn("num_seeds_note", s)

    def test_summary_go_when_high_support(self):
        rows = [{"delta_g": 10.0, "support_verdict": "supported"}] * 10
        s = _build_summary(
            method_requested="xtb",
            method_actual="xtb",
            candidates_path="/tmp/c.csv",
            failure_type="chemoselectivity_error",
            total_loaded=100,
            total_filtered=10,
            n_computed=10,
            rows=rows,
            go_no_go_threshold=0.6,
            seed=42,
        )
        self.assertEqual(s["go_no_go_verdict"], "GO")
        self.assertAlmostEqual(s["support_rate"], 1.0)

    def test_summary_handles_empty(self):
        s = _build_summary(
            method_requested="xtb",
            method_actual="xtb",
            candidates_path="/tmp/c.csv",
            failure_type="chemoselectivity_error",
            total_loaded=0,
            total_filtered=0,
            n_computed=0,
            rows=[],
            go_no_go_threshold=0.6,
            seed=42,
        )
        self.assertEqual(s["n_computed"], 0)
        self.assertEqual(s["support_rate"], 0.0)
        self.assertIsNone(s["mean_dg"])
        self.assertEqual(s["go_no_go_verdict"], "NO_GO_partial_support")

    def test_summary_accepts_dG_reaction_key(self):
        """main() writes ``dG_reaction`` in the row dict (CSV column name),
        not ``delta_g``. The summary builder must accept both keys."""
        rows = [
            {"dG_reaction": 10.0, "support_verdict": "supported"},
            {"dG_reaction": -5.0, "support_verdict": "not_supported"},
        ]
        s = _build_summary(
            method_requested="xtb",
            method_actual="xtb",
            candidates_path="/tmp/c.csv",
            failure_type="chemoselectivity_error",
            total_loaded=2,
            total_filtered=2,
            n_computed=2,
            rows=rows,
            go_no_go_threshold=0.6,
            seed=42,
        )
        self.assertAlmostEqual(s["mean_dg"], 2.5)
        self.assertAlmostEqual(s["min_dg"], -5.0)
        self.assertAlmostEqual(s["max_dg"], 10.0)
        self.assertEqual(s["n_supported"], 1)


class TestCLIMMFF94EndToEnd(unittest.TestCase):
    """Run the CLI end-to-end with method=mmff94 (no subprocess required)."""

    def test_cli_mmff94_full_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            cand_path = os.path.join(tmp, "candidates.csv")
            out_dir = os.path.join(tmp, "out")
            _make_candidates_csv(cand_path, n_chem=5, n_other=2)

            rc = cli_main([
                "--candidates", cand_path,
                "--failure-type", "chemoselectivity_error",
                "--limit", "3",
                "--method", "mmff94",
                "--output-dir", out_dir,
                "--dft-python", "/fake/dft/python",  # not used for mmff94
            ])
            self.assertEqual(rc, 0)

            # Output files
            self.assertTrue(os.path.isfile(os.path.join(out_dir, "per_candidate_results.csv")))
            self.assertTrue(os.path.isfile(os.path.join(out_dir, "dft_validation_summary.json")))
            self.assertTrue(os.path.isdir(os.path.join(out_dir, "detailed_logs")))

            # per_candidate_results.csv structure
            res = pd.read_csv(os.path.join(out_dir, "per_candidate_results.csv"))
            self.assertEqual(len(res), 3)
            expected_cols = {
                "source_id", "failure_type", "candidate_reactants", "candidate_product",
                "dG_reactants", "dG_products", "dG_reaction", "barrier_estimate",
                "method", "status", "supports_negative", "support_verdict",
            }
            self.assertTrue(expected_cols.issubset(set(res.columns)))

            # All rows use mmff94 method
            for m in res["method"].dropna().unique():
                self.assertIn(m, ("mmff94", "uff"))

            # summary.json structure
            with open(os.path.join(out_dir, "dft_validation_summary.json")) as f:
                summary = json.load(f)
            self.assertEqual(summary["method_requested"], "mmff94")
            self.assertEqual(summary["method_actual"], "mmff94")
            self.assertEqual(summary["failure_type"], "chemoselectivity_error")
            self.assertEqual(summary["n_computed"], 3)
            self.assertIn("support_rate", summary)
            self.assertIn("go_no_go_verdict", summary)
            self.assertIn(summary["go_no_go_verdict"], ("GO", "NO_GO_partial_support"))
            self.assertIn("mean_dg", summary)
            self.assertIn("std_dg", summary)

    def test_cli_xtb_with_mocked_subprocess(self):
        """CLI with method=xtb but the dft subprocess mocked."""
        with tempfile.TemporaryDirectory() as tmp:
            cand_path = os.path.join(tmp, "candidates.csv")
            out_dir = os.path.join(tmp, "out")
            _make_candidates_csv(cand_path, n_chem=4, n_other=2)

            def fake_run(cmd, **kwargs):
                # cmd = [dft_python, worker_path, in_path, out_path]
                in_path = cmd[2]
                out_path = cmd[3]
                with open(in_path) as f:
                    cfg = json.load(f)
                results = {}
                for smiles in cfg["smiles_to_xyz"]:
                    results[smiles] = {
                        "energy_hartree": -0.5,
                        "energy_kcal_per_mol": -0.5 * HARTREE_TO_KCAL_MOL,
                        "status": "ok",
                        "method": "xtb",
                    }
                with open(out_path, "w") as f:
                    json.dump({"results": results, "worker_error": None}, f)
                return subprocess.CompletedProcess(cmd, returncode=0)

            with mock.patch("pc_cng.run_dft_validation.subprocess.run", side_effect=fake_run):
                rc = cli_main([
                    "--candidates", cand_path,
                    "--failure-type", "chemoselectivity_error",
                    "--limit", "3",
                    "--method", "xtb",
                    "--output-dir", out_dir,
                    "--dft-python", "/fake/dft/python",
                ])
            self.assertEqual(rc, 0)

            res = pd.read_csv(os.path.join(out_dir, "per_candidate_results.csv"))
            self.assertEqual(len(res), 3)
            # All entries should be xtb method
            for m in res["method"].dropna().unique():
                self.assertEqual(m, "xtb")
            # Every row should have a valid delta_g (mocked uniform energy)
            self.assertTrue(res["dG_reaction"].notna().all())
            # delta_g = E_products - E_reactants; with uniform -0.5 Hartree:
            # multi-component reactions will have nonzero delta_g depending on component counts
            # but same-component reactions (A -> A) would be 0. Here all candidates have
            # different reactants/products, so delta_g != 0 in general.

            with open(os.path.join(out_dir, "dft_validation_summary.json")) as f:
                summary = json.load(f)
            self.assertEqual(summary["method_requested"], "xtb")
            self.assertEqual(summary["method_actual"], "xtb")
            self.assertFalse(summary["degraded_from_requested"])

    def test_cli_orca_stub(self):
        with tempfile.TemporaryDirectory() as tmp:
            cand_path = os.path.join(tmp, "candidates.csv")
            out_dir = os.path.join(tmp, "out")
            _make_candidates_csv(cand_path, n_chem=3, n_other=0)

            rc = cli_main([
                "--candidates", cand_path,
                "--failure-type", "chemoselectivity_error",
                "--limit", "2",
                "--method", "orca",
                "--output-dir", out_dir,
                "--dft-python", "/fake/dft/python",
            ])
            self.assertEqual(rc, 0)
            res = pd.read_csv(os.path.join(out_dir, "per_candidate_results.csv"))
            self.assertEqual(len(res), 2)
            # All energies should be None (stub)
            self.assertTrue(res["dG_reaction"].isna().all())
            for v in res["support_verdict"]:
                self.assertEqual(v, "inconclusive")
            with open(os.path.join(out_dir, "dft_validation_summary.json")) as f:
                summary = json.load(f)
            self.assertEqual(summary["method_actual"], "orca")
            self.assertEqual(summary["n_inconclusive"], 2)
            self.assertEqual(summary["support_rate"], 0.0)
            self.assertIn("ORCA dispatch is a stub", summary["notes"][0])

    def test_cli_missing_candidates_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = os.path.join(tmp, "out")
            rc = cli_main([
                "--candidates", os.path.join(tmp, "nonexistent.csv"),
                "--failure-type", "chemoselectivity_error",
                "--limit", "5",
                "--method", "mmff94",
                "--output-dir", out_dir,
            ])
            self.assertEqual(rc, 2)

    def test_cli_missing_failure_type_column(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad_csv = os.path.join(tmp, "bad.csv")
            pd.DataFrame({"not_failure_type": ["x"], "candidate_reactants": ["CCO"],
                          "candidate_product": ["CC=O"]}).to_csv(bad_csv, index=False)
            out_dir = os.path.join(tmp, "out")
            rc = cli_main([
                "--candidates", bad_csv,
                "--failure-type", "chemoselectivity_error",
                "--limit", "5",
                "--method", "mmff94",
                "--output-dir", out_dir,
            ])
            self.assertEqual(rc, 3)

    def test_cli_clears_cuda_visible_devices(self):
        """CPU-only: CLI must export CUDA_VISIBLE_DEVICES=''."""
        with tempfile.TemporaryDirectory() as tmp:
            cand_path = os.path.join(tmp, "candidates.csv")
            out_dir = os.path.join(tmp, "out")
            _make_candidates_csv(cand_path, n_chem=2, n_other=0)
            old = os.environ.get("CUDA_VISIBLE_DEVICES")
            try:
                os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
                cli_main([
                    "--candidates", cand_path,
                    "--failure-type", "chemoselectivity_error",
                    "--limit", "1",
                    "--method", "mmff94",
                    "--output-dir", out_dir,
                    "--dft-python", "/fake/dft/python",
                ])
                self.assertEqual(os.environ.get("CUDA_VISIBLE_DEVICES"), "")
            finally:
                if old is None:
                    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                else:
                    os.environ["CUDA_VISIBLE_DEVICES"] = old


if __name__ == "__main__":
    unittest.main()
